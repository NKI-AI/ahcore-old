# encoding: utf-8
from pathlib import Path
from types import TracebackType
from typing import Generator, Literal, Optional, Type

from pydantic import AfterValidator, BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing_extensions import Annotated

from ahcore.utils.database_models import Base, Image, Manifest, Patient, Split, SplitDefinitions
from ahcore.utils.io import get_logger


# Custom exceptions
class RecordNotFoundError(Exception):
    pass


def is_positive(v: int | float) -> int | float:
    assert v > 0, f"{v} is not a positive a positive {type(v)}"
    return v


PositiveInt = Annotated[int, AfterValidator(is_positive)]
PositiveFloat = Annotated[float, AfterValidator(is_positive)]


class ImageMetadata(BaseModel):
    class Config:
        allow_mutation = False

    filename: Path
    height: PositiveInt
    width: PositiveInt
    mpp: PositiveFloat


def open_db(database_uri: str):
    engine = create_engine(database_uri)
    create_tables(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def create_tables(engine):
    Base.metadata.create_all(bind=engine)


def insert_record(session, record):
    session.add(record)


def get_or_create_patient(session, patient_code, manifest):
    existing_patient = session.query(Patient).filter_by(patient_code=patient_code).first()
    if not existing_patient:
        patient = Patient(patient_code=patient_code, manifest=manifest)
        session.add(patient)
        session.flush()
        return patient
    return existing_patient


def fetch_image_metadata(image: Image) -> ImageMetadata:
    """Extract metadata from an Image object."""
    return ImageMetadata(
        filename=Path(image.filename), height=int(image.height), width=int(image.width), mpp=float(image.mpp)
    )


class DataManager:
    def __init__(self, database_uri: str) -> None:
        self._database_uri = database_uri
        self.__session = None
        self._logger = get_logger(type(self).__name__)

    @property
    def _session(self):
        if self.__session is None:
            self.__session = open_db(self._database_uri)
        return self.__session

    @staticmethod
    def _ensure_record(record: Type[Base], description: str) -> None:
        """Raises an error if the record is None."""
        if not record:
            raise RecordNotFoundError(f"{description} not found.")

    def get_records_by_split(
        self, manifest_name: str, split_version: str, split_category: Optional[str] = None
    ) -> Generator[Patient, None, None]:
        manifest = self._session.query(Manifest).filter_by(name=manifest_name).first()  # type: ignore
        self._ensure_record(manifest, f"Manifest with name {manifest_name}")

        split_definition = self._session.query(SplitDefinitions).filter_by(version=split_version).first()
        self._ensure_record(split_definition, f"Split definition with version {split_version}")

        query = (
            self._session.query(Patient)
            .join(Split)
            .filter(Patient.manifest_id == manifest.id, Split.split_definition_id == split_definition.id)
        )

        if split_category is not None:
            query = query.filter(Split.category == split_category)

        patients = query.all()

        self._logger.info(
            f"Found {len(patients)} patients for split {split_category if split_category else 'all categories'}"
        )
        for patient in patients:
            yield patient

    def get_image_metadata_by_split(
        self, manifest_name: str, split_version: str, split_category: Optional[str] = None
    ) -> Generator[ImageMetadata, None, None]:
        """
        Yields the metadata of images for a given manifest name, split version, and optional split category.

        Parameters
        ----------
        manifest_name : str
            The name of the manifest.
        split_version : str
            The version of the split.
        split_category : Optional[str], default=None
            The category of the split (e.g., "fit", "validate", "test").

        Yields
        -------
        ImageMetadata
            The metadata of the image.
        """
        for patient in self.get_records_by_split(manifest_name, split_version, split_category):
            for image in patient.images:
                yield fetch_image_metadata(image)

    def get_image_metadata_by_patient(self, patient_code: str) -> list[ImageMetadata]:
        """
        Fetch the metadata for the images associated with a specific patient.

        Parameters
        ----------
        patient_code : str
            The unique code of the patient.

        Returns
        -------
        list[ImageData]
            A list of metadata for all images associated with the patient.
        """
        patient = self._session.query(Patient).filter_by(patient_code=patient_code).first()  # type: ignore
        self._ensure_record(patient, f"Patient with code {patient_code} not found")

        return [fetch_image_metadata(image) for image in patient.images]

    def get_image_by_filename(self, filename: str) -> Type[Image]:
        """
        Fetch the metadata for an image based on its filename.

        Parameters
        ----------
        filename : str
            The filename of the image.

        Returns
        -------
        Image
            The image from the database.
        """
        image = self._session.query(Image).filter_by(filename=filename).first()
        self._ensure_record(image, f"Image with filename {filename} not found")
        assert image
        return image

    def get_image_metadata_by_id(self, image_id: int) -> ImageMetadata:
        """
        Fetch the metadata for an image based on its ID.

        Parameters
        ----------
        image_id : int
            The ID of the image.

        Returns
        -------
        ImageMetadata
            Metadata of the image.
        """
        image = self._session.query(Image).filter_by(id=image_id).first()
        self._ensure_record(image, f"No image found with ID {image_id}")
        return fetch_image_metadata(image)

    def __enter__(self) -> "DataManager":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        if self._session is not None:
            self.close()
        return False

    def close(self):
        if self.__session is not None:
            self.__session.close()
            self.__session = None
