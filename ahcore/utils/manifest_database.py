# encoding: utf-8
from pathlib import Path
from types import TracebackType
from typing import Generator, NamedTuple, Optional, Type

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ahcore.utils.database_models import Base, Image, Manifest, Patient, Split, SplitDefinitions
from ahcore.utils.io import get_logger


# Custom exceptions
class RecordNotFoundError(Exception):
    pass


class ImageMetadata(NamedTuple):
    filename: Path
    height: int
    width: int
    mpp: float


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
        self, manifest_name: str, split_version: str, split_category: str
    ) -> Generator[Patient, None, None]:
        manifest = self._session.query(Manifest).filter_by(name=manifest_name).first()  # type: ignore
        self._ensure_record(manifest, f"Manifest with name {manifest_name}")

        split_definition = self._session.query(SplitDefinitions).filter_by(version=split_version).first()  # type: ignore
        self._ensure_record(split_definition, f"Split definition with version {split_version}")

        patients = (
            self._session.query(Patient)  # type: ignore
            .join(Split)
            .filter(
                Patient.manifest_id == manifest.id,
                Split.split_definition_id == split_definition.id,
                Split.category == split_category,
            )
            .all()
        )

        self._logger.info(f"Found {len(patients)} patients for split {split_category}")
        for patient in patients:
            yield patient

    @staticmethod
    def _fetch_image_metadata(image: Image) -> ImageMetadata:
        """Extract metadata from an Image object."""
        return ImageMetadata(filename=Path(image.filename), height=image.height, width=image.width, mpp=image.mpp)

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

        return [self._fetch_image_metadata(image) for image in patient.images]

    def get_image_metadata_by_filename(self, filename: str) -> ImageMetadata:
        """
        Fetch the metadata for an image based on its filename.

        Parameters
        ----------
        filename : str
            The filename of the image.

        Returns
        -------
        ImageMetadata
            Metadata of the image.
        """
        image = self._session.query(Image).filter_by(filename=filename).first()
        self._ensure_record(image, f"Image with filename {filename} not found")
        return self._fetch_image_metadata(image)

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
        return self._fetch_image_metadata(image)

    def __enter__(self) -> "DataManager":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        if self._session is not None:
            self.close()
        # By not returning anything (implicitly returning None), we're indicating that any exception should continue propagating

    def close(self):
        if self.__session is not None:
            self.__session.close()
            self.__session = None
