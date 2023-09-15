# encoding: utf-8
from pathlib import Path
from typing import NamedTuple

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ahcore.utils.database_models import Base, Image, Manifest, Patient, Split, SplitDefinitions
from ahcore.utils.io import get_logger

DATABASE_URL_TEMPLATE = "sqlite:///{filename}"


class ImageMetadata(NamedTuple):
    filename: Path
    height: int
    width: int
    mpp: float


def open_db(filename: Path):
    engine = create_engine(DATABASE_URL_TEMPLATE.format(filename=filename))
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
    def __init__(self, session):
        self.session = session
        self._logger = get_logger(type(self).__name__)

    def get_records_by_split(self, manifest_name: str, split_version: str, split_category: str):
        """
        Gets the record given the manifest_name, split_version, and split category (e.g. fit, validate, test).

        Parameters
        ----------
        manifest_name : str
        split_version : str
        split_category : str

        Returns
        -------

        """

        # First, we fetch the relevant manifest and split definition
        manifest = self.session.query(Manifest).filter_by(name=manifest_name).first()  # type: ignore
        split_definition = self.session.query(SplitDefinitions).filter_by(version=split_version).first()  # type:ignore

        # Ensure manifest and split_definition exists
        if not manifest or not split_definition:
            raise ValueError("Manifest or Split Definition not found")

        # Fetch patients that belong to the manifest and have the desired split
        patients = (
            self.session.query(Patient)  # type: ignore
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
        patient = self.session.query(Patient).filter_by(patient_code=patient_code).first()  # type: ignore
        if not patient:
            raise ValueError(f"Patient with code {patient_code} not found")

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
        image = self.session.query(Image).filter_by(filename=filename).first()  # type: ignore
        if not image:
            raise ValueError(f"No image found with filename {filename}")

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
        image = self.session.query(Image).filter_by(id=image_id).first()  # type: ignore
        if not image:
            raise ValueError(f"No image found with ID {image_id}")

        return self._fetch_image_metadata(image)
