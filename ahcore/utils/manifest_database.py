# encoding: utf-8
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ahcore.utils.database_models import Base, Manifest, Patient, Split, SplitDefinitions
from ahcore.utils.io import get_logger

logger = get_logger(__name__)

DATABASE_URL_TEMPLATE = "sqlite:///{filename}"


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


def get_records_by_split(session, manifest_name: str, split_version: str, split_category: str):
    # First, we fetch the relevant manifest and split definition
    manifest = session.query(Manifest).filter_by(name=manifest_name).first()  # type: ignore
    split_definition = session.query(SplitDefinitions).filter_by(version=split_version).first()  # type:ignore

    # Ensure manifest and split_definition exists
    if not manifest or not split_definition:
        raise ValueError("Manifest or Split Definition not found")

    # Fetch patients that belong to the manifest and have the desired split
    patients = (
        session.query(Patient)  # type: ignore
        .join(Split)
        .filter(
            Patient.manifest_id == manifest.id,
            Split.split_definition_id == split_definition.id,
            Split.category == split_category,
        )
        .all()
    )

    logger.info(f"Found {len(patients)} patients for split {split_category}")

    for patient in patients:
        yield patient
