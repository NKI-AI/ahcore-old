# encoding: utf-8
import json
import random
from pathlib import Path

from dlup import SlideImage, UnsupportedSlideError
from dlup.experimental_backends import ImageBackend
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ahcore.utils.database_models import (
    Base,
    Image,
    ImageAnnotations,
    ImageLabels,
    Manifest,
    Mask,
    Patient,
    PatientLabels,
    Split,
    SplitDefinitions,
)

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


def get_patient_from_tcga_id(tcga_filename: str) -> str:
    return tcga_filename[:12]


def populate_from_annotated_tcga(session, image_folder: Path, annotation_folder: Path, path_to_mapping: Path):
    # TODO: We should do the mpp as well here

    with open(path_to_mapping, "r") as f:
        mapping = json.load(f)
    manifest = Manifest(name="TCGA Breast Annotations v20230228")
    session.add(manifest)
    session.flush()

    split_definition = SplitDefinitions(version="v1", description="Initial split")
    session.add(split_definition)
    session.flush()

    for folder in annotation_folder.glob("TCGA*"):
        patient_code = get_patient_from_tcga_id(folder.name)

        annotation_path = folder / "annotations.json"
        mask_path = folder / "roi.json"

        # Only add patient if it doesn't exist
        existing_patient = session.query(Patient).filter_by(patient_code=patient_code).first()  # type: ignore
        if existing_patient:
            patient = existing_patient
        else:
            patient = Patient(patient_code=patient_code, manifest=manifest)
            session.add(patient)
            session.flush()

            # For now random.
            split_category = random.choices(["train", "validate", "test"], [70, 20, 10])[0]

            split = Split(category=split_category, patient=patient, split_definition=split_definition)
            session.add(split)

        patient_label = PatientLabels(key="study", value="BRCA", patient=patient)
        session.add(patient_label)
        session.flush()

        filename = mapping[folder.name]

        # TODO: OPENSLIDE doesn't work
        kwargs = {}
        if (
            "TCGA-OL-A5RY-01Z-00-DX1.AE4E9D74-FC1C-4C1E-AE6D-5DF38899BBA6.svs" in filename
            or "TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF.svs" in filename
        ):
            kwargs["overwrite_mpp"] = (0.25, 0.25)

        with SlideImage.from_file_path(image_folder / filename, backend=ImageBackend.PYVIPS, **kwargs) as slide:
            mpp = slide.mpp
            width, height = slide.size
            image = Image(
                filename=str(filename), mpp=mpp, height=height, width=width, reader="OPENSLIDE", patient=patient
            )
        session.add(image)
        session.flush()  # Flush so that Image ID is populated for future records

        mask = Mask(filename=str(mask_path), reader="GEOJSON", image=image)
        session.add(mask)

        image_annotation = ImageAnnotations(filename=str(annotation_path), reader="GEOJSON", image=image)
        session.add(image_annotation)

        label_data = "cancer" if random.choice([True, False]) else "benign"  # Randomly decide if it's cancer or benign
        image_label = ImageLabels(label_data=label_data, image=image)
        session.add(image_label)

        session.commit()


def get_records_by_split(session, manifest_name: str, split_version: str, split_category: str):
    # TODO: Rename this in the db

    if split_category == "fit":
        split_category = "train"


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

if __name__ == "__main__":
    annotation_folder = Path(
        "/data/groups/aiforoncology/derived/pathology/TCGA/gdc_manifest.2021-11-01_diagnostic_breast.txt/tissue_subtypes/v20230228_combined/"
    )
    image_folder = Path("/data/groups/aiforoncology/archive/pathology/TCGA/images/")
    path_to_mapping = Path("/data/groups/aiforoncology/archive/pathology/TCGA/identifier_mapping.json")
    with open_db("manifest.db") as session:
        populate_from_annotated_tcga(session, image_folder, annotation_folder, path_to_mapping)
