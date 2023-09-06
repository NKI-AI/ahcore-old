# encoding: utf-8
import json
import random
from enum import Enum as PyEnum
from pathlib import Path

from sqlalchemy import Column, DateTime, Enum, ForeignKey, Integer, String, create_engine, func
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker


class CategoryEnum(PyEnum):
    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"


class Base(DeclarativeBase):
    pass


class Manifest(Base):
    __tablename__ = "manifest"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    created = Column(DateTime(timezone=True), default=func.now())
    last_updated = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    patients = relationship("Patient", back_populates="manifest")


class Patient(Base):
    __tablename__ = "patient"
    id = Column(Integer, primary_key=True)
    patient_code = Column(String, unique=True)
    manifest_id = Column(Integer, ForeignKey("manifest.id"))

    manifest = relationship("Manifest", back_populates="patients")
    images = relationship("Image", back_populates="patient")
    labels = relationship("PatientLabels", back_populates="patient")
    split = relationship("Split", uselist=False, back_populates="patient")


class Image(Base):
    __tablename__ = "image"
    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True)
    reader = Column(String)
    patient_id = Column(Integer, ForeignKey("patient.id"))

    patient = relationship("Patient", back_populates="images")
    masks = relationship("Mask", back_populates="image")
    annotations = relationship("ImageAnnotations", back_populates="image")
    labels = relationship("ImageLabels", back_populates="image")


class Mask(Base):
    __tablename__ = "mask"
    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True)
    reader = Column(String)
    image_id = Column(Integer, ForeignKey("image.id"))

    image = relationship("Image", back_populates="masks")


class ImageAnnotations(Base):
    __tablename__ = "image_annotations"
    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True)
    image_id = Column(Integer, ForeignKey("image.id"))

    image = relationship("Image", back_populates="annotations")


class ImageLabels(Base):
    __tablename__ = "image_labels"
    id = Column(Integer, primary_key=True)
    label_data = Column(String)  # e.g. "cancer" or "benign"
    image_id = Column(Integer, ForeignKey("image.id"))

    image = relationship("Image", back_populates="labels")


class PatientLabels(Base):
    __tablename__ = "patient_labels"
    id = Column(Integer, primary_key=True)
    key = Column(String)
    value = Column(String)
    patient_id = Column(Integer, ForeignKey("patient.id"))

    patient = relationship("Patient", back_populates="labels")


class SplitDefinitions(Base):
    __tablename__ = "split_definitions"
    id = Column(Integer, primary_key=True)
    version = Column(String, nullable=False)
    description = Column(String)
    splits = relationship("Split", back_populates="split_definition")


class Split(Base):
    __tablename__ = "split"

    id = Column(Integer, primary_key=True)
    category: Column = Column(Enum(CategoryEnum), nullable=False)
    patient_id = Column(Integer, ForeignKey("patient.id"))

    patient = relationship("Patient", back_populates="split")
    split_definition_id = Column(Integer, ForeignKey("split_definitions.id"))
    split_definition = relationship("SplitDefinitions", back_populates="splits")


DATABASE_URL = "sqlite:///tcga.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


def create_tables():
    Base.metadata.create_all(bind=engine)


def insert_record(record):
    with SessionLocal() as session:
        session.add(record)
        session.commit()


def get_patient_from_tcga_id(tcga_filename: str) -> str:
    return tcga_filename[:12]


def populate_from_annotated_tcga(path: Path) -> str:
    annotation_folder = Path(
        "/data/groups/aiforoncology/derived/pathology/TCGA/gdc_manifest.2021-11-01_diagnostic_breast.txt/tissue_subtypes/v20230228_combined/"
    )
    path_to_mapping = Path("/data/groups/aiforoncology/archive/pathology/TCGA/identifier_mapping.json")
    with open(path_to_mapping, "r") as f:
        mapping = json.load(f)

    with SessionLocal() as session:
        manifest = Manifest(name="TCGA Breast Annotations v20230228")
        session.add(manifest)
        session.flush()

        split_definition = SplitDefinitions(version="v1", description="Initial split")
        session.add(split_definition)
        session.flush()

        for folder in annotation_folder.glob("TCGA*"):
            patient_code = get_patient_from_tcga_id(folder.name)

            annotation_path = folder / "annotations.json"
            mask_path = folder / "masks.json"

            # Only add patient if it doesn't exist
            existing_patient = session.query(Patient).filter_by(patient_code=patient_code).first()
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
            image = Image(filename=str(filename), reader="OPENSLIDE", patient=patient)
            session.add(image)
            session.flush()  # Flush so that Image ID is populated for future records

            mask = Mask(filename=str(mask_path), reader="GEOJSON", image=image)
            session.add(mask)

            image_annotation = ImageAnnotations(filename=str(annotation_path), image=image)
            session.add(image_annotation)

            label_data = (
                "cancer" if random.choice([True, False]) else "benign"
            )  # Randomly decide if it's cancer or benign
            image_label = ImageLabels(label_data=label_data, image=image)
            session.add(image_label)


if __name__ == "__main__":
    create_tables()
    populate_from_annotated_tcga("")
