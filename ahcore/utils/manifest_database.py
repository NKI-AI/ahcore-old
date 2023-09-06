# encoding: utf-8
import random
from enum import Enum as PyEnum

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
    filename = Column(String)
    reader = Column(String)
    patient_id = Column(Integer, ForeignKey("patient.id"))

    patient = relationship("Patient", back_populates="images")
    masks = relationship("Mask", back_populates="image")
    annotations = relationship("ImageAnnotations", back_populates="image")
    labels = relationship("ImageLabels", back_populates="image")


class Mask(Base):
    __tablename__ = "mask"
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    reader = Column(String)
    image_id = Column(Integer, ForeignKey("image.id"))

    image = relationship("Image", back_populates="masks")


class ImageAnnotations(Base):
    __tablename__ = "image_annotations"
    id = Column(Integer, primary_key=True)
    annotation_data = Column(String)  # Simplified for example, you might want to use a more complex datatype
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


DATABASE_URL = "sqlite:///your_database_path_here.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


def create_tables():
    Base.metadata.create_all(bind=engine)


def insert_record(record):
    with SessionLocal() as session:
        session.add(record)
        session.commit()


def populate_with_dummies():
    with SessionLocal() as session:
        manifest = Manifest(name="Dummy Manifest")
        session.add(manifest)
        session.flush()

        split_definition = SplitDefinitions(version="v1", description="Initial split")
        session.add(split_definition)
        session.flush()

        for i in range(101):
            patient_code = f"A_{i:03}"
            patient = Patient(patient_code=patient_code, manifest=manifest)
            session.add(patient)
            session.flush()

            patient_age = random.randint(0, 100)
            patient_label = PatientLabels(key="age", value=str(patient_age), patient=patient)
            session.add(patient_label)

            for j in range(2):  # For two images per patient
                image_identifier = f"Image_{patient_code}_{j}.jpg"
                image = Image(filename=image_identifier, reader="OPENSLIDE", patient=patient)
                session.add(image)
                session.flush()  # Flush so that Image ID is populated for future records

                mask_filename = f"{image_identifier}_mask.json"
                mask = Mask(filename=mask_filename, reader="GEOJSON", image=image)
                session.add(mask)

                annotation_data = f"Annotation data for {image_identifier}"  # This is just dummy data
                image_annotation = ImageAnnotations(annotation_data=annotation_data, image=image)
                session.add(image_annotation)

                label_data = (
                    "cancer" if random.choice([True, False]) else "benign"
                )  # Randomly decide if it's cancer or benign
                image_label = ImageLabels(label_data=label_data, image=image)
                session.add(image_label)

            # Assigning splits for each patient
            if i <= 30:
                split_category = "train"
            elif 31 <= i <= 60:
                split_category = "validate"
            else:
                split_category = "test"

            split = Split(category=split_category, patient=patient, split_definition=split_definition)
            session.add(split)

        session.commit()  # Commit all changes to the database


if __name__ == "__main__":
    create_tables()
    populate_with_dummies()
