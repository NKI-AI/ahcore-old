# encoding: utf-8
from enum import Enum as PyEnum

from sqlalchemy import Column, DateTime, Enum, Float, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, relationship


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

    height = Column(Integer)
    width = Column(Integer)
    mpp = Column(Float)

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
    reader = Column(String)
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

    # Add a unique constraint
    __table_args__ = (UniqueConstraint("key", "patient_id", name="uq_patient_label_key"),)

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
