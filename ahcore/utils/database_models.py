"""Database models for ahcore's manifest database."""
from enum import Enum as PyEnum

from sqlalchemy import Column, DateTime, Enum, Float, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped


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

    patients: Mapped["Patient"] = relationship("Patient", back_populates="manifest")


class Patient(Base):
    __tablename__ = "patient"
    id = Column(Integer, primary_key=True)
    patient_code = Column(String, unique=True)
    manifest_id = Column(Integer, ForeignKey("manifest.id"))

    manifest: Mapped["Manifest"] = relationship("Manifest", back_populates="patients")
    images: Mapped["Image"] = relationship("Image", back_populates="patient")
    labels: Mapped["PatientLabels"] = relationship("PatientLabels", back_populates="patient")
    split: Mapped["Split"] = relationship("Split", uselist=False, back_populates="patient")


class Image(Base):
    __tablename__ = "image"
    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True)
    reader = Column(String)
    patient_id = Column(Integer, ForeignKey("patient.id"))

    height = Column(Integer)
    width = Column(Integer)
    mpp = Column(Float)

    patient: Mapped["Patient"] = relationship("Patient", back_populates="images")
    masks: Mapped["Mask"] = relationship("Mask", back_populates="image")
    annotations: Mapped["ImageAnnotations"] = relationship("ImageAnnotations", back_populates="image")
    labels: Mapped["ImageLabels"] = relationship("ImageLabels", back_populates="image")
    cache: Mapped["ImageCache"] = relationship("ImageCache", uselist=False, back_populates="image")


class ImageCache(Base):
    __tablename__ = "image_cache"

    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True)
    reader = Column(String)
    num_tiles = Column(Integer)
    image_id = Column(Integer, ForeignKey("image.id"))

    image: Mapped["Image"] = relationship("Image", back_populates="cache")
    description_id = Column(Integer, ForeignKey("cache_description.id"))
    description: Mapped["CacheDescription"] = relationship("CacheDescription", back_populates="caches")


class CacheDescription(Base):
    __tablename__ = "cache_description"

    id = Column(Integer, primary_key=True)
    mpp = Column(Float)
    tile_size_width = Column(Integer)
    tile_size_height = Column(Integer)
    tile_overlap_width = Column(Integer)
    tile_overlap_height = Column(Integer)
    tile_mode = Column(String)
    crop = Column(Integer, default=False)  # using Integer for boolean for DB compatibility
    mask_threshold = Column(Float)
    grid_order = Column(String)

    caches: Mapped["ImageCache"] = relationship("ImageCache", back_populates="description")


class Mask(Base):
    __tablename__ = "mask"
    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True)
    reader = Column(String)
    image_id = Column(Integer, ForeignKey("image.id"))

    image: Mapped["Image"] = relationship("Image", back_populates="masks")


class ImageAnnotations(Base):
    __tablename__ = "image_annotations"
    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True)
    reader = Column(String)
    image_id = Column(Integer, ForeignKey("image.id"))

    image: Mapped["Image"] = relationship("Image", back_populates="annotations")


class ImageLabels(Base):
    __tablename__ = "image_labels"
    id = Column(Integer, primary_key=True)
    label_data = Column(String)  # e.g. "cancer" or "benign"
    image_id = Column(Integer, ForeignKey("image.id"))

    image: Mapped["Image"] = relationship("Image", back_populates="labels")


class PatientLabels(Base):
    __tablename__ = "patient_labels"
    id = Column(Integer, primary_key=True)
    key = Column(String)
    value = Column(String)
    patient_id = Column(Integer, ForeignKey("patient.id"))

    # Add a unique constraint
    __table_args__ = (UniqueConstraint("key", "patient_id", name="uq_patient_label_key"),)

    patient: Mapped["Patient"] = relationship("Patient", back_populates="labels")


class SplitDefinitions(Base):
    __tablename__ = "split_definitions"
    id = Column(Integer, primary_key=True)
    version = Column(String, nullable=False)
    description = Column(String)
    splits: Mapped["Split"] = relationship("Split", back_populates="split_definition")


class Split(Base):
    __tablename__ = "split"

    id = Column(Integer, primary_key=True)
    category: Column = Column(Enum(CategoryEnum), nullable=False)
    patient_id = Column(Integer, ForeignKey("patient.id"))

    patient: Mapped["Patient"] = relationship("Patient", back_populates="split")
    split_definition_id = Column(Integer, ForeignKey("split_definitions.id"))
    split_definition: Mapped["SplitDefinitions"] = relationship("SplitDefinitions", back_populates="splits")
