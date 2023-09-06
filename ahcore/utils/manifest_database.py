# encoding: utf-8
import sqlite3
from pathlib import Path

import random
from pydantic import BaseModel


class _BaseModel(BaseModel):
    pass

class Manifest(_BaseModel):
    id: int
    name: str


class Patient(_BaseModel):
    id: int
    patient_code: str
    manifest_id: int

class Image(_BaseModel):
    id: int
    filename: str
    reader: str
    patient_id: int


class Mask(_BaseModel):
    id: int
    filename: str
    reader: str
    image_id: int


class ImageAnnotations(_BaseModel):
    id: int
    filename: str
    reader: str
    image_id: int


class ImageLabels(_BaseModel):
    id: int
    key: str
    value: str
    image_id: int

class PatientLabels(_BaseModel):
    id: int
    key: str
    value: str
    patient_id: int

class SplitDefinitions(_BaseModel):
    id: int
    version: str
    description: str


class Split(_BaseModel):
    id: int
    patient_id: int
    split_definition_id: int
    category: str


class DatabaseManager:
    def __init__(self, db_name: Path):
        self.db_name = db_name

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_name)
        # Enabling foreign key constraints
        connection.execute("PRAGMA foreign_keys = ON;")
        return connection

    def create_tables(self) -> None:
        with self._connect() as connection:
            cursor = connection.cursor()

            # Table creations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS manifest (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    created TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patient (
                    id INTEGER PRIMARY KEY,
                    patient_code TEXT UNIQUE NOT NULL,
                    manifest_id INTEGER,
                    created TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (manifest_id) REFERENCES manifest (id)
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    reader TEXT,
                    patient_id INTEGER,
                    created TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patient (id)
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mask (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    reader TEXT,
                    image_id INTEGER,
                    created TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES image (id)
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    reader TEXT,
                    image_id INTEGER,
                    created TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES image (id)
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT,
                    value TEXT,
                    image_id INTEGER,
                    created TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES image (id)
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patient_labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT,
                    value TEXT,
                    patient_id INTEGER,
                    created TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patient (id)
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS split_definitions (
                    id INTEGER PRIMARY KEY,
                    version TEXT NOT NULL,
                    description TEXT,
                    created TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS split (
                    id INTEGER PRIMARY KEY,
                    patient_id INTEGER NOT NULL,
                    split_definition_id INTEGER NOT NULL,
                    category TEXT NOT NULL CHECK (category IN ('train', 'test', 'validate')),
                    created TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patient(id),
                    FOREIGN KEY (split_definition_id) REFERENCES split_definitions(id)
                );
            """)

            # Triggers to update last_updated field for all tables
            tables = [
                "manifest",
                "patient",
                "image",
                "mask",
                "image_annotations",
                "image_labels",
                "patient_labels",
                "split_definitions",
                "split",
            ]

            for table in tables:
                cursor.execute(f"""
                    CREATE TRIGGER IF NOT EXISTS update_{table}_last_updated
                    AFTER UPDATE ON {table}
                    FOR EACH ROW
                    BEGIN
                        UPDATE {table} SET last_updated = CURRENT_TIMESTAMP WHERE id = OLD.id;
                    END;
                """)

    def insert_record(self, table: str, record: _BaseModel) -> None:
        data = record.dict()
        assert "created" not in data
        assert "last_updated" not in data

        with self._connect() as connection:
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?'] * len(data))
            sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            connection.execute(sql, tuple(data.values()))


def populate_with_dummies(db_manager: DatabaseManager) -> None:
    manifest = Manifest(id=1, name="Dummy Manifest")
    db_manager.insert_record("manifest", manifest)

    split_definitions = SplitDefinitions(id=1, version="v1", description="Dummy Split")
    db_manager.insert_record("split_definitions", split_definitions)

    for i in range(101):
        patient_code = f"A_{i:03}"
        patient = Patient(id=i + 1, patient_code=patient_code, manifest_id=1)
        db_manager.insert_record("patient", patient)

        patient_age = random.randint(0, 100)
        patient_label = PatientLabels(id=i + 1, key="age", value=str(patient_age), patient_id=patient.id)
        db_manager.insert_record("patient_labels", patient_label)

        for j in range(2):  # For two images per patient
            image_identifier = f"Image_{patient_code}_{j}.jpg"

            image = Image(id=2 * i + j + 1, filename=image_identifier, reader="OPENSLIDE", patient_id=patient.id)
            db_manager.insert_record("image", image)

            annotation_filename = f"{image_identifier}_annotation.json"
            annotation = ImageAnnotations(id=2 * i + j + 1, filename=annotation_filename, reader="JSON",
                                          image_id=image.id)
            db_manager.insert_record("image_annotations", annotation)

            label_value = random.choice(["cancer", "benign"])
            label = ImageLabels(id=2 * i + j + 1, key="diagnosis", value=label_value, image_id=image.id)
            db_manager.insert_record("image_labels", label)

            mask_filename = f"{image_identifier}_mask.json"
            mask = Mask(id=2 * i + j + 1, filename=mask_filename, reader="GEOJSON", image_id=image.id)
            db_manager.insert_record("mask", mask)

        if i <= 30:
            split_category = "train"
        elif 31 <= i <= 60:
            split_category = "validate"
        else:
            split_category = "test"

        split = Split(id=i + 1, patient_id=patient.id, split_definition_id=1, category=split_category)
        db_manager.insert_record("split", split)


if __name__ == "__main__":
    db_path = Path("your_database_path_here.db")
    db_manager = DatabaseManager(db_path)
    db_manager.create_tables()
    populate_with_dummies(db_manager)