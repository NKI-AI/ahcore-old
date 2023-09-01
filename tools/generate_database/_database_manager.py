# encoding: utf-8
from __future__ import annotations

import random
from typing import Generator

import numpy as np
import sqlite3
from pydantic import BaseModel


def create_dataset_split(
    patient_ids: list[int],
    split_percentages: tuple[float, float, float],
) -> tuple[list[int], list[int], list[int]]:
    """
    Splits data randomly into train, validate and test set using the given ratios.

    Parameters
    ----------
    patient_ids: list[int]
        List of patient identifiers to be split.
    split_percentages: tuple[int, int, int]
        Tuple of ratios (train_ratio, validate_ratio, test_ratio) used to split the data.

    Returns
    -------
    tuple[list[int], list[int], list[int]]
        List of patient identifiers per data subset; train, validate and test
    """

    if len(split_percentages) != 3:
        raise ValueError("Split percentages must contain three values: (train, validate, test)")

    if np.sum(split_percentages) != 100:
        raise ValueError("Split percentages must sum to 100.")

    random.shuffle(patient_ids)

    n_patients = len(patient_ids)
    fit_percentage, validate_percentage, test_percentage = split_percentages

    train_end_idx = int((n_patients * fit_percentage) / 100)
    val_end_idx = train_end_idx + int((n_patients * validate_percentage) / 100)
    test_end_idx = n_patients

    train_split = patient_ids[0:train_end_idx]
    validate_split = patient_ids[train_end_idx:val_end_idx]
    test_split = patient_ids[val_end_idx:test_end_idx]

    return train_split, validate_split, test_split


class PatientInfo(BaseModel):
    id: int
    patient_code: str


class FolderInfo(BaseModel):
    path: str
    num_files: int
    patient_code: str

class SplitDefinition(BaseModel):
    version: str
    description: str

class SplitInfo(BaseModel):
    patient_id: int
    split_definition_id: int
    category: str  # This will hold either "train", "test", or "validate"

class LabelCategoryInfo(BaseModel):
    description: str
    slug: str
    values: list[str]

class LabelValueInfo(BaseModel):
    label_category_slug: str
    value: str

class PatientLabelAssignment(BaseModel):
    patient_code: str
    label_category_slug: str
    label_value: str

class PatientLabelsInfo(BaseModel):
    patient_id: int
    label_value_id: int


class DatabaseManager:
    def __init__(self, db_name: str = "tcga_tiled.sqlite"):
        self.db_name = db_name

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_name)
        # Enabling foreign key constraints
        connection.execute("PRAGMA foreign_keys = ON;")
        return connection

    def create_tables(self) -> None:
        with self._connect() as connection:
            cursor = connection.cursor()

            # patients table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY,
                    patient_code TEXT UNIQUE NOT NULL
                );
            """
            )

            # folder_info table with added patient_id foreign key
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS folder_info (
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL,
                    num_files INTEGER NOT NULL,
                    patient_id INTEGER,
                    FOREIGN KEY (patient_id) REFERENCES patients(id)
                );
            """
            )

            # split_definitions table to hold version and description
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS split_definitions (
                    id INTEGER PRIMARY KEY,
                    version TEXT NOT NULL,
                    description TEXT
                );
            """
            )

            # split_info table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS split_info (
                    id INTEGER PRIMARY KEY,
                    patient_id INTEGER NOT NULL,
                    split_definition_id INTEGER NOT NULL,
                    category TEXT NOT NULL, -- This will hold either "train", "test", or "validate"
                    FOREIGN KEY (patient_id) REFERENCES patients(id),
                    FOREIGN KEY (split_definition_id) REFERENCES split_definitions(id)
                );
            """
            )

            # label_categories table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS label_categories (
                    id INTEGER PRIMARY KEY,
                    description TEXT NOT NULL,
                    slug TEXT UNIQUE NOT NULL
                );
            """
            )

            # label_values table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS label_values (
                    id INTEGER PRIMARY KEY,
                    category_id INTEGER NOT NULL,
                    value TEXT NOT NULL,
                    FOREIGN KEY (category_id) REFERENCES label_categories(id)
                );
            """
            )

            # patient_labels table, mapping patient to label
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_labels (
                    id INTEGER PRIMARY KEY,
                    patient_id INTEGER NOT NULL,
                    label_value_id INTEGER NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients(id),
                    FOREIGN KEY (label_value_id) REFERENCES label_values(id)
                );
                """
            )

    def insert_folder_info(self, folder_infos: list[FolderInfo]) -> None:
        with self._connect() as connection:
            cursor = connection.cursor()

            for info in folder_infos:
                patient_code = info.patient_code

                # Try to get the patient id from the database that matches the patient code.
                cursor.execute("SELECT id FROM patients WHERE patient_code = ?", (patient_code,))
                row = cursor.fetchone()

                # Insert the patient code if it does not already exist in the database.
                if row is not None:
                    patient_id = row[0]
                else:
                    cursor.execute("INSERT INTO patients (patient_code) VALUES (?)", (patient_code,))
                    patient_id = cursor.lastrowid

                cursor.execute(
                    "INSERT INTO folder_info (path, num_files, patient_id) VALUES (?, ?, ?)",
                    (info.path, info.num_files, patient_id),
                )

    def create_random_split(
        self,
        version: str,
        split_ratios: tuple[int, int, int],
        description: str = "",
        patients_generator: Generator | None = None,
    ) -> None:
        with self._connect() as connection:
            cursor = connection.cursor()

            # Check if this version already exists to avoid duplicate splits
            cursor.execute("SELECT id FROM split_definitions WHERE version = ?", (version,))
            row = cursor.fetchone()

            if row:
                print(f"Split version '{version}' already exists. Skipping...")
                return

            definition = SplitDefinition(version=version, description=description)

            cursor.execute(
                "INSERT INTO split_definitions (version, description) VALUES (?, ?)",
                (definition.version, definition.description),
            )
            split_definition_id = cursor.lastrowid

            # Fetch patients either from the generator or all distinct patient IDs if the generator is not provided
            if patients_generator:
                patient_ids = [patient_info.id for patient_info in patients_generator]
            else:
                cursor.execute("SELECT DISTINCT patient_id FROM folder_info")
                patient_ids = [row[0] for row in cursor.fetchall()]

            train_patient_ids, validate_patient_ids, test_patient_ids = create_dataset_split(
                patient_ids=patient_ids, split_percentages=split_ratios
            )

            # Insert into the split_info table
            self._insert_into_split(cursor, train_patient_ids, "train", split_definition_id)
            self._insert_into_split(cursor, validate_patient_ids, "validate", split_definition_id)
            self._insert_into_split(cursor, test_patient_ids, "test", split_definition_id)

    @staticmethod
    def _get_folder_ids_for_patients(cursor: sqlite3.Cursor, patient_ids: list[int]):
        placeholder = ",".join("?" * len(patient_ids))
        cursor.execute(f"SELECT id FROM folder_info WHERE patient_id IN ({placeholder})", tuple(patient_ids))
        return [row[0] for row in cursor.fetchall()]

    @staticmethod
    def _insert_into_split(cursor: sqlite3.Cursor, patient_ids: list[int], category: str, split_definition_id: int) -> None:
        for patient_id in patient_ids:
            info = SplitInfo(patient_id=patient_id, split_definition_id=split_definition_id, category=category)
            cursor.execute(
                "INSERT INTO split_info (patient_id, split_definition_id, category) VALUES (?, ?, ?)",
                (info.patient_id, info.split_definition_id, info.category),
            )

    def insert_label_category(self, category_info: LabelCategoryInfo) -> None:
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO label_categories (description, slug) VALUES (?, ?)",
                (category_info.description, category_info.slug),
            )

    def insert_label_value(self, value_info: LabelValueInfo) -> None:
        with self._connect() as connection:
            cursor = connection.cursor()

            # Fetch the category_id based on the slug
            cursor.execute("SELECT id FROM label_categories WHERE slug = ?", (value_info.label_category_slug,))
            row = cursor.fetchone()
            if not row:
                print(f"No category found for slug '{value_info.label_category_slug}'. Skipping...")
                return

            category_id = row[0]
            cursor.execute(
                "INSERT INTO label_values (category_id, value) VALUES (?, ?)", (category_id, value_info.value)
            )

    def assign_label_to_patient(self, patient_label_assignment: PatientLabelAssignment, strict: bool = False) -> None:
        """
        Assigns a label to a patient using a PatientLabelAssignment model.

        :param patient_label_assignment: The info about the patient and label to assign.
        :param strict: if set to false, the assigment will be skipped if the patient is not in the database.
        """
        with self._connect() as connection:
            cursor = connection.cursor()

            # Fetch label_value_id using slug and label_value
            cursor.execute(
                """
                SELECT lv.id
                FROM label_values lv
                JOIN label_categories lc ON lv.category_id = lc.id
                WHERE lc.slug = ? AND lv.value = ?
                """,
                (patient_label_assignment.label_category_slug, patient_label_assignment.label_value),
            )
            _label_value_id = cursor.fetchone()
            if _label_value_id is None:
                raise ValueError(
                    f"Label '{patient_label_assignment.label_value}' for slug '{patient_label_assignment.label_category_slug}' not found."
                )

            label_value_id = _label_value_id[0]

            # Fetch patient_id using patient_code
            cursor.execute("SELECT id FROM patients WHERE patient_code = ?", (patient_label_assignment.patient_code,))

            _patient_id = cursor.fetchone()
            if _patient_id is None:
                _message = f"Patient code '{patient_label_assignment.patient_code}' not found in database."
                if strict:
                    raise ValueError(_message)
                else:
                    print(f"{_message} Skipping...")
                    return

            patient_id = _patient_id[0]

            patient_labels_info = PatientLabelsInfo(patient_id=patient_id, label_value_id=label_value_id)

            # Insert the relation
            cursor.execute(
                """
                INSERT OR IGNORE INTO patient_labels (patient_id, label_value_id)
                VALUES (?, ?)
                """,
                (patient_labels_info.patient_id, patient_labels_info.label_value_id),
            )

    def get_patients_by_label_category(self, label_category_slug: str) -> Generator[PatientInfo, None, None]:
        with self._connect() as connection:
            cursor = connection.cursor()

            # First, we need to fetch the ID for the given label category slug
            cursor.execute("SELECT id FROM label_categories WHERE slug = ?", (label_category_slug,))
            category_id = cursor.fetchone()

            if not category_id:
                raise ValueError(f"No category found for slug '{label_category_slug}'.")

            category_id = category_id[0]

            # Next, fetch patient IDs that have a label associated with this category
            cursor.execute(
                """
                SELECT p.id, p.patient_code
                FROM patients p
                JOIN patient_labels pl ON p.id = pl.patient_id
                JOIN label_values lv ON pl.label_value_id = lv.id
                WHERE lv.category_id = ?
                """,
                (category_id,),
            )

            # Fetching patients and yielding as generator
            for row in cursor.fetchall():
                yield PatientInfo(id=row[0], patient_code=row[1])
