# encoding: utf-8
import json
import random
import sqlite3
from pathlib import Path
from typing import Generator

from pydantic import BaseModel
from tqdm import tqdm

BATCH_INSERT_SIZE = 100


def find_number_of_tiles(folder_path: Path) -> int:
    with open(folder_path / "meta_data_tiles.json") as json_file:
        meta_data_tiles = json.load(json_file)
        return meta_data_tiles["num_regions_masked"]


class PatientInfo(BaseModel):
    patient_id: int
    patient_name: str


class FolderInfo(BaseModel):
    path: str
    num_files: int
    patient_id: str


class SplitDefinition(BaseModel):
    version: str
    description: str


class SplitInfo(BaseModel):
    folder_id: int
    split_definition_id: int
    category: str  # This will hold either "train", "test", or "validate"


class PatientLabelInfo(BaseModel):
    patient_id: str
    label_slug: str
    label_value: str


class PatientLabelAssignment(BaseModel):
    patient_id: str
    label_slug: str
    label_value: str


class LabelCategoryInfo(BaseModel):
    description: str
    slug: str
    values: list[str]


class LabelValueInfo(BaseModel):
    category_slug: str
    value: str


def get_patient_id(path: Path) -> str:
    return path.name[:12]


class DatabaseManager:
    def __init__(self, db_name="tcga_tiled.sqlite"):
        self.db_name = db_name

    def _connect(self):
        connection = sqlite3.connect(self.db_name)
        # Enabling foreign key constraints
        connection.execute("PRAGMA foreign_keys = ON;")
        return connection

    def create_tables(self):
        with self._connect() as connection:
            cursor = connection.cursor()

            # patients table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY,
                    patient_id TEXT UNIQUE NOT NULL
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
                    folder_id INTEGER NOT NULL,
                    split_definition_id INTEGER NOT NULL,
                    category TEXT NOT NULL, -- This will hold either "train", "test", or "validate"
                    FOREIGN KEY (folder_id) REFERENCES folder_info(id),
                    FOREIGN KEY (split_definition_id) REFERENCES split_definitions(id)
                );
            """
            )

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

            # label mapping table
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

    def insert_folder_info(self, folder_infos: list[FolderInfo]):
        with self._connect() as connection:
            cursor = connection.cursor()

            for info in folder_infos:
                patient_id = get_patient_id(Path(info.path))

                # Check if patient_id exists
                cursor.execute("SELECT id FROM patients WHERE patient_id = ?", (patient_id,))
                row = cursor.fetchone()

                if row:
                    patient_db_id = row[0]
                else:
                    cursor.execute("INSERT INTO patients (patient_id) VALUES (?)", (patient_id,))
                    patient_db_id = cursor.lastrowid

                cursor.execute(
                    "INSERT INTO folder_info (path, num_files, patient_id) VALUES (?, ?, ?)",
                    (info.path, info.num_files, patient_db_id),
                )

    def create_random_split(self, version, split_ratios, description="", patients_generator=None):
        with self._connect() as connection:
            cursor = connection.cursor()

            # Check if this version already exists to avoid duplicate splits
            cursor.execute("SELECT id FROM split_definitions WHERE version = ?", (version,))
            row = cursor.fetchone()

            if row:
                print(f"Split version '{version}' already exists. Skipping...")
                return

            cursor.execute(
                "INSERT INTO split_definitions (version, description) VALUES (?, ?)", (version, description)
            )
            split_definition_id = cursor.lastrowid

            # Fetch patients either from the generator or all distinct patient IDs if the generator is not provided
            if patients_generator:
                patient_ids = [patient_info.patient_id for patient_info in patients_generator]
            else:
                cursor.execute("SELECT DISTINCT patient_id FROM folder_info")
                patient_ids = [row[0] for row in cursor.fetchall()]

            random.shuffle(patient_ids)

            train_end = int(len(patient_ids) * split_ratios[0])
            test_end = train_end + int(len(patient_ids) * split_ratios[1])

            train_patient_ids = patient_ids[:train_end]
            test_patient_ids = patient_ids[train_end:test_end]
            validate_patient_ids = patient_ids[test_end:]

            train_ids = self._get_folder_ids_for_patients(cursor, train_patient_ids)
            test_ids = self._get_folder_ids_for_patients(cursor, test_patient_ids)
            validate_ids = self._get_folder_ids_for_patients(cursor, validate_patient_ids)

            # Insert into the split_info table
            self._insert_into_split(cursor, train_ids, "train", split_definition_id)
            self._insert_into_split(cursor, test_ids, "test", split_definition_id)
            self._insert_into_split(cursor, validate_ids, "validate", split_definition_id)

    @staticmethod
    def _get_folder_ids_for_patients(cursor, patient_ids):
        placeholder = ",".join("?" * len(patient_ids))
        cursor.execute(f"SELECT id FROM folder_info WHERE patient_id IN ({placeholder})", tuple(patient_ids))
        return [row[0] for row in cursor.fetchall()]

    @staticmethod
    def _insert_into_split(cursor, ids, category, split_definition_id):
        for folder_id in ids:
            data = SplitInfo(folder_id=folder_id, split_definition_id=split_definition_id, category=category)
            cursor.execute(
                "INSERT INTO split_info (folder_id, category, split_definition_id) VALUES (?, ?, ?)",
                (data.folder_id, data.category, data.split_definition_id),
            )

    def insert_label_category(self, category_info: LabelCategoryInfo):
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO label_categories (description, slug) VALUES (?, ?)",
                (category_info.description, category_info.slug),
            )

    def insert_label_value(self, value_info: LabelValueInfo):
        with self._connect() as connection:
            cursor = connection.cursor()

            # Fetch the category_id based on the slug
            cursor.execute("SELECT id FROM label_categories WHERE slug = ?", (value_info.category_slug,))
            row = cursor.fetchone()
            if not row:
                print(f"No category found for slug '{value_info.category_slug}'. Skipping...")
                return

            category_id = row[0]
            cursor.execute(
                "INSERT INTO label_values (category_id, value) VALUES (?, ?)", (category_id, value_info.value)
            )

    def assign_label_to_patient(self, patient_label_info: PatientLabelInfo):
        """
        Assigns a label to a patient using a PatientLabelInfo model.

        :param patient_label_info: The info about the patient and label to assign.
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
                (patient_label_info.label_slug, patient_label_info.label_value),
            )
            label_value_id = cursor.fetchone()

            if not label_value_id:
                raise ValueError(
                    f"Label '{patient_label_info.label_value}' for slug '{patient_label_info.label_slug}' not found."
                )
            label_value_id = label_value_id[0]

            # Insert the relation
            cursor.execute(
                """
                INSERT OR IGNORE INTO patient_labels (patient_id, label_value_id)
                VALUES (?, ?)
                """,
                (patient_label_info.patient_id, label_value_id),
            )

    def get_patients_by_label_category(self, label_category: LabelCategoryInfo) -> Generator[PatientInfo, None, None]:
        with self._connect() as connection:
            cursor = connection.cursor()

            # First, we need to fetch the ID for the given label category slug
            cursor.execute("SELECT id FROM label_categories WHERE slug = ?", (label_category.slug,))
            category_id = cursor.fetchone()

            if not category_id:
                raise ValueError(f"No category found for slug '{label_category.slug}'.")

            category_id = category_id[0]

            # Next, fetch patient IDs that have a label associated with this category
            cursor.execute(
                """
                SELECT p.id, p.patient_id
                FROM patients p
                JOIN patient_labels pl ON p.id = pl.patient_id
                JOIN label_values lv ON pl.label_value_id = lv.id
                WHERE lv.category_id = ?
                """,
                (category_id,),
            )

            # Fetching patients and yielding as generator
            for row in cursor.fetchall():
                yield PatientInfo(id=row[0], patient_id=row[1])


def populate_with_tcga_tiled(db):
    original_path = Path("/projects/tcga_tiled/v1/data")

    all_tcgas = original_path.glob("*/*")
    infos_to_insert = []

    for idx, folder_path in tqdm(enumerate(all_tcgas)):
        num_files = find_number_of_tiles(folder_path)
        patient_id = get_patient_id(folder_path)
        info = FolderInfo(path=str(folder_path.relative_to(original_path)), num_files=num_files, patient_id=patient_id)
        infos_to_insert.append(info)

        if len(infos_to_insert) >= BATCH_INSERT_SIZE:
            db.insert_folder_info(infos_to_insert)
            infos_to_insert = []

        # Easy for debugging.
        # if idx > 100:
        #     break

    if infos_to_insert:
        db.insert_folder_info(infos_to_insert)


def populate_label_categories_and_values_with_dummies(db: DatabaseManager):
    # Defining categories and their respective values
    categories = [
        LabelCategoryInfo(description="Tumor Type", slug="tumor_type", values=["Melanoma", "Carcinoma", "Sarcoma"]),
        LabelCategoryInfo(description="Tumor Stage", slug="tumor_stage", values=["I", "II", "III", "IV"]),
        LabelCategoryInfo(
            description="Treatment Response", slug="treatment_response", values=["Positive", "Negative", "Neutral"]
        ),
    ]

    # Inserting the categories and their values into the database
    for category_data in categories:
        db.insert_label_category(category_data)
        for value in category_data.values:
            value_info = LabelValueInfo(category_slug=category_data.slug, value=value)
            db.insert_label_value(value_info)


def assign_dummy_labels_to_patients(db: DatabaseManager):
    # Dummy label assignments for demonstration purposes
    dummy_assignments = (
        [
            PatientLabelAssignment(
                patient_id=str(i),
                label_slug="tumor_type",
                label_value=random.choice(["Melanoma", "Carcinoma", "Sarcoma"]),
            )
            for i in range(1, 11)
        ]
        + [
            PatientLabelAssignment(
                patient_id=str(i), label_slug="tumor_stage", label_value=random.choice(["I", "II", "III", "IV"])
            )
            for i in range(1, 11)
        ]
        + [
            PatientLabelAssignment(
                patient_id=str(i),
                label_slug="treatment_response",
                label_value=random.choice(["Positive", "Negative", "Neutral"]),
            )
            for i in range(1, 11)
        ]
    )

    # Assign the labels to the patients
    for assignment in dummy_assignments:
        label_info = PatientLabelInfo(
            patient_id=assignment.patient_id, label_slug=assignment.label_slug, label_value=assignment.label_value
        )
        db.assign_label_to_patient(label_info)


if __name__ == "__main__":
    db = DatabaseManager()
    db.create_tables()
    populate_with_tcga_tiled(db)

    # Assign splits based on different versions and ratios
    db.create_random_split("v1", (0.8, 0.1, 0.1), "80/10/10 split")
    db.create_random_split("v2", (0.7, 0.2, 0.1), "70/20/10 split")

    populate_label_categories_and_values_with_dummies(db)
    assign_dummy_labels_to_patients(db)

    # Example, get all patients with a specific label.
    label_category = LabelCategoryInfo(
        description="Tumor Type", slug="tumor_type", values=["Melanoma", "Carcinoma", "Sarcoma"]
    )
    filtered_patients = db.get_patients_by_label_category(label_category)
    # Create split that has the patients in there
    db.create_random_split("v3", (0.8, 0.1, 0.1), "label split", filtered_patients)
