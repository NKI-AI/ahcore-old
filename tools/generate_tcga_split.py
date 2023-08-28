# encoding: utf-8
import json
import random
import sqlite3
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm


def find_number_of_tiles(folder_path: Path) -> int:
    with open(folder_path / "meta_data_tiles.json") as json_file:
        meta_data_tiles = json.load(json_file)
        return meta_data_tiles["num_regions_masked"]


class FolderInfo(BaseModel):
    path: str
    num_files: int
    patient_id: str


class SplitInfo(BaseModel):
    folder_id: int
    split: str
    version: str
    description: str


def get_patient_id(path: Path) -> str:
    return path.name[:12]


class DatabaseManager:
    def __init__(self, db_name="tcga_tiled.sqlite"):
        self.db_name = db_name

    def _connect(self):
        return sqlite3.connect(self.db_name)

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

    def create_random_split(self, version, split_ratios, description=""):
        # if sum(split_ratios) != 1.0:
        #     raise ValueError("Split ratios must sum up to 1.")

        with self._connect() as connection:
            cursor = connection.cursor()

            # Check if this version already exists to avoid duplicate splits
            cursor.execute("SELECT DISTINCT version FROM split_info WHERE version = ?", (version,))
            if cursor.fetchone():
                print(f"Split version '{version}' already exists. Skipping...")
                return

            # Fetch all distinct patient IDs
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
            self._insert_into_split(cursor, train_ids, "train", version, description)
            self._insert_into_split(cursor, test_ids, "test", version, description)
            self._insert_into_split(cursor, validate_ids, "validate", version, description)

    @staticmethod
    def _get_folder_ids_for_patients(cursor, patient_ids):
        placeholder = ",".join("?" * len(patient_ids))
        cursor.execute(f"SELECT id FROM folder_info WHERE patient_id IN ({placeholder})", tuple(patient_ids))
        return [row[0] for row in cursor.fetchall()]

    @staticmethod
    def _insert_into_split(cursor, ids, split_name, version, description):
        for folder_id in ids:
            data = SplitInfo(folder_id=folder_id, split=split_name, version=version, description=description)
            cursor.execute(
                "INSERT INTO split_info (folder_id, split, version, description) VALUES (?, ?, ?, ?)",
                (data.folder_id, data.split, data.version, data.description),
            )


if __name__ == "__main__":
    db = DatabaseManager()
    db.create_tables()

    original_path = Path("/projects/tcga_tiled/v1/data")

    all_tcgas = original_path.glob("*/*")
    infos_to_insert = []

    for folder_path in tqdm(all_tcgas):
        num_files = find_number_of_tiles(folder_path)
        patient_id = get_patient_id(folder_path)  # Add this line
        info = FolderInfo(
            path=str(folder_path.relative_to(original_path)), num_files=num_files, patient_id=patient_id
        )  # Update this line
        infos_to_insert.append(info)

        if len(infos_to_insert) >= 100:
            db.insert_folder_info(infos_to_insert)
            infos_to_insert = []

    if infos_to_insert:
        db.insert_folder_info(infos_to_insert)

    # Assign splits based on different versions and ratios
    db.create_random_split("v1", (0.8, 0.1, 0.1), "80/10/10 split")
    db.create_random_split("v2", (0.7, 0.2, 0.1), "70/20/10 split")
