import json
import os
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


class DatabaseManager:
    def __init__(self, db_name="tcga_tiled.sqlite"):
        self.db_name = db_name

    def _connect(self):
        return sqlite3.connect(self.db_name)

    def create_tables(self):
        with self._connect() as connection:
            cursor = connection.cursor()

            # folder_info table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS folder_info (
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL,
                    num_files INTEGER NOT NULL
                );
            """
            )

            # split_info table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS split_info (
                    id INTEGER PRIMARY KEY,
                    folder_id INTEGER NOT NULL,
                    split TEXT NOT NULL,
                    version TEXT,
                    description TEXT,
                    FOREIGN KEY (folder_id) REFERENCES folder_info(id)
                );
            """
            )

    def insert_folder_info(self, folder_infos: list[FolderInfo]):
        with self._connect() as connection:
            cursor = connection.cursor()
            data_tuples = [(info.path, info.num_files) for info in folder_infos]
            cursor.executemany("INSERT INTO folder_info (path, num_files) VALUES (?, ?)", data_tuples)

    def assign_splits(self, version, split_ratios, description=""):
        # if sum(split_ratios) != 1.0:
        #     raise ValueError("Split ratios must sum up to 1.")

        with self._connect() as connection:
            cursor = connection.cursor()

            # Check if this version already exists to avoid duplicate splits
            cursor.execute("SELECT DISTINCT version FROM split_info WHERE version = ?", (version,))
            if cursor.fetchone():
                print(f"Split version '{version}' already exists. Skipping...")
                return

            # Fetch all folder IDs
            cursor.execute("SELECT id FROM folder_info")
            folder_ids = [row[0] for row in cursor.fetchall()]

            import random

            random.shuffle(folder_ids)

            num_folders = len(folder_ids)
            train_end = int(num_folders * split_ratios[0])
            test_end = train_end + int(num_folders * split_ratios[1])

            train_ids = folder_ids[:train_end]
            test_ids = folder_ids[train_end:test_end]
            validate_ids = folder_ids[test_end:]

            # Insert into the split_info table
            self._insert_into_split(cursor, train_ids, "train", version, description)
            self._insert_into_split(cursor, test_ids, "test", version, description)
            self._insert_into_split(cursor, validate_ids, "validate", version, description)

    @staticmethod
    def _insert_into_split(cursor, ids, split_name, version, description):
        for folder_id in ids:
            cursor.execute(
                "INSERT INTO split_info (folder_id, split, version, description) VALUES (?, ?, ?, ?)",
                (folder_id, split_name, version, description),
            )


if __name__ == "__main__":
    db = DatabaseManager()
    db.create_tables()

    original_path = Path("/projects/tcga_tiled/v1/data")

    all_tcgas = original_path.glob("*/*")
    infos_to_insert = []

    for folder_path in tqdm(all_tcgas):
        num_files = find_number_of_tiles(folder_path)
        info = FolderInfo(path=str(folder_path.relative_to(original_path)), num_files=num_files)
        infos_to_insert.append(info)

        if len(infos_to_insert) >= 100:
            db.insert_folder_info(infos_to_insert)
            infos_to_insert = []

    if infos_to_insert:
        db.insert_folder_info(infos_to_insert)

    # Assign splits based on different versions and ratios
    # TODO: We need to pass a function which maps folder to a patient_id (for the split).
    db.assign_splits("v1", (0.8, 0.1, 0.1), "80/10/10 split")
    db.assign_splits("v2", (0.7, 0.2, 0.1), "70/20/10 split")
