import os
import sqlite3
from pathlib import Path
from pydantic import BaseModel
from tqdm import tqdm
import json

def find_number_of_tiles(folder_path: Path) -> int:
    with open(folder_path / "meta_data_tiles.json") as json_file:
        meta_data_tiles = json.load(json_file)
        return meta_data_tiles["num_regions_masked"]


def initialize_database():
    connection = sqlite3.connect("folder_data.db")
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS folder_info (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            num_files INTEGER NOT NULL
        );
    """
    )
    connection.commit()
    connection.close()


class FolderInfo(BaseModel):
    path: str
    num_files: int


class DatabaseManager:
    def __init__(self, db_name="folder_data.db"):
        self.db_name = db_name

    def insert(self, folder_infos: list[FolderInfo]):
        connection = sqlite3.connect(self.db_name)
        cursor = connection.cursor()

        for info in folder_infos:
            cursor.execute("INSERT INTO folder_info (path, num_files) VALUES (?, ?)", (info.path, info.num_files))

        connection.commit()
        connection.close()

    def bulk_insert(self, folder_infos: list[FolderInfo]):
        connection = sqlite3.connect(self.db_name)
        cursor = connection.cursor()

        data_tuples = [(info.path, info.num_files) for info in folder_infos]

        cursor.executemany("INSERT INTO folder_info (path, num_files) VALUES (?, ?)", data_tuples)

        connection.commit()
        connection.close()


if __name__ == "__main__":
    # Using the classes and functions:
    initialize_database()
    db = DatabaseManager()

    all_tcgas = Path("/projects/tcga_tiled/v1/data").glob("*/*")
    infos_to_insert = []  # List to store FolderInfo objects
    print("Will do stuff.")
    for folder_path in tqdm(all_tcgas):
        num_files = find_number_of_tiles(folder_path)

        # Using Pydantic to validate and serialize the data
        info = FolderInfo(path=str(folder_path), num_files=num_files)

        infos_to_insert.append(info)

        # When the list reaches a certain size, perform a bulk insert and clear the list
        if len(infos_to_insert) >= 100:
            db.bulk_insert(infos_to_insert)
            infos_to_insert = []

    # Insert any remaining FolderInfo objects in the list
    if infos_to_insert:
        db.bulk_insert(infos_to_insert)