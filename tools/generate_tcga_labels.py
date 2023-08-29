# encoding: utf-8
"""The eventual goal of this file is to create a manifest (perhaps an SQLite database?) that maps TCGA ids to labels and splits."""

# https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/sample-type-codes
import json
import sqlite3
from pathlib import Path

from rich.progress import Progress
from tqdm import tqdm

CHUNK_SIZE = 250  # This can be adjusted based on your requirements.


def setup_database(root: Path):
    """Setup SQLite database and tables."""
    conn = sqlite3.connect(root / "metadata.db")
    cursor = conn.cursor()

    # Create tables if they don't exist yet.
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS svs_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        svs_folder TEXT UNIQUE
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS tile_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        svs_id INTEGER,
        FOREIGN KEY(svs_id) REFERENCES svs_metadata(id)
    )
    """
    )

    conn.commit()
    return conn


def get_tcga_study_code_mapping():
    tcga_study_codes: dict[str, str] = {}
    with open("tcga_study_codes.txt") as f:
        data = f.readlines()
        for line in data:
            if not line.strip():
                continue
            study_code, study_name = line.strip().split("\t")
            tcga_study_codes[study_code] = study_name
    return tcga_study_codes


def generate_labels(tcga_files_to_study_types, tcga_study_codes):
    # So basically we want to have {case_id}/{file_name},{study_type}
    with open("tcga_labels.txt", "w") as f:
        for case in tcga_files_to_study_types:
            assert len(case["cases"]) == 1
            case_id = case["cases"][0]["case_id"]
            project_id = case["cases"][0]["project"]["project_id"][5:]
            study_type = tcga_study_codes[project_id]
            assert project_id in tcga_study_codes

            file_name = case["file_name"]

            f.write(f"{case_id}/{file_name},{study_type}\n")


def chunk_data(tile_data, chunk_size):
    """Chunk the tile data into smaller pieces for easier processing."""
    keys = list(tile_data.keys())
    for i in range(0, len(keys), chunk_size):
        yield {key: tile_data[key] for key in keys[i : i + chunk_size]}


def write_to_database(conn, chunks_gen):
    cursor = conn.cursor()
    for tile_dict in chunks_gen:
        for svs_folder, tiles in tile_dict.items():
            cursor.execute(
                "INSERT OR IGNORE INTO svs_metadata (svs_folder) VALUES (?)",
                (svs_folder,),
            )
            cursor.execute("SELECT id FROM svs_metadata WHERE svs_folder=?", (svs_folder,))
            svs_id = cursor.fetchone()[0]
            cursor.executemany(
                "INSERT INTO tile_files (filename, svs_id) VALUES (?, ?)",
                [(tile, svs_id) for tile in tiles],
            )
    conn.commit()


if __name__ == "__main__":
    meta_root = Path("/projects/tcga_tiled/v1/meta")
    conn = setup_database(meta_root)
    tcga_study_codes = get_tcga_study_code_mapping()

    # Now we need to map identifer/tcga-code to the label.
    with open("tcga_files_to_study_types.2023-08-20.json", "r", encoding="utf-8") as json_file:
        tcga_files_to_study_types = json.load(json_file)

    generate_labels(tcga_files_to_study_types, tcga_study_codes)

    print("Collecting metadata...")
    tile_chunks_gen = construct_tile_dict(CHUNK_SIZE)
    print("Tile data loaded")
    write_to_database(conn, tile_chunks_gen)

    conn.close()
