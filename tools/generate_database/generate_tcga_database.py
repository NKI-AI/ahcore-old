# encoding: utf-8
from __future__ import annotations

import json
import random
import os
from pathlib import Path
from tqdm import tqdm

from _database_manager import DatabaseManager
from _database_manager import FolderInfo, LabelCategoryInfo, LabelValueInfo, PatientLabelAssignment

TILES_ROOT_PATH = "/projects/tcga_tiled/v1/data"
META_ROOT_PATH = "/data/groups/aiforoncology/archive/pathology/TCGA/metadata"
BATCH_INSERT_SIZE = 100
DEBUG = False


def find_number_of_tiles(folder_path: Path) -> int:
    with open(folder_path / "meta_data_tiles.json") as json_file:
        meta_data_tiles = json.load(json_file)
        return meta_data_tiles["num_regions_masked"]


def get_patient_code(path: Path) -> str:
    return path.name[:12]


def populate_with_tcga_tiled(db: DatabaseManager) -> None:
    tiles_root_path = Path(TILES_ROOT_PATH)
    all_tcgas = tiles_root_path.glob("*/*")
    infos_to_insert = []

    for idx, folder_path in tqdm(enumerate(all_tcgas)):
        num_files = find_number_of_tiles(folder_path)
        patient_code = get_patient_code(folder_path)
        info = FolderInfo(
            path=str(folder_path.relative_to(tiles_root_path)), num_files=num_files, patient_code=patient_code
        )
        infos_to_insert.append(info)

        if len(infos_to_insert) >= BATCH_INSERT_SIZE:
            db.insert_folder_info(infos_to_insert)
            infos_to_insert = []

        if DEBUG:
            # Easy for debugging.
            if idx > 100:
                break

    if infos_to_insert:
        db.insert_folder_info(infos_to_insert)



def populate_with_tcga_labels(db: DatabaseManager) -> None:
    root_label_data = os.path.join(os.path.dirname(__file__), "database_label_data")
    with open(Path(root_label_data) / "tcga_study_codes.txt" , "r", encoding="utf-8") as file:
        study_codes = [line.strip().split("\t")[0] for line in file.readlines() if line != ""]

    # Defining categories and their respective values
    categories = [
        LabelCategoryInfo(description="TCGA study codes", slug="tcga_study_codes", values=study_codes),
    ]

    # Inserting the categories and their values into the database
    for category_info in categories:
        if len(category_info.values) != len(set(category_info.values)):
            print(f"Could not add label category {category_info.slug}. Label values within a category must be unique. Skipping...")
            continue

        db.insert_label_category(category_info)

        for value in category_info.values:
            value_info = LabelValueInfo(label_category_slug=category_info.slug, value=value)
            db.insert_label_value(value_info)


def assign_labels_to_patients(db: DatabaseManager) -> None:
    meta_root_path = Path(META_ROOT_PATH)
    meta_basic_path = meta_root_path / "metadata_basic"

    meta_basic_files = meta_basic_path.glob("**/*diagnostic*.txt")
    patient_study_code_mapping = {}
    for meta_file_path in meta_basic_files:
        with open(meta_file_path, "r") as txt_file:
            lines = txt_file.readlines()
            for line in lines[1:]: # Skip header
                columns = line.split("\t")
                patient_code = columns[2]
                study_code = columns[3][5:]

                patient_study_code_mapping[patient_code] = study_code

    assignments = []
    for patient_code, study_code in patient_study_code_mapping.items():
        assignments.append(PatientLabelAssignment(
                                patient_code=patient_code,
                                label_category_slug="tcga_study_codes",
                                label_value=study_code,
                            ))

    # Assign the labels to the patients in the database.
    for patient_label_assignment in assignments:
        db.assign_label_to_patient(patient_label_assignment, strict=False)



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
    for category_info in categories:
        db.insert_label_category(category_info)
        for value in category_info.values:
            value_info = LabelValueInfo(label_category_slug=category_info.slug, value=value)
            db.insert_label_value(value_info)


def assign_dummy_labels_to_patients(db: DatabaseManager) -> None:
    # Dummy label assignments for demonstration purposes
    dummy_assignments = (
        [
            PatientLabelAssignment(
                patient_code=patient_code,
                label_category_slug="tumor_type",
                label_value=random.choice(["Melanoma", "Carcinoma", "Sarcoma"]),
            )
            for patient_code in ["TCGA-V4-A9EC", "TCGA-06-0148", "TCGA-AO-A12F"]
        ]
        + [
            PatientLabelAssignment(
                patient_code=patient_code,
                label_category_slug="tumor_stage",
                label_value=random.choice(["I", "II", "III", "IV"])
            )
            for patient_code in ["TCGA-V4-A9EC", "TCGA-06-0148", "TCGA-AO-A12F"]
        ]
        + [
            PatientLabelAssignment(
                patient_code=patient_code,
                label_category_slug="treatment_response",
                label_value=random.choice(["Positive", "Negative", "Neutral"]),
            )
            for patient_code in ["TCGA-V4-A9EC", "TCGA-06-0148", "TCGA-AO-A12F"]
        ]
    )

    # Assign the labels to the patients
    for assignment in dummy_assignments:
        db.assign_label_to_patient(assignment)


if __name__ == "__main__":
    db = DatabaseManager()
    db.create_tables()
    populate_with_tcga_tiled(db)

    # Assign splits based on different versions and ratios
    db.create_random_split("v1", (80, 10, 10), "80/10/10 split")
    db.create_random_split("v2", (70, 20, 10), "70/20/10 split")

    populate_with_tcga_labels(db)
    assign_labels_to_patients(db)


    # populate_label_categories_and_values_with_dummies(db)
    # assign_dummy_labels_to_patients(db)

    # Example, get all patients with a specific label.
    label_category_slug = "tcga_study_codes"

    filtered_patients = db.get_patients_by_label_category(label_category_slug)

    # Create split that has the patients in there
    db.create_random_split("v3", (80, 10, 10), "label split", filtered_patients)