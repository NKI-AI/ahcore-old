# encoding: utf-8
"""Module to create dataset manifests"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Union

import numpy as np
from tqdm import tqdm

from ahcore.cli import dir_path, file_path


def create_tiger_manifest(args: argparse.Namespace):
    """Perform the manifest generation for the TIGER dataset."""
    base_directory = args.base_directory
    output_file = args.output
    manifests = []
    all_images = (base_directory / "images").glob("*.tif")
    for image_fn in tqdm(all_images):
        identifier = image_fn.name[:-4]
        tissue_mask_fn = _parse_path(base_directory / "tissue-masks" / f"{identifier}_tissue.tif", base_directory)
        annotation_mask_fn = _parse_path(
            base_directory / "annotations-tissue-bcss-masks" / f"{identifier}.tif",
            base_directory,
        )
        annotation_xml_fn = _parse_path(
            base_directory / "annotations-tissue-bcss-xmls" / f"{identifier}.xml",
            base_directory,
        )
        if args.use_mask:
            annotation_xml_fn = None
        else:
            annotation_mask_fn = None
        manifests.append(
            _create_tiger_sample(
                image_fn.relative_to(base_directory),
                identifier,
                tissue_mask_fn,
                annotation_mask_fn,
                annotation_xml_fn,
            )
        )

    with open(output_file, "w") as json_file:
        json.dump(manifests, json_file, indent=2)


def _create_tiger_sample(
    image_fn: Path,
    identifier: str,
    tissue_mask_fn: Path | None,
    annotation_mask_fn: Path | None,
    annotation_xml_mask_fn: Path | None,
) -> dict[str, Any]:
    if annotation_mask_fn is not None and annotation_xml_mask_fn is not None:
        raise RuntimeError("Not supported to have both an image mask and an image xml.")

    annotation_image_template = {
        "name": "annotation_mask",
        "filename": str(annotation_mask_fn) if annotation_mask_fn else None,
        "reader": ("IMAGE", (), {"backend": "TIFFFILE"}),
    }
    annotation_xml_template: dict[str, Union[str, None, tuple]] = {
        "name": "annotation_from_xml",
        "filename": str(annotation_xml_mask_fn) if annotation_xml_mask_fn else None,
        "reader": ("ASAP_XML", (), {}),
    }

    template = {
        "image": (str(image_fn), "PYVIPS"),
        "mask": (str(tissue_mask_fn) if tissue_mask_fn else None, "TIFFFILE"),
        "annotations": annotation_xml_template if annotation_xml_mask_fn is not None else annotation_image_template,
        "identifier": identifier,
    }
    return template


def _parse_path(path: Path, base_path: Path):
    if not path.is_file():
        return None
    return path.relative_to(base_path)


def _parse_geojsons(directory, do_not_use_roi):
    geojsons = []
    roi_json = None
    if directory.is_dir():
        _geojsons = directory.glob("*.json")
        for _geojson in _geojsons:
            if _geojson.name == "roi.json":
                if not do_not_use_roi:
                    roi_json = _geojson
                    geojsons.append(_geojson)
                continue
            geojsons.append(_geojson)

    return geojsons, roi_json


def create_tcga_manifest(args: argparse.Namespace):
    """Perform the manifest generation for a TCGA dataset with possible tissue masks and GeoJSON annotations."""
    base_directory = args.base_directory
    annotations_path = args.annotations
    output_file = args.output
    manifests = []
    all_images = base_directory.glob("**/*.svs")
    images_dict = {fn.name[:-4]: fn for fn in tqdm(all_images)}

    for identifier in tqdm(images_dict):
        image_fn = images_dict[identifier]
        annotation_geojson_tcga = annotations_path / identifier

        geojsons, roi_json = _parse_geojsons(annotation_geojson_tcga, args.do_not_use_roi)

        if annotations_path is not None and geojsons == []:
            continue

        annotation_template = {}
        if annotations_path:
            annotation_template = {
                "filenames": [str(_.relative_to(annotations_path)) for _ in geojsons],
                "reader": "GEOJSON",
            }
        mask_template = None
        if roi_json:
            mask_template = {
                "filenames": [str(roi_json.relative_to(annotations_path))],
                "reader": "GEOJSON",
            }

        template = {
            "image": (str(image_fn.relative_to(base_directory)), "PYVIPS"),
            "mask": mask_template,
            "annotations": annotation_template,
            "identifier": identifier,
        }

        manifests.append(template)

    with open(output_file, "w") as json_file:
        json.dump(manifests, json_file, indent=2)


def compute_train_test_split(data_ids, split_train, split_val, shuffle=True):
    # First we need to split across patients
    patient_ids = list(data_ids.keys())
    if shuffle:
        random.shuffle(patient_ids)

    number_of_samples = len(data_ids)

    train_max = math.floor(split_train * number_of_samples / 100)
    val_max = math.floor((split_train + split_val) * number_of_samples / 100)

    train_ids, val_ids, test_ids = patient_ids_to_sample_ids(data_ids, train_max, val_max)
    return train_ids, val_ids, test_ids


def patient_ids_to_sample_ids(data_ids, train_max, val_max):
    patient_ids = data_ids.keys()

    train_ids = []
    val_ids = []
    test_ids = []
    curr_num = 0
    for curr_id in patient_ids:
        if curr_num < train_max:
            train_ids += data_ids[curr_id]
        elif train_max <= curr_num < val_max:
            val_ids += data_ids[curr_id]
        else:
            test_ids += data_ids[curr_id]
        curr_num += len(data_ids[curr_id])

    return train_ids, val_ids, test_ids


def compute_distribution(statistics, data_ids):
    # Filter on data_ids
    statistics = {k: v for k, v in statistics.items() if k in data_ids}
    total_areas = defaultdict(list)

    # Accumulate statistics
    for key in statistics:
        labels = statistics[key]["labels"] + ["background"]
        areas = statistics[key]["areas"]
        for label in labels:
            total_areas[label].append(areas[label])

    mean_area = {k: np.mean(v) for k, v in total_areas.items()}
    total_area = sum(mean_area.values())
    proportion = {k: v / total_area * 100 for k, v in mean_area.items()}

    return proportion


def check_tolerance(
    target_proportions,
    output_proportions,
    abs_tol,
    rel_tol,
    labels,
):
    def is_close(a, b, a_tol, r_tol):
        return np.abs(a - b) <= (a_tol + r_tol * np.abs(b))

    for key in output_proportions:
        if labels and key not in labels:
            continue

        if not is_close(
            target_proportions[key],
            output_proportions[key],
            a_tol=abs_tol,
            r_tol=rel_tol / 100.0,
        ):
            return False

    return True


def compute_train_test_split_mc(
    data_ids,
    statistics,
    split_train,
    split_val,
    abs_tol: float,
    rel_tol: float,
):
    NUM_TRIES = 5000000
    number_of_patients = len(data_ids)

    patient_ids = list(data_ids.keys())

    train_max = math.floor(split_train * number_of_patients / 100)
    val_max = math.floor((split_train + split_val) * number_of_patients / 100)

    average_proportions = compute_distribution(statistics, patient_ids)

    _check_tolerance = partial(
        check_tolerance,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        labels=None,
    )

    for _ in tqdm(range(NUM_TRIES)):
        random.shuffle(patient_ids)

        _train_ids = patient_ids[0:train_max]
        _val_ids = patient_ids[train_max:val_max]
        _test_ids = patient_ids[val_max:]
        train_prop = compute_distribution(statistics, _train_ids)
        train_match = _check_tolerance(average_proportions, train_prop)
        val_prop = compute_distribution(statistics, _val_ids)
        val_match = _check_tolerance(average_proportions, val_prop)
        test_prop = compute_distribution(statistics, _test_ids)
        test_match = _check_tolerance(average_proportions, test_prop)

        if train_match and val_match and test_match:
            output = {
                "avg_proportions": average_proportions,
                "train_proportions": train_prop,
                "val_proportions": val_prop,
                "test_proportions": test_prop,
            }
            train_ids, val_ids, test_ids = patient_ids_to_sample_ids(data_ids, train_max, val_max)
            return output, train_ids, val_ids, test_ids

    raise RuntimeError(f"Could not find a appropriate split within {NUM_TRIES} tries.")


def parse_args_split(args):
    split = args.split
    split_train = 0
    split_val = 0
    split_test = 0
    if len(split) >= 1:
        split_train = split[0]
    if len(split) >= 2:
        split_val = split[1]
    if len(split) == 3:
        split_test = split[2]

    if not split_train + split_test + split_val == 100:
        raise argparse.ArgumentTypeError("Split must add up to 100.")

    return split_train, split_test, split_val


def train_test_split(args: argparse.Namespace):
    with open(args.manifest, "r") as f:
        manifest = json.load(f)

    split_train, split_test, split_val = parse_args_split(args)

    data_ids = defaultdict(list)
    for curr_manifest in manifest:
        identifier = curr_manifest["identifier"]
        patient_identifier = identifier[:12]
        if not args.use_all:
            if not curr_manifest["annotations"]:
                continue
        data_ids[patient_identifier].append(identifier)
    _data_ids = dict(data_ids)

    output = {}
    if args.dataset_statistics is None:
        train_ids, val_ids, test_ids = compute_train_test_split(_data_ids, split_train, split_val, shuffle=args.shuffle)

    else:
        with open(args.dataset_statistics, "r") as json_file:
            statistics = json.load(json_file)
            computed_stats, train_ids, val_ids, test_ids = compute_train_test_split_mc(
                _data_ids,
                statistics,
                split_train,
                split_val,
                abs_tol=args.abs_tol,
                rel_tol=args.rel_tol,
            )
            output["metadata"] = computed_stats

    ids_train = {k: "fit" for k in train_ids}
    ids_val = {k: "validate" for k in val_ids}
    ids_test = {k: "test" for k in test_ids}

    output["split"] = {**ids_train, **ids_val, **ids_test}

    print(json.dumps(output, indent=2))


def register_parser(parser: argparse._SubParsersAction):
    """Register manifest commands to a root parser."""
    manifest_parser = parser.add_parser("manifest", help="manifest parser")
    manifest_subparsers = manifest_parser.add_subparsers(help="manifest subparser")
    manifest_subparsers.required = True
    manifest_subparsers.dest = "subcommand"

    _parser: argparse.ArgumentParser = manifest_subparsers.add_parser(
        "create-tiger", help="Create manifest for the TIGER dataset."
    )

    _parser.add_argument(
        "base_directory",
        type=dir_path,
        help="base directory to the TIGER dataset",
    )
    _parser.add_argument(
        "--output",
        type=pathlib.Path,
        default="tiger_manifest.json",
        help="Where to write the JSON output.",
    )
    _parser.add_argument(
        "--use-mask",
        action="store_true",
        help="Use mask rather than XML.",
    )
    _parser.set_defaults(subcommand=create_tiger_manifest)

    # Parser for tcga.
    _parser = manifest_subparsers.add_parser("create-tcga", help="Create manifest for a TCGA dataset.")
    _parser.add_argument(
        "base_directory",
        type=dir_path,
        help="base directory to the TCGA dataset",
    )
    _parser.add_argument(
        "--annotations",
        type=dir_path,
        required=False,
        help="Directory to the GeoJSONs dataset. If not set no annotations will be added. Will only add annotated"
        " images if set",
    )
    _parser.add_argument(
        "--do-not-use-roi",
        action="store_true",
        help="Ignore the ROI. Always assuming there is a file `roi.json` in the directory.",
    )
    _parser.add_argument(
        "--output",
        type=pathlib.Path,
        default="tcga_manifest.json",
        help="Where to write the JSON output.",
    )
    _parser.set_defaults(subcommand=create_tcga_manifest)

    _parser = manifest_subparsers.add_parser("train-test-split", help="Create a simple train-test-split.")
    _parser.add_argument(
        "manifest",
        type=pathlib.Path,
        help="Path to manifest.",
    )
    _parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle before creating the split.",
    )
    _parser.add_argument(
        "--use-all",
        action="store_true",
        help="Use all of data description, also the ones without annotations.",
    )
    _parser.add_argument(
        "--split",
        nargs="+",
        type=float,
        help="Give the percentage of samples in train, val, test. If missing assumed to be 0.",
        required=True,
    )
    _parser.add_argument(
        "--dataset-statistics",
        type=file_path,
        help="If given, will create balanced split based on the areas (in mm2) of the dataset",
        required=False,
    )
    _parser.add_argument(
        "--abs-tol",
        type=float,
        help="Absolute tolerance for proportions. "
        "Only used when `--dataset-statistics` is set. Value between 0 and 100.",
        default=5.0,
    )
    _parser.add_argument(
        "--rel-tol",
        type=float,
        help="Relative tolerance for proportions. "
        "Only used when `--dataset-statistics` is set. Value between 0 and 100.",
        default=5.0,
    )

    _parser.set_defaults(subcommand=train_test_split)
