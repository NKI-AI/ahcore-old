# encoding: utf-8
"""
Utility to merge a collection of geojson into one.
"""
import argparse
import json
import shutil
from pathlib import Path

from dlup.annotations import WsiAnnotations


def convert_to_combined(path: Path) -> dict:
    """
    Convert a single geojson file to the combined format.
    """
    all_geojsons = list([_ for _ in path.glob("*.json") if _.name != "roi.json"])

    annotations = WsiAnnotations.from_geojson(all_geojsons)
    return annotations.as_geojson(split_per_label=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input directory containing the geojson files.",
    )
    parser.add_argument("output", type=Path, help="Path to the output file.")
    args = parser.parse_args()

    output = convert_to_combined(args.input)
    args.output.mkdir(exist_ok=True, parents=True)
    with open(args.output / "annotations.json", "w") as f:
        json.dump(output, f, indent=2)

    # copy over the roi.json
    roi = args.input / "roi.json"
    if roi.exists():
        roi_destination = args.output / "roi.json"
        shutil.copy(roi, roi_destination)
