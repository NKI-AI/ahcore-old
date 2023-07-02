# encoding: utf-8
"""Module to write masks from annotation files"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from dlup._image import Resampling
from dlup.writers import TiffCompression, TifffileImageWriter
from tqdm import tqdm

from ahcore.cli import dir_path, file_path
from ahcore.transforms.pre_transforms import ConvertAnnotationsToMask
from ahcore.utils.data import DataDescription, GridDescription
from ahcore.utils.manifest import ImageManifest, image_manifest_to_dataset


def write_mask(args: argparse.Namespace):
    with open(args.manifest, "r") as json_file:
        json_manifests = json.load(json_file)

    for json_manifest in tqdm(json_manifests):
        iterator, metadata = create_wsi_iterator(args, json_manifest)
        path = args.output_directory / str(metadata["identifier"] + "_mask.tif")

        if args.mpp is None:
            mpp = (metadata["slide_mpp"], metadata["slide_mpp"])
        else:
            mpp = (args.mpp, args.mpp)

        tiffwriter = TifffileImageWriter(
            path,
            size=metadata["scaled_size"],
            mpp=mpp,
            tile_size=(args.tile_size, args.tile_size),
            pyramid=True,
            compression=TiffCompression.LZW,
            interpolator=Resampling.NEAREST,
        )
        tiffwriter.from_tiles_iterator(iterator())


def create_wsi_iterator(args: argparse.Namespace, json_manifest: dict):
    curr_manifest = ImageManifest(**json_manifest)
    # Disable the mask
    curr_manifest.mask = None

    index_map = {}
    for line in args.label_map.split(","):
        key, value = line.split("=")
        if not value.isdigit():
            raise argparse.ArgumentTypeError("Need a key value pair of string, int. Got {key}, {value}.")
        index_map[key.strip()] = int(value.strip())

    transform = ConvertAnnotationsToMask(roi_name="roi", index_map=index_map)

    # TODO: Enable in dlup not to load the image (or pass to the output)
    # TODO: Allow empty values for most of the below
    data_description = DataDescription(
        data_dir=args.image_directory,
        manifest_path=Path(""),
        dataset_split_path=Path(""),
        mask_label="mask",
        mask_threshold=0.5,
        roi_name="roi",
        num_classes=5,
        annotations_dir=Path(""),
        training_grid=GridDescription(mpp=None, tile_size=(1024, 1024), tile_overlap=(0, 0), output_tile_size=None),
        inference_grid=GridDescription(mpp=None, tile_size=(1024, 1024), tile_overlap=(0, 0), output_tile_size=None),
        index_map={},
    )

    ds = image_manifest_to_dataset(
        data_description,
        manifest=curr_manifest,
        mpp=args.manifest,
        tile_size=(args.tile_size, args.tile_size),
        tile_overlap=(0, 0),
        output_tile_size=None,
        transform=transform,
    )

    scaling = ds.slide_image.get_scaling(args.mpp)

    metadata = {
        "output_mpp": args.mpp,
        "slide_mpp": ds.slide_image.mpp,
        "shape": ds.slide_image.size,
        "scaled_size": ds.slide_image.get_scaled_size(scaling),
        "identifier": curr_manifest.identifier,
    }

    def iterator():
        for sample in tqdm(ds, leave=False):
            roi = sample["annotation_data"]["roi"]
            mask = sample["annotation_data"]["mask"]
            mask = np.where(roi == 1, mask, 0).astype(np.uint8)

            yield mask

    return iterator, metadata


def register_parser(parser: argparse._SubParsersAction):
    """Register inspect commands to a root parser."""
    inspect_parser = parser.add_parser("mask", help="Utilities to work with masks")
    inspect_subparsers = inspect_parser.add_subparsers(help="Mask subparser")
    inspect_subparsers.required = True
    inspect_subparsers.dest = "subcommand"

    _parser: argparse.ArgumentParser = inspect_subparsers.add_parser(
        "convert", help="Convert annotations to tiff masks"
    )
    _parser.add_argument(
        "image_directory",
        type=dir_path,
        help="Directory to the images.",
    )
    _parser.add_argument(
        "annotation_directory",
        type=dir_path,
        help="Directory to the annotations.",
    )
    _parser.add_argument(
        "manifest",
        type=file_path,
        help="Path to the manifest.",
    )
    _parser.add_argument(
        "output_directory",
        type=dir_path,
        help="Directory to write output masks.",
    )
    _parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tiff output tile size",
    )
    _parser.add_argument(
        "--label-map",
        type=str,
        help="Map of the form label1=output_number,label2=... denoting the labels to be used.",
        required=True,
    )
    _parser.add_argument(
        "--mpp",
        type=float,
        help="mpp of the output. If not set will select native slide mpp.",
    )
    _parser.set_defaults(subcommand=write_mask)


if __name__ == "__main__":
    root_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    root_subparsers = root_parser.add_subparsers(help="Possible ahcore CLI utils to run.")
    root_subparsers.required = True
    root_subparsers.dest = "subcommand"
    register_parser(root_subparsers)

    args = root_parser.parse_args()
    args.subcommand(args)
