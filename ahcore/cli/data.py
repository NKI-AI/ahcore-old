# encoding: utf-8
"""Module to write copy manifests files over to SCRATCH directory"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from rich.progress import Progress

from ahcore.cli import dir_path, file_path


def _quick_hash(file_path: Path, max_bytes: int = 10**6) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        block = f.read(max_bytes)
        hasher.update(block)
    return hasher.hexdigest()


def copy_data(args: argparse.Namespace):
    manifest_fn = args.manifest_fn
    base_dir = args.base_dir
    dataset_name = args.dataset_name
    target_dir = os.environ.get("SCRATCH", None)

    if target_dir is None or not os.access(target_dir, os.W_OK):
        print("Please set the SCRATCH environment variable to a writable directory.")
        sys.exit(1)

    with open(manifest_fn, "r") as f:
        manifests = json.load(f)

    with Progress() as progress:
        task = progress.add_task("[cyan]Copying...", total=len(manifests))

        for manifest in manifests:
            image_fn = manifest["image"][0]
            get_from = base_dir / image_fn
            write_to = Path(target_dir) / dataset_name / image_fn

            write_to.parent.mkdir(parents=True, exist_ok=True)
            if write_to.exists():
                # compute the hash of previous and new file
                old_hash = _quick_hash(write_to)
                new_hash = _quick_hash(get_from)
                if old_hash == new_hash:
                    # Skip if they are the same
                    progress.console.log("Skipping file as it already exists: {}".format(image_fn))
                    progress.update(task, advance=1)
                    continue

            # Copy file from get_from to write_to
            shutil.copy(get_from, write_to)
            progress.update(task, advance=1)


def register_parser(parser: argparse._SubParsersAction):
    """Register inspect commands to a root parser."""
    data_parser = parser.add_parser("data", help="Data utilities")
    data_subparsers = data_parser.add_subparsers(help="Data subparser")
    data_subparsers.required = True
    data_subparsers.dest = "subcommand"

    _parser: argparse.ArgumentParser = data_subparsers.add_parser(
        "copy-data-from-manifest",
        help="Copy the data to a different drive based on the manifest. The data will be copied over to $SCRATCH / DATASET_NAME",
    )

    _parser.add_argument(
        "manifest_fn",
        type=file_path,
        help="Path to the ahcore manifest.",
    )
    _parser.add_argument(
        "base_dir",
        type=dir_path(require_writable=True),
        help="Directory to which the paths defined in the manifest are relative to.",
    )
    _parser.add_argument(
        "dataset_name",
        type=file_path,
        help="Name of the dataset to copy the data to. The.",
    )

    _parser.set_defaults(subcommand=copy_data)