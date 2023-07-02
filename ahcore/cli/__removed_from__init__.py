# coding=utf-8
# Copyright (c) dlup contributors
"""Ahcore Command-line interface. This is the file which builds the main parser."""
# ORIGINAL FILE WHICH HAS SHRUNKEN.
from __future__ import annotations

import argparse
import pathlib


def dir_path(path: str) -> pathlib.Path:
    """Check if the path is a valid directory.

    Parameters
    ----------
    path : str

    Returns
    -------
    pathlib.Path
        The path as a pathlib.Path object.
    """
    _path = pathlib.Path(path)
    if _path.is_dir():
        return _path
    raise argparse.ArgumentTypeError(f"{path} is not a valid directory.")


def file_path(path: str) -> pathlib.Path:
    """Check if the path is a valid file.

    Parameters
    ----------
    path : str

    Returns
    -------
    pathlib.Path
        The path as a pathlib.Path object.

    """
    _path = pathlib.Path(path)
    if _path.is_file():
        return _path
    raise argparse.ArgumentTypeError(f"{path} is not a valid file.")


def main() -> None:
    """
    Console script for ahcore.
    """
    # From https://stackoverflow.com/questions/17073688/how-to-use-argparse-subparsers-correctly
    root_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    root_subparsers = root_parser.add_subparsers(help="Possible ahcore CLI utils to run.")
    root_subparsers.required = True
    root_subparsers.dest = "subcommand"

    # Prevent circular import
    from ahcore.cli.manifest import register_parser as register_manifest_subcommand

    # Whole manifest related commands.
    register_manifest_subcommand(root_subparsers)

    # Prevent circular import
    # from ahcore.cli.inspect import register_parser as register_inspect_parser

    # Data inspection commands
    # register_inspect_parser(root_subparsers)

    # Prevent circular import
    from ahcore.cli.mask import register_parser as register_mask_parser

    # Mask writing commands
    register_mask_parser(root_subparsers)

    args = root_parser.parse_args()
    args.subcommand(args)


if __name__ == "__main__":
    main()
