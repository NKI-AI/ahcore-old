"""Utilities to describe the dataset to be used and the way it should be parsed."""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def dataclass_to_uuid(data_class) -> str:
    """Create a unique identifier for a dataclass.

    This is done by pickling the object, and computing the sha256 hash of the pickled object.
    There is a very small probability that there is a hash collision, but this is very unlikely, so we ignore this
    possibility.

    Arguments
    ---------
    data_class: dataclass
        The dataclass to create a unique identifier for.

    Returns
    -------
    str
        A unique identifier for the dataclass.
    """
    serialized = pickle.dumps(data_class.__dict__)
    # probability of collision is very small with sha256
    hashed = hashlib.sha256(serialized).hexdigest()
    return hashed


@dataclass(frozen=False)
class DataDescription:
    """General description of the dataset and settings on how these should be sampled."""

    mask_label: Optional[str]
    mask_threshold: Optional[float]  # This is only used for training
    roi_name: Optional[str]
    num_classes: int
    data_dir: Path
    # TODO maybe this is just an env var?
    manifest_database_uri: str
    manifest_name: str
    split_version: str

    annotations_dir: Path

    training_grid: GridDescription
    inference_grid: GridDescription

    index_map: Optional[dict[str, int]]
    remap_labels: Optional[dict[str, str]] = None
    use_class_weights: Optional[bool] = False
    convert_mask_to_rois: bool = True
    use_roi: bool = True


@dataclass(frozen=False)
class GridDescription:
    mpp: Optional[float]
    tile_size: tuple[int, int]
    tile_overlap: tuple[int, int]
    output_tile_size: Optional[tuple[int, int]] = None
