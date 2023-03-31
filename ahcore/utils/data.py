# encoding: utf-8
"""Utilities to describe the dataset to be used and the way it should be parsed."""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Optional, Protocol, TypedDict

from dlup.data.dataset import TiledROIsSlideImageDataset


class _IsDataclass(Protocol):
    # From https://stackoverflow.com/a/55240861
    __dataclass_fields__: ClassVar[Dict]


def dataclass_to_uuid(data_class: _IsDataclass) -> str:
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
    manifest_path: Path
    dataset_split_path: Optional[Path]
    center_info_path: Optional[Path]
    centers: Optional[list[str]]
    annotations_dir: Path
    compute_staining: bool

    training_grid: GridDescription
    inference_grid: GridDescription

    index_map: Optional[dict[str, int]]
    max_val_tiffs: Optional[int] = None
    extract_center: Optional[bool] = False
    remap_labels: Optional[dict[str, str]] = None
    colors: Optional[dict[str, str]] = None
    use_class_weights: Optional[bool] = False
    normalize_mean: Optional[list[float]] = None
    normalize_std: Optional[list[float]] = None
    convert_mask_to_rois: bool = True
    use_roi: bool = True


@dataclass(frozen=False)
class GridDescription:
    mpp: Optional[float]
    tile_size: tuple[int, int]
    tile_overlap: tuple[int, int]
    output_tile_size: Optional[tuple[int, int]] = None


class InferenceMetadata(TypedDict):
    mpp: float | None
    size: tuple[int, int] | None
    tile_size: tuple[int, int] | None
    filename: Path | None


def create_inference_metadata(
    dataset: TiledROIsSlideImageDataset, target_mpp: float | None, tile_size: tuple[int, int]
) -> InferenceMetadata:
    """Create the metadata for inference.

    Arguments
    ---------
    dataset : TiledROIsSlideImageDataset
        The dataset to create the metadata for.
    target_mpp : float | None
        The microns-per-pixel to be used for the tiles. If not set, will select the level 0 resolution.
    tile_size : tuple[int, int]
        The size of the tiles.
    """

    scaling = dataset.slide_image.get_scaling(target_mpp)
    path = Path(dataset.slide_image.identifier)
    filename = Path(path.parent.stem) / path.stem
    scaled_size = dataset.slide_image.get_scaled_size(scaling)

    mpp = target_mpp if target_mpp is not None else dataset.slide_image.mpp
    metadata = InferenceMetadata(filename=filename, tile_size=tile_size, mpp=mpp, size=scaled_size)

    return metadata
