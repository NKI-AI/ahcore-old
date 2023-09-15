"""
All utilities to parse manifests into datasets. A manifest is a database containing the description of a dataset.
See the documentation for more information and examples.

TODO: Definitely imporove the logging here.

"""

from __future__ import annotations

import functools
import warnings
from enum import Enum
from pathlib import Path
from typing import Callable, TypedDict

from dlup import SlideImage
from dlup.annotations import WsiAnnotations
from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.experimental_backends import ImageBackend
from dlup.tiling import GridOrder, TilingMode
from pytorch_lightning.trainer.states import TrainerFn

from ahcore.utils.data import DataDescription
from ahcore.utils.io import get_logger
from ahcore.utils.manifest_database import DataManager, open_db
from ahcore.utils.rois import compute_rois

logger = get_logger(__name__)


class _AnnotationReadersDict(TypedDict):
    ASAP_XML: Callable
    GEOJSON: Callable
    PYVIPS: Callable
    TIFFFILE: Callable
    OPENSLIDE: Callable


_AnnotationReaders: _AnnotationReadersDict = {
    "ASAP_XML": WsiAnnotations.from_asap_xml,
    "GEOJSON": WsiAnnotations.from_geojson,
    "PYVIPS": functools.partial(SlideImage.from_file_path, backend=ImageBackend.PYVIPS),
    "TIFFFILE": functools.partial(SlideImage.from_file_path, backend=ImageBackend.TIFFFILE),
    "OPENSLIDE": functools.partial(SlideImage.from_file_path, backend=ImageBackend.OPENSLIDE),
}

_ImageBackends = {
    "PYVIPS": ImageBackend.PYVIPS,
    "TIFFFILE": ImageBackend.TIFFFILE,
    "OPENSLIDE": ImageBackend.OPENSLIDE,
}

_AnnotationReaders_names: list[tuple[str, str]] = [(field, field) for field in _AnnotationReaders.keys()]
_ImageBackends_names: list[tuple[str, str]] = [(field, field) for field in _ImageBackends.keys()]

AnnotationReaders = Enum(value="AnnotationReaders", names=_AnnotationReaders_names)  # type: ignore
ImageBackends = Enum(value="ImageBackends", names=_ImageBackends_names)  # type: ignore

_Stages = Enum("Stages", [(_, _) for _ in ["fit", "validate", "test", "predict"]])  # type: ignore


def _parse_annotations(annotations_root: Path, record):
    if record is None:
        return
    assert len(record) == 1

    reader = record[0].reader
    filename = record[0].filename

    if reader == "GEOJSON":
        return WsiAnnotations.from_geojson(annotations_root / filename)
    else:
        raise NotImplementedError


def _get_rois(mask, data_description: DataDescription, stage: str):
    if (mask is None) or (stage != TrainerFn.FITTING) or (not data_description.convert_mask_to_rois):
        return None

    tile_size = data_description.training_grid.tile_size
    tile_overlap = data_description.training_grid.tile_overlap

    return compute_rois(mask, tile_size=tile_size, tile_overlap=tile_overlap, centered=True)


def datasets_from_data_description(db_manager: DataManager, data_description: DataDescription, transform, stage: str):
    logger.info(
        f"Reading manifest from {data_description.manifest_database_path} for stage {stage} (type={type(stage)})"
    )

    image_root = data_description.data_dir
    annotations_root = data_description.annotations_dir

    if stage == TrainerFn.FITTING:
        grid_description = data_description.training_grid
    else:
        grid_description = data_description.inference_grid

    records = db_manager.get_records_by_split(
        manifest_name=data_description.manifest_name,
        split_version=data_description.split_version,
        split_category=stage,
    )

    for record in records:
        labels = [(label.key, label.value) for label in record.labels] if record.labels else None

        for image in record.images:
            mask = _parse_annotations(annotations_root, image.masks)
            annotations = _parse_annotations(annotations_root, image.annotations)
            rois = _get_rois(mask, data_description, stage)
            mask_threshold = 0.0 if stage != TrainerFn.FITTING else data_description.mask_threshold

            dataset = TiledROIsSlideImageDataset.from_standard_tiling(
                path=image_root / image.filename,
                mpp=grid_description.mpp,
                tile_size=grid_description.tile_size,
                tile_overlap=grid_description.tile_overlap,
                tile_mode=TilingMode.overflow,
                grid_order=GridOrder.C,
                crop=False,
                mask=mask,
                mask_threshold=mask_threshold,
                output_tile_size=getattr(grid_description, "output_tile_size", None),
                rois=rois,  # type: ignore
                annotations=annotations if stage != TrainerFn.PREDICTING else None,
                labels=labels,
                transform=transform,
                backend=ImageBackend[image.reader],
                overwrite_mpp=(image.mpp, image.mpp),
                limit_bounds=False if rois is not None else True,
            )

            logger.info(
                "Added dataset with length %s (filename=%s, original mpp=%s).",
                len(dataset),
                image.filename,
                image.mpp,
            )

            yield dataset
