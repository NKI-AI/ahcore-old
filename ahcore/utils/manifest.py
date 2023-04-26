"""
All utilities to parse manifests into datasets. A manifest is a JSON file containing the description of a dataset.
See the documentation for more information and examples.

"""

from __future__ import annotations

import functools
import json
import warnings
from enum import Enum
from pathlib import Path
from typing import Callable, NamedTuple, Optional, TypedDict, Union

import pydantic
from dlup import SlideImage
from dlup.annotations import WsiAnnotations
from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.experimental_backends import ImageBackend
from dlup.tiling import GridOrder, TilingMode
from pydantic import BaseModel
from pytorch_lightning.trainer.states import TrainerFn

from ahcore.utils.data import DataDescription
from ahcore.utils.io import get_logger
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

AnnotationReaders = Enum(
    value="AnnotationReaders", names=[(field, field) for field in _AnnotationReaders.keys()]
)  # type: ignore
ImageBackends = Enum(value="ImageBackends", names=[(field, field) for field in _ImageBackends.keys()])  # type: ignore


class AnnotationModel(BaseModel):
    filenames: list[Path]
    reader: AnnotationReaders


class ImageManifest(BaseModel):
    image: tuple[Path, ImageBackends]
    identifier: str
    mask: Optional[AnnotationModel]
    annotations: Optional[AnnotationModel]
    labels: Optional[dict[str, Union[int, float, str, bool]]]
    mpp: Optional[float]


_Stages = Enum("Stages", [(_, _) for _ in ["fit", "validate", "test", "predict"]])  # type: ignore


class DataSplit(BaseModel):
    metadata: Optional[dict]
    split: dict[str, _Stages]


class SplittedManifest(NamedTuple):
    """The manifest split across the different model phases fit, validate, test and predict"""

    fit: list[ImageManifest]
    validate: list[ImageManifest]
    test: list[ImageManifest]
    predict: list[ImageManifest]


def read_json_manifest(json_fn: Path) -> list[ImageManifest]:
    """Read a json manifest and return the data as a list of `ImageManifest`'s.

    Arguments
    ----------
    json_fn: Path
        The path to the json manifest file.

    Returns
    -------
    list[ImageManifest]
        A list of `ImageManifest`'s.
    """
    with open(json_fn, "r") as json_file:
        data = json.load(json_file)

    return pydantic.parse_obj_as(list[ImageManifest], data)


def read_json_split_file(json_fn: Path) -> DataSplit:
    """Read a json split file and return the data as a `DataSplit`.

    Arguments
    ----------
    json_fn: Path
        The path to the json split file.

    Returns
    -------
    DataSplit
        A `DataSplit` object.
    """
    with open(json_fn, "r") as json_file:
        data = json.load(json_file)

    _parsed_file = DataSplit.parse_obj(data)

    return _parsed_file


def manifests_from_data_description(data_description: DataDescription) -> SplittedManifest:
    """Create a `SplittedManifest` from a `DataDescription`. Basically this returns a NamedTuple with "fit", "validate",
    "test" and "predict" as keys and a list of `ImageManifest`'s as values.

    Parameters
    ----------
    data_description: DataDescription
        The `DataDescription` object. This contains the values of the absolute paths to the data.
        If the data_description does not contain a split_path, the data is assumed to be in the predict stage.

    Returns
    -------
    SplittedManifest
        A `SplittedManifest` object.

    """
    logger.info(f"Reading manifest from {data_description.manifest_path}")
    manifest = read_json_manifest(data_description.manifest_path)

    splitted_manifest = SplittedManifest(fit=[], validate=[], test=[], predict=[])
    if data_description.dataset_split_path is not None:
        logger.info(f"Reading split from {data_description.dataset_split_path}")
        dataset_split = read_json_split_file(data_description.dataset_split_path)
        split = dataset_split.split
        for image_manifest in manifest:
            if image_manifest.identifier not in split:
                warnings.warn(f"Image identifier {image_manifest.identifier} not found in split. Skipping image.")
                continue
            stage = split[image_manifest.identifier]
            if stage == _Stages["predict"]:
                raise RuntimeError(
                    f"Image {image_manifest.identifier} is in the predict stage. "
                    f"This is not allowed, when a dataset split is provided."
                )
            getattr(splitted_manifest, stage.name).append(image_manifest)
    else:
        logger.info(f"No split provided. Assuming predict stage.")
        splitted_manifest.predict.extend(manifest)

    return splitted_manifest


def parse_wsi_attributes_from_manifest(
    data_description: DataDescription, manifest: ImageManifest
) -> tuple[Path, _ImageBackends, float | None]:
    overwrite_mpp = None
    if manifest.mpp:
        overwrite_mpp = (manifest.mpp, manifest.mpp)
    image_fn, _image_backend = manifest.image
    image_fn = data_description.data_dir / image_fn
    image_backend = _ImageBackends[_image_backend.name]
    return image_fn, image_backend, overwrite_mpp


def _parse_annotations(
    annotation_model: AnnotationModel | None, *, base_dir: Path
) -> WsiAnnotations | SlideImage | None:
    _annotations = None
    if not annotation_model:
        return _annotations

    filenames = [base_dir / curr_path for curr_path in annotation_model.filenames]
    if len(filenames) == 0:
        raise ValueError("No annotation files found. If `annotations` key is given, expect at least one file.")

    try:
        # FIXME: This is ugly. We should have a better way to handle this.
        reader = _AnnotationReaders[str(annotation_model.reader.name)]  # type: ignore
        logger.debug("Reading annotations with %s (%s)", reader, annotation_model.reader.name)
    except KeyError:
        raise ValueError(f"Annotation reader {annotation_model.reader.name} not supported.")

    logger.debug("Reading annotations from %s", filenames)
    return reader(filenames)


def image_manifest_to_dataset(
    data_description: DataDescription,
    manifest: ImageManifest,
    mpp: Optional[float],
    tile_size: tuple[int, int],
    tile_overlap: tuple[int, int],
    output_tile_size: tuple[int, int] | None = None,
    transform: Optional[Callable] = None,
    stage: str = TrainerFn.FITTING,
) -> TiledROIsSlideImageDataset:
    """Create a `TiledROIsSlideImageDataset` from an `ImageManifest`.

    Arguments
    ----------
    data_description: DataDescription
        The `DataDescription` object.
    manifest: ImageManifest
        The `ImageManifest` object.
    mpp: Optional[float]
        The microns per pixel of the image. If None, the mpp is read from the image (level 0).
    tile_size: tuple[int, int]
        The size of the tiles to extract from the image.
    tile_overlap: tuple[int, int]
        The overlap of the tiles to extract from the image.
    output_tile_size: tuple[int, int] | None
        The size of the output tiles. If None, the output tile size is the same as the input tile size.
    transform: Optional[Callable]
        A transform to apply to the tiles. These are the pre-transforms.
    stage: str
        The stage of the pipeline (training, validating, testing)

    Returns
    -------
    TiledROIsSlideImageDataset
        A `TiledROIsSlideImageDataset` object.
    """
    # This line parses image path and backend from the manifest
    image_fn, image_backend, overwrite_mpp = parse_wsi_attributes_from_manifest(data_description, manifest)
    # This block parses the annotations
    _annotations = _parse_annotations(manifest.annotations, base_dir=data_description.annotations_dir)

    # This block parses the mask
    # FIXME: A different manifest needed during inference.
    _mask = _parse_annotations(manifest.mask, base_dir=data_description.annotations_dir)
    rois = None
    if data_description.convert_mask_to_rois and _mask is not None and stage == TrainerFn.FITTING:
        if isinstance(_mask, SlideImage):
            # TODO: You could run a connected components here.
            raise ValueError("Cannot convert mask to ROIs if the reader is SlideImage.")

        # in inference mode, we just select the grid as is
        # TODO: Do we need to align the grid to (0, 0)?
        rois = compute_rois(_mask, tile_size=tile_size, tile_overlap=tile_overlap, centered=True)

    labels = None
    if manifest.labels:
        labels = [(k, v) for k, v in manifest.labels.items()]

    kwargs = {}
    if overwrite_mpp is not None:
        logger.info("Overriding mpp with value from manifest: %s", manifest.mpp)
        kwargs["overwrite_mpp"] = (manifest.mpp, manifest.mpp)

    if rois is not None:
        kwargs["limit_bounds"] = False

    # FIXME: rois has correct type, but not what dlup expects (Optional[Tuple[Tuple[int, ...]]])
    # If we are in inference mode, this means that we need to select *all* tiles covered by the ROI in contrast
    # to the case in training where we can select a subsample based on the threshold.
    mask_threshold = 0.0 if stage != TrainerFn.FITTING else data_description.mask_threshold
    dataset = TiledROIsSlideImageDataset.from_standard_tiling(
        path=image_fn,
        mpp=mpp,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        tile_mode=TilingMode.overflow,
        grid_order=GridOrder.C,
        crop=False,
        mask=_mask,
        mask_threshold=mask_threshold,
        output_tile_size=output_tile_size,
        rois=rois,  # type: ignore
        annotations=_annotations if stage != TrainerFn.PREDICTING else None,
        labels=labels,
        transform=transform,
        backend=image_backend,
        **kwargs,
    )

    return dataset
