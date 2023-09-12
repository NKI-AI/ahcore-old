# encoding: utf-8
"""Utility to create tiles from the TCGA FFPE H&E slides.

Other models uses 0.5um/pixel at 224 x 224 size.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import partial
from logging import getLogger
from multiprocessing import Pool
from pathlib import Path
from pprint import pformat
from typing import Union

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import PIL.Image
from dlup import SlideImage
from dlup.data.dataset import RegionFromSlideDatasetSample, TiledROIsSlideImageDataset
from dlup.tiling import GridOrder, TilingMode
from PIL import Image
from pydantic import BaseModel
from rich.progress import Progress

from ahcore.cli import dir_path, file_path
from ahcore.readers import H5FileImageReader
from ahcore.writers import H5FileImageWriter

logger = getLogger(__name__)
import sys

logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def read_mask(path: Path) -> np.ndarray:
    return iio.imread(path)[..., 0]


def load_json(path: Path, encoding: str = "utf-8", **kwargs):
    """Load json file."""
    with open(path, "r", encoding=encoding) as json_file:
        json_obj = json.load(json_file, **kwargs)
    return json_obj


def write_json(
    obj,
    path: Path,
    indent: int | None = 2,
    **kwargs,
):
    """Save in json format."""
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(obj, json_file, indent=indent, **kwargs)


def make_serializable(obj):
    """Makes object serializable"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (str, float, int)) or obj is None:
        return obj
    elif isinstance(obj, (tuple, list)):
        return [make_serializable(_) for _ in obj]
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    else:
        raise RuntimeError(f"Object of type {type(obj)} is not supported.")


@dataclass
class SlideImageMetaData:
    """Metadata of a whole slide image."""

    path: Path
    mpp: float
    aspect_ratio: float
    magnification: float | None
    size: tuple[int, int]
    vendor: str | None

    @classmethod
    def from_dataset(cls, dataset: TiledROIsSlideImageDataset):
        _relevant_keys = ["aspect_ratio", "magnification", "mpp", "size", "vendor"]
        return cls(
            **{
                "path": dataset.path,
                **{key: getattr(dataset.slide_image, key) for key in _relevant_keys},
            }
        )

    def __iter__(self):
        for k, v in asdict(self).items():
            yield k, v


@dataclass()
class TileMetaData:
    """Metadata of a tile."""

    coordinates: tuple[int, int]
    region_index: int
    grid_local_coordinates: tuple[int, int]
    grid_index: int

    @classmethod
    def from_sample(cls, sample: RegionFromSlideDatasetSample):
        _relevant_keys = [
            "coordinates",
            "region_index",
            "grid_local_coordinates",
            "grid_index",
        ]
        return cls(**{k: v for k, v in sample.items() if k in _relevant_keys})

    def __iter__(self):
        for k, v in asdict(self).items():
            yield k, v


class DatasetConfigs(BaseModel):
    """Configurations of the TiledROIsSlideImageDataset dataset"""

    mpp: float
    tile_size: tuple[int, int]
    tile_overlap: tuple[int, int]
    tile_mode: str
    crop: bool
    mask_threshold: float
    grid_order: str


#
#
# def _save_thumbnail(
#     image_fn: Path,
#     dataset_cfg: DatasetConfigs,
#     mask: npt.NDArray[np.uint8],
#     h5_writer: H5FileImageWriter,
# ):
#     target_mpp = max(dataset_cfg.mpp * 30, 30)
#     tile_size = (
#         min(30, dataset_cfg.tile_size[0] // 30),
#         min(30, dataset_cfg.tile_size[1] // 30),
#     )
#
#     dataset = TiledROIsSlideImageDataset.from_standard_tiling(
#         image_fn,
#         target_mpp,
#         tile_size,
#         (0, 0),
#         mask=mask,
#         mask_threshold=dataset_cfg.mask_threshold,
#     )
#     scaled_region_view = dataset.slide_image.get_scaled_view(dataset.slide_image.get_scaling(target_mpp))
#
#     # Let us write the mask too.
#     mask_fn = output_fn / "mask.jpg"
#     mask = PIL.Image.fromarray(mask * 255, mode="L")
#     mask.save(mask_fn, quality=75)
#
#     thumbnail = dataset.slide_image.get_thumbnail(tuple(scaled_region_view.size))
#     thumbnail_fn = output_fn / "thumbnail.jpg"
#     thumbnail.convert("RGB").save(thumbnail_fn, quality=75)
#
#     background = Image.new("RGBA", tuple(scaled_region_view.size), (255, 255, 255, 255))
#
#     for d in dataset:
#         tile = d["image"]
#         coords = np.array(d["coordinates"])
#         box = tuple(np.array((*coords, *(coords + tile_size))).astype(int))
#         background.paste(tile, box)
#         # draw = ImageDraw.Draw(background)
#         # draw.rectangle(box, outline="red")
#     overlay_fn = output_fn / "overlay.jpg"
#     background.convert("RGB").save(overlay_fn, quality=75)
#


def create_slide_image_dataset(
    slide_image_path: Path,
    mask: SlideImage | np.ndarray | None,
    cfg: DatasetConfigs,
    overwrite_mpp: tuple[float, float] | None = None,
) -> TiledROIsSlideImageDataset:
    """
    Initializes and returns a slide image dataset.

    Parameters
    ----------
    slide_image_path : Path
        Path to a whole slide image file.
    mask : np.ndarray | None
        Binary mask used to filter each tile.
    cfg : DatasetConfigs
        Dataset configurations.
    overwrite_mpp : tuple[float, float] | None
        Tuple of (mpp_x, mpp_y) used to overwrite the mpp of the loaded slide image.

    Returns
    -------
    TiledROIsSlideImageDataset
        Initialized slide image dataset.

    """

    return TiledROIsSlideImageDataset.from_standard_tiling(
        path=slide_image_path,
        mpp=cfg.mpp,
        tile_size=cfg.tile_size,
        tile_overlap=cfg.tile_overlap,
        grid_order=GridOrder[cfg.grid_order],
        tile_mode=TilingMode[cfg.tile_mode],
        crop=cfg.crop,
        mask=mask,
        mask_threshold=cfg.mask_threshold,
        overwrite_mpp=overwrite_mpp,
    )


def _generator(dataset, quality: int | None = 80, compression: str = "JPEG"):
    for idx, sample in enumerate(dataset):
        buffered = io.BytesIO()
        if quality is not None:
            # If we just cast the PIL.Image to RGB, the alpha channel is set to black
            # which is a bit unnatural if you look in the image pyramid where it would be white in lower resolutions
            # this is why we take the following approach.
            tile = sample["image"]
            background = PIL.Image.new("RGB", tile.size, (255, 255, 255))  # Create a white background
            background.paste(tile, mask=tile.split()[3])  # Paste the image using the alpha channel as mask
            background.convert("RGB").save(buffered, format=compression, quality=quality)
        else:
            sample["image"].save(buffered, format=compression, quality=quality)

        # Now we have the image bytes
        coordinates = sample["coordinates"]
        array = np.frombuffer(buffered.getvalue(), dtype="uint8")
        yield [coordinates], array[np.newaxis, :]


def save_tiles(
    dataset: TiledROIsSlideImageDataset,
    h5_writer: H5FileImageWriter,
    quality: int | None = 80,
):
    """
    Saves the tiles in the given image slide dataset to disk.

    Parameters
    ----------
    dataset : TiledROIsSlideImageDataset
        The image slide dataset containing tiles of a single whole slide image.
    h5_writer : H5FileImageWriter
        The H5 writer to write the tiles to.
    quality : int | None
        If not None, the compression quality of the saved tiles in jpg, otherwise png

    """
    compression = "JPEG" if quality is not None else "PNG"
    generator = _generator(dataset, quality, compression)
    h5_writer.consume(generator)


def tiling_pipeline(
    image_path: Path,
    mask_path: Path,
    output_file: Path,
    dataset_cfg: DatasetConfigs,
    save_thumbnail: bool = False,
) -> None:
    try:
        # TODO: Come up with a way to inject the mask later on as well.
        mask = read_mask(mask_path)
        dataset = create_slide_image_dataset(
            slide_image_path=image_path,
            mask=mask,
            cfg=dataset_cfg,
        )
        _scaling = dataset.slide_image.get_scaling(dataset_cfg.mpp)

        h5_writer = H5FileImageWriter(
            filename=output_file,
            size=dataset.slide_image.get_scaled_size(_scaling),
            mpp=dataset_cfg.mpp,
            tile_size=dataset_cfg.tile_size,
            tile_overlap=dataset_cfg.tile_overlap,
            num_samples=len(dataset),
            is_binary=True,
        )

    except Exception as e:
        logger.error(f"Failed: {image_path} with exception {e}")
        return

    logger.debug("Working on %s. Writing to %s", image_path, output_file)

    # if save_thumbnail:
    #     _save_thumbnail(image_path, dataset_cfg, mask, h5_writer)


def wrapper(dataset_cfg, save_dir_data, save_thumbnail, args):
    image_path, mask_path, path_id = args
    _output_directory = save_dir_data / path_id
    return tiling_pipeline(image_path, mask_path, _output_directory, dataset_cfg, save_thumbnail)


def main():
    parser = argparse.ArgumentParser("Tiling of whole slide images.")
    # Assume a comma separated format from image_file,mask_file
    parser.add_argument(
        "--file-list",
        type=file_path,
        required=True,
        help="Path to the file list. Each comma-separated line is of the form `<image_fn>,<mask_fn>,<output_directory>`"
        " where the output directory is with request to --output-dir",
    )
    parser.add_argument(
        "--output-file",
        type=file_path,
        required=True,
        help="Path to the output file",
    )
    parser.add_argument(
        "--mpp",
        type=float,
        required=True,
        help="Resolution (microns per pixel) at which the slides should be tiled.",
    )
    parser.add_argument("--tile-size", type=int, required=True, help="Size of the tiles in pixels.")
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=0,
        help="Overlap of the tile in pixels (default=0).",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.6,
        help="0 every tile is discarded, 1 requires the whole tile to be foreground (default=0.6).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers to use for tiling. 0 disables the tiling (default: 8)",
    )
    parser.add_argument(
        "--save-thumbnail",
        action="store_true",
        help="Save a thumbnail of the slide, including the filtered tiles and the mask itself.",
    )
    parser.add_argument(
        "--simple-check",
        action="store_true",
        help="Filter the list based on if the folders already exist.",
    )

    args = parser.parse_args()
    images_list = []

    with open(args.file_list, "r") as file_list:
        for line in file_list:
            image_file, mask_file, output_directory = line.split(",")
            if (args.output_dir / "data" / Path(output_directory.strip())).is_dir() and args.simple_check:
                continue
            images_list.append(
                (
                    Path(image_file.strip()),
                    Path(mask_file.strip()),
                    Path(output_directory.strip()),
                )
            )

    logger.info(f"Number of slides: {len(images_list)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("Tiling...")

    save_dir_data = args.output_dir / "data"
    save_dir_data.mkdir(parents=True, exist_ok=True)

    save_dir_meta = args.output_dir / "metadata"
    save_dir_meta.mkdir(parents=True, exist_ok=True)

    crop = False
    tile_mode = TilingMode.overflow
    tile_size = (args.tile_size, args.tile_size)
    tile_overlap = (args.tile_overlap, args.tile_overlap)
    mpp = args.mpp
    mask_threshold = args.mask_threshold

    dataset_cfg = DatasetConfigs(
        mpp=mpp,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        tile_mode=tile_mode,
        crop=crop,
        mask_threshold=mask_threshold,
        grid_order="C",
    )

    logger.info(f"Dataset configurations: {pformat(dataset_cfg)}")

    write_json(obj=dataset_cfg.dict(), path=save_dir_meta / "dataset_configs.json", indent=2)

    if args.num_workers > 0:
        # Convert list of tuples into list of lists
        images_list = [list(item) for item in images_list]
        # Create a partially applied function with dataset_cfg
        partial_wrapper = partial(wrapper, dataset_cfg, save_dir_data, args.save_thumbnail)

        with Progress() as progress:
            task = progress.add_task("[cyan]Tiling...", total=len(images_list))
            with Pool(processes=args.num_workers) as pool:
                for _ in pool.imap_unordered(partial_wrapper, images_list):
                    progress.update(task, advance=1)
    else:
        with Progress() as progress:
            task = progress.add_task("[cyan]Tiling...", total=len(images_list))
            for idx, (image_path, mask_path, path_id) in enumerate(images_list):
                tiling_pipeline(
                    image_path,
                    mask_path,
                    args.output_file,
                    dataset_cfg,
                    save_thumbnail=args.save_thumbnail,
                )
                progress.update(task, advance=1)


if __name__ == "__main__":
    main()
