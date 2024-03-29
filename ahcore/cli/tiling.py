"""Utility to create tiles from the TCGA FFPE H&E slides.

Many models uses 0.5um/pixel at 224 x 224 size.
"""
from __future__ import annotations

import argparse
import io
from functools import partial
from logging import getLogger
from multiprocessing import Pool
from pathlib import Path
from pprint import pformat
from typing import Any, Generator

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import PIL.Image
from dlup import SlideImage
from dlup.data.dataset import TiledWsiDataset
from dlup.tiling import GridOrder, TilingMode
from PIL import Image
from pydantic import BaseModel
from rich.progress import Progress

from ahcore.cli import dir_path, file_path
from ahcore.writers import H5FileImageWriter

logger = getLogger(__name__)


def read_mask(path: Path) -> npt.NDArray[np.int_]:
    return iio.imread(path)[..., 0]


class SlideImageMetaData(BaseModel):
    """Metadata of a whole slide image."""

    path: Path
    mpp: float
    aspect_ratio: float
    magnification: float | None
    size: tuple[int, int]
    vendor: str | None

    @classmethod
    def from_dataset(cls, dataset: TiledWsiDataset) -> "SlideImageMetaData":
        _relevant_keys = ["aspect_ratio", "magnification", "mpp", "size", "vendor"]
        return cls(
            **{
                "path": dataset.path,
                **{key: getattr(dataset.slide_image, key) for key in _relevant_keys},
            }
        )

    def __iter__(self) -> Generator[tuple[str, Any], None, None]:
        for k, v in self.model_dump().items():
            yield k, v


class TileMetaData(BaseModel):
    coordinates: tuple[int, int]
    region_index: int
    grid_local_coordinates: tuple[int, int]
    grid_index: int

    def __iter__(self) -> Generator[tuple[str, Any], None, None]:
        for k, v in self.model_dump().items():
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


def _save_thumbnail(
    image_fn: Path,
    dataset_cfg: DatasetConfigs,
    mask: npt.NDArray[np.int_] | None,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None, npt.NDArray[np.uint8]]:
    target_mpp = max(dataset_cfg.mpp * 30, 30)
    tile_size = (
        min(30, dataset_cfg.tile_size[0] // 30),
        min(30, dataset_cfg.tile_size[1] // 30),
    )

    dataset = TiledWsiDataset.from_standard_tiling(
        image_fn,
        target_mpp,
        tile_size,
        (0, 0),
        mask=mask,
        mask_threshold=dataset_cfg.mask_threshold,
    )
    scaled_region_view = dataset.slide_image.get_scaled_view(dataset.slide_image.get_scaling(target_mpp))

    if mask is not None:
        # Let us write the mask too.
        mask_io = io.BytesIO()
        pil_mask = PIL.Image.fromarray(mask * 255, mode="L")
        pil_mask.save(mask_io, quality=75)
        mask_arr = np.frombuffer(mask_io.getvalue(), dtype="uint8")
    else:
        mask_arr = None

    thumbnail_io = io.BytesIO()

    # TODO: This needs to change in dlup, the scaled_region_view needs to return size in int, int.
    _tile_size = tuple(scaled_region_view.size)
    tile_size = (_tile_size[0], _tile_size[1])

    thumbnail = dataset.slide_image.get_thumbnail(tile_size)
    thumbnail.convert("RGB").save(thumbnail_io, quality=75)
    thumbnail_arr = np.frombuffer(thumbnail_io.getvalue(), dtype="uint8")

    region_size = tuple(scaled_region_view.size)
    background = Image.new("RGBA", (region_size[0], region_size[1]), (255, 255, 255, 255))

    overlay_io = io.BytesIO()
    for d in dataset:
        tile = d["image"]
        coords = np.array(d["coordinates"])
        box = tuple(np.array((*coords, *(coords + tile_size))).astype(int))
        background.paste(tile, (box[0], box[1]))
        # You could uncomment this to plot the boxes of the grid as well, but this can quickly become crowded.
        # draw = ImageDraw.Draw(background)
        # draw.rectangle(box, outline="red")
    background.convert("RGB").save(overlay_io, quality=75)
    overlay_arr = np.frombuffer(overlay_io.getvalue(), dtype="uint8")

    return thumbnail_arr, mask_arr, overlay_arr


def create_slide_image_dataset(
    slide_image_path: Path,
    mask: SlideImage | npt.NDArray[np.int_] | None,
    cfg: DatasetConfigs,
    overwrite_mpp: tuple[float, float] | None = None,
) -> TiledWsiDataset:
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

    return TiledWsiDataset.from_standard_tiling(
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


def _generator(
    dataset: TiledWsiDataset, quality: int | None = 80, compression: str = "JPEG"
) -> Generator[Any, Any, Any]:
    for idx in range(len(dataset)):
        # TODO: To use:
        # for idx, sample in enumerate(dataset):
        # The following needs to be added to TiledWsiDataset:
        # def __iter__(self) -> Iterator[RegionFromWsiDatasetSample]:
        #     for i in range(len(self)):
        #         yield self[i]
        sample = dataset[idx]
        buffered = io.BytesIO()
        if quality is not None:
            # If we just cast the PIL.Image to RGB, the alpha channel is set to black
            # which is a bit unnatural if you look in the image pyramid where it would be white in lower resolutions
            # this is why we take the following approach.
            tile: PIL.Image.Image = sample["image"]
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
    dataset: TiledWsiDataset,
    h5_writer: H5FileImageWriter,
    quality: int | None = 80,
) -> None:
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


def _tiling_pipeline(
    image_path: Path,
    mask_path: Path | None,
    output_file: Path,
    dataset_cfg: DatasetConfigs,
    quality: int,
    save_thumbnail: bool = False,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # TODO: Come up with a way to inject the mask later on as well.
        mask = read_mask(mask_path) if mask_path else None
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
        save_tiles(dataset, h5_writer, quality)
        if save_thumbnail:
            thumbnail, thumbnail_mask, overlay = _save_thumbnail(image_path, dataset_cfg, mask)

            if thumbnail_mask is not None:
                h5_writer.add_associated_images(
                    images=(("thumbnail", thumbnail), ("mask", thumbnail_mask), ("overlay", overlay)),
                    description="thumbnail, mask and overlay",
                )
            else:
                h5_writer.add_associated_images(images=(("thumbnail", thumbnail),), description="thumbnail")

    except Exception as e:
        logger.error(f"Failed: {image_path} with exception {e}")
        return

    logger.debug("Working on %s. Writing to %s", image_path, output_file)


def _wrapper(dataset_cfg: DatasetConfigs, quality: int, save_thumbnail: bool, args: tuple[Path, Path, Path]) -> None:
    image_path, mask_path, output_file = args
    return _tiling_pipeline(image_path, mask_path, output_file, dataset_cfg, quality, save_thumbnail)


def _do_tiling(args: argparse.Namespace) -> None:
    images_list: list[tuple[Path, Path | None, Path]] = []

    with open(args.file_list, "r") as file_list:
        for line in file_list:
            if line.startswith("#"):
                continue
            image_file, mask_file, output_filename = line.split(",")
            if (args.output_directory / "data" / Path(output_filename.strip())).is_file() and args.simple_check:
                continue

            mask_fn = Path(mask_file.strip()) if mask_file != "" else None
            images_list.append(
                (
                    Path(image_file.strip()),
                    mask_fn,
                    args.output_directory / "data" / Path(output_filename.strip()),
                )
            )

    logger.info(f"Number of slides: {len(images_list)}")
    logger.info(f"Output directory: {args.output_directory}")
    logger.info("Tiling...")

    save_dir_data = args.output_directory / "data"
    save_dir_data.mkdir(parents=True, exist_ok=True)

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

    if args.num_workers > 0:
        # Create a partially applied function with dataset_cfg
        partial_wrapper = partial(_wrapper, dataset_cfg, args.quality, args.save_thumbnail)

        with Progress() as progress:
            task = progress.add_task("[cyan]Tiling...", total=len(images_list))
            with Pool(processes=args.num_workers) as pool:
                for _ in pool.imap_unordered(partial_wrapper, images_list):
                    progress.update(task, advance=1)
    else:
        with Progress() as progress:
            task = progress.add_task("[cyan]Tiling...", total=len(images_list))
            for idx, (image_path, mask_path, output_file) in enumerate(images_list):
                _tiling_pipeline(
                    image_path,
                    mask_path,
                    output_file,
                    dataset_cfg,
                    quality=args.quality,
                    save_thumbnail=args.save_thumbnail,
                )
                progress.update(task, advance=1)


def register_parser(parser: argparse._SubParsersAction[Any]) -> None:
    """Register inspect commands to a root parser."""
    tiling_parser = parser.add_parser("tiling", help="Tiling utilities")
    tiling_subparsers = tiling_parser.add_subparsers(help="Tiling subparser")
    tiling_subparsers.required = True
    tiling_subparsers.dest = "subcommand"

    _parser: argparse.ArgumentParser = tiling_subparsers.add_parser(
        "tile-to-h5",
        help="Tiling WSI images to h5",
    )

    # Assume a comma separated format from image_file,mask_file
    _parser.add_argument(
        "--file-list",
        type=file_path,
        required=True,
        help="Path to the file list. Each comma-separated line is of the form `<image_fn>,<mask_fn>,<output_directory>`"
        " where the output directory is with request to --output-dir. mask_fn can be empty.",
    )
    _parser.add_argument(
        "--output-directory",
        type=dir_path(require_writable=True),
        required=True,
        help="Path to the output directory.",
    )
    _parser.add_argument(
        "--mpp",
        type=float,
        required=True,
        help="Resolution (microns per pixel) at which the slides should be tiled.",
    )
    _parser.add_argument("--tile-size", type=int, required=True, help="Size of the tiles in pixels.")
    _parser.add_argument(
        "--tile-overlap",
        type=int,
        default=0,
        help="Overlap of the tile in pixels (default=0).",
    )
    _parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.6,
        help="0 every tile is discarded, 1 requires the whole tile to be foreground (default=0.6).",
    )
    _parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers to use for tiling. 0 disables the tiling (default: 8)",
    )
    _parser.add_argument(
        "--save-thumbnail",
        action="store_true",
        help="Save a thumbnail of the slide, including the filtered tiles and the mask itself.",
    )
    _parser.add_argument(
        "--simple-check",
        action="store_true",
        help="Filter the list based on if the h5 images already exist.",
    )
    _parser.add_argument(
        "--quality",
        type=int,
        default=80,
        help="Quality of the saved tiles in jpg, otherwise png (default: 80)",
    )

    _parser.set_defaults(subcommand=_do_tiling)
