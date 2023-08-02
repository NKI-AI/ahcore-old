# encoding: utf-8
"""Utility to create tiles from the TCGA FFPE H&E slides.

Other models uses 0.5um/pixel at 224 x 224 size.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import partial
from logging import getLogger
from multiprocessing import Pool
from pathlib import Path
from typing import Union

import imageio.v3 as iio
import numpy as np
import PIL.Image
from dlup import SlideImage
from dlup.data.dataset import RegionFromSlideDatasetSample, TiledROIsSlideImageDataset
from dlup.tiling import GridOrder, TilingMode
from pydantic import BaseModel
from rich.progress import Progress

logger = getLogger(__name__)
import sys

logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
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

    path: Path
    coordinates: tuple[int, int]
    region_index: int
    grid_local_coordinates: tuple[int, int]
    grid_index: int

    @classmethod
    def from_sample(cls, sample: RegionFromSlideDatasetSample):
        _relevant_keys = [
            "coordinates",
            "path",
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


def save_tiles(
    dataset: TiledROIsSlideImageDataset,
    save_dir: Path,
    quality: int | None = 80,
) -> dict[str : dict[str : Union[Path, float, int, bool, tuple[int, int]]]]:
    """
    Saves the tiles in the given image slide dataset to disk.

    Parameters
    ----------
    dataset : TiledROIsSlideImageDataset
        The image slide dataset containing tiles of a single whole slide image.
    save_dir : Path
        The directory to which the old_data should be saved.
    quality : int | None
        If not None, the compression quality of the saved tiles in jpg, otherwise png

    """
    extension = "jpg" if quality is not None else "png"

    tile_meta_data_dict = defaultdict(dict)
    slide_image_name = dataset.path.stem
    for idx, sample in enumerate(dataset):
        _tile_filename_suffix = "_".join([str(co) for co in sample["grid_local_coordinates"]])
        tile_filename = f"{slide_image_name}_tile_{_tile_filename_suffix}.{extension}"
        if quality is not None:
            # If we just cast the PIL.Image to RGB, the alpha channel is set to black
            # which is a bit unnatural if you look in the image pyramid where it would be white in lower resolutions
            # this is why we take the following approach.
            tile = sample["image"]
            background = PIL.Image.new("RGB", tile.size, (255, 255, 255))  # Create a white background
            background.paste(tile, mask=tile.split()[3])  # Paste the image using the alpha channel as mask
            background.convert("RGB").save(save_dir / tile_filename, quality=quality)
        else:
            sample["image"].save(save_dir / tile_filename)
        tile_meta_data_dict[tile_filename] = dict(TileMetaData.from_sample(sample))

    return tile_meta_data_dict


def tiling_pipeline(
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    dataset_cfg: DatasetConfigs,
) -> None:
    logger.debug("Working on %s. Writing to %s", image_path, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # mask = SlideImage.from_file_path(mask_path, interpolator="NEAREST")
    mask = read_mask(mask_path)

    dataset = create_slide_image_dataset(
        slide_image_path=image_path,
        mask=mask,
        cfg=dataset_cfg,
    )

    _tile_meta_data_dict = save_tiles(dataset=dataset, save_dir=output_dir)
    _scaling = dataset.slide_image.get_scaling(dataset_cfg.mpp)
    tile_meta_data = {
        "path": image_path,
        "scaled_size": dataset.slide_image.get_scaled_size(_scaling),
        "num_regions_total": len(dataset.regions),
        "num_regions_masked": len(dataset.masked_indices),
        **dict(dataset_cfg),
        "tiles": _tile_meta_data_dict,
    }
    write_json(
        obj=make_serializable(tile_meta_data),
        path=output_dir / "meta_data_tiles.json",
        indent=2,
    )

    _slide_meta_data = dict(SlideImageMetaData.from_dataset(dataset))
    write_json(
        obj=make_serializable(_slide_meta_data),
        path=output_dir / "meta_data_original_slide.json",
        indent=2,
    )


def wrapper(dataset_cfg, save_dir_data, args):
    image_path, mask_path, path_id = args
    _output_directory = save_dir_data / path_id
    return tiling_pipeline(image_path, mask_path, _output_directory, dataset_cfg)


def main():
    parser = argparse.ArgumentParser("Tiling of whole slide images.")
    # Assume a comma separated format from image_file,mask_file
    parser.add_argument("--file-list", type=Path, required=True, help="Path to the file list")
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to the output directory")
    parser.add_argument(
        "--mpp", type=float, required=True, help="Resolution (microns per pixel) at which the slides should be tiled."
    )
    parser.add_argument("--tile-size", type=int, required=True, help="Size of the tiles in pixels.")
    parser.add_argument("--tile-overlap", type=int, default=0, help="Overlap of the tile in pixels (default=0).")
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.6,
        help="0 every tile is discarded, 1 requires the whole tile to be foreground (default=0.6).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of workers to use for tiling. " "0 disables the tiling"
    )
    args = parser.parse_args()
    images_list = []

    with open(args.file_list, "r") as file_list:
        for line in file_list:
            image_file, mask_file, output_directory = line.split(",")
            images_list.append((Path(image_file.strip()), Path(mask_file.strip()), Path(output_directory.strip())))

    logger.info(f"Number of slides: {len(images_list)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("Tiling...")

    save_dir_data = args.output_dir / "data"
    save_dir_data.mkdir(parents=True, exist_ok=True)

    save_dir_meta = args.output_dir / "meta"
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

    write_json(obj=dataset_cfg.dict(), path=save_dir_meta / "dataset_configs.json", indent=2)

    if args.num_workers > 0:
        # Convert list of tuples into list of lists
        images_list = [list(item) for item in images_list]
        # Create a partially applied function with dataset_cfg
        partial_wrapper = partial(wrapper, dataset_cfg, save_dir_data)

        with Progress() as progress:
            task = progress.add_task("[cyan]Tiling...", total=len(images_list))
            with Pool(processes=args.num_workers) as pool:
                for _ in pool.imap_unordered(partial_wrapper, images_list):
                    progress.update(task, advance=1)
    else:
        with Progress() as progress:
            task = progress.add_task("[cyan]Tiling...", total=len(images_list))
            for idx, (image_path, mask_path, path_id) in enumerate(images_list):
                _output_directory = save_dir_data / path_id
                tiling_pipeline(image_path, mask_path, _output_directory, dataset_cfg)
                progress.update(task, advance=1)


if __name__ == "__main__":
    main()
