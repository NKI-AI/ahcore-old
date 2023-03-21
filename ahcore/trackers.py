# encoding: utf-8
"""
All trackers, classes which are used during reference, for instance to compute specific things such as
tumor-stroma-ratio, but also writing tiff's as output.

"""
from __future__ import annotations

import abc
import os
from pathlib import Path
from typing import Any, Union

import h5py
import torch
from dlup.writers import TiffCompression, TifffileImageWriter
from torch import Tensor

from ahcore.utils.io import get_logger
from dlup._image import Resampling
logger = get_logger(__name__)


class Tracker(abc.ABC):
    """Abstract tracker class. Trackers are objects that are created and called only during inference.
    They should all implement a call method that is called at the end of a prediction epoch."""

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(self, predictions: list[list[Tensor]], metadata: dict[str, Any], *args, **kwargs):
        raise NotImplementedError()


class TiffWriter(Tracker):
    def __init__(
        self,
        save_dir: Union[str, Path],
        pyramid: bool = False,
        compression: str | None = "jpeg",
        quality: int | None = 100,
        is_mask: bool = True,
    ):
        self.save_dir = save_dir
        self.pyramid = pyramid
        self.compression = TiffCompression(compression)
        self.quality = quality

        self._interpolator = Resampling.NEAREST if is_mask else None

    @staticmethod
    def create_pred_iterator(predictions: list[list[Tensor]]):
        cat_predictions = torch.cat(predictions[0])
        arg_predictions = torch.argmax(cat_predictions, dim=1)
        for pred in arg_predictions:
            yield pred.cpu().numpy().astype("uint8")

    def __call__(self, predictions: list[list[Tensor]], metadata: dict[str, Any], *args, **kwargs):
        filename = Path(self.save_dir) / Path(str(metadata["filename"]) + ".tiff")
        logger.info(f"Writing Tiff for {filename.stem}")
        os.makedirs(filename.parent)
        tiff_writer = TifffileImageWriter(
            filename=filename,
            size=metadata["size"],
            mpp=metadata["mpp"],
            tile_size=metadata["tile_size"],
            pyramid=self.pyramid,
            compression=self.compression,
            quality=self.quality,
            interpolator=self._interpolator,
        )

        pred_iter = self.create_pred_iterator(predictions)
        tiff_writer.from_tiles_iterator(pred_iter)


class DumpToDisk(Tracker):
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def __call__(self, predictions: list[list[Tensor]], metadata: dict[str, Any], *args, **kwargs):
        filename = Path(self.save_dir) / Path(str(metadata["filename"]) + ".h5")
        logger.info(f"Dumping Predictions to disk for {filename.stem}")
        os.makedirs(filename.parent, exist_ok=True)
        dump = torch.cat(predictions[0])
        dump = torch.argmax(dump, dim=1).numpy().astype("uint8")
        with h5py.File(filename, "w") as hf:
            hf.create_dataset(filename.stem, data=dump, compression="lzf", chunks=True)  # B, C, W, H
