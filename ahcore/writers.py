# encoding: utf-8
import json
from pathlib import Path
from typing import Any, Generator, Optional, Tuple

import h5py
import numpy as np
import numpy.typing as npt
from dlup.tiling import Grid, GridOrder, TilingMode

from ahcore.utils.io import get_logger

logger = get_logger(__name__)


class H5FileImageWriter:
    """Image writer that writes tile-by-tile to h5."""

    def __init__(
        self,
        filename: Path,
        size: tuple[int, int],
        mpp: float,
        tile_size: tuple[int, int],
        tile_overlap: tuple[int, int],
        num_samples: int,
        progress: Any | None = None,
    ) -> None:
        self._grid: Optional[Grid] = None
        self._grid_coordinates: Optional[npt.NDArray] = None
        self._filename: Path = filename
        self._size: tuple[int, int] = size
        self._mpp: float = mpp
        self._tile_size: tuple[int, int] = tile_size
        self._tile_overlap: Tuple[int, int] = tile_overlap
        self._num_samples: int = num_samples
        self._progress = progress
        self._image_dataset: Optional[h5py.Dataset] = None
        self._coordinates_dataset: Optional[h5py.Dataset] = None
        self._tile_indices: Optional[h5py.Dataset] = None
        self._current_index: int = 0

        self._logger = get_logger(type(self).__name__)
        self._logger.debug("Writing h5 to %s", self._filename)

    def init_writer(self, first_batch: np.ndarray, h5file: h5py.File) -> None:
        """Initializes the image_dataset based on the first tile."""
        batch_shape = np.asarray(first_batch).shape
        batch_dtype = np.asarray(first_batch).dtype

        self._current_index = 0

        self._coordinates_dataset = h5file.create_dataset(
            "coordinates",
            shape=(self._num_samples, 2),
            dtype=int,
            compression="gzip",
        )

        # TODO: We only support a single Grid
        # And GridOrder C
        self._grid = Grid.from_tiling(
            (0, 0),
            size=self._size,
            tile_size=self._tile_size,
            tile_overlap=self._tile_overlap,
            mode=TilingMode.overflow,
            order=GridOrder.C,
        )

        self._tile_indices = h5file.create_dataset(
            "tile_indices",
            shape=(len(self._grid),),
            dtype=int,
            compression="gzip",
        )
        # Initialize to -1, which is the default value
        self._tile_indices[:] = -1

        self._image_dataset = h5file.create_dataset(
            "data",
            shape=(self._num_samples,) + batch_shape[1:],
            dtype=batch_dtype,
            compression="gzip",
            chunks=(1,) + batch_shape[1:],
        )

        # This only works when the mode is 'overflow' and in 'C' order.
        metadata = {
            "mpp": self._mpp,
            "dtype": str(batch_dtype),
            "shape": tuple(batch_shape[1:]),
            "size": (int(self._size[0]), int(self._size[1])),
            "num_channels": batch_shape[1],
            "num_samples": self._num_samples,
            "tile_size": tuple(self._tile_size),
            "tile_overlap": tuple(self._tile_overlap),
            "num_tiles": len(self._grid),
            "grid_order": "C",
            "mode": "overflow",
        }
        metadata_json = json.dumps(metadata)
        h5file.attrs["metadata"] = metadata_json

    def consume(self, batch_generator: Generator[tuple[np.ndarray, np.ndarray], None, None]) -> None:
        """Consumes tiles one-by-one from a generator and writes them to the h5 file."""
        grid_counter = 0
        try:
            with h5py.File(self._filename.with_suffix(".h5.partial"), "w") as h5file:
                first_coordinates, first_batch = next(batch_generator)
                self.init_writer(first_batch, h5file)

                batch_generator = self._batch_generator((first_coordinates, first_batch), batch_generator)
                # progress bar will be used if self._progress is not None
                if self._progress:
                    batch_generator = self._progress(batch_generator, total=self._num_samples)

                for coordinates, batch in batch_generator:
                    # We take a coordinate, and step through the grid until we find it.
                    # Note that this assumes that the coordinates come in C-order, so we will always hit it
                    for idx, curr_coordinates in enumerate(coordinates):
                        # As long as our current coordinates are not equal to the grid coordinates, we make a step
                        while not np.all(curr_coordinates == self._grid[grid_counter]):
                            grid_counter += 1
                        # If we find it, we set it to the index, so we can find it later on
                        # This can be tested by comparing the grid evaluated at a grid index with the tile index
                        # mapped to its coordinates
                        self._tile_indices[grid_counter] = self._current_index + idx
                        grid_counter += 1

                    batch_size = batch.shape[0]
                    self._image_dataset[self._current_index : self._current_index + batch_size] = batch
                    self._coordinates_dataset[self._current_index : self._current_index + batch_size] = coordinates
                    self._current_index += batch_size

        except Exception as e:
            self._logger.error("Error in consumer thread for %s: %s", self._filename, exc_info=e)

        # When done writing rename the file.
        self._filename.with_suffix(".h5.partial").rename(self._filename)

    @staticmethod
    def _batch_generator(first_coordinates_batch, batch_generator):
        # We yield the first batch too so the progress bar takes the first batch also into account
        yield first_coordinates_batch
        for tile in batch_generator:
            if tile is None:
                break
            yield tile
