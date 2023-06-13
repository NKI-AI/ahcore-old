# encoding: utf-8
import json
import math
import numpy.typing as npt
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Optional, Tuple, Union
from dlup.tiling import Grid, GridOrder, TilingMode
import h5py
import numpy as np
import PIL.Image
from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.tiling import TilingMode
from dlup.writers import Resampling, TifffileImageWriter

from ahcore.utils.io import get_logger

logger = get_logger(__name__)


class StitchingMode(Enum):
    CROP = 0
    AVERAGE = 1
    MAXIMUM = 2


class PasteMode(Enum):
    OVERWRITE = 0
    ADD = 1


class _DatasetIterator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.dataset):
            item = self.dataset[self.index]
            self.index += 1
            return item
        else:
            raise StopIteration


def crop_to_bbox(array, bbox):
    (x, y), (h, w) = bbox
    return array[y : y + h, x : x + w]


def paste_region(array, region, location, paste_mode=PasteMode.OVERWRITE):
    x, y = location
    h, w = region.shape[:2]

    if paste_mode == PasteMode.OVERWRITE:
        array[y : y + h, x : x + w] = region
    elif paste_mode == PasteMode.ADD:
        array[y : y + h, x : x + w] += region
    else:
        raise ValueError("Unsupported paste mode")


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

        logger.info("Writing h5 to %s", self._filename)

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
            "shape": tuple(batch_shape),
            "size": (int(self._size[0]), int(self._size[1])),
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

                logger.info("Done writing tiles for %s", self._filename)

        except Exception as e:
            logger.error("Error in consumer thread for %s: %s", self._filename, exc_info=e)

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


class H5FileImageReader:
    def __init__(self, filename, size, tile_size, tile_overlap, stitching_mode):
        self._filename = filename
        self._tile_overlap = tile_overlap
        self._tile_size = tile_size
        self._size = size
        self._stitching_mode = stitching_mode

    def read_region(self, location, size):
        with h5py.File(self._filename, "r") as h5file:
            image_dataset = h5file["data"]
            metadata = h5file["metadata"]
            num_tiles = metadata["num_tiles"]
            tile_height, tile_width, num_channels = image_dataset.shape[1:]

            stride_height = tile_height - self._tile_overlap[1]
            stride_width = tile_width - self._tile_overlap[0]

            total_rows = math.ceil((self._size[1] - self._tile_overlap[1]) / stride_height)
            total_cols = math.ceil((self._size[0] - self._tile_overlap[0]) / stride_width)

            assert total_rows * total_cols == num_tiles

            x, y = location
            w, h = size
            if x < 0 or y < 0 or x + w > self._size[0] or y + h > self._size[1]:
                raise ValueError("Requested region is out of bounds")

            start_row = y // stride_height
            end_row = min((y + h - 1) // stride_height + 1, total_rows)
            start_col = x // stride_width
            end_col = min((x + w - 1) // stride_width + 1, total_cols)

            if self._stitching_mode == StitchingMode.AVERAGE:
                divisor_array = np.zeros((h, w), dtype=np.uint8)
            stitched_image = np.zeros((h, w, num_channels), dtype=np.uint8)
            for i in range(start_row, end_row):
                for j in range(start_col, end_col):
                    tile_idx = (i * total_cols) + j
                    tile = image_dataset[tile_idx]

                    start_y = i * stride_height - y
                    end_y = start_y + tile_height
                    start_x = j * stride_width - x
                    end_x = start_x + tile_width

                    img_start_y = max(0, start_y)
                    img_end_y = min(h, end_y)
                    img_start_x = max(0, start_x)
                    img_end_x = min(w, end_x)

                    if self._stitching_mode == StitchingMode.CROP:
                        crop_start_y = img_start_y - start_y
                        crop_end_y = img_end_y - start_y
                        crop_start_x = img_start_x - start_x
                        crop_end_x = img_end_x - start_x

                        # TODO: Simplify this
                        bbox = (crop_start_y, crop_start_x), (crop_end_y - crop_start_y, crop_end_x - crop_start_x)
                        #                         cropped_tile = tile[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
                        cropped_tile = crop_to_bbox(tile, bbox)
                        stitched_image[img_start_y:img_end_y, img_start_x:img_end_x] = cropped_tile

                    elif self._stitching_mode == StitchingMode.AVERAGE:
                        tile_start_y = max(0, -start_y)
                        tile_end_y = img_end_y - img_start_y
                        tile_start_x = max(0, -start_x)
                        tile_end_x = img_end_x - img_start_x

                        # TODO: Replace this with crop_to_bbox
                        cropped_tile = tile[tile_start_y:tile_end_y, tile_start_x:tile_end_x]
                        stitched_image[img_start_y:img_end_y, img_start_x:img_end_x] += cropped_tile
                        divisor_array[img_start_y:img_end_y, img_start_x:img_end_x] += 1
                    else:
                        raise ValueError("Unsupported stitching mode")

        if self._stitching_mode == StitchingMode.AVERAGE:
            stitched_image = np.round(stitched_image / divisor_array[..., np.newaxis], 0).astype(np.uint8)

        return PIL.Image.fromarray(stitched_image)


if __name__ == "__main__":
    import tqdm

    class RandomColoredTilesDataset:
        def __init__(self, num_tiles=500, tile_width=128, tile_height=128):
            self.num_tiles = num_tiles
            self.tile_width = tile_width
            self.tile_height = tile_height

        def __len__(self):
            return self.num_tiles

        def __getitem__(self, idx):
            if idx >= self.num_tiles or idx < 0:
                raise IndexError("Index out of range")

            color = tuple(np.random.randint(0, 256, size=3))
            image = PIL.Image.new("RGB", (self.tile_width, self.tile_height), color)
            return image

    tile_width, tile_height = 128, 128
    image_width, image_height = 128 * 20, 128 * 25
    tiles_per_row = image_width // tile_width
    tiles_per_column = image_height // tile_height
    total_tiles = tiles_per_row * tiles_per_column

    # Create the dataset with the required number of tiles
    dataset = RandomColoredTilesDataset(num_tiles=total_tiles)

    # Initialize an empty array for the output image
    output_image_array = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Fill the output image with the tiles from the dataset
    tile_idx = 0
    for y in range(0, image_height, tile_height):
        for x in range(0, image_width, tile_width):
            tile_image = dataset[tile_idx]
            tile_array = np.array(tile_image)
            output_image_array[y : y + tile_height, x : x + tile_width] = tile_array
            tile_idx += 1

    # Convert the output array to a PIL Image
    output_image = PIL.Image.fromarray(output_image_array)

    writer = TifffileImageWriter(
        "test.tiff",
        interpolator=Resampling.NEAREST,
        pyramid=False,
        tile_size=(256, 256),
        mpp=(1.0, 1.0),
        size=(2560, 3200, 3),
    )

    writer.from_pil(output_image)  # will be transposed but doesn't matter

    image_fn = Path("test.tiff")
    tile_size = (128, 128)
    tile_overlap = (10, 10)
    mpp = 1.0

    ds = TiledROIsSlideImageDataset.from_standard_tiling(
        image_fn,
        mpp=mpp,
        tile_mode=TilingMode.overflow,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        crop=False,
    )

    slide = ds.slide_image
    print(slide.mpp)
    scale = slide.get_scaling(mpp)
    size = slide.get_scaled_size(scale)
    print(size)

    writer = H5FileImageWriter(
        "output.h5", size=size, mpp=mpp, tile_size=tile_size, tile_overlap=tile_overlap, progress=tqdm.tqdm
    )
    writer.from_iterator(ds)

    # TRY AVERAGE too, you can inspect the division_array what it does.
    reader = H5FileImageReader(
        "output.h5", tile_size=tile_size, tile_overlap=tile_overlap, size=size, stitching_mode=StitchingMode.CROP
    )

    a = reader.read_region((0, 0), (2400, 2400))
    b = slide.read_region((0, 0), 1, (2400, 2400))
