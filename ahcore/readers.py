# encoding: utf-8
import json
import math
from enum import Enum

import h5py
import numpy as np


class StitchingMode(Enum):
    CROP = 0
    AVERAGE = 1
    MAXIMUM = 2


class PasteMode(Enum):
    OVERWRITE = 0
    ADD = 1


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


class H5FileImageReader:
    def __init__(self, filename, stitching_mode):
        self._filename = filename
        self._stitching_mode = stitching_mode

        self.__empty_tile = None

        self._h5file = None
        self._metadata = None
        self._mpp = None
        self._tile_size = None
        self._tile_overlap = None
        self._size = None
        self._num_channels = None
        self._dtype = None
        self._stride = None

    def get_mpp(self):
        return self._mpp

    def get_scaling(self, mpp: float | None) -> float:
        """Inverse of get_mpp()."""
        if not mpp:
            return 1.0
        return self._mpp / mpp

    def _open_file(self):
        self._h5file = h5py.File(self._filename, "r")
        self._metadata = json.loads(self._h5file.attrs["metadata"])
        self._mpp = self._metadata["mpp"]
        self._tile_size = self._metadata["tile_size"]
        self._tile_overlap = self._metadata["tile_overlap"]
        self._size = self._metadata["size"]
        self._num_channels = self._metadata["num_channels"]
        self._dtype = self._metadata["dtype"]
        self._stride = (self._tile_size[0] - self._tile_overlap[0], self._tile_size[1] - self._tile_overlap[1])

    def __enter__(self):
        if self._h5file is None:
            self._open_file()
        return self

    def _empty_tile(self):
        if self.__empty_tile is not None:
            return self.__empty_tile

        self.__empty_tile = np.zeros((self._num_channels, *self._tile_size), dtype=self._dtype)
        return self.__empty_tile

    def read_region(self, location, size):
        if self._h5file is None:
            self._open_file()

        image_dataset = self._h5file["data"]
        num_tiles = self._metadata["num_tiles"]
        tile_indices = self._h5file["tile_indices"]

        total_rows = math.ceil((self._size[1] - self._tile_overlap[1]) / self._stride[1])
        total_cols = math.ceil((self._size[0] - self._tile_overlap[0]) / self._stride[0])

        assert total_rows * total_cols == num_tiles

        x, y = location
        w, h = size
        if x < 0 or y < 0 or x + w > self._size[0] or y + h > self._size[1]:
            raise ValueError("Requested region is out of bounds")

        start_row = y // self._stride[1]
        end_row = min((y + h - 1) // self._stride[1] + 1, total_rows)
        start_col = x // self._stride[0]
        end_col = min((x + w - 1) // self._stride[0] + 1, total_cols)

        if self._stitching_mode == StitchingMode.AVERAGE:
            divisor_array = np.zeros((h, w), dtype=np.uint8)
        stitched_image = np.zeros((self._num_channels, h, w), dtype=np.uint8)
        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                tile_idx = (i * total_cols) + j
                # Map through tile indices
                tile_index_in_image_dataset = tile_indices[tile_idx]
                tile = (
                    self._empty_tile()
                    if tile_index_in_image_dataset == -1
                    else image_dataset[tile_index_in_image_dataset]
                )
                start_y = i * stride_height - y
                end_y = start_y + tile_height
                start_x = j * stride_width - x
                end_x = start_x + tile_width

                img_start_y = max(0, start_y)
                img_end_y = min(h, end_y)
                img_start_x = max(0, start_x)
                img_end_x = min(w, end_x)

                if self._stitching_mode == StitchingMode.CROP:
                    raise NotImplementedError
                    crop_start_y = img_start_y - start_y
                    crop_end_y = img_end_y - start_y
                    crop_start_x = img_start_x - start_x
                    crop_end_x = img_end_x - start_x

                    bbox = (crop_start_y, crop_start_x), (crop_end_y - crop_start_y, crop_end_x - crop_start_x)
                    cropped_tile = crop_to_bbox(tile, bbox)
                    stitched_image[:, img_start_y:img_end_y, img_start_x:img_end_x] = cropped_tile

                elif self._stitching_mode == StitchingMode.AVERAGE:
                    raise NotImplementedError
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

        return stitched_image

    def close(self):
        if self._h5file is not None:
            self._h5file.close()  # Close the file in close
        self._h5file = None  # Reset the h5file attribute

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
