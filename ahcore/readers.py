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
    def __init__(self, filename, size, tile_size, tile_overlap, stitching_mode):
        self._filename = filename
        self._tile_overlap = tile_overlap
        self._tile_size = tile_size
        self._size = size
        self._stitching_mode = stitching_mode

        self.__empty_tile = None

    def _empty_tile(self, metadata):
        if self.__empty_tile is not None:
            return self.__empty_tile

        self.__empty_tile = np.zeros(metadata["shape"], dtype=metadata["dtype"])
        return self.__empty_tile

    def read_region(self, location, size):
        with h5py.File(self._filename, "r") as h5file:
            image_dataset = h5file["data"]
            metadata = json.loads(h5file.attrs["metadata"])
            num_tiles = metadata["num_tiles"]
            tile_indices = h5file["tile_indices"]

            num_channels, tile_height, tile_width = image_dataset.shape[1:]

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
            stitched_image = np.zeros((num_channels, h, w), dtype=np.uint8)
            for i in range(start_row, end_row):
                for j in range(start_col, end_col):
                    tile_idx = (i * total_cols) + j
                    # Map through tile indices
                    tile_index_in_image_dataset = tile_indices[tile_idx]
                    if tile_index_in_image_dataset == -1:
                        tile = self._empty_tile(metadata)
                    else:
                        tile = image_dataset[tile_index_in_image_dataset]
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
