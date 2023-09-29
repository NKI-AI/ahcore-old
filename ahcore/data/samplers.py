"""
Module implementing the samplers. These are used for instance to create batches of the same WSI.
"""
from __future__ import annotations

import math
from typing import List

from dlup.data.dataset import ConcatDataset
from torch.utils.data import Sampler

from ahcore.utils.io import get_logger

logger = get_logger()


class WsiBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset: ConcatDataset, batch_size: int):
        super().__init__(data_source=dataset)
        self._dataset = dataset
        self._batch_size = batch_size

        self._slices: List[slice] = []
        self._compute_slices()

    def _compute_slices(self):
        for idx, _ in enumerate(self._dataset.datasets):
            slice_start = 0 if len(self._slices) == 0 else self._slices[-1].stop
            slice_stop = self._dataset.cumulative_sizes[idx]
            self._slices.append(slice(slice_start, slice_stop))

    def __iter__(self):
        for slice_ in self._slices:
            batch = []
            # Within each slice, create batches of size self._batch_size
            for idx in range(slice_.start, slice_.stop):
                batch.append(idx)
                if len(batch) == self._batch_size:
                    yield batch
                    batch = []

            # If there are remaining items that couldn't form a full batch, yield them as a smaller batch
            if len(batch) > 0:
                yield batch

    def __len__(self):
        # The total number of batches is the sum of the number of batches in each slice
        return sum(math.ceil((s.stop - s.start) / self._batch_size) for s in self._slices)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"batch_size={self._batch_size}, "
            f"num_batches={self.__len__()}, "
            f"num_wsis={len(self._dataset.datasets)})"
        )
