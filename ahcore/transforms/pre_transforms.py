# encoding: utf-8
"""Module for the pre-transforms, which are the transforms that are applied before samples are outputted in a
dataset."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Union

import numpy as np
import torch
from dlup.data.transforms import ContainsPolygonToLabel, ConvertAnnotationsToMask, RenameLabels
from torchvision.transforms import Compose
from torchvision.transforms import functional as F

from ahcore.exceptions import ConfigurationError
from ahcore.utils.data import DataDescription
from ahcore.utils.io import get_logger

logger = get_logger(__name__)


class PreTransformTaskFactory:
    def __init__(self, transforms: list[Callable]):
        """
        Pre-transforms are transforms that are applied to the samples directly originating from the dataset.
        These transforms are typically the same for the specific tasks (e.g., segmentation,
        detection or whole-slide classification).

        Each of these tasks has a specific constructor. In all cases, the final transforms convert the PIL image
        (as the image key of the output sample) to a tensor, and ensure that the sample dictionary can be collated.

        In ahcore, the augmentations are done separately and are part of the model in the forward function.

        Parameters
        ----------
        transforms : list
            List of transforms to be used.
        """
        # These are always finally added.
        transforms += [
            ImageToTensor(),
            PathToString(),
        ]
        self._transforms = Compose(transforms)

    @classmethod
    def for_segmentation(cls, data_description: DataDescription, requires_target: bool = True):
        """
        Pretransforms for segmentation tasks. If the target is required these transforms are applied as follows:
        - Labels are renamed (for instance if you wish to map several labels to on specific class)
        - `Polygon` and `Point` annotations are converted to a mask
        - The mask is one-hot encoded.

        Parameters
        ----------
        data_description : DataDescription
        requires_target : bool

        Returns
        -------
        PreTransformTaskFactory
            The `PreTransformTaskFactory` initialized for segmentation tasks.
        """
        transforms: list[Callable] = []
        if data_description.extract_center:
            transforms.append(ExtractTCGACenter(meta_path=data_description.center_info_path, centers=data_description.centers))
        if not requires_target:
            return cls(transforms)

        if data_description.index_map is None:
            raise ConfigurationError("`index_map` is required for segmentation models when the target is required.")

        if data_description.remap_labels is not None:
            transforms.append(RenameLabels(remap_labels=data_description.remap_labels))

        transforms.append(
            ConvertAnnotationsToMask(roi_name=data_description.roi_name, index_map=data_description.index_map)
        )
        transforms.append(OneHotEncodeMask(index_map=data_description.index_map))

        return cls(transforms)

    @classmethod
    def for_wsi_classification(cls, data_description: DataDescription, requires_target: bool = True):
        transforms: list[Callable] = []
        if not requires_target:
            return cls(transforms)

        index_map = data_description.index_map
        if index_map is None:
            raise ConfigurationError("`index_map` is required for classification models when the target is required.")

        transforms.append(LabelToClassIndex(index_map=index_map))

        return cls(transforms)

    @classmethod
    def for_tile_classification(cls, roi_name: str, label: str, threshold: float):
        """Tile classification is based on a transform which checks if a polygon is present for a given threshold"""
        convert_annotations = ContainsPolygonToLabel(roi_name=roi_name, label=label, threshold=threshold)
        return cls([convert_annotations])

    def __call__(self, data):
        return self._transforms(data)

    def __repr__(self):
        return f"PreTransformTaskFactory(transforms={self._transforms})"


class LabelToClassIndex:
    """
    Maps label values to class indices according to the index_map specified in the data description.

    Example:
        If there are two tasks:
            - Task1 with classes {A, B, C}
            - Task2 with classes {X, Y}
        Then an input sample could look like: {{"labels": {"Task1": "C", "Task2: "Y"}, ...}
        If the index map is: {"A": 0, "B": 1, "C": 2, "X": 0, "Y": 1}
        The returned sample will look like: {"labels": {"task1": 2, "task2": 1}, ...}
    """

    def __init__(self, index_map: dict[str, int]):
        self._index_map = index_map

    def __call__(self, sample):
        sample["labels"] = {
            label_name: self._index_map[label_value] for label_name, label_value in sample["labels"].items()
        }

        return sample


class OneHotEncodeMask:
    def __init__(self, index_map: dict[str, int]):
        """Create the one-hot encoding of the mask for segmentation.
        If we have `N` classes, the result will be an `(B, N + 1, H, W)` tensor, where the first sample is the
        background.

        Parameters
        ----------
        index_map : dict[str, int]
            Index map mapping the label name to the integer value it has in the mask.

        """
        self._index_map = index_map

        # Check the max value in the mask
        self._largest_index = max(index_map.values())

    def __call__(self, sample):
        mask = sample["annotation_data"]["mask"]

        new_mask = np.zeros((self._largest_index + 1, *mask.shape))
        for idx in range(self._largest_index + 1):
            new_mask[idx] = (mask == idx).astype(np.float32)

        sample["annotation_data"]["mask"] = new_mask
        return sample


class PathToString:
    """Path objects cannot be collated in the standard pytorch collate function.
    This transform converts the path to a string.
    """

    def __call__(self, sample):
        # Path objects cannot be collated
        sample["path"] = str(sample["path"])

        return sample


class ImageToTensor:
    """
    Transform to translate the output of a dlup dataset to data_description supported by AhCore
    """

    def __call__(self, sample):
        sample["image"] = F.pil_to_tensor(sample["image"].convert("RGB")).float()

        if sample["image"].sum() == 0:
            raise RuntimeError(f"Empty tile for {sample['path']} at {sample['coordinates']}")

        # annotation_data is added by the ConvertPolygonToMask transform.
        if "annotation_data" not in sample:
            return sample

        if "mask" in sample["annotation_data"]:
            mask = sample["annotation_data"]["mask"]
            if len(mask.shape) == 2:
                # Mask is not one-hot encoded
                mask = mask[np.newaxis, ...]
            sample["target"] = torch.from_numpy(mask).float()

        if "roi" in sample["annotation_data"]:
            roi = sample["annotation_data"]["roi"]
            sample["roi"] = torch.from_numpy(roi[np.newaxis, ...]).float()

        # Not required anymore
        del sample["annotation_data"]
        # This might be empty.
        del sample["annotations"]

        return sample

    def __repr__(self):
        return f"{type(self).__name__}()"


class ExtractTCGACenter:
    """Extracts center metadata for a TCGA WSI, given a metadata csv file

    Args:
        path: path to csv file containing 2 columns, TSS Code and Source Site (see example below)
        centers: list of centers to index. If a center is encountered that is not part of
            the provided list, it automatically gets assigned index len(centers)

    An example content of a metadata csv file would be
    TSS Code,Source Site
    01,International Genomics Consortium
    02,MD Anderson Cancer Center
    """

    def __init__(self, meta_path: Union[Path, str], centers: list[str]) -> None:
        # extract slide-to-center mapping from meta file
        self._center_map = {}
        self._meta_path = Path(meta_path)
        with open(self._meta_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._center_map[row["TSS Code"]] = row["Source Site"]

        # extract center encodings:
        self._center_encoding = {center: i for i, center in enumerate(centers)}
        self._num_encodings = len(self._center_encoding)

    def __call__(self, sample: dict) -> dict:
        # first get center as string from the slide_id (= last part of wsi filename)
        slide_id = Path(sample["path"]).stem
        center = self._center_map[slide_id.split("-")[1]]
        # add both center and its encoding
        encoding = self._center_encoding.get(center, self._num_encodings)
        sample["center"] = (center, encoding)
        return sample
