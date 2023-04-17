# encoding: utf-8
"""
Augmentations factory

"""
from __future__ import annotations

from typing import cast

import kornia.augmentation as K
import torch
from kornia.constants import DataKey, Resample
from omegaconf import ListConfig
from torch import nn

from ahcore.utils.data import DataDescription
from ahcore.utils.io import get_logger

logger = get_logger(__name__)


def cast_list_to_tensor(obj, default=0.0):
    """
    Converts a list to a tensor

    Parameters
    ----------
    obj : Any
    default : float
        Default value
    Returns
    -------
    torch.Tensor
    """
    if obj is None:
        obj = [default] * 3

    if isinstance(obj, (float, int)):
        obj = [float(obj)] * 3

    # TODO: Figure out why ListConfig is needed.
    if isinstance(obj, (list, tuple, ListConfig)):
        obj = torch.Tensor(obj)[:, None, None]

    return obj


class MeanStdNormalizer(nn.Module):
    """
    Normalizes the mean and standard deviation of the input image. Assumes the original range is `[0, 255]`.
    """

    def __init__(self, mean: tuple[float, float, float] | None = None, std: tuple[float, float, float] | None = None):
        """
        Parameters
        ----------
        mean : tuple[float, float, float], optional
        std : tuple[float, float, float], optional
        """
        super().__init__()
        if mean is None:
            self._mean = nn.Parameter(torch.Tensor([0.0] * 3), requires_grad=False)
        else:
            self._mean = nn.Parameter(torch.Tensor(mean), requires_grad=False)

        if std is None:
            self._std = nn.Parameter(torch.Tensor([1.0] * 3), requires_grad=False)
        else:
            self._std = nn.Parameter(torch.Tensor(std), requires_grad=False)

    def forward(self, *args: torch.Tensor, **kwargs):
        output = []
        data_keys = kwargs["data_keys"]
        for sample, data_key in zip(args, data_keys):
            if data_key in [DataKey.INPUT, 0, "INPUT"]:
                sample = sample / 255.0
                sample = (sample - self._mean[..., None, None].to(sample.device)) / self._std[..., None, None].to(
                    sample.device
                )
            output.append(sample)

        if len(output) == 1:
            return output[0]

        return output


class CenterCrop(nn.Module):
    """Perform a center crop of the image and target"""

    def __init__(self, size: int | tuple[int, int], **kwargs):
        super().__init__()
        _size = size
        if isinstance(size, int):
            _size = (size, size)

        if isinstance(size, ListConfig):
            _size = tuple(size)

        self._cropper = K.CenterCrop(
            size=_size, align_corners=True, p=1.0, keepdim=False, cropping_mode="slice"
        )

    def forward(self, *sample: torch.Tensor, data_keys: list[str | int | DataKey] = None, **kwargs):
        output = [self._cropper(item) for item in sample]

        if len(output) == 1:
            return output[0]
        return output


def _parse_random_apply(random_apply: int | bool | tuple[int, int] | ListConfig) -> int | bool | tuple[int, int]:
    if isinstance(random_apply, (int, bool)):
        return random_apply

    if isinstance(random_apply, ListConfig):
        return cast(tuple[int, int], tuple(random_apply))

    return random_apply


def _parse_random_apply_weights(random_apply_weights: list[float] | ListConfig | None) -> list[float] | None:
    if isinstance(random_apply_weights, ListConfig):
        return cast(list[float], list(random_apply_weights))

    return random_apply_weights


class AugmentationFactory(nn.Module):
    """Factory for the augmentation. There are three classes of augmentations:
    - `initial_transforms`: Transforms which are the first to always be applied to the sample
    - `intensity_augmentations`: Transforms which only affect the intensity and not the geometry. Only applied to the
    image.
    - `geometric_augmentations`: Transforms which affect the geometry. They are applied to both the image, ROI and mask.
    """

    DATA_KEYS = {"image": DataKey.INPUT, "target": DataKey.MASK, "roi": DataKey.MASK}

    def __init__(
        self,
        data_description: DataDescription,
        initial_transforms: list | None = None,
        random_apply_intensity: int | bool | ListConfig = False,
        random_apply_weights_intensity: list[float] | None = None,
        intensity_augmentations: list | None = None,
        random_apply_geometric: int | bool | ListConfig = False,
        random_apply_weights_geometric: list[float] | ListConfig | None = None,
        geometric_augmentations: list | None = None,
        final_transforms: list | None = None,
    ):
        super().__init__()

        self._transformable_keys = ["image", "target"]
        if data_description.use_roi:
            self._transformable_keys.append("roi")
        self._data_keys = [self.DATA_KEYS[key] for key in self._transformable_keys]

        # Initial transforms will be applied sequentially
        if initial_transforms:
            for transform in initial_transforms:
                logger.info(f"Using initial transform {transform}")
        self._initial_transforms = nn.ModuleList(initial_transforms)

        # Intensity augmentations will be selected in random order
        if intensity_augmentations:
            for transform in intensity_augmentations:
                logger.info(f"Adding intensity augmentation {transform}")

        self._intensity_augmentations = None
        if intensity_augmentations:
            self._intensity_augmentations = K.AugmentationSequential(
                *intensity_augmentations,
                data_keys=list(self.DATA_KEYS.values()),
                same_on_batch=False,
                random_apply=_parse_random_apply(random_apply_intensity),
                random_apply_weights=_parse_random_apply_weights(random_apply_weights_intensity),
            )

        # Geometric augmentations will be selected in random order.
        if geometric_augmentations:
            for transform in geometric_augmentations:
                logger.info(f"Adding geometric augmentation {transform}")

        self._geometric_augmentations = None
        if geometric_augmentations:
            self._geometric_augmentations = K.AugmentationSequential(
                *geometric_augmentations,
                data_keys=list(self.DATA_KEYS.values()),
                same_on_batch=False,
                random_apply=_parse_random_apply(random_apply_geometric),
                random_apply_weights=_parse_random_apply_weights(random_apply_weights_geometric),
                extra_args={DataKey.MASK: dict(resample=Resample.NEAREST, align_corners=True)},
            )

        # Final transforms will be applied sequentially
        if final_transforms:
            for transform in final_transforms:
                logger.info(f"Using final transform {transform}")
        self._final_transforms = nn.ModuleList(final_transforms)

    def forward(self, sample):
        output_data = [sample[key] for key in self._transformable_keys if key in sample]
        if self._initial_transforms:
            kwargs = {"data_keys": self._data_keys, "filenames": sample["path"]}
            for transform in self._initial_transforms:
                output_data = transform(*output_data, **kwargs)

        if isinstance(output_data, torch.Tensor):
            output_data = [output_data]

        if self._intensity_augmentations:
            output_data[0] = self._intensity_augmentations(*output_data[:1], data_keys=[DataKey.INPUT])

        if self._geometric_augmentations:
            output_data = self._geometric_augmentations(*output_data, data_keys=self._data_keys)

        if isinstance(output_data, torch.Tensor):
            output_data = [output_data]

        if self._final_transforms:
            for transform in self._final_transforms:
                output_data = transform(*output_data, data_keys=self._data_keys)

        if isinstance(output_data, torch.Tensor):
            output_data = [output_data]

        # Add the output data back into the sample
        for key, curr_output in zip(self._transformable_keys, output_data):
            sample[key] = curr_output

        return sample
