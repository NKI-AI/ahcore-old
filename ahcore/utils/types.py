from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from dlup.data.dataset import Dataset
from pydantic import AfterValidator
from typing_extensions import Annotated


def is_positive(v: int | float) -> int | float:
    assert v > 0, f"{v} is not a positive {type(v)}"
    return v


PositiveInt = Annotated[int, AfterValidator(is_positive)]
PositiveFloat = Annotated[float, AfterValidator(is_positive)]
_Roi = tuple[tuple[int, int], tuple[int, int]]
Rois = list[_Roi]
GenericArray = npt.NDArray[np.generic]

DlupDatasetSample = dict[str, Any]
_DlupDataset = Dataset[DlupDatasetSample]
