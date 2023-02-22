"""Metrics that are calculated and tracked only at the end of an epoch"""
from typing import List

import numpy as np
import torch
import torchmetrics
from torchmetrics.classification.auroc import AUROC

from ahcore.metrics.auc import robustness_auc  # noqa
from ahcore.metrics.linear_probing import LinearProbing

_TORCH: str = "torch"
_NUMPY: str = "numpy"
_STR_TO_FUNC = {
    "mean": {_TORCH: torch.mean, _NUMPY: np.mean},
    "max": {_TORCH: torch.max, _NUMPY: np.max},
}
_FEATURE = "feature"
_PRED = "pred"


EpochMetric = torchmetrics.Metric
"""Base class of epoch metrics."""


class PerTargetMetric(EpochMetric):
    """Base class for per target epoch metrics.

    Args:
        target_key: indicates the target with respect to which the robustness metrics are computed,
            e.g. "center"
        target_aggregations: list of strings indicating the aggregation statistics to be computed
            over targets. For example, when `target_key`="center", on top of having the desired
            metric per target, one will additionally also compute statistics (e.g. mean, std) over
            all centers and add these as separate metrics.

    Note: The base class expects that every derived class will implement the following methods:
        get_over_targets_metric_prefix - constructs the prefix of the metric names that will be used
            for tracking robustness metrics over all targets, e.g. "robustness-of-pred-over-centers"
        get_aggregation_metric_suffix - for each over targets aggregation statistic (e.g. mean of
            all per center values) constructs the suffix of the metric corresponding to that
            aggregation (e.g. "auc-mean")
    """

    def __init__(self, target_key: str, target_aggregations: List[str]) -> None:
        super().__init__()
        self.target_key = target_key
        self.target_aggregations = target_aggregations
        self.encoding_to_name: dict[int, str] = {}
        self.over_targets_metric_prefix = self.get_over_targets_metric_prefix()

    def compute_statistics_over_targets(self, per_target_metrics: List[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Adds the statistics by aggregating over all targets"""
        stats = {}
        per_target_metrics = [
            a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a for a in per_target_metrics
        ]
        for aggregation in self.target_aggregations:
            metric_name_suffix = self.get_aggregation_metric_suffix(aggregation)
            metric_name = self.over_targets_metric_prefix + metric_name_suffix
            agg_func = getattr(np, aggregation)
            metric_value = agg_func(per_target_metrics)
            stats[metric_name] = metric_value
        return stats

    def get_over_targets_metric_prefix(self) -> str:
        """To be overwritten by derived classes"""
        raise NotImplementedError

    def get_aggregation_metric_suffix(self, aggregation: str) -> str:
        """To be overwritten by derived classes"""
        raise NotImplementedError

    def get_encoding_to_name(self, batch: dict) -> None:
        """Updates self.econding_to_name using the values in `batch`.
        If an encoding is used for multiple targets, all those targets are aggregated under
        a default name 'all_other_targets'.
        """

        for name, encoding in zip(*batch[self.target_key]):  # type: ignore
            encoding = encoding.item()
            if encoding in self.encoding_to_name and self.encoding_to_name[encoding] != name:
                self.encoding_to_name[encoding] = f"all_other_{self.target_key}s"
            else:
                self.encoding_to_name[encoding] = name


class PredictionAUCRobustness(PerTargetMetric):
    """Tracks AUC between predictions and a chosen target (e.g. center)

    On top of the base-class arguments, the following args are expected
        predictions_aggregation: indicates how an image batch of shape (B, * H, W) should be
            aggregated per-patch (e.g. mean of all values in (C, H, W) dimensions). The output after
            applying this operation is of shape (B,)

    """

    def __init__(
        self,
        target_key: str = "center",
        predictions_aggregation: str = "mean",
        target_aggregations: List[str] = ["min", "max", "mean", "std"],
    ):
        super().__init__(target_key=target_key, target_aggregations=target_aggregations)
        self.predictions_aggregation = predictions_aggregation
        self.add_state(name="targets", default=[])
        self.add_state(name="predictions", default=[])

        self.metric = AUROC(task="binary")
        self.per_target_metric_prefix = f"robustness-of-pred-per-{self.target_key}/"

    def update(self, batch: dict) -> None:
        self.get_encoding_to_name(batch)
        self.targets += [torch.flatten(batch[self.target_key][1])]  # type: ignore  # shape (B,)
        # aggregate the (h,w) component of each input according to the input aggregation
        aggregation_func = _STR_TO_FUNC[self.predictions_aggregation][_TORCH]
        predictions_batch = batch[_PRED]
        dim = tuple(i for i in range(1, len(predictions_batch.shape)))
        self.predictions += [aggregation_func(predictions_batch, dim=dim)]  # type: ignore

    def compute(self) -> dict:
        predictions = torch.cat(self.predictions, dim=0)  # type: ignore
        y = torch.cat(self.targets, dim=0)  # type: ignore
        aucs = {}
        for target_encoding in torch.unique(y):
            targets = target_encoding == y
            target_name = self.encoding_to_name[target_encoding.item()]
            metric_name_suffix = f"auc-{target_name}"
            metric_name = self.per_target_metric_prefix + metric_name_suffix
            aucs[metric_name] = self.metric(predictions, targets)

        aucs.update(self.compute_statistics_over_targets(list(aucs.values())))
        return aucs

    def get_aggregation_metric_suffix(self, aggregation: str) -> str:
        """Overwrites base class with desired suffix"""
        return f"auc-{aggregation}"

    def get_over_targets_metric_prefix(self) -> str:
        """Overwrites base class with desired prefix"""
        return f"robustness-of-pred-over-{self.target_key}s/"


class FeatureAUCRobustness(PerTargetMetric):
    """Tracks AUC between patch-level feature aggregation and chosen target

    On top of the base-class arguments, the following args are expected
        feature_layer: integer indicating which features layer to be used from the encoder
            (0 = last layer)
        features_patch_aggregation: for a features batch of shape (B, C, H, W), it indicates the
            aggregation over the last two dimensions. The output after applying this operation is of
            shape (B, C)
        features_per_target_aggregation: after applying features_patch_aggregation and computing the
            auc per target, we get for each target a tensor of dimension equal to C = num features.
            This string indicates how to aggregate this tensor into one statistic.

    """

    def __init__(
        self,
        target_key: str = "center",
        feature_layer: int = 0,
        features_patch_aggregation: str = "mean",
        features_per_target_aggregation: str = "mean",
        target_aggregations: List[str] = ["min", "max", "mean", "std"],
    ):
        super().__init__(target_key=target_key, target_aggregations=target_aggregations)
        self.feature_layer = feature_layer
        self.features_patch_aggregation = features_patch_aggregation
        self.features_per_target_aggregation = features_per_target_aggregation
        self.add_state(name="features", default=[])
        self.add_state(name="targets", default=[])

        self.per_target_metric_prefix = f"robustness-of-feat-per-{self.target_key}/"

    def update(self, batch: dict) -> None:
        self.get_encoding_to_name(batch)
        self.targets += [torch.flatten(batch[self.target_key][1])]  # type: ignore  # shape (B,)
        # aggregate the (h,w) component of each input according to the input aggregation
        aggregation_func = _STR_TO_FUNC[self.features_patch_aggregation][_TORCH]
        input_batch = batch[_FEATURE][self.feature_layer]
        self.features += [aggregation_func(input_batch, dim=(-2, -1))]  # type: ignore

    def compute(self) -> dict:
        features = torch.cat(self.features, dim=0).detach().cpu().numpy()  # type: ignore
        y = torch.cat(self.targets, dim=0).detach().cpu().numpy()  # type: ignore
        aucs = {}
        for target_encoding in np.unique(y):
            targets = target_encoding == y
            target_name = self.encoding_to_name[target_encoding.item()]
            metric_name_suffix = f"auc-{target_name}"
            metric_name = self.per_target_metric_prefix + metric_name_suffix
            aucs[metric_name] = robustness_auc(features, targets)
            # reduce per-target metric from a tensor of (num_features) to one value
            per_target_reduction = _STR_TO_FUNC[self.features_per_target_aggregation][_NUMPY]
            aucs[metric_name] = per_target_reduction(aucs[metric_name])

        aucs.update(self.compute_statistics_over_targets(list(aucs.values())))
        return aucs

    def get_aggregation_metric_suffix(self, aggregation: str) -> str:
        """Overwrites base class with desired suffix"""
        return f"auc-{aggregation}"

    def get_over_targets_metric_prefix(self) -> str:
        """Overwrites base class with desired prefix"""
        return f"robustness-of-feat-over-{self.target_key}s/"


class LinearModelFeatureRobustness(PerTargetMetric):
    """Model-level AUC between features and a chosen target (e.g. center) using a linear model.
    The class reports metrics per target and aggregation across targets by desired statistics (e.g.
    min, max, mean, std)

    On top of the base-class arguments, the following args are expected:
        feature_layer: integer indicating which features layer to be used from the encoder
            (0 = last layer)
        features_patch_aggregation: for a features batch of shape (B, C, H, W), it indicates the
            aggregation over the last two dimensions. The output after applying this operation is of
            shape (B, C)
        model_name: name of the linear model to be used
        test_size: test size to be used by the linear model
        scoring: scoring to be used by the linear model
        **kwargs: further kwargs to be passed to the linear model
    """

    def __init__(
        self,
        target_key: str = "center",
        feature_layer: int = 0,
        features_patch_aggregation: str = "mean",
        target_aggregations: List[str] = ["min", "max", "mean", "std"],
        model_name: str = "LogisticRegression",
        test_size: float = 0.3,
        scoring: str = "accuracy",
        **kwargs,
    ):
        super().__init__(target_key=target_key, target_aggregations=target_aggregations)
        self.feature_layer = feature_layer
        self.features_patch_aggregation = features_patch_aggregation
        self.add_state(name="targets", default=[])
        self.add_state(name="features", default=[])

        self.linear_prober = LinearProbing(model_name=model_name, test_size=test_size, scoring=scoring, **kwargs)

        self.per_target_metric_prefix = f"robustness-of-linear-model-per-{self.target_key}/"

    def update(self, batch: dict) -> None:
        self.get_encoding_to_name(batch)
        self.targets += [torch.flatten(batch[self.target_key][1])]  # type: ignore  # shape (B,)
        # aggregate the (h,w) component of each input according to the input aggregation
        aggregation_func = _STR_TO_FUNC[self.features_patch_aggregation][_TORCH]
        input_batch = batch[_FEATURE][self.feature_layer]
        self.features += [aggregation_func(input_batch, dim=(-2, -1))]  # type: ignore

    def compute(self) -> dict:
        # identify number of features (=num of channels)
        inputs = torch.cat(self.features, dim=0)  # type: ignore
        targets = torch.cat(self.targets, dim=0)  # type: ignore

        # compute auc per center
        aucs = {}
        for target_encoding in torch.unique(targets):
            y = target_encoding == targets
            target_name = self.encoding_to_name[target_encoding.item()]
            metric_name_suffix = f"patch-{self.features_patch_aggregation}-feat-rob-auc-{target_name}"
            metric_name = self.per_target_metric_prefix + metric_name_suffix
            aucs[metric_name] = self.linear_prober(inputs, y)

        aucs.update(self.compute_statistics_over_targets(list(aucs.values())))
        return aucs

    def get_aggregation_metric_suffix(self, aggregation: str) -> str:
        """Overwrites base class with desired suffix"""
        return f"patch-{self.features_patch_aggregation}-feat-rob-{aggregation}"

    def get_over_targets_metric_prefix(self) -> str:
        """Overwrites base class with desired prefix"""
        return f"robustness-of-linear-model-over-{self.target_key}s/"
