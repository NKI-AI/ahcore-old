"""Metrics that are calculated and tracked only at the end of an epoch"""
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np
import torch
import torchmetrics
from torchmetrics.classification.auroc import AUROC

from ahcore.metrics.auc import robustness_auc  # noqa
from ahcore.metrics.linear_probing import LinearProbing
from ahcore.utils.data import DataDescription
from torchmetrics.functional.classification import binary_auroc

_TORCH: str = "torch"
_NUMPY: str = "numpy"
_STR_TO_FUNC = {
    "mean": {_TORCH: torch.mean, _NUMPY: np.mean},
    "max": {_TORCH: torch.max, _NUMPY: np.max},
}
_FEATURE = "feature"
_PRED = "prediction"

EpochMetric = torchmetrics.Metric
"""Base class of epoch metrics."""

# _INTERSECTION = "intersection"
# _UNION = "union"
#
#
# def get_dice_from_intersection_and_union(intersection: torch.Tensor, union: torch.Tensor, smooth: torch.tensor = 1) -> torch.Tensor:
#     dice_score = ((2.0 * intersection) + smooth) / (union + smooth)
#     return dice_score
#
#
# class WSISoftDice(EpochMetric):
#     """WSI Dice metric class, computes the dice score over the whole WSI
#     Args:
#         from_logits: does the input contain logits? If so, softmax (for n-dimensional input) or
#             sigmoid (for 1d) will be applied, default False
#         smooth: a smoothing factor to be added to both the calculation of intersection
#             and total area, default 1
#         reduction: how to aggregate the dice scores from all WSIs in the Dataloader
#             into a single score, defaults to 'mean'. Comma separated string is accepted, if more
#             than one reduction specified, values for each reduction method are returned.
#         keep_individual_scores: whether to keep individual scores for each WSI
#     """
#
#     def __init__(
#             self,
#             from_logits: bool = False,
#             smooth: int = 1,
#             reduction: List[str] = ["mean"],
#             keep_individual_scores: bool = False,
#     ) -> None:
#         super().__init__()
#         self.from_logits = from_logits
#         self.smooth = smooth
#         self.reduction = reduction
#         self.keep_individual_scores = keep_individual_scores
#         self.wsis = set()  # keeps track of seen wsis
#
#     @staticmethod
#     def _get_intersection_and_union_attr(wsi_id: str) -> Tuple[str, str]:
#         return f"{_INTERSECTION}_{wsi_id}", f"{_UNION}_{wsi_id}"
#
#     def update(self, batch: dict) -> None:
#         """Computes intersection and union components for a dice score"""
#         # retrieve intersection and union at batch level
#         batch_intersection, batch_union = self._get_intersection_and_cardinality(
#             batch["pred"], batch["target"], roi=batch["roi"], num_classes=batch["target"].shape[1])
#         # loop through batch, as one batch may contain patches from multiple wsis
#         for wsi_path, intersection, union in zip(batch["path"], batch_intersection, batch_union):
#             wsi_id = Path(wsi_path).stem
#             self._add_state_if_needed(wsi_id)
#             intersection_attr, union_attr = self._get_intersection_and_union_attr(wsi_id)
#             self._update_intersection_and_union(intersection_attr, union_attr, intersection, union)
#
#     def compute(self) -> Dict[str, torch.Tensor]:
#         """Aggregates slide-level intersections and unions to calculate dice scores"""
#         intersections = []
#         unions = []
#         wsis = []
#         for wsi_id in self.wsis:  # type: ignore
#             wsis.append(wsi_id)  # need this as the set iteration order might change
#             intersection_attr, union_attr = self._get_intersection_and_union_attr(wsi_id)
#             intersection = getattr(self, intersection_attr)
#             union = getattr(self, union_attr)
#             # check if states were actually calculated for this wsi
#             if isinstance(intersection, torch.Tensor) and isinstance(union, torch.Tensor):
#                 intersections.append(intersection)
#                 unions.append(union)
#             else:
#                 continue
#         intersections = torch.stack(intersections)  # (num_wsi, C)
#         unions = torch.stack(unions)  # (num_wsi, C)
#         dice_per_wsi, *_ = get_dice_from_intersection_and_union(
#             intersection=intersections,
#             union=unions,
#             smooth=self.smooth,
#         )  # (num_wsi,)
#         scores = {reduction: reduce_loss(dice_per_wsi, reduction) for reduction in self.reduction}
#         if self.keep_individual_scores:
#             scores.update({wsi_id: score for wsi_id, score in zip(wsis, dice_per_wsi)})
#         return scores
#
#     def _add_state_if_needed(self, wsi_id: str) -> None:
#         intersection_attr, union_attr = self._get_intersection_and_union_attr(wsi_id)
#         if not hasattr(self, intersection_attr):
#             self.add_state(intersection_attr, default=[])
#             self.wsis.add(wsi_id)
#         if not hasattr(self, union_attr):
#             self.add_state(union_attr, default=[])
#             self.wsis.add(wsi_id)
#
#     def _update_intersection_and_union(
#             self,
#             intersection_attr: str,
#             union_attr: str,
#             intersection: torch.Tensor,
#             union: torch.Tensor,
#     ) -> None:
#         """Updates intersection and union attributes for a given WSI with the given input values
#         It handles both the case when this is the first patch seen, as well as when we only need to
#         add the current values to the running sums for intersection and union
#         """
#         intersection_state = getattr(self, intersection_attr)
#         union_state = getattr(self, union_attr)
#         if not isinstance(
#                 intersection_state, torch.Tensor
#         ):  # handles both None and empty list (=default) cases
#             setattr(self, intersection_attr, intersection)
#         else:
#             setattr(self, intersection_attr, intersection_state + intersection)
#         if not isinstance(
#                 union_state, torch.Tensor
#         ):  # handles both None and empty list (=default) cases
#             setattr(self, union_attr, union)
#         else:
#             setattr(self, union_attr, union_state + union)
#
#     def _get_intersection_and_cardinality(self, predictions: torch.Tensor, target: torch.Tensor, roi: torch.Tensor | None, num_classes: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
#
#         soft_predictions = F.softmax(predictions, dim=1)
#         # if roi is not None:
#         #     soft_predictions = soft_predictions * roi
#         #     target = target * roi
#
#         predictions = soft_predictions.argmax(dim=1)
#         _target = target.argmax(dim=1)
#
#         dice_components = []
#         for class_idx in range(num_classes):
#             curr_predictions = (predictions == class_idx).int()
#             curr_target = (_target == class_idx).int()
#             # Compute the dice score
#             if roi is not None:
#                 intersection = torch.sum((curr_predictions * curr_target) * roi.squeeze(1), dim=(0, 1, 2))
#                 cardinality = torch.sum(curr_predictions * roi.squeeze(1), dim=(0, 1, 2)) + torch.sum(
#                     curr_target * roi.squeeze(1), dim=(0, 1, 2)
#                 )
#             else:
#                 intersection = torch.sum((curr_predictions * curr_target), dim=(0, 1, 2))
#                 cardinality = torch.sum(curr_predictions, dim=(0, 1, 2)) + torch.sum(curr_target, dim=(0, 1, 2))
#             dice_components.append((intersection, cardinality))
#         return dice_components


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
            data_description: DataDescription = None,
    ):
        super().__init__(target_key=target_key, target_aggregations=target_aggregations)
        self.predictions_aggregation = predictions_aggregation
        self.add_state(name="targets", default=[])
        self.add_state(name="predictions", default=[])

        self.metric = binary_auroc
        self.per_target_metric_prefix = f"robustness-of-pred-per-{self.target_key}/"
        self.name = "PredictionAUCRobustness"

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
