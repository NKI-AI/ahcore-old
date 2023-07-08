# encoding: utf-8
"""
This module contains the core Lightning module for ahcore. This module is responsible for:
- Training, Validation and Inference
- Wrapping models"""
from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch.optim.optimizer
from dlup.data.dataset import ConcatDataset
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ahcore.utils.data import DataDescription, InferenceMetadata
from ahcore.utils.io import get_logger
from ahcore.utils.model import ExtractFeaturesHook

logger = get_logger(__name__)


class AhCoreLightningModule(pl.LightningModule):
    # FIXME: This can be achieved using .name
    STAGE_MAP = {TrainerFn.FITTING: "train", TrainerFn.VALIDATING: "val"}
    RELEVANT_KEYS = [
        "coordinates",
        "mpp",
        "path",
        "region_index",
        "grid_local_coordinates",
        "grid_index",
    ]
    INFERENCE_DICT: InferenceMetadata = {"mpp": None, "size": None, "tile_size": None, "filename": None}

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,  # noqa
        data_description: DataDescription,
        loss: nn.Module | None = None,
        augmentations: dict[str, nn.Module] | None = None,
        metrics: dict[str, nn.Module] | None = None,
        scheduler: Any | None = None,  # noqa
        attach_feature_layers: list[str] | None = None,
        trackers: list[Any] | None = None,
    ):
        super().__init__()

        self.save_hyperparameters(
            logger=False, ignore=["model", "augmentations", "metrics", "data_description", "loss", "trackers"]
        )  # TODO: we should send the hyperparams to the logger elsewhere

        self._model = model(out_channels=data_description.num_classes)
        self._augmentations = augmentations

        self._loss = loss
        self._robustness_metrics = []
        if metrics is not None:
            self._metrics = metrics.get("tile_level")
            self._wsi_metrics = metrics.get("wsi_level")
            if "prediction_robustness" in metrics:
                self._robustness_metrics.append(metrics["prediction_robustness"])
            if "feature_robustness" in metrics:
                self._robustness_metrics.append(metrics["feature_robustness"])
            if "linear_probing" in metrics:
                self._robustness_metrics.append(metrics["linear_probing"])

        if not trackers:
            self._trackers = []
        else:
            self._trackers = trackers

        self._data_description = data_description
        self._attach_feature_layers = attach_feature_layers

        self.predict_metadata: InferenceMetadata = self.INFERENCE_DICT  # Used for saving metadata during prediction
        self._validation_dataset: ConcatDataset | None = None

    @property
    def wsi_metrics(self):
        return self._wsi_metrics

    def forward(self, sample):
        """This function is only used during inference"""
        self._model.eval()
        return self._model.forward(sample)

    @property
    def data_description(self) -> DataDescription:
        return self._data_description

    @property
    def validation_dataset(self) -> ConcatDataset:
        return self._validation_dataset

    @property
    def _tensorboard(self) -> SummaryWriter | None:
        _tensorboard = [_ for _ in self.loggers if isinstance(_, pl.loggers.tensorboard.TensorBoardLogger)]
        if not _tensorboard:
            return None
        return _tensorboard[0].experiment

    def _compute_metrics(
        self, prediction: torch.Tensor, target: torch.Tensor, roi: torch.Tensor | None, stage: TrainerFn
    ) -> dict[str, torch.Tensor]:
        if not self._metrics:
            return {}
        metrics = {f"{self.STAGE_MAP[stage]}/{k}": v for k, v in self._metrics(prediction, target, roi).items()}
        return metrics

    def do_step(self, batch, batch_idx: int, stage: TrainerFn):
        if self._augmentations and stage in self._augmentations:
            batch = self._augmentations[stage](batch)

        if self._loss is None:
            raise RuntimeError(
                f"Loss is not defined for {self.__class__.__name__}. "
                f"This is required during training and validation"
            )

        _input = batch["image"]
        _target = batch["target"]
        # Batch size is required for accurate loss calculation and logging
        batch_size = _input.shape[0]
        # ROIs can reduce the usable area of the inputs, the loss should be scaled appropriately
        roi = batch.get("roi", None)

        # Extract features only when not training
        layer_names = [] if stage == TrainerFn.FITTING else self._attach_feature_layers
        with ExtractFeaturesHook(self._model, layer_names=layer_names) as hook:
            _prediction = self._model(_input)
            if layer_names is not []:  # Only add the features if they are requested
                batch["features"] = hook.features

        batch["prediction"] = _prediction
        loss = self._loss(_prediction, _target, roi)
        # The relevant_dict contains values to know where the tiles originate.
        _relevant_dict = {k: v for k, v in batch.items() if k in self.RELEVANT_KEYS}

        _metrics = self._compute_metrics(_prediction, _target, roi, stage=stage)
        _loss = loss.mean()
        output = {"loss": _loss, "loss_per_sample": loss.clone().detach(), "metrics": _metrics, **_relevant_dict}

        # Log the loss
        self.log(f"{self.STAGE_MAP[stage]}/loss", _loss, batch_size=batch_size, sync_dist=True, on_epoch=True)
        # Log the metrics
        self.log_dict(_metrics, batch_size=batch_size, sync_dist=True, prog_bar=False, on_epoch=True, on_step=False)

        if stage == stage.VALIDATING:  # Create tiles iterator and process metrics
            for robustness_metric in self._robustness_metrics:
                robustness_metric.update(batch)

        return output

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        output = self.do_step(batch, batch_idx, stage=TrainerFn.FITTING)
        if self.global_step == 0:
            if self._tensorboard:
                self._tensorboard.add_graph(self._model, batch["image"])
        return output

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        output = self.do_step(batch, batch_idx, stage=TrainerFn.VALIDATING)

        # This is a sanity check. We expect the filenames to be constant across the batch.
        filename = batch["path"][0]
        if any([filename != f for f in batch["path"]]):
            raise ValueError("Filenames are not constant across the batch.")
        return output

    def on_validation_start(self) -> None:
        super().on_validation_start()
        self._validation_dataset = self.trainer.datamodule.val_concat_dataset

    def on_validation_epoch_end(self) -> None:
        if len(self._robustness_metrics) > 0:
            for robustness_metric in self._robustness_metrics:
                self.log_dict(robustness_metric.compute(), sync_dist=True, prog_bar=True)
                robustness_metric.reset()

    def on_predict_start(self) -> None:
        """Check that the metadata exists (necessary for saving output) exists before going through the WSI"""
        if not self.predict_metadata["filename"]:
            raise ValueError("Empty predict_metadata found")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if self._augmentations:
            batch = self._augmentations["predict"](batch)

        inputs = batch["image"]
        preds = self._model(inputs)
        gathered_preds = self.all_gather(preds)
        return gathered_preds

    def on_predict_epoch_end(self, results) -> None:
        """Call all the inference trackers to update"""
        self.update_predict_trackers(results)
        self.predict_metadata = self.INFERENCE_DICT  # reset the metadata

    @rank_zero_only
    def update_predict_trackers(self, results):
        """On rank zero we update the trackers"""
        for tracker in self._trackers:
            tracker(results, self.predict_metadata)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
