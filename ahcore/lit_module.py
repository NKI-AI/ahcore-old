# encoding: utf-8
"""
This module contains the core Lightning module for ahcore. This module is responsible for:
- Training, Validation and Inference
- Wrapping models"""
from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch.optim.optimizer
from pytorch_lightning.trainer.states import TrainerFn
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ahcore.utils.data import DataDescription
from ahcore.utils.io import get_logger

logger = get_logger(__name__)


class AhCoreLightningModule(pl.LightningModule):
    RELEVANT_KEYS = [
        "coordinates",
        "mpp",
        "path",
        "region_index",
        "grid_local_coordinates",
        "grid_index",
    ]

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,  # noqa
        data_description: DataDescription,
        loss: nn.Module | None = None,
        augmentations: dict[str, nn.Module] | None = None,
        metrics: dict[str, nn.Module] | None = None,
        scheduler: Any | None = None,  # noqa
    ):
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=[
                "model",
                "augmentations",
                "metrics",
                "data_description",
                "loss",
            ],
        )  # TODO: we should send the hyperparams to the logger elsewhere

        self._num_classes = data_description.num_classes
        self._model = model(out_channels=self._num_classes)
        self._augmentations = augmentations

        self._loss = loss
        if metrics is not None:
            self._metrics = metrics.get("tile_level")
            self._wsi_metrics = metrics.get("wsi_level")

        self._data_description = data_description

    @property
    def wsi_metrics(self):
        return self._wsi_metrics

    @property
    def name(self):
        return self._model.__class__.__name__

    def forward(self, sample):
        """This function is only used during inference"""
        self._model.eval()
        return self._model.forward(sample)

    @property
    def data_description(self) -> DataDescription:
        return self._data_description

    @property
    def _tensorboard(self) -> SummaryWriter | None:
        _tensorboard = [_ for _ in self.loggers if isinstance(_, pl.loggers.tensorboard.TensorBoardLogger)]
        if not _tensorboard:
            return None
        return _tensorboard[0].experiment

    def _compute_metrics(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        roi: torch.Tensor | None,
        stage: TrainerFn | str,
    ) -> dict[str, torch.Tensor]:
        if not self._metrics:
            return {}

        _stage = stage.value if isinstance(stage, TrainerFn) else stage
        metrics = {f"{_stage}/{k}": v for k, v in self._metrics(prediction, target, roi).items()}
        return metrics

    def do_step(self, batch, batch_idx: int, stage: TrainerFn | str):
        if self._augmentations and stage in self._augmentations:
            batch = self._augmentations[stage](batch)

        if self._loss is None:
            raise RuntimeError(
                f"Loss is not defined for {self.__class__.__name__}. "
                f"This is required during training and validation"
            )

        _target = batch["target"]
        # Batch size is required for accurate loss calculation and logging
        batch_size = batch["image"].shape[0]
        # ROIs can reduce the usable area of the inputs, the loss should be scaled appropriately
        roi = batch.get("roi", None)

        if stage == TrainerFn.FITTING:
            _prediction = self._model(batch["image"])
            batch["prediction"] = _prediction
        else:
            batch = {**batch, **self._get_inference_prediction(batch["image"])}
            _prediction = batch["prediction"]

        loss = self._loss(_prediction, _target, roi)

        # The relevant_dict contains values to know where the tiles originate.
        _relevant_dict = {k: v for k, v in batch.items() if k in self.RELEVANT_KEYS}
        _metrics = self._compute_metrics(_prediction, _target, roi, stage=stage)
        _loss = loss.mean()
        output = {
            "loss": _loss,
            "loss_per_sample": loss.clone().detach(),
            "metrics": _metrics,
            **_relevant_dict,
        }
        if stage != TrainerFn.FITTING:
            output["prediction"] = _prediction

        _stage = stage.value if isinstance(stage, TrainerFn) else stage

        self.log(
            f"{_stage}/loss",
            _loss,
            batch_size=batch_size,
            sync_dist=True,
            on_epoch=True,
            prog_bar=True,
        )

        # Log the metrics
        self.log_dict(
            _metrics,
            batch_size=batch_size,
            sync_dist=True,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )

        return output

    def _get_inference_prediction(self, _input: torch.Tensor) -> dict[str, torch.Tensor]:
        output = {}
        output["prediction"] = self._model(_input)
        return output

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        # TODO: This is problematic as you first need to pass through the augmentations to get the correct shape
        # if self.global_step == 0:
        #     if self._tensorboard:
        #         self._tensorboard.add_graph(self._model, batch["image"])

        output = self.do_step(batch, batch_idx, stage=TrainerFn.FITTING)
        return output

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        output = self.do_step(batch, batch_idx, stage=TrainerFn.VALIDATING)

        # This is a sanity check. We expect the filenames to be constant across the batch.
        filename = batch["path"][0]
        if any([filename != f for f in batch["path"]]):
            raise ValueError("Filenames are not constant across the batch.")
        return output

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if self._augmentations:
            batch = self._augmentations["predict"](batch)

        inputs = batch["image"]
        predictions = self._model(inputs)
        gathered_predictions = self.all_gather(predictions)
        return gathered_predictions

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "validate/loss",
                    "interval": "epoch",
                    "frequency": self.trainer.check_val_every_n_epoch,
                },
            }
        return {"optimizer": optimizer}
