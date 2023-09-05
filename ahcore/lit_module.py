# encoding: utf-8
"""
This module contains the core Lightning module for ahcore. This module is responsible for:
- Training, Validation and Inference
- Wrapping models"""
from __future__ import annotations

from typing import Any, Optional, cast

import kornia as K
import pytorch_lightning as pl
import torch.optim.optimizer
from dlup.data.dataset import ConcatDataset
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import ahcore.transforms.augmentations
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
    INFERENCE_DICT: InferenceMetadata = {
        "mpp": None,
        "size": None,
        "tile_size": None,
        "filename": None,
    }

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
            logger=False,
            ignore=[
                "model",
                "augmentations",
                "metrics",
                "data_description",
                "loss",
                "trackers",
            ],
        )  # TODO: we should send the hyperparams to the logger elsewhere

        self._num_classes = data_description.num_classes
        self._model = model(out_channels=self._num_classes)
        self._augmentations = augmentations

        self._loss = loss
        if metrics is not None:
            self._metrics = metrics.get("tile_level")
            self._wsi_metrics = metrics.get("wsi_level")

        if not trackers:
            self._trackers = []
        else:
            self._trackers = trackers

        self._data_description = data_description
        self._attach_feature_layers = attach_feature_layers

        self.predict_metadata: InferenceMetadata = self.INFERENCE_DICT  # Used for saving metadata during prediction
        self._validation_dataset: ConcatDataset | None = None

        # Setup test-time augmentation
        self._tta_augmentations = [
            ahcore.transforms.augmentations.Identity(),
            K.augmentation.RandomHorizontalFlip(p=1.0),
            K.augmentation.RandomVerticalFlip(p=1.0),
        ]
        self._use_test_time_augmentation = False
        self._tta_steps = len(self._tta_augmentations)

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
    def validation_dataset(self) -> Optional[ConcatDataset]:
        return self._validation_dataset

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
        stage: TrainerFn,
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

        # Log the loss
        self.log(
            f"{self.STAGE_MAP[stage]}/loss",
            _loss,
            batch_size=batch_size,
            sync_dist=True,
            on_epoch=True,
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

        output_size = (_input.shape[0], self._num_classes, *_input.shape[2:])
        _predictions = torch.zeros([self._tta_steps, *output_size], device=self.device)
        _collected_features = {k: None for k in self._attach_feature_layers}

        with ExtractFeaturesHook(self._model, layer_names=self._attach_feature_layers) as hook:
            for idx, augmentation in enumerate(self._tta_augmentations):
                model_prediction = self._model(augmentation(_input))
                _predictions[idx] = augmentation.inverse(model_prediction)

                if self._attach_feature_layers:
                    _features = hook.features
                    for key in _features:
                        if _collected_features[key] is None:
                            _collected_features[key] = torch.zeros(
                                [self._tta_steps, *_features[key].size()],
                                device=self.device,
                            )
                        _features[key] = _collected_features[key]

            output["prediction"] = _predictions.mean(dim=0)

        if self._attach_feature_layers:
            output["features"] = _collected_features

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
        assert hasattr(
            self.trainer, "datamodule"
        ), "Datamodule is not defined for the trainer. Required for validation"
        datamodule: AhCoreLightningModule = cast(AhCoreLightningModule, getattr(self.trainer, "datamodule"))
        self._validation_dataset = datamodule.val_concat_dataset

    def on_predict_start(self) -> None:
        """Check that the metadata exists (necessary for saving output) exists before going through the WSI"""
        if not self.predict_metadata["filename"]:
            raise ValueError("Empty predict_metadata found")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if self._augmentations:
            batch = self._augmentations["predict"](batch)

        inputs = batch["image"]
        predictions = self._model(inputs)
        gathered_predictions = self.all_gather(predictions)
        return gathered_predictions

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
