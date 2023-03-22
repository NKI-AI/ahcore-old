# encoding: utf-8
"""
This module contains the core Lightning module for ahcore. This module is responsible for:
- Training, Validation and Inference
- Wrapping models"""
from __future__ import annotations

import multiprocessing
import os
from functools import partial
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any
from dlup._image import Resampling
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim.optimizer
from dlup.data.dataset import ConcatDataset
from dlup.writers import TiffCompression, TifffileImageWriter
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ahcore.transforms.augmentations import cast_list_to_tensor
from ahcore.utils.data import DataDescription, InferenceMetadata
from ahcore.utils.io import get_logger
from ahcore.utils.plotting import plot_batch

logger = get_logger(__name__)

DIFFERENCE_INDEX_MAP = {
    "under": 1,
    "correct": 2,
    "over": 3,
}
DIFFERENCE_COLORS = {
    "under": "yellow",
    "correct": "green",
    "over": "red",
}

_LOOP_DONE = "LOOP_DONE"
_WSI_PROCESSING = "PROCESSING_WSI"
_WSI_DONE = "WSI_DONE"
_TIFF_SAVE_PATH = Path(os.environ.get("SCRATCH", "/tmp")) / Path("tiffs")


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
        trackers: list[Any] | None = None,
    ):
        super().__init__()

        self.save_hyperparameters(
            logger=False, ignore=["model", "augmentations", "metrics", "data_description", "loss", "trackers"]
        )  # TODO: we should send the hyperparams to the logger elsewhere

        self._model = model(num_classes=data_description.num_classes)
        self._augmentations = augmentations

        self._loss = loss
        if metrics is not None:
            self._metrics = metrics.get("tile_level")
            self._wsi_metrics = metrics.get("wsi_level")
            self._robustness_metrics = metrics.get("robustness")

        self._plot_batch = partial(plot_batch, index_map=data_description.index_map, colors=data_description.colors)
        if not trackers:
            self._trackers = []
        else:
            self._trackers = trackers

        self._index_map = data_description.index_map
        self._data_description = data_description

        self.predict_metadata: InferenceMetadata = self.INFERENCE_DICT  # Used for saving metadata during prediction

        self._new_val_wsi: bool | None  # indicates when we start a new WSI in val_dataloader
        self._last_val_wsi_filename: str | None = None  # keeps track of last processed WSI
        self._max_val_tiffs: int = self._data_description.max_val_tiffs if self._data_description.max_val_tiffs else 1
        self._written_val_tiffs: int = 0
        self._validation_index: int | None = None  # keeps track of running indices during validation loop
        self._validation_dataset: ConcatDataset | None = None
        self._predictions_queue: Queue | None = None  # queue holding the predictions to be processed by tiffwriter
        self._tiles_process: Process | None = None
        self._tile_shape: tuple[int, int] | None = None
        multiprocessing.set_start_method("spawn")  # Set spawn method since cuda is incompatible with fork

    def forward(self, sample):
        """This function is only used during inference"""
        self._model.eval()
        return self._model.forward(sample)

    @property
    def _tensorboard(self) -> SummaryWriter | None:
        _tensorboard = [_ for _ in self.loggers if isinstance(_, pl.loggers.tensorboard.TensorBoardLogger)]
        if not _tensorboard:
            return None
        return _tensorboard[0].experiment

    def log_images(self, image: torch.Tensor, target: torch.Tensor, step: int, name: str, plotting_fn=None):
        if not plotting_fn:
            plotting_fn = self._plot_batch

        mean = cast_list_to_tensor(self._data_description.normalize_mean, 0.0)
        std = cast_list_to_tensor(self._data_description.normalize_std, 1.0)
        _image = (image.cpu() * std) + mean
        sample = plotting_fn(_image, mask_batch=target)
        if self._tensorboard is not None:
            self._tensorboard.add_image(f"{name}", sample, step)

    def _compute_metrics(
        self, prediction: torch.Tensor, target: torch.Tensor, roi: torch.Tensor | None, stage: TrainerFn
    ) -> dict[str, torch.Tensor]:
        if not self._metrics:
            return {}
        metrics = {f"{self.STAGE_MAP[stage]}/{k}": v for k, v in self._metrics(prediction, target, roi).items()}
        return metrics

    def _process_wsi_metrics(
        self, prediction: torch.Tensor, target: torch.Tensor, wsi_name: str, roi: torch.Tensor | None
    ) -> None:
        if not self._wsi_metrics:
            return None
        self._wsi_metrics.process_batch(prediction, target, roi=roi, wsi_name=wsi_name)

    def _compute_wsi_metrics(self) -> dict[str, torch.Tensor]:
        if not self._wsi_metrics:
            return {}
        metrics = self._wsi_metrics.get_average_score()
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

        _prediction = self._model(_input)
        batch["prediction"] = _prediction
        loss = self._loss(_prediction, _target, roi)
        # The relevant_dict contains values to know where the tiles originate.
        _relevant_dict = {k: v for k, v in batch.items() if k in self.RELEVANT_KEYS}

        _metrics = self._compute_metrics(_prediction, _target, roi, stage=stage)
        self._robustness_metrics.update(batch)
        _loss = loss.mean()
        output = {"loss": _loss, "loss_per_sample": loss.clone().detach(), "metrics": _metrics, **_relevant_dict}

        # Log the loss
        self.log(f"{self.STAGE_MAP[stage]}/loss", _loss, batch_size=batch_size, sync_dist=True, on_epoch=True)
        # Log the metrics
        self.log_dict(_metrics, batch_size=batch_size, sync_dist=True, prog_bar=False, on_epoch=True, on_step=False)

        if stage == stage.VALIDATING:  # Create tiles iterator and process metrics
            current_wsi_filename = self._get_current_val_wsi_filename()
            self._process_wsi_metrics(_prediction, _target, str(current_wsi_filename), roi)
            self._robustness_metrics.update(batch)

            # add the current predictions to the queue, to be processed by the tiff writer process
            curr_val_dataset, num_grid_tiles = self._get_current_val_dataset(return_num_tiles=True)

            self._enqueue_prediction(
                {**batch, "prediction": _prediction, "tile_shape": self._tile_shape, "num_grid_tiles": num_grid_tiles},
                current_wsi_filename,
            )
            # prepare the validation index for the next batch's step
            self._validation_index += batch_size
            if batch_idx == 0:  # Log the images of the first step
                predictions = F.softmax(_prediction, dim=1).detach()
                # TODO: Can we extract the current stage from the trainer?
                name = f"{self.STAGE_MAP[stage]}/images"
                self.log_images(_input, target=predictions, step=self.global_step, name=f"{name}/prediction")
                self.log_diff_image(_input, predictions=predictions, target=_target, roi=roi, name=name)

        return output

    def log_diff_image(
        self,
        input: torch.Tensor,
        predictions: torch.Tensor,
        target: torch.Tensor,
        roi: torch.Tensor | None,
        name: str,
    ) -> None:
        if self._index_map is None:
            return None

        _target = target.detach().argmax(dim=1)
        _predictions = predictions.argmax(dim=1)
        _difference_plotting_function = partial(
            plot_batch, index_map=DIFFERENCE_INDEX_MAP, colors=DIFFERENCE_COLORS, mask_as_polygon=True
        )
        for label in self._index_map:
            index = self._index_map[label]
            # TODO: Create a new function that generates this "mistake map"
            is_predicted = (_predictions == index).int()
            is_target = (_target == index).int()
            is_one_of_both = ((is_predicted + is_target) > 0).int()
            difference = ((is_predicted - is_target) + 2) * is_one_of_both
            if roi is not None:
                difference = difference * roi[:, 0, ...]

            self.log_images(
                input if roi is None else input * roi,  # Mask out the input, so it's clearer what is what
                target=difference.int(),
                step=self.global_step,
                name=f"{name}/{label}",
                plotting_fn=_difference_plotting_function,
            )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        output = self.do_step(batch, batch_idx, stage=TrainerFn.FITTING)
        if self.global_step == 0:
            if self._tensorboard:
                self._tensorboard.add_graph(self._model, batch["image"])
            self.log_images(batch["image"], target=batch["target"], step=self.global_step, name="train/images/batch_0")
            # TODO: Log ROI
        return output

    def on_validation_start(self) -> None:
        super().on_validation_start()
        self._initialize_validation_loop_attributes()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        output = self.do_step(batch, batch_idx, stage=TrainerFn.VALIDATING)
        if self.current_epoch == 0 and batch_idx == 0:
            self.log_images(batch["image"], target=batch["target"], step=self.global_step, name="val/images/batch_0")
        return output

    def on_validation_epoch_end(self) -> None:
        if self._robustness_metrics:
            robustness_metrics = self._robustness_metrics.compute()
            self._robustness_metrics.reset()
            self.log_dict(robustness_metrics, prog_bar=True, sync_dist=True)
        "Adds a marker _WSI_DONE for the last WSI and LOOP_DONE flag to the queue and waits for child process to finish"
        self._predictions_queue.put(
            [{"tile_shape": self._tile_shape}, _WSI_DONE, None]
        )  # mark the end of the last WSI in val loop
        self._predictions_queue.put([None, _LOOP_DONE, None])  # mark the end of the validation loop
        self._tiles_process.join()  # wait for the queue processing to complete

        # Log the WSI level metrics
        avg_scores = self._compute_wsi_metrics()
        self.log_dict(avg_scores, prog_bar=True, sync_dist=True)
        # Reset the metric container for the next iteration
        self._wsi_metrics.reset()
        self._written_val_tiffs = 0  # Reset the written tiffs counter

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

    def _initialize_validation_loop_attributes(self) -> None:
        """Initializes all the instance variables that are used for tracking WSI-level info during a validation loop."""
        self._predictions_queue = Queue()
        self._new_val_wsi = True
        self._validation_index = 0
        self._set_val_loop_dataset()
        self._start_new_process()

    def _start_new_process(self) -> None:
        """Starts a new child process and adds it to the self._tiles_process"""
        self._tiles_process = Process(target=_write_predictions_from_queue, args=(self._predictions_queue,))
        self._tiles_process.start()

    def _enqueue_prediction(self, batch: dict[str, torch.Tensor], filename: Path) -> None:
        """Adds the required information to self._predictions_queue, to be consumed by the TiffWriter process.

        Notes:
            1. when we start processing a new WSI, we also create and push a new tiffwriter
            2. in the queue, we mark the processing status of a WSI by either _WSI_PROCESSING or _WSI_DONE
            3. we also keep track of when a WSI is done through self._new_val_wsi flag
        """
        if self._written_val_tiffs >= self._max_val_tiffs:
            return
        _tiff_writer = None
        status = _WSI_PROCESSING
        if self._new_val_wsi:  # we are starting batches from a new WSI
            self._last_val_wsi_filename = filename
            size = self._get_current_val_wsi_size()
            save_name = _TIFF_SAVE_PATH / f"step_{self.global_step}" / Path(str(filename) + ".tiff")
            os.makedirs(save_name.parent, exist_ok=True)
            # create a new tiff writer for this WSI
            _tiff_writer = TifffileImageWriter(
                filename=str(save_name),
                size=size,
                mpp=self._data_description.inference_grid.mpp,
                tile_size=self._data_description.inference_grid.tile_size,
                pyramid=True,
                compression=TiffCompression("jpeg"),
                quality=100,
                interpolator=Resampling.NEAREST,
            )
            self._new_val_wsi = False
            self._predictions_queue.put(
                [{"num_grid_tiles": batch["num_grid_tiles"]}, status, _tiff_writer]
            )  # mark the start of a new WSI
            self._predictions_queue.put([batch, status, None])  # add the first batch
        elif filename != self._last_val_wsi_filename:  # done with a WSI
            self._written_val_tiffs += 1
            self._new_val_wsi = True
            self._last_val_wsi_filename = filename
            self._predictions_queue.put([{"tile_shape": self._tile_shape}, _WSI_DONE, None])  # mark the end of a WSI
            self._enqueue_prediction(batch, filename)  # enqueue the prediction for the new WSI
        else:  # still processing the same WSI
            status = _WSI_PROCESSING
            self._predictions_queue.put([batch, status, _tiff_writer])

    def _set_val_loop_dataset(self) -> None:
        """Fixes a reference to the validation ConcatDataset that is used in the current val loop.

        To be called at the beginning of each validation run.
        """
        self._validation_dataset = self.trainer.datamodule.val_concat_dataset

    def _get_current_val_wsi_filename(self) -> Path:
        """Retrieves the filename of the WSI that is currently processed in the validation loop"""
        batch_dataset, _ = self._get_current_val_dataset()
        return _get_filename_from_dataset(batch_dataset)

    def _get_current_val_wsi_size(self) -> tuple[int, int]:
        """Retrieves the size of the WSI processed at current val step, to be used by the tiffwriter"""
        # retrieve the dataset corresponding to this batch
        batch_dataset, _ = self._get_current_val_dataset()
        # retrieve the size for the current mpp
        mpp = self._data_description.inference_grid.mpp
        scaling = batch_dataset.slide_image.get_scaling(mpp)
        size = batch_dataset.slide_image.get_scaled_size(scaling)
        return size

    def _get_current_val_dataset(self, return_num_tiles: bool = False):
        """Retrieves the validation dataset that is processed at the current step in the val loop"""
        concat_dataset = self._validation_dataset
        curr_val_dataset = concat_dataset.index_to_dataset(self._validation_index)
        self._tile_shape = curr_val_dataset[0].grids[0][1]
        if return_num_tiles:
            num_grid_tiles = len(curr_val_dataset[0].grids[0][0])
            return curr_val_dataset, num_grid_tiles
        return curr_val_dataset


def _get_filename_from_dataset(dataset) -> Path:
    path = Path(dataset.slide_image.identifier)
    return Path(path.parent.stem) / path.stem


def _process_prediction(prediction: torch.Tensor) -> np.ndarray:
    argmax_prediction = torch.argmax(prediction, dim=1)
    return argmax_prediction.cpu().numpy().astype("uint8")


def _write_predictions_from_queue(queue: Queue) -> None:
    """Creates tile generators from elements in `queue`, and consumes them through the underlying tiffwriters."""

    def _tile_generator(num_tiles: int):
        "create generator of predictions from the queue"
        prev_index = -1  # keep track of the previous index, start before the first index
        curr_region_index = 0
        while True:
            batch, status, _ = queue.get()

            if status == _WSI_DONE:  # If we are done with WSI, check if there are any tiles left to write
                while curr_region_index < num_tiles - 1:  # add empty tiles until the end of the WSI
                    curr_region_index += 1
                    yield np.zeros(batch["tile_shape"], dtype="uint8")
                break

            if status == _LOOP_DONE:
                # if we are done with the validation loop, add a signal to the processor
                queue.put([None, _LOOP_DONE, None])
                break

            prediction = batch["prediction"]
            batch_prediction = _process_prediction(prediction)
            for pred, curr_region_index in zip(batch_prediction, batch["region_index"]):
                tile_step = curr_region_index - prev_index

                if tile_step > 1:
                    for _ in range(tile_step - 1):
                        yield np.zeros(batch["tile_shape"], dtype="uint8")
                prev_index = curr_region_index
                yield pred

    while True:
        # retrieve first element for status inspection
        batch, status, tiff_writer = queue.get()

        if status == _WSI_DONE and tiff_writer is None:
            continue  # skip the WSI_DONE signal -- this means there were no empty tiles to write

        if status == _LOOP_DONE:
            break

        # create generator based on rest of the elements
        curr_num_tiles = batch["num_grid_tiles"]
        tiles_generator = _tile_generator(curr_num_tiles)
        tiff_writer.from_tiles_iterator(tiles_generator)
