import os
import sys
from dataclasses import dataclass
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim.optimizer
from dlup.writers import TiffCompression, TifffileImageWriter
from pytorch_lightning.callbacks import Callback

from ahcore.utils.process import Process

_LOOP_DONE = "LOOP_DONE"
_WSI_PROCESSING = "PROCESSING_WSI"
_WSI_DONE = "WSI_DONE"


@dataclass
class TileWriterInput:
    """
    Message format for the tiffwriter queue.
    """

    status: str
    filename: Optional[str] = None
    size: Optional[Union[Tuple[int, int], Tuple[int, int, int]]] = None
    mpp: Optional[Union[float, Tuple[float, float]]] = None
    prediction: Any = None
    region_index: Optional[int] = None
    num_grid_tiles: Optional[int] = None
    global_step: Optional[int] = None


class TiffWriterCallback(Callback):
    """
    Callback to write tiffs on validation/test/prediction. Data is required to be loaded in order
    and some additional identifier keys need to be provided.

    The prediction batch input should contain the following dict values. Keyword arguments can be
    used to overwrite the default names:
    pred: Prediction mask.
    path: A unique name of the WSI.
    region_index: The region index of the tile.
    num_grid_tiles: The total number of grid tiles in the wsi.
    layer_size: The size in pixels of the layer to write.

    Args:
        save_dir: Directory to save the predictions, if None will store with logs
        prediction_key: The batch key used for predictions
        filename_key: The batch key used for identifying filenames
        num_tiles_key: The batch key used for identifying the number of tiles in the wsi
        layer_size_key: The batch key used for identifying the size of the layer to write
        region_index_key: The batch key used for identifying the region index of the tile
        mpp_key: The batch key used for identifying the mpp of the layer to write
        max_full_tiffs: Maximum number of tiff files to write
        write_on_predict: Write tiff files on prediction step
        write_on_test: Write tiff files on test step
        write_on_validation: Write tiff files on validation step
        compression: Tiff compression to use (if None will use JPEG compression)
    """

    def __init__(
        self,
        save_dir: Optional[str] = None,
        prediction_key: str = "pred",
        filename_key: str = "path",
        num_tiles_key: str = "num_grid_tiles",
        layer_size_key: str = "layer_size",
        region_index_key: str = "region_index",
        mpp_key: str = "mpp",
        max_full_tiffs: Optional[int] = None,
        write_on_validation: bool = False,
        write_on_test: bool = True,
        write_on_predict: bool = True,
        compression: TiffCompression = None,
    ):
        self._save_dir = save_dir
        self._prediction_key = prediction_key
        self._filename_key = filename_key
        self._num_tiles_key = num_tiles_key
        self._layer_size_key = layer_size_key
        self._region_index_key = region_index_key
        self._write_on_validation = write_on_validation
        self._write_on_test = write_on_test
        self._write_on_predict = write_on_predict
        self._mpp_key = mpp_key
        self._max_full_tiffs = max_full_tiffs
        self._predictions_queue: Optional[
            Queue
        ] = None  # queue holding the predictions to be processed by tiffwriter
        self._tiles_process = None
        self._compression = compression
        self._written_tiffs = []
        self._last_wsi = None

    @property
    def tiles_process(self):
        if self._tiles_process is None:
            raise ValueError("TiffWriterProcess is not initialized")
        return self._tiles_process

    @property
    def predictions_queue(self):
        if self._predictions_queue is None:
            raise ValueError("predictions_queue is not initialized")
        return self._predictions_queue

    def _start_new_process(self, trainer: "pl.Trainer", phase: str) -> None:
        """Starts a new child process and adds it to the self._tiles_process"""
        self._written_tiffs = []
        self._predictions_queue = Queue()
        self._tiles_process = _TiffWriterProcess(
            phase=phase,
            save_dir=self._resolve_save_dir(trainer),
            queue=self._predictions_queue,
            compression=self._compression,
        )
        self._tiles_process.start()

    def _resolve_save_dir(self, trainer: "pl.Trainer") -> str:
        """Determines model checkpoint save directory at runtime. Reference attributes from the
        trainer's logger to determine where to save checkpoints. The path for saving weights is
        set in this priority:

        1.  The ``TiffWriterCallback``'s ``save_dir`` if passed
        2.  The ``Logger``'s ``log_dir`` if the trainer has loggers
        3.  The ``Trainer``'s ``default_root_dir`` if the trainer has no loggers

        The path gets extended with subdirectory "checkpoints".
        """
        if self._save_dir is not None:
            # short circuit if save_dir was passed
            return self._save_dir

        if len(trainer.loggers) > 0:
            if trainer.loggers[0].save_dir is not None:
                save_dir = trainer.loggers[0].save_dir
            else:
                save_dir = trainer.default_root_dir
            name = trainer.loggers[0].name
            version = trainer.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
            pred_dir = os.path.join(save_dir, str(name), version, "predictions")
        else:
            # if no loggers, use default_root_dir
            pred_dir = os.path.join(trainer.default_root_dir, "predictions")

        return pred_dir

    def _enqueue_prediction(self, batch: Dict[str, Any], global_step: int) -> None:
        """
        Adds the required information to self.predictions_queue, to be consumed by the TiffWriter
        process.

        Notes:
            in the queue, we mark the processing status of a WSI by either _WSI_PROCESSING or
            _WSI_DONE
        """
        if self._reached_maximum_tiles():
            return
        if not isinstance(batch, dict):
            raise ValueError("Prediction batch must be a dict")
        status = _WSI_PROCESSING
        if self.tiles_process.is_alive():
            for idx, filename in enumerate(batch[self._filename_key]):
                if filename != self._last_wsi:
                    self._written_tiffs.append(filename)
                    self.predictions_queue.put(TileWriterInput(status=_WSI_DONE))
                    if self._reached_maximum_tiles():
                        return
                self._last_wsi = filename
                self.predictions_queue.put(
                    TileWriterInput(
                        filename=filename,
                        region_index=batch[self._region_index_key][idx].cpu().item(),
                        prediction=batch[self._prediction_key][idx].cpu(),
                        num_grid_tiles=batch[self._num_tiles_key][idx].cpu(),
                        size=batch[self._layer_size_key][idx].cpu(),
                        mpp=batch[self._mpp_key][idx].cpu().item(),
                        status=status,
                        global_step=global_step,
                    )
                )

    def _reached_maximum_tiles(self):
        """
        Return:
            True if maximum number of tiff tiles is reached
        """
        return self._max_full_tiffs is not None and len(self._written_tiffs) > self._max_full_tiffs

    def on_validation_batch_end(self, *args, **kwargs):
        if self._write_on_validation:
            self._write_batch(*args, **kwargs)

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self._write_on_validation:
            self._start_new_process(trainer, phase="validation")

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self._write_on_validation:
            self._stop_process()

    def on_test_batch_end(self, *args, **kwargs):
        if self._write_on_test:
            self._write_batch(*args, **kwargs)

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._write_on_test:
            self._start_new_process(trainer, phase="test")

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._write_on_test:
            self._stop_process()

    def on_predict_batch_end(self, *args, **kwargs):
        if self._write_on_predict:
            self._write_batch(*args, **kwargs)

    def on_predict_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self._write_on_predict:
            self._start_new_process(trainer, phase="predict")

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._write_on_predict:
            self._stop_process()

    def _write_batch(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        self.tiles_process.check_exceptions()
        self._enqueue_prediction(outputs, global_step=trainer.global_step)

    def _stop_process(self):
        """
        Adds a marker _WSI_DONE for the last WSI and LOOP_DONE flag to the queue and waits
        for child process to finish
        """
        self.predictions_queue.put(TileWriterInput(status=_WSI_DONE))
        # mark the end of the writing process
        self.predictions_queue.put(TileWriterInput(status=_LOOP_DONE))
        self.tiles_process.join()


def auto_convert_to_mask(prediction: torch.Tensor) -> np.ndarray:
    """
    Auto detects multi-class output, and returns a prediction mask.

    Args:
        prediction: Prediction mask (H, W), (1, H, W) or (C, H, W)

    Returns:
        Returns prediction mask (H, W)
    """
    prediction = prediction.squeeze(0)
    if prediction.ndim == 2:
        # Use binary prediction
        prediction = prediction.sigmoid().round()
    else:
        prediction = torch.argmax(prediction, dim=0)
    return prediction.numpy().astype("uint8")


class _TiffWriterProcess:
    def __init__(
        self,
        *,
        phase: str,
        save_dir: str,
        queue: Queue,
        compression: Optional[TiffCompression] = None,
    ):
        self._queue = queue
        self._phase = phase
        self._save_dir = Path(save_dir)
        self._process = None
        self._tile_size = None
        self._last_wsi = None
        self._compression = TiffCompression("jpeg") if compression is None else compression

    def _tile_generator(self, msg: TileWriterInput, tile_size, num_grid_tiles):
        "create generator of predictions from the queue"
        prev_index = -1  # keep track of the previous index, start before the first index
        curr_region_index = 0
        while msg.status != _LOOP_DONE:
            # If we are done with WSI, check if there are any tiles left to write
            if msg.status == _WSI_DONE:
                # add empty tiles until the end of the WSI
                while curr_region_index < num_grid_tiles:
                    curr_region_index += 1
                    yield np.zeros(tile_size, dtype="uint8")
                break
            if msg.region_index is None:
                raise ValueError("Region index should not be None")
            if prev_index > msg.region_index:
                raise ValueError(
                    f"Tiles returned out of order prev index: {prev_index}"
                    f" current index: {msg.region_index}"
                )
            tile_step = msg.region_index - prev_index
            if tile_step > 1:
                for _ in range(tile_step - 1):
                    yield np.zeros(tile_size, dtype="uint8")
            prediction = auto_convert_to_mask(msg.prediction)
            prev_index = msg.region_index
            # The writer expects (tile_height, tile_width, [num_channels])
            if prediction.ndim > 3:
                raise ValueError(f"Prediction has too many dimensions, shape: {prediction.shape}")
            if prediction.shape[:2] != tile_size:
                raise ValueError(
                    f"Tile size expected to be {tuple(tile_size)}, but is {prediction.shape[:2]}"
                )
            yield prediction

            # First message is passed to the function
            msg = self._queue.get()
        # if we are done with the validation loop, add a signal to the processor
        self._queue.put(TileWriterInput(status=_LOOP_DONE))

    def _write_predictions_from_queue(self) -> None:
        """
        Creates tile generators from elements in `queue`, and consumes them through the underlying
        tiffwriters.
        """
        tiff_writer = None
        while True:
            # retrieve first element for status inspection
            msg = self._queue.get()
            if msg.status == _WSI_DONE:
                tiff_writer = None
                # skip the WSI_DONE signal -- this means there were no empty tiles to write
                continue

            if msg.status == _LOOP_DONE:
                break
            # We prefer the extension .tif as this is what ASAP expects for overlays.
            save_name = (
                self._save_dir
                / self._phase
                / f"step_{msg.global_step}"
                / Path(msg.filename + ".tif").name
            )
            os.makedirs(save_name.parent, exist_ok=True)
            tile_size = msg.prediction.shape[1:]
            tiff_writer = TifffileImageWriter(
                filename=str(save_name),
                size=tuple(msg.size),
                mpp=msg.mpp,
                tile_size=tile_size,  # We just take the shape of a prediction
                pyramid=True,
                compression=self._compression,
                quality=100,
            )
            # create generator based on rest of the elements
            tiles_generator = self._tile_generator(
                msg, tile_size, num_grid_tiles=msg.num_grid_tiles
            )
            tiff_writer.from_tiles_iterator(tiles_generator)

    def start(self):
        """
        Start the tiff writer process
        """
        if self._process is not None and self._process.is_alive():
            raise Exception("Process is already running")
        self._process = Process(target=self._write_predictions_from_queue)
        self._process.start()

    def join(self):
        """
        Waits for the queue processing to complete, checks for exceptions
        """
        if self._process is None:
            return
        self._process.join()
        self.check_exceptions()

    def is_alive(self):
        """
        Checks if process is still alive.
        """
        if self._process is None:
            return False
        return self._process.is_alive()

    def check_exceptions(self):
        """
        Checks child process for exceptions, prints exceptions if they occur and re-raises them.
        """
        if self._process is None:
            return
        if not self._process.is_alive():
            if self._process.exception:
                error, traceback = self._process.exception
                sys.stderr.write(traceback + "\n")
                # Re-raise the exception to parent process
                raise error
