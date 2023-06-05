# encoding: utf-8
import hashlib
import queue
import threading
from pathlib import Path
from threading import Semaphore
from typing import TypedDict

import torch
from pytorch_lightning.callbacks import Callback

from ahcore.utils.io import get_cache_dir, get_logger
from ahcore.writers import H5FileImageWriter

logger = get_logger(__name__)


class _WriterMessage(TypedDict):
    queue: queue.Queue
    writer: H5FileImageWriter
    thread: threading.Thread


class WriteH5Callback(Callback):
    def __init__(self, max_queue_size: int, max_concurrent_writers: int):
        super().__init__()
        self._writers: dict[str, _WriterMessage] = {}
        self._current_filename = None
        self._max_queue_size = max_queue_size
        self._semaphore = Semaphore(max_concurrent_writers)
        self._validation_index = 0

    @property
    def writers(self):
        return self._writers

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        filename = batch["path"][0]  # Filenames are constant across the batch.
        if any([filename != path for path in batch["path"]]):
            raise ValueError(
                "All paths in a batch must be the same. "
                "Either use batch_size=1 or ahcore.data.samplers.WsiBatchSampler."
            )

        if filename != self._current_filename:
            # TODO: This filename might contain 'global_step', or only give the last one depending on settings
            # TODO: These files can be very large
            # TODO: The outputs also has a metrics dictionary, so you could use that to figure out if its better or not
            output_filename = self._create_output_filename(filename, step=pl_module.global_step)
            output_filename.parent.mkdir(parents=True, exist_ok=True)

            logger.info("Got new filename in WriteH5Callback %s. Will write to %s", filename, output_filename)
            if self._current_filename is not None:
                self._writers[self._current_filename]["queue"].put(None)  # Add None to writer's queue
                self._writers[self._current_filename]["thread"].join()
                self._semaphore.release()

            self._semaphore.acquire()

            current_dataset, _ = pl_module.validation_dataset.index_to_dataset(self._validation_index)
            slide_image = current_dataset.slide_image
            # We need a sanity check for now
            # TODO: Remove when all works
            if slide_image.identifier != filename:
                raise ValueError("Identifier should be the same as filename.")

            mpp = pl_module.data_description.inference_grid.mpp
            size = slide_image.get_scaled_size(slide_image.get_scaling(mpp))
            num_samples = len(current_dataset)

            # Let's get the data_description, so we can figure out the tile size and things like that
            tile_size = pl_module.data_description.inference_grid.tile_size
            tile_overlap = pl_module.data_description.inference_grid.tile_overlap

            new_queue = queue.Queue()
            new_writer = H5FileImageWriter(
                output_filename,
                size=size,
                mpp=mpp,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                num_samples=num_samples,
                progress=None,
            )
            new_thread = threading.Thread(target=new_writer.consume, args=(self.generator(new_queue),))
            new_thread.start()
            self._writers[filename] = {"queue": new_queue, "writer": new_writer, "thread": new_thread}
            self._current_filename = filename

        prediction = batch["prediction"].detach().cpu().numpy()
        coordinates_x, coordinates_y = batch["coordinates"]
        coordinates = torch.stack([coordinates_x, coordinates_y]).T.detach().cpu().numpy()
        self._writers[filename]["queue"].put((coordinates, prediction))
        self._validation_index += prediction.shape[0]

    def on_validation_end(self, trainer, pl_module):
        if self._current_filename is not None:
            self._writers[self._current_filename]["queue"].put(None)
            self._writers[self._current_filename]["thread"].join()
            self._semaphore.release()
            self._validation_index = 0

    def generator(self, queue: queue.Queue):
        while True:
            batch = queue.get()
            if batch is None:
                break
            yield batch

    @staticmethod
    def _create_output_filename(input_path: Path, step: None | int | str = None) -> Path:
        # Get the absolute path of the file
        input_path = Path(input_path).resolve()

        # Create a SHA256 hash of the file path
        hash_object = hashlib.sha256(str(input_path).encode())
        hex_dig = hash_object.hexdigest()

        # Return the hashed filename with the new extension
        if step:
            return get_cache_dir() / "h5s" / f"step_{step}" / f"{hex_dig}.h5"
        return get_cache_dir() / "h5s" / f"{hex_dig}.h5"


class ComputeWsiMetricsCallback(Callback):
    def __init__(self, reader=None):
        self._reader = reader
        self._metrics = None
        self.metrics_thread = None

    @property
    def metrics(self):
        return self._metrics

    def on_validation_start(self, trainer, pl_module):
        # Start the metrics computation in a separate thread
        # self.metrics_thread = threading.Thread(target=self.compute_metrics)
        # self.metrics_thread.start()
        pass

    def on_validation_end(self, trainer, pl_module):
        # Ensure that all h5 files have been written
        # for writer in trainer.callbacks[WriteH5Callback].writers.values():
        #     writer["thread"].join()
        #
        self._metrics = self.compute_metrics()

    # def on_validation_end(self, trainer, pl_module):
    #     # Ensure that all h5 files have been written
    #     for writer in trainer.callbacks[WriteH5Callback]._writers.values():
    #         writer['thread'].join()
    #
    #     # Wait for the metrics computation to finish
    #     self.metrics_thread.join()
    #
    #     pl_module.log('custom_metric', self.metrics)

    def compute_metrics(self):
        pass  # use h5_reader here.

    class ComputeMetricsCallback(Callback):
        def __init__(self, h5_reader):
            self.h5_reader = h5_reader
            self.metrics = None
            self.metrics_thread = None
