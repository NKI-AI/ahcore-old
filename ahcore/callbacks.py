# encoding: utf-8
import concurrent.futures
import hashlib
import multiprocessing
import queue
import threading
from pathlib import Path
from threading import Semaphore
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
from dlup import SlideImage
from dlup.annotations import WsiAnnotations
from dlup.tiling import Grid, GridOrder, TilingMode
from dlup.writers import TiffCompression, TifffileImageWriter
from pytorch_lightning.callbacks import Callback
from torch.utils.data import Dataset

from ahcore.readers import H5FileImageReader, StitchingMode
from ahcore.utils.data import DataDescription
from ahcore.utils.io import get_cache_dir, get_logger
from ahcore.utils.manifest import AnnotationModel, AnnotationReaders, ImageManifest, _ImageBackends, _parse_annotations
from ahcore.writers import H5FileImageWriter

logger = get_logger(__name__)


class ValidationDataset(Dataset):
    def __init__(self, data_description: DataDescription, manifest: ImageManifest, reader: H5FileImageReader):
        self._native_mpp = manifest.mpp

        self._scaling = reader.get_mpp() / self._native_mpp

        mask = _parse_annotations(manifest.mask, base_dir=data_description.annotations_dir)
        annotations = _parse_annotations(manifest.annotations, base_dir=data_description.annotations_dir)

        self._reader = reader

        # TODO: WsiAnnotations should have a .size property
        if isinstance(annotations, WsiAnnotations):
            ann_origin, ann_size = annotations.bounding_box
        else:
            raise NotImplementedError

        grid = Grid.from_tiling(
            ann_origin,
            ann_size,
            tile_size=(1024, 1024),
            tile_overlap=(0, 0),
            mode=TilingMode.overflow,
            order=GridOrder.C,
        )

        # We need to filter the grid, perhaps check how this is done in dlup.
        self._regions = []
        for grid_elem in grid:
            coordinates = grid_elem * self._scaling
            # Now read the region etc.

    def __getitem__(self, idx):
        pass


class _WriterMessage(TypedDict):
    queue: queue.Queue
    writer: H5FileImageWriter
    thread: threading.Thread


def _get_uuid_for_filename(input_path: Path) -> str:
    # Get the absolute path of the file
    input_path = Path(input_path).resolve()

    # Create a SHA256 hash of the file path
    hash_object = hashlib.sha256(str(input_path).encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig


def _get_output_filename(input_path: Path, step: None | int | str = None) -> Path:
    hex_dig = _get_uuid_for_filename(input_path=input_path)

    # Return the hashed filename with the new extension
    if step:
        return get_cache_dir() / "h5s" / f"step_{step}" / f"{hex_dig}.h5"
    return get_cache_dir() / "h5s" / f"{hex_dig}.h5"


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

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx, dataloader_idx=0
    ):
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
            output_filename = _get_output_filename(filename, step=pl_module.global_step)
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

        # prediction = batch["prediction"].detach().cpu().numpy()
        # TODO: We store temporarily the target rather than the prediction for easy comparison
        prediction = batch["annotation_data"]["mask"].detach().cpu().numpy()

        coordinates_x, coordinates_y = batch["coordinates"]
        coordinates = torch.stack([coordinates_x, coordinates_y]).T.detach().cpu().numpy()
        self._writers[filename]["queue"].put((coordinates, prediction))
        self._validation_index += prediction.shape[0]

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self._current_filename is not None:
            self._writers[self._current_filename]["queue"].put(None)
            self._writers[self._current_filename]["thread"].join()
            self._semaphore.release()
            self._validation_index = 0

    @staticmethod
    def generator(queue: queue.Queue):
        while True:
            batch = queue.get()
            if batch is None:
                break
            yield batch


class ComputeWsiMetricsCallback(Callback):
    def __init__(self, max_threads=10):
        self._reader = H5FileImageReader
        self._metrics = []
        self._filenames: dict[Path, Path] = {}
        self._logger = get_logger(type(self).__name__)
        self._semaphore = Semaphore(max_threads)  # Limit the number of threads

        self.__write_h5_callback_index = -1
        self._wsi_metrics = None

        self._validation_manifests: dict[str, ImageManifest] = {}

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        has_write_h5_callback = None
        for idx, callback in enumerate(trainer.callbacks):
            if isinstance(callback, WriteH5Callback):
                has_write_h5_callback = True
                self.__write_h5_callback_index = idx
                self._logger.info("Found WriteH5Callback at index %s: %s", idx, trainer.callbacks[idx])
                break

        if not has_write_h5_callback:
            raise ValueError(
                "WriteH5Callback is not in the trainer's callbacks. This is required before WSI metrics can be computed using this Callback"
            )

        self._wsi_metrics = pl_module.wsi_metrics

        # We should also attach the validation manifest to the class, but convert it to a dictionary mapping
        # the UUID
        data_dir = trainer.datamodule.data_description.data_dir
        for manifest in trainer.datamodule.val_manifest:
            image_fn = data_dir / manifest.image[0]
            self._validation_manifests[_get_uuid_for_filename(image_fn)] = manifest

            if not manifest.mpp:
                # TODO: Put this elsewhere
                # In this case we need to figure it out.
                with SlideImage.from_file_path(
                    image_fn, backend=_ImageBackends[manifest.image[1].name]
                ) as slide_image:
                    manifest.mpp = slide_image.mpp

        self._logger.info("Added %s images to validation manifest.", len(self._validation_manifests))

    @property
    def metrics(self):
        return self._metrics

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx, dataloader_idx=0
    ):
        filename = Path(batch["path"][0])  # Filenames are constant across the batch.
        if filename not in self._filenames:
            self._filenames[_get_output_filename(filename, step=pl_module.global_step)] = filename

    def compute_metrics(self):
        metrics = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_filename = {
                executor.submit(self.compute_metrics_for_case, filename): filename for filename in self._filenames
            }

            for future in concurrent.futures.as_completed(future_to_filename):
                filename = future_to_filename[future]
                try:
                    metric = future.result()
                except Exception as exc:
                    self._logger.error("%r generated an exception: %s" % (filename, exc))
                else:
                    metrics.append(metric)
                    self._logger.info("Metric for %r is %f" % (filename, metric))
        return metrics

    def compute_metrics_for_case(self, filename):
        wsi_filename = self._filenames[filename]
        validation_manifest = self._validation_manifests[_get_uuid_for_filename(wsi_filename)]

        with self._semaphore:  # Only allow a certain number of threads to compute metrics concurrently
            # Compute the metric for one filename here...
            with H5FileImageReader(filename, stitching_mode=StitchingMode.CROP) as h5reader:

                predictions = None
                target = None
                roi = None
                # self._wsi_metrics.process_batch(predictions=predictions, target=target, roi=roi)

                metrics = self._wsi_metrics.get_average_score()

            return self._wsi_metrics

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Ensure that all h5 files have been written
        self._logger.info("Computing metrics for %s predictions", len(self._filenames))

        self._metrics = self.compute_metrics()

        trainer.logger.log_metrics({"custom_metric": 1.0}, step=trainer.global_step)


class WriteTiffCallback(Callback):
    def __init__(self, max_concurrent_writers: int):
        self._pool = multiprocessing.Pool(max_concurrent_writers)
        self._logger = get_logger(type(self).__name__)
        self.__write_h5_callback_index = -1

        self._h5_reader = H5FileImageReader
        self._tiff_writer = TifffileImageWriter

        self._tile_size = (1024, 1024)

        self._tile_process_function = None  # function that is applied to the tile.
        self._filename_mapping = None  # Function that maps h5 name to something else

        self._filenames = []  # This has all the h5 files

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        has_write_h5_callback = None
        for idx, callback in enumerate(trainer.callbacks):
            if isinstance(callback, WriteH5Callback):
                has_write_h5_callback = True
                self.__write_h5_callback_index = idx
                self._logger.info("Found WriteH5Callback at index %s: %s", idx, trainer.callbacks[idx])
                break
        if not has_write_h5_callback:
            raise ValueError("WriteH5Callback required before tiff images can be written using this Callback.")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        results = []
        for filename in self._filenames:
            result = self._pool.apply_async(self._write_tiff, (filename,))
            results.append(result)

        for result in results:
            result.get()  # Wait for the process to complete.

    def _write_tiff(self, filename):
        with H5FileImageReader(filename, stitching_mode=StitchingMode.CROP) as h5_reader:
            writer = TifffileImageWriter(self._filename_mapping(filename), tile_size=self._tile_size)
            writer.from_tiles_iterator(self._iterator_from_reader(h5_reader))

    def _iterator_from_reader(self, h5_reader: H5FileImageReader):
        grid = Grid.from_tiling(
            (0, 0),
            h5_reader.size,
            tile_size=self._tile_size,
            tile_overlap=(0, 0),
            mode=TilingMode.overflow,
            order=GridOrder.C,
        )

        for location in grid:
            region = h5_reader.read_region(location, self._tile_size)
            yield region if self._tile_process_function is None else self._tile_process_function(region)
