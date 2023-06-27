# encoding: utf-8
from __future__ import annotations
import concurrent.futures
import hashlib
import multiprocessing
import queue
import threading
from pathlib import Path
from threading import Semaphore
from typing import TypedDict, Optional

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
from dlup import SlideImage
from ahcore.utils.manifest import DataDescription
from dlup.annotations import WsiAnnotations
from dlup.data.transforms import RenameLabels, convert_annotations
from dlup.tiling import Grid, GridOrder, TilingMode
from dlup.writers import TiffCompression, TifffileImageWriter
from pytorch_lightning.callbacks import Callback
from torch.utils.data import Dataset

from ahcore.readers import H5FileImageReader, StitchingMode
from ahcore.transforms.pre_transforms import one_hot_encoding
from ahcore.utils.io import get_cache_dir, get_logger
from ahcore.utils.manifest import ImageManifest, _ImageBackends, _parse_annotations
from ahcore.writers import H5FileImageWriter

logger = get_logger(__name__)

_TORCH: str = "torch"
_NUMPY: str = "numpy"
_TILE_OPERATION = {
    "argmax": {_TORCH: torch.argmax, _NUMPY: np.argmax}
}


class _ValidationDataset(Dataset):
    """Helper dataset to compute the validation metrics in `ahcore.callbacks.ComputeWsiMetricsCallback`."""
    def __init__(
        self,
        data_description: DataDescription,
        native_mpp: float,
        reader: H5FileImageReader,
        annotations: WsiAnnotations,
        mask: WsiAnnotations | None = None,
        region_size: tuple[int, int] = (1024, 1024),
    ):
        """
        Parameters
        ----------
        data_description : DataDescription
        native_mpp : float
            The actual mpp of the underlying image. Is likely different from the `reader` mpp.
        reader : H5FileImageReader
        annotations : WsiAnnotations
        mask : WsiAnnotations (optional)
        region_size : tuple[int, int]
            The region size to use to split up the image into regions.
        """
        super().__init__()
        self._data_description = data_description
        self._native_mpp = native_mpp
        self._scaling = self._native_mpp / reader.mpp
        self._reader = reader
        self._region_size = region_size

        self._logger = get_logger(type(self).__name__)

        if not isinstance(annotations, WsiAnnotations):
            raise NotImplementedError
        if mask is not None and not isinstance(mask, WsiAnnotations):
            raise NotImplemented

        self._annotations = annotations
        self._grid = Grid.from_tiling(
            (0, 0),
            reader.size,
            tile_size=self._region_size,
            tile_overlap=(0, 0),
            mode=TilingMode.overflow,
            order=GridOrder.C,
        )

        self._regions = []
        for coordinates in self._grid:
            if mask is None:
                self._regions.append(coordinates)
                continue

            mask_area = mask.read_region(coordinates, self._scaling, self._region_size)
            if sum([_.area for _ in mask_area]) > 0:
                self._regions.append(coordinates)
        self._logger.debug("Number of validation regions: %s", len(self._regions))

    def __getitem__(self, idx):
        coordinates = self._regions[idx]

        x, y = coordinates
        width, height = self._region_size

        # Check if the region exceeds the reader's dimensions
        if x + width > self._reader.size[0] or y + height > self._reader.size[1]:
            # Calculate new region size
            new_width = min(width, self._reader.size[0] - x)
            new_height = min(height, self._reader.size[1] - y)
            clipped_region = self._reader.read_region_raw((x, y), (new_height, new_width))
            prediction = np.zeros((clipped_region.shape[0], *self._region_size), dtype=clipped_region.dtype)
            prediction[:new_height, :new_width] = clipped_region
        else:
            prediction = self._reader.read_region_raw(coordinates, self._region_size)
        ground_truth = self._annotations.read_region(coordinates, self._scaling, self._region_size)

        ground_truth = RenameLabels(remap_labels=self._data_description.remap_labels)({"annotations": ground_truth})[
            "annotations"
        ]
        points, region, roi = convert_annotations(
            ground_truth,
            self._region_size,
            index_map=self._data_description.index_map,
            roi_name="roi",
        )

        region = one_hot_encoding(index_map=self._data_description.index_map, mask=region)

        return region, prediction, roi[np.newaxis, ...]

    def __len__(self):
        return len(self._regions)


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


def _get_output_filename(input_path: Path, epoch: None | int | str = None) -> Path:
    hex_dig = _get_uuid_for_filename(input_path=input_path)

    # Return the hashed filename with the new extension
    if epoch is not None:
        return get_cache_dir() / "h5s" / f"epoch_{epoch}" / f"{hex_dig}.h5"
    return get_cache_dir() / "h5s" / f"{hex_dig}.h5"


class WriteH5Callback(Callback):
    def __init__(self, max_queue_size: int, max_concurrent_writers: int):
        super().__init__()
        self._writers: dict[str, _WriterMessage] = {}
        self._current_filename = None
        self._max_queue_size = max_queue_size
        self._semaphore = Semaphore(max_concurrent_writers)
        self._validation_index = 0

        self._logger = get_logger(type(self).__name__)

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
            output_filename = _get_output_filename(filename, epoch=pl_module.current_epoch)
            output_filename.parent.mkdir(parents=True, exist_ok=True)
            self._logger.debug("%s -> %s", filename, output_filename)
            if self._current_filename is not None:
                self._writers[self._current_filename]["queue"].put(None)  # Add None to writer's queue
                self._writers[self._current_filename]["thread"].join()
                self._semaphore.release()

            self._semaphore.acquire()

            current_dataset, _ = pl_module.validation_dataset.index_to_dataset(self._validation_index)
            slide_image = current_dataset.slide_image

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
    def __init__(self, max_threads=10, tile_operation: Optional[str] = "argmax"):
        self._data_description = None
        self._reader = H5FileImageReader
        self._metrics = []
        self._tile_operation = tile_operation
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
                break

        if not has_write_h5_callback:
            raise ValueError(
                "WriteH5Callback is not in the trainer's callbacks. This is required before WSI metrics can be computed using this Callback"
            )

        self._wsi_metrics = pl_module.wsi_metrics
        self._data_description = trainer.datamodule.data_description

        # We should also attach the validation manifest to the class, but convert it to a dictionary mapping
        # the UUID
        data_dir = self._data_description.data_dir
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
            output_filename = _get_output_filename(filename, epoch=pl_module.current_epoch)
            self._logger.debug("%s -> %s", filename, output_filename)
            self._filenames[output_filename] = filename

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
                    self._logger.debug("Metric for %r is %s" % (filename, metric))
        return metrics

    def compute_metrics_for_case(self, filename):
        wsi_filename = self._filenames[filename]
        validation_manifest = self._validation_manifests[_get_uuid_for_filename(wsi_filename)]
        native_mpp = validation_manifest.mpp
        with self._semaphore:  # Only allow a certain number of threads to compute metrics concurrently
            # Compute the metric for one filename here...
            with H5FileImageReader(filename, stitching_mode=StitchingMode.CROP) as h5reader:
                mask = _parse_annotations(validation_manifest.mask, base_dir=self._data_description.annotations_dir)
                annotations = _parse_annotations(
                    validation_manifest.annotations, base_dir=self._data_description.annotations_dir
                )
                dataset_of_validation_image = _ValidationDataset(
                    data_description=self._data_description,
                    native_mpp=native_mpp,
                    mask=mask,
                    annotations=annotations,
                    reader=h5reader,
                )
                for idx in range(len(dataset_of_validation_image)):
                    prediction, ground_truth, roi = dataset_of_validation_image[idx]
                    _prediction = _TILE_OPERATION[self._tile_operation][_NUMPY](prediction, 1)
                    _prediction = one_hot_encoding(index_map=self._data_description.index_map, mask=_prediction)
                    _prediction = torch.from_numpy(prediction).unsqueeze(0)
                    _ground_truth = torch.from_numpy(ground_truth).unsqueeze(0)
                    _roi = torch.from_numpy(roi).unsqueeze(0)

                    self._wsi_metrics.process_batch(
                        predictions=_prediction, target=_ground_truth, roi=_roi, wsi_name=str(filename)
                    )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Ensure that all h5 files have been written
        self._logger.debug("Computing metrics for %s predictions", len(self._filenames))

        self.compute_metrics()
        metrics = self._wsi_metrics.get_average_score()
        self._wsi_metrics.reset()

        self._logger.info("Metrics: %s", metrics)
        pl_module.log_dict(metrics, prog_bar=True)
        trainer.logger.log_metrics(metrics, step=trainer.global_step)


class WriteTiffCallback(Callback):
    def __init__(self, max_concurrent_writers: int):
        self._pool = multiprocessing.Pool(max_concurrent_writers)
        self._logger = get_logger(type(self).__name__)
        self.__write_h5_callback_index = -1

        self._h5_reader = H5FileImageReader
        self._tiff_writer = TifffileImageWriter

        self._tile_size = (1024, 1024)

        # TODO: Handle tile operation such that we avoid repetitions.
        self._tile_process_function = None  # function that is applied to the tile.
        # TODO: Map H5 to a different filename
        self._filename_mapping = None  # Function that maps h5 name to something else

        self._filenames = []  # This has all the h5 files

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        has_write_h5_callback = None
        for idx, callback in enumerate(trainer.callbacks):
            if isinstance(callback, WriteH5Callback):
                has_write_h5_callback = True
                self.__write_h5_callback_index = idx
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
