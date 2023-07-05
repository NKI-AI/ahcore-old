# encoding: utf-8
from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import multiprocessing
import queue
import threading
from pathlib import Path
from threading import Semaphore
from typing import Iterator, Optional, TypedDict

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from dlup import SlideImage
from dlup._image import Resampling
from dlup.annotations import WsiAnnotations
from dlup.data.transforms import RenameLabels, convert_annotations
from dlup.tiling import Grid, GridOrder, TilingMode
from dlup.writers import TiffCompression, TifffileImageWriter
from pytorch_lightning.callbacks import Callback
from torch.utils.data import Dataset

from ahcore.readers import H5FileImageReader, StitchingMode
from ahcore.transforms.pre_transforms import one_hot_encoding
from ahcore.utils.io import get_logger
from ahcore.utils.manifest import DataDescription, ImageManifest, _ImageBackends, _parse_annotations
from ahcore.writers import H5FileImageWriter

logger = get_logger(__name__)

logging.getLogger("pyvips").setLevel(logging.ERROR)


class _ValidationDataset(Dataset):
    """Helper dataset to compute the validation metrics."""

    def __init__(
        self,
        data_description: Optional[DataDescription],
        native_mpp: float,
        reader: H5FileImageReader,
        annotations: Optional[WsiAnnotations] = None,
        mask: Optional[WsiAnnotations] = None,
        region_size: tuple[int, int] = (1024, 1024),
    ):
        """
        Parameters
        ----------
        data_description : DataDescription
        native_mpp : float
            The actual mpp of the underlying image.
        reader : H5FileImageReader
        annotations : WsiAnnotations
        mask : WsiAnnotations
        region_size : Tuple[int, int]
            The region size to use to split up the image into regions.
        """
        super().__init__()
        self._data_description = data_description
        self._native_mpp = native_mpp
        self._scaling = self._native_mpp / reader.mpp
        self._reader = reader
        self._region_size = region_size
        self._logger = get_logger(type(self).__name__)

        self._annotations = self._validate_annotations(annotations)
        self._mask = self._validate_annotations(mask)

        self._grid = Grid.from_tiling(
            (0, 0),
            reader.size,
            tile_size=self._region_size,
            tile_overlap=(0, 0),
            mode=TilingMode.overflow,
            order=GridOrder.C,
        )

        self._regions = self._generate_regions()
        self._logger.debug(f"Number of validation regions: {len(self._regions)}")

    def _validate_annotations(self, annotations: Optional[WsiAnnotations]) -> Optional[WsiAnnotations]:
        if annotations is not None and self._data_description is None:
            raise ValueError(
                "Annotations are provided but no data description is given. This is required to map the"
                "labels to indices."
            )

        if annotations is not None and not isinstance(annotations, WsiAnnotations):
            raise NotImplementedError
        return annotations

    def _generate_regions(self) -> list[tuple[int, int]]:
        regions = []
        for coordinates in self._grid:
            if self._mask is None or self._is_masked(coordinates):
                regions.append(coordinates)
        return regions

    def _is_masked(self, coordinates: tuple[int, int]) -> bool:
        mask_area = self._mask.read_region(coordinates, self._scaling, self._region_size)
        return sum(_.area for _ in mask_area) > 0

    def __getitem__(self, idx: int) -> dict[str, npt.NDArray[np.uint8 | int | float]]:
        sample = {}
        coordinates = self._regions[idx]

        sample["prediction"] = self._get_h5_region(coordinates)

        if self._annotations is not None:
            target, roi = self._get_annotation_data(coordinates)
            sample["roi"] = roi
            sample["target"] = target

        return sample

    def _get_h5_region(self, coordinates: tuple[int, int]) -> npt.NDArray[np.uint8 | int | float]:
        x, y = coordinates
        width, height = self._region_size

        if x + width > self._reader.size[0] or y + height > self._reader.size[1]:
            region = self._read_and_pad_region(coordinates)
        else:
            region = self._reader.read_region_raw(coordinates, self._region_size)
        return region

    def _read_and_pad_region(self, coordinates: tuple[int, int]) -> npt.NDArray[np.uint8 | int | float]:
        x, y = coordinates
        width, height = self._region_size
        new_width = min(width, self._reader.size[0] - x)
        new_height = min(height, self._reader.size[1] - y)
        clipped_region = self._reader.read_region_raw((x, y), (new_width, new_height))

        prediction = np.zeros((clipped_region.shape[0], *self._region_size), dtype=clipped_region.dtype)
        prediction[:, :new_height, :new_width] = clipped_region
        return prediction

    def _get_annotation_data(
        self, coordinates: tuple[int, int]
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        annotations = self._annotations.read_region(coordinates, self._scaling, self._region_size)
        annotations = RenameLabels(remap_labels=self._data_description.remap_labels)({"annotations": annotations})[
            "annotations"
        ]

        points, region, roi = convert_annotations(
            annotations,
            self._region_size,
            index_map=self._data_description.index_map,
            roi_name="roi",
        )
        region = one_hot_encoding(index_map=self._data_description.index_map, mask=region)
        roi = roi[np.newaxis, ...]
        return region, roi

    def __iter__(self) -> Iterator[dict[str, npt.NDArray[np.uint8 | int | float]]]:
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self) -> int:
        return len(self._regions)


class _WriterMessage(TypedDict):
    queue: queue.Queue
    writer: H5FileImageWriter
    thread: threading.Thread


def _get_uuid_for_filename(input_path: Path) -> str:
    """Get a unique filename for the given input path. This is done by hashing the absolute path of the file.
    This is required because we cannot assume any input format. We hash the complete input path.

    Parameters
    ----------
    input_path : Path
        The input path to hash.

    Returns
    -------
    str
        The hashed filename.
    """
    # Get the absolute path of the file
    input_path = Path(input_path).resolve()

    # Create a SHA256 hash of the file path
    hash_object = hashlib.sha256(str(input_path).encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig


def _get_output_filename(dump_dir: Path, input_path: Path, step: None | int | str = None) -> Path:

    hex_dig = _get_uuid_for_filename(input_path=input_path)

    # Return the hashed filename with the new extension
    if step is not None:
        return dump_dir / "outputs" / f"step_{step}" / f"{hex_dig}.h5"
    return dump_dir / "outputs" / f"{hex_dig}.h5"


class WriteH5Callback(Callback):
    def __init__(self, max_queue_size: int, max_concurrent_writers: int, dump_dir: Path):
        """
        Callback to write predictions to H5 files. This callback is used to write whole-slide predictions to single H5
        files in a separate thread.

        TODO:
            - Add support for distributed data parallel

        Parameters
        ----------
        max_queue_size : int
            The maximum number of items to store in the queue (i.e. tiles).
        max_concurrent_writers : int
            The maximum number of concurrent writers.
        dump_dir : pathlib.Path
            The directory to dump the H5 files to.
        """
        super().__init__()
        self._writers: dict[str, _WriterMessage] = {}
        self._current_filename = None
        self._dump_dir = Path(dump_dir)
        self._max_queue_size = max_queue_size
        self._semaphore = Semaphore(max_concurrent_writers)
        self._validation_index = 0

        self._logger = get_logger(type(self).__name__)

    @property
    def writers(self):
        return self._writers

    @property
    def dump_dir(self) -> Path:
        return self._dump_dir

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
            output_filename = _get_output_filename(self._dump_dir, filename, step=pl_module.global_step)
            output_filename.parent.mkdir(parents=True, exist_ok=True)
            with open(self._dump_dir / "outputs" / f"step_{pl_module.global_step}" / "image_h5_link.txt", "a") as file:
                file.write(f"{filename},{output_filename}\n")

            self._logger.debug("%s -> %s", filename, output_filename)
            if self._current_filename is not None:
                self._writers[self._current_filename]["queue"].put_nowait(None)  # Add None to writer's queue
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
        self._writers[filename]["queue"].put_nowait((coordinates, prediction))
        self._validation_index += prediction.shape[0]

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self._current_filename is not None:
            self._writers[self._current_filename]["queue"].put_nowait(None)
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
        """
        Callback to compute metrics on whole-slide images. This callback is used to compute metrics on whole-slide
        images in separate threads.

        Parameters
        ----------
        max_threads : int
            The maximum number of concurrent threads.
        """
        self._data_description = None
        self._reader = H5FileImageReader
        self._metrics = []
        self._dump_dir = None
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

        self._dump_dir = trainer.callbacks[self.__write_h5_callback_index].dump_dir

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
            output_filename = _get_output_filename(
                dump_dir=self._dump_dir, input_path=filename, step=pl_module.global_step
            )
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
                for sample in dataset_of_validation_image:
                    prediction = torch.from_numpy(sample["prediction"]).unsqueeze(0).float()
                    target = torch.from_numpy(sample["target"]).unsqueeze(0)
                    roi = torch.from_numpy(sample["roi"]).unsqueeze(0)

                    self._wsi_metrics.process_batch(
                        predictions=prediction, target=target, roi=roi, wsi_name=str(filename)
                    )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Ensure that all h5 files have been written
        self._logger.debug("Computing metrics for %s predictions", len(self._filenames))
        self.compute_metrics()
        metrics = self._wsi_metrics.get_average_score()
        self._wsi_metrics.reset()

        self._logger.debug("Metrics: %s", metrics)
        pl_module.log_dict(metrics, prog_bar=True)
        # TODO(jt): I think this is not strictly required.
        # trainer.logger.log_metrics(metrics, step=trainer.global_step)


# Separate because this cannot be pickled.
def _iterator_from_reader(h5_reader: H5FileImageReader, tile_size, tile_process_function):

    validation_dataset = _ValidationDataset(
        data_description=None,
        native_mpp=h5_reader.mpp,
        reader=h5_reader,
        annotations=None,
        mask=None,
        region_size=(1024, 1024),
    )

    for sample in validation_dataset:
        region = sample["prediction"]
        yield region if tile_process_function is None else tile_process_function(region)


def _write_tiff(filename, tile_size, tile_process_function, _iterator_from_reader):
    logger.info("Writing TIFF %s", filename.with_suffix(".tiff"))
    with H5FileImageReader(filename, stitching_mode=StitchingMode.CROP) as h5_reader:
        writer = TifffileImageWriter(
            filename.with_suffix(".tiff"),
            size=h5_reader.size,
            mpp=h5_reader.mpp,
            tile_size=tile_size,
            pyramid=True,
            interpolator=Resampling.NEAREST,
        )
        writer.from_tiles_iterator(_iterator_from_reader(h5_reader, tile_size, tile_process_function))


def tile_process_function(x):
    return np.argmax(x, axis=0).astype(np.uint8)


class WriteTiffCallback(Callback):
    def __init__(self, max_concurrent_writers: int):
        self._pool = multiprocessing.Pool(max_concurrent_writers)
        self._logger = get_logger(type(self).__name__)
        self._dump_dir = None
        self.__write_h5_callback_index = -1

        self._tile_size = (1024, 1024)

        # TODO: Handle tile operation such that we avoid repetitions.

        self._tile_process_function = tile_process_function  # function that is applied to the tile.

        self._filenames = {}  # This has all the h5 files

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        has_write_h5_callback = None
        for idx, callback in enumerate(trainer.callbacks):
            if isinstance(callback, WriteH5Callback):
                has_write_h5_callback = True
                self.__write_h5_callback_index = idx
                break
        if not has_write_h5_callback:
            raise ValueError("WriteH5Callback required before tiff images can be written using this Callback.")

        self._dump_dir = trainer.callbacks[self.__write_h5_callback_index].dump_dir

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx, dataloader_idx=0
    ):
        filename = Path(batch["path"][0])  # Filenames are constant across the batch.
        if filename not in self._filenames:
            output_filename = _get_output_filename(
                dump_dir=self._dump_dir, input_path=filename, step=pl_module.global_step
            )
            self._filenames[filename] = output_filename

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        results = []
        for image_filename, h5_filename in self._filenames.items():
            print(f"Started writing tiff for {image_filename}")
            self._logger.debug("Writing image output %s to %s", image_filename, image_filename.with_suffix(".tiff"))
            with open(
                self._dump_dir / "outputs" / f"step_{pl_module.global_step}" / "image_tiff_link.txt", "a"
            ) as file:
                file.write(f"{image_filename},{h5_filename.with_suffix('.tiff')}\n")
            if not h5_filename.exists():
                self._logger.warning("H5 file %s does not exist. Skipping", h5_filename)
                continue

            result = self._pool.apply_async(
                _write_tiff, (h5_filename, self._tile_size, self._tile_process_function, _iterator_from_reader)
            )
            results.append(result)

        for result in results:
            result.get()  # Wait for the process to complete.
