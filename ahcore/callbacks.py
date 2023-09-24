from __future__ import annotations

import hashlib
import itertools
import json
import logging
import multiprocessing
import time
from collections import namedtuple
from multiprocessing import Pipe, Process, Queue, Semaphore
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Generator, Iterator, Optional, TypedDict, cast

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
from dlup import SlideImage
from dlup._image import Resampling
from dlup.annotations import WsiAnnotations
from dlup.data.dataset import ConcatDataset, TiledROIsSlideImageDataset
from dlup.data.transforms import RenameLabels, convert_annotations
from dlup.tiling import Grid, GridOrder, TilingMode
from dlup.writers import TiffCompression, TifffileImageWriter
from pytorch_lightning.callbacks import Callback
from shapely.geometry import MultiPoint, Point
from torch.utils.data import Dataset

from ahcore.lit_module import AhCoreLightningModule
from ahcore.readers import H5FileImageReader, StitchingMode
from ahcore.transforms.pre_transforms import one_hot_encoding
from ahcore.utils.data import DataDescription, GridDescription
from ahcore.utils.io import get_logger
from ahcore.utils.manifest import get_mask_and_annotations_from_record
from ahcore.utils.manifest_database import DataManager, ImageMetadata, fetch_image_metadata
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
        if annotations is None:
            return None

        if isinstance(annotations, WsiAnnotations):
            if self._data_description is None:
                raise ValueError(
                    "Annotations as a `WsiAnnotations` class are provided but no data description is given."
                    "This is required to map the labels to indices."
                )
        elif isinstance(annotations, SlideImage):
            pass  # We do not need a specific test for this
        else:
            raise NotImplementedError(f"Annotations of type {type(annotations)} are not supported.")

        return annotations

    def _generate_regions(self) -> list[tuple[int, int]]:
        """Generate the regions to use. These regions are filtered grid cells where there is a mask.

        Returns
        -------
        List[Tuple[int, int]]
            The list of regions.
        """
        regions = []
        for coordinates in self._grid:
            if self._mask is None or self._is_masked(coordinates):
                regions.append(coordinates)
        return regions

    def _is_masked(self, coordinates: tuple[int, int]) -> bool:
        """Check if the region is masked. This works with any masking function that supports a `read_region` method or
        returns a list of annotations with an `area` attribute. In case there are elements of the form `Point` in the
        annotation list, these are also added.

        Parameters
        ----------
        coordinates : Tuple[int, int]
            The coordinates of the region to check.

        Returns
        -------
        bool
            True if the region is masked, False otherwise. Will also return True when there is no mask.
        """
        if self._mask is None:
            return True

        region_mask = self._mask.read_region(coordinates, self._scaling, self._region_size)

        if isinstance(region_mask, np.ndarray):
            return region_mask.sum() > 0

        # We check if the region is not a Point, otherwise this annotation is always included
        # Else, we compute if there is a positive area in the region.
        return sum(_.area if _ is not isinstance(_, (Point, MultiPoint)) else 1.0 for _ in region_mask) > 0

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = {}
        coordinates = self._regions[idx]

        sample["prediction"] = self._get_h5_region(coordinates)

        if self._annotations is not None:
            target, roi = self._get_annotation_data(coordinates)
            if roi is not None:
                sample["roi"] = roi
            sample["target"] = target

        return sample

    def _get_h5_region(self, coordinates: tuple[int, int]) -> npt.NDArray:
        x, y = coordinates
        width, height = self._region_size

        if x + width > self._reader.size[0] or y + height > self._reader.size[1]:
            region = self._read_and_pad_region(coordinates)
        else:
            region = self._reader.read_region_raw(coordinates, self._region_size)
        return region

    def _read_and_pad_region(self, coordinates: tuple[int, int]) -> npt.NDArray:
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
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None]:
        if not self._annotations:
            raise ValueError("No annotations are provided.")

        if not self._data_description:
            raise ValueError("No data description is provided.")

        if not self._data_description.remap_labels:
            raise ValueError("Remap labels are not provided.")

        if not self._data_description.index_map:
            raise ValueError("Index map is not provided.")

        annotations = self._annotations.read_region(coordinates, self._scaling, self._region_size)
        annotations = RenameLabels(remap_labels=self._data_description.remap_labels)({"annotations": annotations})[
            "annotations"
        ]

        points, boxes, region, roi = convert_annotations(
            annotations,
            self._region_size,
            index_map=self._data_description.index_map,
            roi_name="roi",
        )
        region = one_hot_encoding(index_map=self._data_description.index_map, mask=region)
        if roi is not None:
            return region, roi[np.newaxis, ...]
        return region, None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self) -> int:
        return len(self._regions)


class _WriterMessage(TypedDict):
    queue: Queue
    writer: H5FileImageWriter
    process: Process
    connection: Connection


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


def _get_h5_output_filename(dump_dir: Path, input_path: Path, model_name: str, step: None | int | str = None) -> Path:
    hex_dig = _get_uuid_for_filename(input_path=input_path)

    # Return the hashed filename with the new extension
    if step is not None:
        return dump_dir / "outputs" / model_name / f"step_{step}" / f"{hex_dig}.h5"
    return dump_dir / "outputs" / model_name / f"{hex_dig}.h5"


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
    def dump_dir(self) -> Path:
        return self._dump_dir

    def __check_process(self) -> bool | None:
        """
        This module communicates with all the child processes spawned by the main process while writing h5 files.
        It monitors if the target function of the child process has been correctly executed.

        Returns
        -------
        bool | None
        """
        # This is for mypy
        assert self._current_filename, "_current_filename shouldn't be None here"

        connection = self._writers[self._current_filename]["connection"]
        if connection.poll():  # Check if there's a message from the child process
            status, filename, message = connection.recv()  # Receive the message
            if status is True:
                self._logger.debug(f"Successfully processed {self._current_filename}")
            elif status is False:
                raise Exception(f"Failed processing {self._current_filename} due to {message}")
            connection.close()
            return status
        else:
            # Sometimes, we may encounter EOF if the communication from the child process has already been processed.
            return None

    def __process_management(self) -> None:
        """
        Handle the graceful termination of multiple processes at the end of h5 writing.
        This block ensures proper release of resources allocated during multiprocessing.

        Returns
        -------
        None
        """
        assert self._current_filename, "_current_filename shouldn't be None here"

        self._writers[self._current_filename]["queue"].put(None)
        self._writers[self._current_filename]["process"].join()
        self.__check_process()
        # TODO: Are these really needed?
        self._writers[self._current_filename]["process"].terminate()
        self._writers[self._current_filename]["process"].close()
        self._writers[self._current_filename]["queue"].close()

    @property
    def writers(self):
        return self._writers

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        filename = batch["path"][0]  # Filenames are constant across the batch.
        if any([filename != path for path in batch["path"]]):
            raise ValueError(
                "All paths in a batch must be the same. "
                "Either use batch_size=1 or ahcore.data.samplers.WsiBatchSampler."
            )

        if filename != self._current_filename:
            output_filename = _get_h5_output_filename(
                self.dump_dir,
                filename,
                model_name=str(pl_module.name),
                step=pl_module.global_step,
            )
            output_filename.parent.mkdir(parents=True, exist_ok=True)
            link_fn = output_filename.parent / "image_h5_link.txt"
            with open(link_fn, "a" if link_fn.is_file() else "w") as file:
                file.write(f"{filename},{output_filename}\n")

            self._logger.debug("%s -> %s", filename, output_filename)
            if self._current_filename is not None:
                self.__process_management()
                self._semaphore.release()

            self._semaphore.acquire()
            validate_dataset: ConcatDataset = trainer.datamodule.validate_dataset  # type: ignore

            current_dataset: TiledROIsSlideImageDataset
            current_dataset, _ = validate_dataset.index_to_dataset(self._validation_index)  # type: ignore
            slide_image = current_dataset.slide_image

            data_description: DataDescription = pl_module.data_description  # type: ignore
            inference_grid: GridDescription = data_description.inference_grid

            mpp = inference_grid.mpp
            if mpp is None:
                mpp = slide_image.mpp

            size = slide_image.get_scaled_size(slide_image.get_scaling(mpp))
            num_samples = len(current_dataset)

            # Let's get the data_description, so we can figure out the tile size and things like that
            tile_size = inference_grid.tile_size
            tile_overlap = inference_grid.tile_overlap

            new_queue: Queue = Queue()
            parent_conn, child_conn = Pipe()
            new_writer = H5FileImageWriter(
                output_filename,
                size=size,
                mpp=mpp,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                num_samples=num_samples,
                progress=None,
            )
            new_process = Process(target=new_writer.consume, args=(self.generator(new_queue), child_conn))
            new_process.start()
            self._writers[filename] = {
                "queue": new_queue,
                "writer": new_writer,
                "process": new_process,
                "connection": parent_conn,
            }
            self._current_filename = filename

        prediction = outputs["prediction"].detach().cpu().numpy()
        coordinates_x, coordinates_y = batch["coordinates"]
        coordinates = torch.stack([coordinates_x, coordinates_y]).T.detach().cpu().numpy()
        self._writers[filename]["queue"].put((coordinates, prediction))
        self._validation_index += prediction.shape[0]

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self._current_filename is not None:
            self.__process_management()
            self._semaphore.release()
            self._validation_index = 0
        # Reset current filename to None for correct execution of subsequent validation loop
        self._current_filename = None
        # Clear all the writers from the current epoch
        self._writers = {}

    @staticmethod
    def generator(queue: Queue):
        while True:
            batch = queue.get()
            if batch is None:
                break
            yield batch


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
    logger.debug("Writing TIFF %s", filename.with_suffix(".tiff"))
    with H5FileImageReader(filename, stitching_mode=StitchingMode.CROP) as h5_reader:
        writer = TifffileImageWriter(
            filename.with_suffix(".tiff"),
            size=h5_reader.size,
            mpp=h5_reader.mpp,
            tile_size=tile_size,
            pyramid=True,
            compression=TiffCompression.ZSTD,
            interpolator=Resampling.NEAREST,
        )
        writer.from_tiles_iterator(_iterator_from_reader(h5_reader, tile_size, tile_process_function))


def tile_process_function(x):
    return np.argmax(x, axis=0).astype(np.uint8)


class WriteTiffCallback(Callback):
    def __init__(self, max_concurrent_writers: int):
        self._pool = multiprocessing.Pool(max_concurrent_writers)
        self._logger = get_logger(type(self).__name__)
        self._dump_dir: Optional[Path] = None
        self.__write_h5_callback_index = -1

        self._tile_size = (1024, 1024)

        # TODO: Handle tile operation such that we avoid repetitions.

        self._tile_process_function = tile_process_function  # function that is applied to the tile.
        self._filenames: dict[Path, Path] = {}  # This has all the h5 files

    @property
    def dump_dir(self) -> Optional[Path]:
        return self._dump_dir

    def _validate_parameters(self):
        dump_dir = self._dump_dir
        if not dump_dir:
            raise ValueError("Dump directory is not set.")

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        _callback: Optional[WriteH5Callback] = None
        for idx, callback in enumerate(trainer.callbacks):  # type: ignore
            if isinstance(callback, WriteH5Callback):
                _callback = cast(WriteH5Callback, trainer.callbacks[idx])  # type: ignore
                break
        if _callback is None:
            raise ValueError("WriteH5Callback required before tiff images can be written using this Callback.")

        # This is needed for mypy
        assert _callback, "_callback should never be None after the setup."
        assert _callback.dump_dir, "_callback.dump_dir should never be None after the setup."
        self._dump_dir = _callback.dump_dir

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        assert self.dump_dir, "dump_dir should never be None here."

        filename = Path(batch["path"][0])  # Filenames are constant across the batch.
        if filename not in self._filenames:
            output_filename = _get_h5_output_filename(
                dump_dir=self.dump_dir,
                input_path=filename,
                model_name=str(pl_module.name),
                step=pl_module.global_step,
            )
            self._filenames[filename] = output_filename

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        assert self.dump_dir, "dump_dir should never be None here."
        self._logger.info("Writing TIFF files to %s", self.dump_dir / "outputs" / f"{pl_module.name}")
        results = []
        for image_filename, h5_filename in self._filenames.items():
            self._logger.debug(
                "Writing image output %s to %s",
                Path(image_filename),
                Path(image_filename).with_suffix(".tiff"),
            )
            output_path = self.dump_dir / "outputs" / f"{pl_module.name}" / f"step_{pl_module.global_step}"
            with open(output_path / "image_tiff_link.txt", "a") as file:
                file.write(f"{image_filename},{h5_filename.with_suffix('.tiff')}\n")
            if not h5_filename.exists():
                self._logger.warning("H5 file %s does not exist. Skipping", h5_filename)
                continue

            result = self._pool.apply_async(
                _write_tiff,
                (
                    h5_filename,
                    self._tile_size,
                    self._tile_process_function,
                    _iterator_from_reader,
                ),
            )
            results.append(result)

        for result in results:
            result.get()  # Wait for the process to complete.
        self._filenames = {}  # Reset the filenames


# Create a data structure to hold all required information for each task
TaskData = namedtuple("TaskData", ["filename", "h5_filename", "metadata", "mask", "annotations"])


def prepare_task_data(filename, dump_dir, pl_module, data_description, data_manager):
    h5_filename = _get_h5_output_filename(
        dump_dir=dump_dir,
        input_path=data_description.data_dir / filename,
        model_name=str(pl_module.name),
        step=pl_module.global_step,
    )
    image = data_manager.get_image_by_filename(str(filename))
    metadata = fetch_image_metadata(image)
    mask, annotations = get_mask_and_annotations_from_record(data_description.annotations_dir, image)

    return TaskData(filename, h5_filename, metadata, mask, annotations)


def compute_metrics_for_case(
    task_data: TaskData,
    class_names,
    data_description,
    wsi_metrics,
    save_per_image: bool,
):
    # Extract the data from the namedtuple
    filename, h5_filename, metadata, mask, annotations = task_data

    dump_list = []

    logger.info("Computing metrics for %s", filename)

    with H5FileImageReader(h5_filename, stitching_mode=StitchingMode.CROP) as h5reader:
        dataset_of_validation_image = _ValidationDataset(
            data_description=data_description,
            native_mpp=metadata.mpp,
            mask=mask,
            annotations=annotations,
            reader=h5reader,
        )
        for sample in dataset_of_validation_image:
            prediction = torch.from_numpy(sample["prediction"]).unsqueeze(0).float()
            target = torch.from_numpy(sample["target"]).unsqueeze(0)
            roi = torch.from_numpy(sample["roi"]).unsqueeze(0)

            wsi_metrics.process_batch(
                predictions=prediction,
                target=target,
                roi=roi,
                wsi_name=str(filename),
            )
    if save_per_image:
        wsi_metrics_dictionary = {
            "image_fn": str(data_description.data_dir / metadata.filename),
            "uuid": filename.stem,
        }
        if filename.with_suffix(".tiff").is_file():
            wsi_metrics_dictionary["tiff_fn"] = str(filename.with_suffix(".tiff"))
        if filename.is_file():
            wsi_metrics_dictionary["h5_fn"] = str(filename)
        for metric in wsi_metrics._metrics:
            metric.get_wsi_score(str(filename))
            wsi_metrics_dictionary[metric.name] = {
                class_names[class_idx]: metric.wsis[str(filename)][class_idx][metric.name].item()
                for class_idx in range(data_description.num_classes)
            }
        dump_list.append(wsi_metrics_dictionary)

    return dump_list


# Adjusted stand-alone function.
def schedule_task(
    task_data,
    pool,
    results_dict,
    class_names,
    data_description,
    wsi_metrics,
    save_per_image,
):
    result = pool.apply_async(
        compute_metrics_for_case,
        args=(task_data, class_names, data_description, wsi_metrics, save_per_image),
    )
    results_dict[result] = task_data.filename


class ComputeWsiMetricsCallback(Callback):
    def __init__(self, max_processes=10, save_per_image: bool = True):
        """
        Callback to compute metrics on whole-slide images. This callback is used to compute metrics on whole-slide
        images in separate processes.

        Parameters
        ----------
        max_processes : int
            The maximum number of concurrent processes.
        """
        self._data_description: Optional[DataDescription] = None
        self._reader = H5FileImageReader
        self._max_processes: int = max_processes
        self._dump_dir: Optional[Path] = None
        self._save_per_image = save_per_image
        self._filenames: dict[Path, Path] = {}

        self._wsi_metrics = None
        self._class_names: dict[int, str] = {}
        self._data_manager = None
        self._validate_filenames_gen = None

        self._validate_metadata_gen = None

        self._dump_list: list[dict[str, str]] = []
        self._logger = get_logger(type(self).__name__)

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        pl_module = cast(AhCoreLightningModule, pl_module)

        _callback: Optional[WriteH5Callback] = None
        for idx, callback in enumerate(trainer.callbacks):  # type: ignore
            if isinstance(callback, WriteH5Callback):
                _callback = cast(WriteH5Callback, trainer.callbacks[idx])  # type: ignore
                break

        if _callback is None:
            raise ValueError(
                "WriteH5Callback is not in the trainer's callbacks. "
                "This is required before WSI metrics can be computed using this Callback"
            )

        self._dump_dir = _callback.dump_dir

        self._wsi_metrics = pl_module.wsi_metrics
        self._data_description = trainer.datamodule.data_description  # type: ignore

        # For mypy
        assert self._data_description
        index_map = self._data_description.index_map
        assert index_map

        if not self._data_description:
            raise ValueError("Data description is not set.")

        self._class_names = dict([(v, k) for k, v in index_map.items()])
        self._class_names[0] = "background"

        # Here we can query the database for the validation images
        self._data_manager: DataManager = trainer.datamodule.data_manager  # type: ignore

    def _create_validate_image_metadata_gen(
        self,
    ) -> Generator[ImageMetadata, None, None]:
        assert self._data_description
        assert self._data_manager
        gen = self._data_manager.get_image_metadata_by_split(
            manifest_name=self._data_description.manifest_name,
            split_version=self._data_description.split_version,
            split_category="validate",
        )
        for image_metadata in gen:
            yield image_metadata

    @property
    def _validate_metadata(self) -> Generator[ImageMetadata, None, None]:
        return self._validate_metadata_gen

    @property
    def metrics(self):
        return self._metrics

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._validate_metadata_gen = self._create_validate_image_metadata_gen()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if not self._dump_dir:
            raise ValueError("Dump directory is not set.")

        filenames = batch["path"]  # Filenames are constant across the batch.
        if len(set(filenames)) != 1:
            raise ValueError(
                "All paths in a batch must be the same. "
                "Either use batch_size=1 or ahcore.data.samplers.WsiBatchSampler."
            )

    def compute_metrics(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        assert self._dump_dir
        assert self._data_description
        metrics = []

        with multiprocessing.Pool(processes=self._max_processes) as pool:
            results_to_filename = {}
            completed_tasks = 0

            # Fill up the initial task pool
            for image_metadata in itertools.islice(self._validate_metadata, self._max_processes):
                logger.info("Metadata: %s", image_metadata)
                # Assemble the task data
                # filename", "h5_filename", "metadata", "mask", "annotations
                task_data = prepare_task_data(
                    image_metadata.filename,
                    self._dump_dir,
                    pl_module,
                    self._data_description,
                    self._data_manager,
                )

                # Schedule task
                schedule_task(
                    task_data,
                    pool,
                    results_to_filename,
                    self._class_names,
                    self._data_description,
                    self._wsi_metrics,
                    self._save_per_image,
                )

            while results_to_filename:
                time.sleep(0.1)  # Reduce excessive polling
                # Check for completed tasks
                for result in list(results_to_filename.keys()):
                    if result.ready():
                        filename = results_to_filename.pop(result)
                        try:
                            metric = result.get()
                        except Exception as exc:
                            self._logger.error("%r generated an exception: %s" % (filename, exc))
                        else:
                            metrics.append(metric)
                            self._logger.debug("Metric for %r is %s" % (filename, metric))

                        completed_tasks += 1

                        # Schedule a new task if there are more filenames left in the generator
                        next_metadata = next(self._validate_metadata, None)
                        while next_metadata:
                            task_data = prepare_task_data(
                                next_metadata.filename,  # <-- Changed from image_metadata.filename
                                self._dump_dir,
                                pl_module,
                                self._data_description,
                                self._data_manager,
                            )

                            # Schedule task
                            schedule_task(
                                task_data,
                                pool,
                                results_to_filename,
                                self._class_names,
                                self._data_description,
                                self._wsi_metrics,
                                self._save_per_image,
                            )

                            next_metadata = next(self._validate_metadata, None)
        return metrics

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self._dump_dir:
            raise ValueError("Dump directory is not set.")
        if not self._wsi_metrics:
            raise ValueError("WSI metrics are not set.")

        # Ensure that all h5 files have been written
        self._logger.debug("Computing metrics for %s predictions", len(self._filenames))
        self.compute_metrics(trainer, pl_module)
        metrics = self._wsi_metrics.get_average_score()
        with open(
            self._dump_dir / "outputs" / pl_module.name / f"step_{pl_module.global_step}" / "results.json",
            "w",
            encoding="utf-8",
        ) as json_file:
            json.dump(self._dump_list, json_file, indent=2)
        self._wsi_metrics.reset()
        # Reset stuff
        self._dump_list = []
        self._filenames = {}

        self._logger.debug("Metrics: %s", metrics)

        # TODO: Maybe put this elsewhere?
        metrics = {f"validate/{k}": v for k, v in metrics.items()}
        pl_module.log_dict(metrics, prog_bar=True)
