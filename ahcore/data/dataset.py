# encoding: utf-8
"""
Utilities to construct datasets and DataModule's from manifests.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterator

import pytorch_lightning as pl
import torch
from torchvision import transforms
from dlup import SlideImage
from dlup.data.dataset import ConcatDataset, Dataset
from pytorch_lightning.trainer.states import TrainerFn
from torch.utils.data import DataLoader, Sampler

import ahcore.data.samplers
from ahcore.transforms.image_normalization import MacenkoNormalizer
from ahcore.utils.data import DataDescription, create_inference_metadata, dataclass_to_uuid
from ahcore.utils.io import get_logger
from ahcore.utils.manifest import (
    image_manifest_to_dataset,
    manifests_from_data_description,
    parse_wsi_attributes_from_manifest,
)

logger = get_logger(__name__)


class DlupDataModule(pl.LightningDataModule):
    """Datamodule for the Ahcore framework. This datamodule is based on `dlup`."""

    def __init__(
            self,
            data_description: DataDescription,
            pre_transform: Callable,
            batch_size: int = 32,  # noqa,pylint: disable=unused-argument
            validate_batch_size: int | None = None,  # noqa,pylint: disable=unused-argument
            num_workers: int = 16,
            persistent_workers: bool = False,
            pin_memory: bool = False,
            compute_staining: bool = False,
    ) -> None:
        """
        Construct a DataModule based on a manifest.


        Parameters
        ----------
        data_description : DataDescription
            See `ahcore.utils.data.DataDescription` for more information.
        pre_transform : Callable
            A pre-transform is a callable which is directly applied to the output of the dataset before collation in
            the dataloader. The transforms typically convert the image in the output to a tensor, convert the
            `WsiAnnotations` to a mask or similar.
        batch_size : int
            The batch size of the data loader.
        validate_batch_size : int, optional
            Sometimes the batch size for validation can be larger. If so, set this variable. Will also use this for
            prediction.
        num_workers : int
            The number of workers used to fetch tiles.
        persistent_workers : bool
            Whether to use persistent workers. Check the pytorch documentation for more information.
        pin_memory : bool
            Whether to use cuda pin workers. Check the pytorch documentation for more information.
        """
        super().__init__()

        self.save_hyperparameters(
            logger=True,
            ignore=[
                "data_description",
                "pre_transform",
                "data_dir",
                "annotations_dir",
                "num_workers",
                "persistent_workers",
                "pin_memory",
            ],
        )  # save all relevant hyperparams

        # Data settings
        self.data_description = data_description
        self._manifests = manifests_from_data_description(data_description)

        self._batch_size = self.hparams.batch_size  # type: ignore
        self._validate_batch_size = self.hparams.validate_batch_size  # type: ignore
        self._compute_staining = compute_staining

        mask_threshold = data_description.mask_threshold
        if mask_threshold is None:
            mask_threshold = 0.0
        self._mask_threshold = mask_threshold

        self._pre_transform = pre_transform

        # DataLoader settings
        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._pin_memory = pin_memory

        self._fit_data_iterator: Iterator[Dataset] | None = None
        self._validate_data_iterator: Iterator[Dataset] | None = None
        self._test_data_iterator: Iterator[Dataset] | None = None
        self._predict_data_iterator: Iterator[Dataset] | None = None

        # Variables to keep track if a dataset has already be constructed (it's a slow operation)
        self._already_called: dict[str, bool] = {"fit": False, "validate": False, "test": False, "predict": False}

        for field in self._manifests._fields:
            logger.info("Number of images for stage %s: %s", field, getattr(self._manifests, field).__len__())

        self._num_classes = data_description.num_classes

    def setup(self, stage: TrainerFn | None = None) -> None:
        if not stage:
            return

        if stage and self._already_called[stage]:
            return
        logger.info("Constructing dataset iterator for stage %s...", stage)
        manifests = getattr(self._manifests, stage)
        if not manifests:
            setattr(self, f"_{stage}_data_iterator", None)
            return

        grid = getattr(self.data_description, "training_grid" if stage == TrainerFn.FITTING else "inference_grid")
        mpp = getattr(grid, "mpp", None)
        tile_size = grid.tile_size
        tile_overlap = grid.tile_overlap
        output_tile_size = getattr(grid, "output_tile_size", None)

        if self._compute_staining is True:
            path_to_dump_stains = (
                    Path(os.environ.get("SCRATCH", "/tmp")) / "ahcore_cache" / "staining_parameters"
            )
            stain_computer = MacenkoNormalizer(return_stains=False)
            wsi_transformation = transforms.Compose([transforms.PILToTensor()])
            for manifest in manifests:
                image_fn, _, overwrite_mpp = parse_wsi_attributes_from_manifest(self.data_description, manifest)
                if (path_to_dump_stains / (image_fn.stem + ".h5")).is_file():
                    logger.info(f"Staining vectors for {image_fn.stem} are already cached...")
                    continue
                else:
                    slide_image = SlideImage.from_file_path(image_fn, overwrite_mpp=overwrite_mpp)
                    rescaled_wsi = slide_image.get_scaled_view(slide_image.get_scaling(16))
                    thumbnail = slide_image.get_thumbnail(size=rescaled_wsi.size).convert("RGB")
                    rescaled_wsi_tensor = wsi_transformation(thumbnail).unsqueeze(0)
                    stain_computer.fit(
                        wsi=rescaled_wsi_tensor, wsi_name=image_fn.stem, dump_to_folder=path_to_dump_stains
                    )

        def dataset_iterator() -> Iterator[Dataset]:
            for image_manifest in manifests:
                dataset = image_manifest_to_dataset(
                    manifest=image_manifest,
                    data_description=self.data_description,
                    mpp=mpp,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    output_tile_size=output_tile_size,
                    transform=self._pre_transform(requires_target=True if stage != TrainerFn.PREDICTING else False),
                    stage=stage,
                )

                logger.info("Added dataset for %s with length %s.", image_manifest.identifier, len(dataset))
                yield dataset

        setattr(self, f"_{stage}_data_iterator", dataset_iterator())

    def _construct_concatenated_dataloader(self, data_iterator, batch_size: int, stage: TrainerFn | None = None):
        if not data_iterator:
            return None

        def construct_dataset() -> ConcatDataset:
            datasets = []
            for _, ds in enumerate(data_iterator):
                datasets.append(ds)
            return ConcatDataset(datasets)

        dataset = self._load_from_cache(construct_dataset, stage=stage)

        batch_sampler: Sampler
        if stage == TrainerFn.FITTING:
            batch_sampler = torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(data_source=dataset, replacement=True),
                batch_size=batch_size,
                drop_last=True,
            )

        else:
            batch_sampler = ahcore.data.samplers.WsiBatchSampler(
                sampler=torch.utils.data.SequentialSampler(data_source=dataset),
                dataset=dataset,
                batch_size=batch_size,
            )

        return DataLoader(
            dataset,  # type: ignore
            num_workers=self._num_workers,
            batch_sampler=batch_sampler,
            persistent_workers=self._persistent_workers,
            pin_memory=self._pin_memory,
        )

    def _load_from_cache(self, func: Callable, stage, *args, **kwargs):
        name = func.__qualname__.split("<locals>.")[-1]
        path = Path(os.environ.get("SCRATCH", "/tmp")) / "ahcore_cache" / str(stage.value) / name
        filename = path / f"{self.uuid}.pkl"
        if not filename.is_file():
            path.mkdir(exist_ok=True, parents=True)
            logger.info("Caching %s", name)

            obj = func(*args, **kwargs)

            with open(filename, "wb") as file:
                torch.save(obj, file)
        else:
            with open(filename, "rb") as file:
                logger.info(f"Loading %s from cache %s file", name, filename)
                obj = torch.load(file)

        return obj

    def _construct_dataloader_iterator(
            self, data_iterator, batch_size: int
    ) -> Iterator[tuple[dict[str, Any], DataLoader]] | None:
        if not data_iterator:
            return None

        test_description = self.data_description.inference_grid
        # TODO: This should be somewhere where we validate the configuration
        if (
                test_description.output_tile_size is not None
                and test_description.output_tile_size != test_description.tile_size
        ):
            raise ValueError(f"`output_tile_size should be equal to tile_size in inference or set to None.")

        for dataset in data_iterator:
            metadata = create_inference_metadata(dataset, test_description.mpp, test_description.tile_size)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
            )

            yield metadata, dataloader

    def train_dataloader(self):
        if not self._fit_data_iterator:
            self.setup(TrainerFn.FITTING)
        return self._construct_concatenated_dataloader(
            self._fit_data_iterator, batch_size=self._batch_size, stage=TrainerFn.FITTING
        )

    def val_dataloader(self):
        if not self._validate_data_iterator:
            self.setup(TrainerFn.VALIDATING)

        batch_size = self._validate_batch_size if self._validate_batch_size else self._batch_size
        val_dataloader = self._construct_concatenated_dataloader(
            self._validate_data_iterator, batch_size=batch_size, stage=TrainerFn.VALIDATING
        )
        setattr(self, f"val_concat_dataset", val_dataloader.dataset)
        return val_dataloader

    def test_dataloader(self):
        if not self._test_data_iterator:
            self.setup(TrainerFn.TESTING)
        batch_size = self._validate_batch_size if self._validate_batch_size else self._batch_size
        return self._construct_concatenated_dataloader(
            self._validate_data_iterator, batch_size=batch_size, stage=TrainerFn.TESTING
        )

    def predict_dataloader(self):
        if not self._predict_data_iterator:
            self.setup(TrainerFn.PREDICTING)

        batch_size = self._validate_batch_size if self._validate_batch_size else self._batch_size
        return self._construct_dataloader_iterator(self._predict_data_iterator, batch_size=batch_size)

    def teardown(self, stage: str | None = None) -> None:
        getattr(self, f"_{stage}_data_iterator").__del__()

    @property
    def uuid(self) -> str:
        """This property is used to create a unique cache file for each dataset. The constructor of this dataset
        is completely determined by the data description, including the pre_transforms. Therefore, we can use the
        data description to create an uuid that is unique for each datamodule.

        The uuid is computed by hashing the data description using the `dataclass_to_uuid` function, which uses
        a sha256 hash of the pickled object. As pickles can change with python versions, this uuid will be different
        when using different python versions.

        Returns
        -------
        str
            A unique identifier for this datamodule.
        """
        return dataclass_to_uuid(self.data_description)
