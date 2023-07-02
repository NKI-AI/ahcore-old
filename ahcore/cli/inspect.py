# # encoding: utf-8
# """Module to inspect a dataset such as the total area"""
# from __future__ import annotations
#
# import argparse
# import json
# from collections import defaultdict
#
# import numpy as np
# import numpy.typing as npt
# from dlup import SlideImage
# from dlup.annotations import WsiAnnotations
# from dlup.data.dataset import TiledROIsSlideImageDataset
# from dlup.data.transforms import ConvertAnnotationsToMask
# from dlup.tiling import TilingMode
# from tqdm import tqdm
#
# from ahcore.cli import dir_path, file_path
# from ahcore.transforms.pre_transforms import Compose, ImageToTensor, RenameLabels
# from ahcore.utils.manifest import ImageManifest, manifests_from_data_description, DataDescription
#
#
# # FIXME: This does not work right now as we have a different way to read the manifests/
# def get_from_manifests(args, curr_manifests, label_map: dict[str, str]):
#     mpp = 2.0
#     pixel_area = mpp * mpp
#     num_rois = 0
#
#     total_area: defaultdict[str, float] = defaultdict(float)
#     total_roi = 0.0
#
#     available_labels = []
#     total_annotated_area = 0.0
#
#     for curr_manifest in curr_manifests:
#         parsed_manifest = parse_manifest(args.image_directory, args.annotation_directory, curr_manifest)
#         image_fn = parsed_manifest.image_fn
#         image_backend = parsed_manifest.backend
#         _mask = parsed_manifest.mask
#         _annotations = parsed_manifest.annotations
#         labels = parsed_manifest.labels
#
#         if _mask is None:
#             raise RuntimeError(f"Mask is required to be able to compute statistics.")
#         else:
#             mask = _mask
#
#         if _annotations is None:
#             raise RuntimeError(f"Annotations are required to be able to compute statistics.")
#         else:
#             annotations = _annotations
#
#         if isinstance(annotations, SlideImage):
#             raise NotImplementedError("SlideImage's are not supported for statistics.")
#
#         try:
#             with SlideImage.from_file_path(image_fn, backend=image_backend) as slide_image:
#                 dimensions = slide_image.size
#         except Exception:  # TODO: better exception in dlup
#             return None
#
#         if isinstance(mask, WsiAnnotations):
#             roi_labels = mask.available_labels
#             if len(roi_labels) != 1:
#                 raise RuntimeError("Can only understand one label.")
#
#             rois = mask.read_region((0, 0), 1.0, dimensions)
#             num_rois += len(rois)
#             available_labels += [k for k in annotations.available_labels if k not in roi_labels]
#             available_labels.sort()
#         elif isinstance(mask, SlideImage):
#             raise NotImplementedError("SlideImage's as masks are not supported for statistics.")
#         else:
#             raise NotImplementedError(f"Mask of type {type(mask)} is not supported for statistics.")
#
#         # Rename the available labels
#         available_labels = [label if label not in label_map else label_map[label] for label in available_labels]
#
#         index_map = {v: idx + 1 for idx, v in enumerate(available_labels)}
#         index_map["background"] = 0
#
#         transforms = Compose(
#             [
#                 RenameLabels(remap_labels=label_map),
#                 ConvertAnnotationsToMask(roi_name=roi_labels[0], index_map=index_map),
#                 ImageToTensor(),
#             ]
#         )
#         try:
#             dataset = TiledROIsSlideImageDataset.from_standard_tiling(
#                 image_fn,
#                 backend=image_backend,
#                 tile_size=(512, 512),
#                 tile_overlap=(0, 0),
#                 mpp=mpp,
#                 mask=mask,
#                 mask_threshold=0.01,
#                 annotations=annotations,
#                 tile_mode=TilingMode.overflow,
#                 transform=transforms,
#             )
#         except Exception:
#             return None
#
#         # TODO: Mask threshold should be >, no >=
#         if len(dataset) == 0:
#             raise RuntimeError(f"Empty dataset. Strange that this even happens.")
#
#         for sample in dataset:
#             roi = np.asarray(sample["roi"])
#             total_roi += float(roi.sum())
#             target = np.asarray(sample["target"])
#             for key in index_map:
#                 mask_array = target == index_map[key]
#                 if key == "background":
#                     mask_array[roi == 0] = 0
#
#                 total_area[key] += float(mask_array.sum())
#
#         total_annotated_area += sum(total_area.values())
#
#     proportions = {k: v / total_roi * 100 for k, v in total_area.items()}
#     areas = {k: v * pixel_area for k, v in total_area.items()}
#
#     output = {
#         "labels": available_labels,
#         "num_rois": num_rois,
#         "roi_area": total_roi * pixel_area,
#         "areas": areas,
#         "total_annotated_area": total_annotated_area * pixel_area,
#         "proportions": proportions,
#     }
#     if label_map:
#         output["label_map"] = label_map
#
#     return output
#
#
# def compute_statistics(args: argparse.Namespace):
#     """
#     Compute statistics of the dataset given a manifest. Will select on PatientID (first 12 digits of TCGA string).
#
#     Parameters
#     ----------
#     args : args.Namespace
#
#     Returns
#     -------
#     None
#     """
#     with open(args.manifest, "r") as json_file:
#         json_manifests = json.load(json_file)
#
#     # Create dictionary from the label map
#     label_map = {}
#     if args.label_map:
#         for pair in args.label_map.split(","):
#             try:
#                 key, value = pair.split("=")
#             except ValueError:
#                 raise argparse.ArgumentTypeError(f"Cannot parse key, value pairs. Got {args.label_map}.")
#
#             label_map[key.strip()] = value.strip()
#
#     output = {}
#
#     # Combine manifests per patient
#     combined_manifests = defaultdict(list)
#     for json_manifest in tqdm(json_manifests):
#         curr_manifest = ImageManifest(**json_manifest)
#         combined_manifests[curr_manifest.identifier[:12]].append(curr_manifest)
#
#     for curr_manifests in tqdm(combined_manifests):
#         _curr_manifest = combined_manifests[curr_manifests]
#         curr_output = get_from_manifests(args, combined_manifests[curr_manifests], label_map)
#         if curr_output is None:
#             continue
#
#         output[_curr_manifest[0].identifier[:12]] = curr_output
#
#     print(json.dumps(output, indent=2))
#
#
# def register_parser(parser: argparse._SubParsersAction):
#     """Register inspect commands to a root parser."""
#     inspect_parser = parser.add_parser("inspect", help="Inspect a dataset")
#     inspect_subparsers = inspect_parser.add_subparsers(help="Inspect subparser")
#     inspect_subparsers.required = True
#     inspect_subparsers.dest = "subcommand"
#
#     _parser: argparse.ArgumentParser = inspect_subparsers.add_parser(
#         "compute-statistics", help="Compute statistics for dataset"
#     )
#
#     _parser.add_argument(
#         "image_directory",
#         type=dir_path,
#         help="Directory to the images.",
#     )
#     _parser.add_argument(
#         "annotation_directory",
#         type=dir_path,
#         help="Directory to the annotations.",
#     )
#     _parser.add_argument(
#         "manifest",
#         type=file_path,
#         help="Path to the manifest.",
#     )
#
#     _parser.add_argument(
#         "--label-map",
#         type=str,
#         help="Map labels onto a different class in the form: `label=new_label,label2=new_label2,...`. "
#         "It is not required that all labels are present, only the ones in the list will be overwritten.",
#     )
#
#     _parser.set_defaults(subcommand=compute_statistics)
