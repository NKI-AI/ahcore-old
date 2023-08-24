# encoding: utf-8
import argparse
import json
import multiprocessing

import tqdm

from ahcore.cli import dir_path


def is_jpeg(filename):
    with open(filename, "rb") as f:
        start_bytes = f.read(2)
        f.seek(-2, 2)
        end_bytes = f.read(2)
    return start_bytes == b"\xFF\xD8" and end_bytes == b"\xFF\xD9"


def process_svs_folder(args):
    svs_folder, lock = args
    metadata_fn = svs_folder / "meta_data_original_slide.json"
    if not metadata_fn.is_file():
        with lock:
            with open("errors.txt", "a") as f:
                f.write(f"{svs_folder},missing_metadata\n")
        return

    with open(svs_folder / "meta_data_tiles.json") as json_file:
        meta_data_tiles = json.load(json_file)
        max_images = meta_data_tiles["num_regions_masked"]
        tiles_dict = meta_data_tiles["tiles"]
        for i in range(max_images):
            if (f"img_{i}.jpg" not in tiles_dict) or not (svs_folder / f"img_{i}.jpg").is_file():
                with lock:
                    with open("errors.txt", "a") as f:
                        f.write(f"{svs_folder},metadata\n")
                return
            else:
                if not is_jpeg(svs_folder / f"img_{i}.jpg"):
                    with lock:
                        with open("errors.txt", "a") as f:
                            f.write(f"{svs_folder},broken_jpeg\n")
                    return


def detect_problems():
    parser = argparse.ArgumentParser("Tiling of whole slide images.")
    # Assume a comma separated format from image_file,mask_file
    parser.add_argument(
        "ROOT",
        type=dir_path(require_writable=False),
        help="Path to the output folder",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    root_folder = args.ROOT
    svs_folders = list((root_folder / "data").glob("*/*.svs"))

    if args.num_workers == 0:
        for folder in tqdm.tqdm(svs_folders):
            process_svs_folder((folder, None))
    else:
        lock = multiprocessing.Manager().Lock()
        with multiprocessing.Pool(args.num_workers) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(process_svs_folder, [(folder, lock) for folder in svs_folders]), total=len(svs_folders)
                )
            )


if __name__ == "__main__":
    detect_problems()
