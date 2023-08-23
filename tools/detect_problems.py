from pathlib import Path
import json
import tqdm
from multiprocessing import Pool, cpu_count

# TODO: Make much more elaborate, check each file!


def process_svs_folder(svs_folder):
    metadata_fn = svs_folder / "meta_data_original_slide.json"
    if not metadata_fn.is_file():
        with open("errors.txt", "a") as f:
            f.write(str(svs_folder) + ",missing_metadata\n")
        return

    with open(svs_folder / "meta_data_tiles.json") as json_file:
        meta_data_tiles = json.load(json_file)
        max_images = meta_data_tiles["num_regions_masked"]
        tiles_dict = meta_data_tiles["tiles"]
        for i in range(max_images):
            if (f"img_{i}.jpg" not in tiles_dict) or not (svs_folder/ f"img_{i}.jpg").is_file():
                with open("errors.txt", "a") as f:
                    f.write(str(svs_folder) + ",metadata\n")
                return

def construct_tile_dict():
    root_folder = Path("/projects/tcga_tiled/v1/data")
    svs_folders = list(root_folder.glob("*/*.svs"))

    for svs_folder in tqdm.tqdm(svs_folders):
        process_svs_folder(svs_folder)


if __name__ == "__main__":
    construct_tile_dict()