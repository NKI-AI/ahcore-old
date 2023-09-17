import csv
import subprocess
from pathlib import Path

import submitit

metadata_path = Path("/projects/ellogon_tils/metadata/paths/2023-09-06")
slurm_block_paths = Path("/projects/ellogon_tils/slurm_blocks/")

MIN_GB = 7
MIN_CORES = 8
PARTITION = "a6000"
QOS = "a6000_qos"

studies = ["basis", "matador", "n4plus", "paradigm", "train"]
NUM_SLIDES_PER_JOB = 10


ENROOT_TEMPLATE = """
    enroot start --rw \
        -m /projects/ellogon_tils/:/projects/ellogon_tils/ \
        -m /data:/data
        tils_container.sqsh python app/scripts/run.py \
            --slides_csv {csv_path} \
            --unique_column slidename \
            --models_dir /appdata/models \
            --output_dir /projects/ellogon_tils/outputs/ \
            --external_data_path ""
"""

"""
    enroot start --rw \
        -m /projects/ellogon_tils/:/projects/ellogon_tils/ \
        -m /data:/data
        tils_container.sqsh python app/scripts/run.py \
            --slides_csv /projects/ellogon_tils/slurm_blocks/train/block_13.csv \
            --unique_column slidename \
            --models_dir /appdata/models \
            --output_dir /projects/ellogon_tils/outputs/ \
            --external_data_path ""
"""


def read_list(study: str):
    output = []
    if study not in studies:
        raise ValueError("Study needs to be in %s", studies)
    path = metadata_path / f"{study}_paths.csv"

    for line in path.read_text().split("\n"):
        if line.strip() == "":
            continue
        image_fn, _, _ = line.split(",")
        image_fn = Path(image_fn)

        output.append((f"{study}-{image_fn.name}", image_fn))

    return output


def generate_lists():
    slurm_block_paths.mkdir(exist_ok=True)

    output_dict = {}
    for study in studies:
        output_dir = slurm_block_paths / study
        output_dir.mkdir(exist_ok=True)

        full_lists = read_list(study)
        number_of_blocks = -(-len(full_lists) // NUM_SLIDES_PER_JOB)  # ceiling division

        # Calculate the padding length based on the number of blocks
        padding_length = len(str(number_of_blocks))

        for idx in range(number_of_blocks):
            start = idx * NUM_SLIDES_PER_JOB
            end = start + NUM_SLIDES_PER_JOB

            block = full_lists[start:end]

            file_path = output_dir / f"block_{str(idx).zfill(padding_length)}.csv"

            if not file_path.exists():  # Check if file already exists
                with file_path.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["slidename", "slidepath"])  # header
                    writer.writerows(block)


def run_container_with_file(filename):
    # TODO: Replace this
    cmd = ENROOT_TEMPLATE.format(filename)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout


def execute_slurm(data):
    for study_id, file_list in data:
        results = {}
        executor = submitit.AutoExecutor(folder=f"submitit_logs/TILs/{study_id}")

        executor.update_parameters(
            timeout_min=60,
            slurm_partition=PARTITION,
            slurm_qos=QOS,
            slurm_comment=f"sTILs",
            slurm_mem_gb=MIN_GB,
            slurm_gpus=1,  # Requesting 1 GPU
            slurm_cpus_on_node=8,  # Requesting minimum of 8 cores (CPUs)
        )

        jobs = []
        for filename in file_list:
            job = executor.submit(run_container_with_file, filename, study_id)
            jobs.append(job)

        # Collect results (optional)
        for job in jobs:
            results[job] = job.result()

        # Print the results
        for job, result in results.items():
            print(f"Result for {job}: {result}")


if __name__ == "__main__":
    generate_lists()

    file_lists = []
    # Generate the file list by scrolling over the dictories
    for study in studies:
        all_csv = list((slurm_block_paths / study).glob("*.csv"))
        file_lists.append((study, all_csv))

    print(file_lists)
