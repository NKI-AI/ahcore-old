#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2
#SBATCH --time=14-00:00:00
#SBATCH --output=tumor_stroma_attention_unet_macenko%A.out
#SBATCH --error=tumor_stroma_attention_unet_macenko%A.err
#SBATCH --job-name=tumor_stroma_attention_unet_macenko
# -----------------------------------------------------------
# Parameters to change for different hyper parameter searches
# -----------------------------------------------------------
# Machine parameters per hyperparameters configuration
NUM_GPUS_PER_NODE=1
PARTITION=a100
ACCOUNT=a100
NUM_CPUS_PER_NODE=16
MEM_GB_PER_NODE=100
NUM_NODES=1
BATCH_SIZE=8
NUM_WORKERS=16
TIME_PER_NODE=6000  # this is in minutes

# export MASTER_PORT=5012  # necessary if multi-node per hyperparameters
# Mlflow configs
EXPERIMENT_NAME="tumor_stroma_monai"  # change this
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST")  # there is only one node at this point -- the master node
MLFLOW_PORT=5008
MLFLOW_ADDRESS="http://$MASTER_NODE:$MLFLOW_PORT"
MLFLOW_BACKEND_STORE_DIR="/home/a.karkala/experiments/mlflow"
MLFLOW_ARTIFACT_DIR="/home/a.karkala/experiments/mlflow/artifacts"
# Hyperparameter config -- multirun example:  # Use the non-optimize bandwidth experiment
HYPERPARAMETERS="augmentations=segmentation_macenko,segmentation"

RUN_CONFIG="lit_module=attention_unet lit_module.model.depth=4 lit_module.model.kernel_multiplier=7 trainer.max_epochs=500 trainer=default losses=segmentation_ce datamodule=dataset data_description=tissue_subtypes/training task=segmentation"
# -----------------------------------------------------------
# Fixed config below, propagating the above parameters to the code
# -----------------------------------------------------------
# activate environment
spack load openjpeg jpeg/3l3onbs libtiff/3e242tv libxml2 sqlite glib/7wugotn cairo+pdf gdk-pixbuf/q4sxq6b

. /sw/spack/share/spack/setup-env.sh
unset LD_LIBRARY_PATH
spack load cuda@11.7
spack load openslide/njs7ivd
spack load pixman@0.40.0
spack load libvips@8.14.1
spack unload python py-pip
conda activate ahcore
spack load glib/7wugotn

python /home/j.teuwen/tmp/pycharm_project_ahcore_393/tools/train.py lit_module=monai_attention_unet augmentations=segmentation data_description=tissue_subtypes/segmentation task=segmentation
# Slurm setup
SLURM_PARAMETERS="hydra/launcher=submitit_slurm hydra.launcher.gpus_per_node=$NUM_GPUS_PER_NODE hydra.launcher.tasks_per_node=$NUM_GPUS_PER_NODE hydra.launcher.nodes=$NUM_NODES hydra.launcher.cpus_per_task=$NUM_CPUS_PER_NODE hydra.launcher.mem_gb=$MEM_GB_PER_NODE hydra.launcher.partition=$PARTITION +hydra.launcher.additional_parameters={account:$ACCOUNT} hydra.launcher.timeout_min=$TIME_PER_NODE"
# Standard setup
DATA_PARAMETERS="datamodule.num_workers=$NUM_WORKERS datamodule.batch_size=$BATCH_SIZE"
MLFLOW_PARAMETERS="+logger.mlflow.experiment_name=$EXPERIMENT_NAME logger.mlflow.tracking_uri=$MLFLOW_ADDRESS"
MACHINE_PARAMETERS="trainer.devices=$NUM_GPUS_PER_NODE trainer.num_nodes=$NUM_NODES"
# nccl environment
#export NCCL_NSOCKS_PERTHREAD=4
#export NCCL_SOCKET_NTHREADS=2
# start mlflow server
#export MLFLOW_TRACKING_URI="http://0.0.0.0:$MLFLOW_PORT"  # needed for creating experiment below
#mlflow server --backend-store-uri $MLFLOW_BACKEND_STORE_DIR --default-artifact-root $MLFLOW_ARTIFACT_DIR --host 0.0.0.0:$MLFLOW_PORT &
#sleep 3
#mlflow experiments create -n $EXPERIMENT_NAME
# debugging flags (optional)
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export PYTHONFAULTHANDLER=1
# the hyperparam search below will spawn nodes and report to mlflow automatically
#python /home/a.karkala/ahcore/tools/train.py --multirun $RUN_CONFIG $SLURM_PARAMETERS $MACHINE_PARAMETERS $DATA_PARAMETERS $MLFLOW_PARAMETERS $HYPERPARAMETERS &
#wait
python /home/j.teuwen/tmp/pycharm_ahcore/tools/train.py lit_module=monai_attention_unet augmentations=segmentation data_description=tissue_subtypes/segmentation task=segmentation
