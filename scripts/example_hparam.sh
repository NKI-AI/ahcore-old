#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2
#SBATCH --time=14-00:00:00
#SBATCH --output=hyperparam_master_%A.out
#SBATCH --error=hyperparam_master_%A.err

# -----------------------------------------------------------
# Parameters to change for different hyper parameter searches
# -----------------------------------------------------------

# Machine parameters per hyperparam configuration
NUM_GPUS_PER_NODE=1
PARTITION=a6000
ACCOUNT=a6000
NUM_CPUS_PER_NODE=16
MEM_GB_PER_NODE=60
NUM_NODES=1
NUM_WORKERS=16
BATCH_SIZE=16
TIME_PER_NODE=60  # this is in minutes

# export MASTER_PORT=5012  # necessary if multi-node per hyperparam

# Mlflow configs
EXPERIMENT_NAME="change_this"  # change this
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST")  # there is only one node at this point -- the master node
MLFLOW_PORT=5002
MLFLOW_ADDRESS="http://$MASTER_NODE:$MLFLOW_PORT"
MLFLOW_BACKEND_STORE_DIR=/path/to_mlflow_outputs
MLFLOW_ARTIFACT_DIR=/path/to/mlflow_artifacts

# Hyperparameter config -- multirun example:  # Use the non-optimize bandwidth experiment
HYPERPARAMETERS="lit_module=new_unet,unet,attention_unet"
RUN_CONFIG="trainer.max_epochs=20 trainer=default datamodule=debug_dataset data_description=debug task=segmentation transforms=segmentation"

# -----------------------------------------------------------
# Fixed config below, propagating the above parameters to the code
# -----------------------------------------------------------

# activate environment
unset LD_LIBRARY_PATH
spack load cuda@11.3.1
spack load openslide@3.4.1
spack load pixman@0.40.0
spack load libvips@8.13.0
spack unload py-pip py-wheel py-setuptools python
source activate "my_conda_env"

# Slurm setup
SLURM_PARAMETERS="hydra/launcher=submitit_slurm hydra.launcher.gpus_per_node=$NUM_GPUS_PER_NODE hydra.launcher.tasks_per_node=$NUM_GPUS_PER_NODE hydra.launcher.nodes=$NUM_NODES hydra.launcher.cpus_per_task=$NUM_CPUS_PER_NODE hydra.launcher.mem_gb=$MEM_GB_PER_NODE hydra.launcher.partition=$PARTITION +hydra.launcher.additional_parameters={account:$ACCOUNT} hydra.launcher.timeout_min=$TIME_PER_NODE"

# Standard setup
DATA_PARAMETERS="datamodule.batch_size=$BATCH_SIZE datamodule.num_workers=$NUM_WORKERS"
MLFLOW_PARAMETERS="+logger.mlflow.experiment_name=$EXPERIMENT_NAME logger.mlflow.tracking_uri=$MLFLOW_ADDRESS"
MACHINE_PARAMETERS="trainer.devices=$NUM_GPUS_PER_NODE trainer.num_nodes=$NUM_NODES"

# nccl environment
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

# start mlflow server
export MLFLOW_TRACKING_URI="http://0.0.0.0:$MLFLOW_PORT"  # needed for creating experiment below
mlflow server --backend-store-uri $MLFLOW_BACKEND_STORE_DIR --default-artifact-root $MLFLOW_ARTIFACT_DIR --host 0.0.0.0:$MLFLOW_PORT &
sleep 3
mlflow experiments create -n $EXPERIMENT_NAME

# debugging flags (optional)
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export PYTHONFAULTHANDLER=1

# the hyperparam search below will spawn nodes and report to mlflow automatically
python tools/train.py --multirun $RUN_CONFIG $SLURM_PARAMETERS $MACHINE_PARAMETERS $DATA_PARAMETERS $MLFLOW_PARAMETERS $HYPERPARAMETERS &
wait
