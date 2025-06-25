#!/bin/bash
# FILENAME:  anvil_train_grpo

#SBATCH -A ai250102-ai       # allocation name
#SBATCH --nodes=1             # Total # of nodes 
#SBATCH --ntasks-per-node=4   # Number of MPI ranks per node (one rank per GPU)
#SBATCH --gpus-per-node=4     # Number of GPUs per node
#SBATCH --time=24:00:00        # Total run time limit (hh:mm:ss)
#SBATCH -J traingrpo          # Job name
#SBATCH --output=/anvil/scratch/x-kluo5/phantom-reasoning/slurm_logs/%j.out
#SBATCH --error=/anvil/scratch/x-kluo5/phantom-reasoning/slurm_logs/%j.out
#SBATCH -p ai                # Queue (partition) name
#SBATCH --mail-user=kzl6@cornell.edu
#SBATCH --mail-type=all       # Send email to above address at begin and end of job

# Manage processing environment, load compilers, and applications.
module purge
module load modtree/gpu
module load cuda/12.0.1
module load conda
module use $HOME/privatemodules
module load conda-env/phantom-reasoning-py3.12.8
source activate phantom-reasoning

module list

# First argument should be path to the accelerate config file
# Second argument should be the path to the training config file
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_accelerate_config> <path_to_config_file> [additional_args]"
    exit 1
fi
ACCELERATE_CONFIG_FILE_PATH=$1
GRPO_CONFIG_FILE_PATH=$2

# Get NUM_GPUS from nvidia-smi. It repeats the number of GPUs a few times, take the first one.
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n 1)


# Get the model name from config file
MODEL_NAME=$(grep "model_name_or_path" $GRPO_CONFIG_FILE_PATH | cut -d '"' -f 2)

# Get CUDA visible devices as 0,...,NUM_GPUS-1 (0 indexing)
CUDA_DEVICES_TRAINING=$(seq -s, 0 $((NUM_GPUS - 1)))

# Get additional arguments
shift 2
cmd_args=$@
echo "-------------------------------"
echo "Additional arguments: $cmd_args"
echo "-------------------------------"

echo "-------------------------------"
echo "Starting GRPO training with"
echo "- accelerate config=$ACCELERATE_CONFIG_FILE_PATH"
echo "- training config=$GRPO_CONFIG_FILE_PATH"
echo "- on GPUs $CUDA_DEVICES_TRAINING"
echo "-------------------------------"

NUM_PROCESSES=$((NUM_GPUS))
export WANDB_PROJECT="grpo"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_TRAINING ACCELERATE_LOG_LEVEL=info accelerate launch \
    --num_processes=$NUM_PROCESSES \
    --config_file $ACCELERATE_CONFIG_FILE_PATH \
	src/phantom_reasoner/grpo.py \
	--config $GRPO_CONFIG_FILE_PATH \
    $@
