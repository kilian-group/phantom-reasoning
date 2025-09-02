#!/usr/bin/env bash

# Script to create a merged training script for GRPO with VLLM colocation
# Usage: ./scripts/create_train_grpo_vllm_colocate.sh <cluster_name>

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <cluster_name>"
    echo "Supported cluster names: aida, anvil, empire"
    exit 1
fi

CLUSTER_NAME=$1
OUTPUT_FILE="scripts/train_grpo__vllm_colocate.sub"

# Validate cluster name
if [[ "$CLUSTER_NAME" != "aida" && "$CLUSTER_NAME" != "anvil" && "$CLUSTER_NAME" != "empire" ]]; then
    echo "Error: Unsupported cluster name '$CLUSTER_NAME'"
    echo "Supported cluster names: aida, anvil, empire"
    exit 1
fi

echo "Creating training script for cluster: $CLUSTER_NAME"
echo "Output file: $OUTPUT_FILE"

mkdir -p logs

# Set default SBATCH parameters
SBATCH_JOB_NAME="grpo"
SBATCH_OUTPUT="logs/grpo-%j.out"
SBATCH_ERROR="logs/grpo-%j.err"
SBATCH_NODES="1"
SBATCH_NTASKS="8"
SBATCH_MEM="50GB"
SBATCH_TIME="24:00:00"
SBATCH_MAIL_USER=$USER_EMAIL

# Define SBATCH_PARTITION based on cluster
if [[ "$CLUSTER_NAME" == "aida" ]]; then
    SBATCH_PARTITION="full"
elif [[ "$CLUSTER_NAME" == "anvil" ]]; then
    SBATCH_PARTITION="ai"
elif [[ "$CLUSTER_NAME" == "empire" ]]; then
    SBATCH_PARTITION="cornell,priority"
fi

# Create the merged script, substituting the variables
cat << EOT > "$OUTPUT_FILE"
#!/usr/bin/env bash
#SBATCH --job-name=$SBATCH_JOB_NAME
#SBATCH --output=$SBATCH_OUTPUT
#SBATCH --error=$SBATCH_ERROR
#SBATCH -p $SBATCH_PARTITION
#SBATCH -N $SBATCH_NODES
#SBATCH -n $SBATCH_NTASKS
#SBATCH --gres=gpu:4
#SBATCH --mem=$SBATCH_MEM
#SBATCH --time=$SBATCH_TIME
#SBATCH --mail-user=$SBATCH_MAIL_USER
#SBATCH --mail-type=all
EOT

# On aida, select H100 or A100 GPUs by adding SBATCH -C=gpu-h100|gpu-a100&no-gpu-1g.10gb
# NOTE: the &no-gpu-1g.10gb part is necessary to avoid the aida nodes with 1GB of memory, which are not supported by GRPO
if [[ "$CLUSTER_NAME" == "aida" ]]; then
    cat >> "$OUTPUT_FILE" << EOT
#SBATCH --constraint="gpu-h100|gpu-a100&no-gpu-1g.10gb"
EOT
fi

# Add account information for each cluster
if [[ "$CLUSTER_NAME" == "aida" ]]; then
    # If aida, no need to add anything
    echo "No sbatch -A information needed for AIDA"
elif [[ "$CLUSTER_NAME" == "anvil" ]]; then
    # If anvil, add SBATCH -A $ANVIL_PROJECT_ID-ai
    cat >> "$OUTPUT_FILE" << EOT
#SBATCH -A $ANVIL_PROJECT_ID-ai
EOT
elif [[ "$CLUSTER_NAME" == "empire" ]]; then
    # If empire, add SBATCH -A cornell
    cat >> "$OUTPUT_FILE" << EOT
#SBATCH -A cornell
EOT
fi

# Add conda environment loading
if [[ "$CLUSTER_NAME" == "aida" ]]; then
    cat >> "$OUTPUT_FILE" << 'EOF'
source $HOME/.bashrc
EOF
elif [[ "$CLUSTER_NAME" == "anvil" ]]; then
    cat >> "$OUTPUT_FILE" << 'EOF'
module load conda
source $(pwd)/scripts/anvil/load_modules_cuda.sh
EOF
elif [[ "$CLUSTER_NAME" == "empire" ]]; then
    cat >> "$OUTPUT_FILE" << 'EOF'
source $HOME/.bashrc
EOF
fi

# Add CONDA_ENV_NAME checker to the script for all clusters
cat >> "$OUTPUT_FILE" << 'EOF'
# Set CONDA_ENV_NAME to default if not set
if [ -z "$CONDA_ENV_NAME" ]; then
    CONDA_ENV_NAME="phantom-reasoning"
fi
conda activate $CONDA_ENV_NAME
EOF

# Add the common part for launching the training script
# Escape the dollar sign in this part
cat >> "$OUTPUT_FILE" << 'EOF'

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
NUM_PROCESSES=$((NUM_GPUS))

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

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_TRAINING ACCELERATE_LOG_LEVEL=info accelerate launch \
    --num_processes=$NUM_PROCESSES \
    --config_file $ACCELERATE_CONFIG_FILE_PATH \
    src/phantom_reasoner/grpo.py \
    --config $GRPO_CONFIG_FILE_PATH \
    $cmd_args
EOF

# Make the output file executable
chmod a+x "$OUTPUT_FILE"

echo "--------------------------------"
echo "Successfully created $OUTPUT_FILE for cluster: $CLUSTER_NAME"
echo "Usage: sbatch $OUTPUT_FILE <path_to_accelerate_config> <path_to_config_file> [additional_args]"
