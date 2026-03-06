#!/usr/bin/env bash
# Set up conda environment variables for phantom-reasoning.
# Persists CONDA_ENV_NAME, RUN_BASE_DIR, HF_HOME (cluster-dependent), ANVIL_PROJECT_ID
# (when cluster=anvil), and prompts for USER_EMAIL, WANDB_ENTITY, WANDB_PROJECT.
# Run with: bash scripts/setup_conda_env_vars.sh <conda_env_name> [cluster]
# After completion, run: conda deactivate && conda activate <conda_env_name>

if [ $# -lt 1 ]; then
    echo "Usage: $0 <conda_env_name> [cluster]"
    echo ""
    echo "  conda_env_name  Name of the conda environment to configure"
    echo "  cluster         Optional. Supported values: anvil, unicorn"
    exit 1
fi

CONDA_ENV_NAME="$1"
shift
CLUSTER="${1:-}"
shift
cmd_args=$@

# Helper: set a conda env var without printing the "reactivate" notice
conda_set() {
    conda env config vars set --name "$CONDA_ENV_NAME" "$1" > /dev/null
}

# --- Always-set variables ---
conda_set CONDA_ENV_NAME="$CONDA_ENV_NAME"

# --- Cluster-specific variables ---
if [ "$CLUSTER" = "anvil" ]; then
    conda_set RUN_BASE_DIR="$SCRATCH/phantom-reasoning"
    conda_set HF_HOME="$SCRATCH/huggingface"
    conda_set ANVIL_PROJECT_ID="nairr250102"
elif [ "$CLUSTER" = "unicorn" ]; then
    conda_set RUN_BASE_DIR="/scratch/$USER/phantom-reasoning"
    conda_set HF_HOME="/scratch/$USER/huggingface"
else
    echo "Warning: no cluster specified — setting RUN_BASE_DIR to '.'"
    conda_set RUN_BASE_DIR="."
fi

# --- Interactive prompts ---
read -rp "USER_EMAIL (for SLURM job notifications): " USER_EMAIL
read -rp "WANDB_ENTITY (W&B organization name): " WANDB_ENTITY
read -rp "WANDB_PROJECT [phantom-reasoning]: " WANDB_PROJECT
WANDB_PROJECT="${WANDB_PROJECT:-phantom-reasoning}"

conda_set USER_EMAIL="$USER_EMAIL"
conda_set WANDB_ENTITY="$WANDB_ENTITY"
conda_set WANDB_PROJECT="$WANDB_PROJECT"

# --- Summary ---
echo ""
echo "Environment variables set for conda env '$CONDA_ENV_NAME':"
echo "  CONDA_ENV_NAME = $CONDA_ENV_NAME"
if [ "$CLUSTER" = "anvil" ]; then
    echo "  RUN_BASE_DIR     = $SCRATCH/phantom-reasoning"
    echo "  HF_HOME          = $SCRATCH/huggingface"
    echo "  ANVIL_PROJECT_ID = nairr250102"
elif [ "$CLUSTER" = "unicorn" ]; then
    echo "  RUN_BASE_DIR   = /scratch/$USER/phantom-reasoning"
    echo "  HF_HOME        = /scratch/$USER/huggingface"
else
    echo "  RUN_BASE_DIR   = ."
fi
echo "  USER_EMAIL     = $USER_EMAIL"
echo "  WANDB_ENTITY   = $WANDB_ENTITY"
echo "  WANDB_PROJECT  = $WANDB_PROJECT"

echo ""
echo "Reactivate your environment for changes to take effect:"
echo "  conda deactivate && conda activate $CONDA_ENV_NAME"
echo ""
echo "Then complete W&B setup:"
echo "  wandb login"
