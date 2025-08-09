# Setup instructions for Anvil Purdue

We run all these commands from the root of this repository `./phantom-reasoning/` (and not `./docs/` for 
instance).

## Installing Conda environment and dependencies 

The Anvil cluster provides shared conda installation, which we recommend over installing your own personal 
conda (Anmol: there were issues with python paths with personal conda installations).

```bash
module load conda
./scripts/anvil/load_modules_cuda.sh
```
Since these need to be run on every startup and on getting a node allocation, these could also be added to 
`.bashrc`:

```bash
echo "module load conda" >> ~/.bashrc
echo "module load modtree/gpu" >> ~/.bashrc
echo "module load cuda/12.0.1" >> ~/.bashrc
```

Now follow the instructions in [README.md] to install the repository in development mode.
For convenience, we write them down here.

```bash
# Assuming you are in ./phantom-reasoning git root repository

export CONDA_ENV_NAME="phantom-reasoning" # or whatever the name of your conda environment is

conda create -n $CONDA_ENV_NAME python=3.12
conda activate $CONDA_ENV_NAME

conda install conda-forge::swi-prolog

# If need an editable version add as a submodule
pip install phantom-wiki[eval]

# Install phantom-reasoning in editable mode
pip install -e ".[dev]"

pip install flash-attn --no-build-isolation

pre-commit install
```

## Setting environment variables

> \[!NOTE\]
> Home paths `~/` only have 25GB on Anvil, so it's extremely important that you set huggingface datasets, 
> models, checkpoints to scratch. If anything needs to be shared (e.g. datasets), we save them to the shared 
> directory. The conda and pip environments (folders `~/.conda/` and `~/.cache/pip`) will take up 10GB or so 
> with just this 1 project.

```bash
conda env config vars set ANVIL_PROJECT_ID="nairr250102"
conda env config vars set RUN_BASE_DIR="$SCRATCH/phantom-reasoning/"
conda env config vars set WANDB_ENTITY="mlcore"
conda env config vars set WANDB_PROJECT="phantom-reasoning"
conda env config vars set HF_HOME="$SCRATCH/huggingface"
conda env config vars set CONDA_ENV_NAME=$CONDA_ENV_NAME # so the env name is available automatically when activated

conda deactivate
conda activate $CONDA_ENV_NAME
```

## Setting up Weights&Biases

Setup wandb login: `wandb login` and paste the API key from the `mlcore` organization. 
Contact Anmol if you don't have access to the `mlcore` org.

## Creating symlinks to the data and output folders

```bash
ln -s /anvil/projects/x-$ANVIL_PROJECT_ID/phantom-reasoning/data .
ln -s /anvil/projects/x-$ANVIL_PROJECT_ID/phantom-reasoning/runs .
ln -s /anvil/projects/x-$ANVIL_PROJECT_ID/phantom-reasoning/eval .
mkdir $RUN_BASE_DIR
```

## Running experiments

For a GRPO experiment on Qwen3-1.7B model:

```bash
module load conda
conda activate $CONDA_ENV_NAME # to get ANVIL_PROJECT_ID variable

# Option 1: Interactive
salloc -A $ANVIL_PROJECT_ID-ai -p ai --gres=gpu:4 -n 16 -N 1 --mem=100GB -t 12:00:00
# After getting an allocation:
module load conda
./scripts/anvil/load_modules_cuda.sh
conda activate $CONDA_ENV_NAME

./scripts/anvil/train_grpo__vllm_server.sh \
    recipes/accelerate_configs/zero1.yaml \
    recipes/Qwen/Qwen3-1.7B/grpo/config_4gpu__vllm_server.yaml

# Option 2: Batch job
sbatch -A $ANVIL_PROJECT_ID-ai scripts/anvil/train_grpo__vllm_server.sh \
    recipes/accelerate_configs/zero1.yaml \
    recipes/Qwen/Qwen3-1.7B/grpo/config_4gpu__vllm_server.yaml
```

Script for running the Llama model with the colocate setup

```bash
./scripts/anvil/train_grpo__vllm_colocate.sh \
    recipes/accelerate_configs/zero1.yaml \
    recipes/meta-llama/Llama-3.2-1B-Instruct/grpo/config_4gpu__vllm_colocate.yaml
```

