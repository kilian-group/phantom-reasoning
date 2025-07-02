# Setup instructions for Anvil Purdue

We run all these commands from the root of this repository `./phantom-reasoning/` (and not `./docs/` for instance).

1. First, install the conda environment at `~/miniconda`.

Now follow the instructions in [README.md] to install the repository in development mode.
For convenience, we write them down here.

```bash
# Assuming you are in ./phantom-reasoning git root repository

export CONDA_ENV_NAME="phantom-reasoning" # or whatever the name of your conda environment is

conda create -n $CONDA_ENV_NAME python=3.12
conda activate $CONDA_ENV_NAME

conda install conda-forge::swi-prolog

# Install phantom-wiki and phantom-reasoning in editable modes
git clone git@github.com:kilian-group/phantom-wiki.git
cd phantom-wiki
pip install -e ".[eval]"

cd ..
git clone git@github.com:anmolkabra/phantom-reasoning.git
cd phantom-reasoning
pip install -e ".[dev]"

pre-commit install
```

2. Set environment vars in the conda environment.

> \[!NOTE\]
> Home paths `~/` only have 25GB on Anvil, so it's extremely important that you set huggingface datasets, models, checkpoints to scratch. If anything needs to be shared (e.g. datasets), we save them to the shared directory. The conda and pip environments (folders `~/.conda/` and `~/.cache/pip`) will take up 10GB or so with just this 1 project.

```bash
conda env config vars set WANDB_ENTITY="mlcore"
conda env config vars set WANDB_ENTITY="mlcore"
conda env config vars set WANDB_PROJECT="phantom-reasoning"
conda env config vars set CONDA_ENV_NAME=$CONDA_ENV_NAME # so the env name is available automatically when activated

conda deactivate
conda activate $CONDA_ENV_NAME
```

3. Setup wandb login: `wandb login` and paste the API key from the `mlcore` organization. Contact Anmol if you don't have access to the `mlcore` org.

4. Run a GRPO experiment on Qwen3-1.7B model:

```bash
conda activate $CONDA_ENV # for sbatch to pull in user-defined env vars

# Option 1: Interactive
salloc -p ai --gres=gpu:a100:4 -n 16 -N 1 --mem=100GB -t 12:00:00
# After getting an allocation:
conda activate $CONDA_ENV_NAME

# the anvil config scripts should also work on aida

./scripts/aida/train_grpo.sh \
    recipes/accelerate_configs/zero1.yaml \
    recipes/Qwen/Qwen3-1.7B/grpo/config_anvil.yaml

# Option 2: Batch job
sbatch scripts/anvil/train_grpo.sh \
    recipes/accelerate_configs/zero1.yaml \
    recipes/Qwen/Qwen3-1.7B/grpo/config_anvil.yaml
```
