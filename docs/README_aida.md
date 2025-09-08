# Setup instructions for AIDA

We run all these commands from the root of this repository `./phantom-reasoning/` (and not `./docs/` for instance).

1. First, install the conda environment at `~/miniconda`.

Now follow the instructions in [README.md] to install the repository in development mode.
For convenience, we write them down here.

```bash
# Assuming you are in ./phantom-reasoning git root repository

export CONDA_ENV_NAME="phantom-reasoning" # or whatever the name of your conda environment is

conda create -n $CONDA_ENV_NAME
conda activate $CONDA_ENV_NAME

conda install conda-forge::swi-prolog
conda install python=3.12
pip install uv

# Install phantom-wiki and phantom-reasoning in editable modes
git clone git@github.com:kilian-group/phantom-wiki.git
cd phantom-wiki
uv pip install -e ".[eval]"

cd ..
git clone git@github.com:anmolkabra/phantom-reasoning.git
cd phantom-reasoning
uv pip install -e ".[dev]"

uv pip install flash-attn --no-build-isolation

pre-commit install
```

2. Set environment vars in the conda environment.

```bash
conda env config vars set WANDB_ENTITY="mlcore"
conda env config vars set WANDB_PROJECT="phantom-reasoning"
conda env config vars set CONDA_ENV_NAME=$CONDA_ENV_NAME # so the env name is available automatically when activated
conda env config vars set USER_EMAIL="user@email.com" # for emailing when allocations become available

conda deactivate
conda activate $CONDA_ENV_NAME
```

3. Setup wandb login: `wandb login` and paste the API key from the `mlcore` organization. Contact Anmol if you don't have access to the `mlcore` org.

4. Run a GRPO experiment on Qwen3-1.7B model:

```bash
conda activate $CONDA_ENV_NAME # for sbatch to pull in user-defined env vars

# Option 1: Interactive
salloc -p full --gres=gpu:a100:4 -n 16 -N 1 --mem=100GB -t 12:00:00 --mail-type=all --mail-user=$USER_EMAIL

# After getting an allocation:
conda activate $CONDA_ENV_NAME

./scripts/create_train_grpo__vllm_colocate.sh aida

./scripts/train_grpo__vllm_colocate.sub \
    recipes/accelerate_configs/zero1.yaml \
    recipes/Qwen/Qwen3-1.7B/grpo/config_pw_4gpu.yaml

# Option 2: Batch job
sbatch scripts/train_grpo__vllm_colocate.sub \
    recipes/accelerate_configs/zero1.yaml \
    recipes/Qwen/Qwen3-1.7B/grpo/config_pw_4gpu.yaml
```
