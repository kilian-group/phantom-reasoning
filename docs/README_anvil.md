# Setup instructions for Anvil Purdue

We run all these commands from the root of this repository `./phantom-reasoning/` (and not `./docs/` for instance).

1. First, install the conda environment. The Anvil cluster provides shared conda installation, which we recommend over installing your own personal conda (Anmol: there were issues with python paths with personal conda installations).

```bash
module load conda
./scripts/anvil/load_modules_cuda.sh
```

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

# NOTE as of 2025-08-11: flash-attn does not seem to work on Anvil because of old GLIBC version 2.28
# (flash-attn==2.8.2 requires GLIBC 2.32 or higher)
# NOTE: installing flash-attn will require a GPU allocation
# so skip to the end for getting an interactive GPU allocation to install flash-attn
uv pip install flash-attn --no-build-isolation

pre-commit install
```

2. Set environment vars in the conda environment.

> \[!NOTE\]
> Home paths `~/` only have 25GB on Anvil, so it's extremely important that you set huggingface datasets, models, checkpoints to scratch. If anything needs to be shared (e.g. datasets), we save them to the shared directory. The conda and pip environments (folders `~/.conda/` and `~/.cache/pip`) will take up 10GB or so with just this 1 project.

```bash
conda env config vars set ANVIL_PROJECT_ID="250102"
conda env config vars set RUN_BASE_DIR="$SCRATCH/phantom-reasoning/"
conda env config vars set WANDB_ENTITY="mlcore"
conda env config vars set WANDB_PROJECT="phantom-reasoning"
conda env config vars set HF_HOME="$SCRATCH/huggingface"
conda env config vars set CONDA_ENV_NAME=$CONDA_ENV_NAME # so the env name is available automatically when activated
conda env config vars set USER_EMAIL="user@email.com" # for emailing when allocations become available

conda deactivate
conda activate $CONDA_ENV_NAME
```

3. Setup wandb login: `wandb login` and paste the API key from the `mlcore` organization. Contact Anmol if you don't have access to the `mlcore` org.

4. Create a symlink to the data, runs, and eval repositories and set up the output directory.

```bash
# TODO: Nannan's email says the project folder will migrated soon
# We should ideally use $PROJECT/phantom-reasoning/... when the migration is complete
ln -s /anvil/projects/ai$ANVIL_PROJECT_ID/phantom-reasoning/data .
ln -s /anvil/projects/ai$ANVIL_PROJECT_ID/phantom-reasoning/runs .
ln -s /anvil/projects/ai$ANVIL_PROJECT_ID/phantom-reasoning/eval .
mkdir $RUN_BASE_DIR
```

5. Run a GRPO experiment on Qwen3-1.7B model:

```bash
module load conda
conda activate $CONDA_ENV_NAME # to get ANVIL_PROJECT_ID variable

# Option 1: Interactive
salloc -A nairr$ANVIL_PROJECT_ID-ai -p ai --gres=gpu:4 -n 16 -N 1 --mem=100GB -t 12:00:00 --mail-type=all --mail-user=$USER_EMAIL

# After getting an allocation:
module load conda
./scripts/anvil/load_modules_cuda.sh
conda activate $CONDA_ENV_NAME

./scripts/aida/train_grpo__vllm_colocate.sh \
    recipes/accelerate_configs/zero1.yaml \
    recipes/Qwen/Qwen3-1.7B/grpo/config_4gpu__vllm_colocate.yaml

# Option 2: Batch job
sbatch -A nairr$ANVIL_PROJECT_ID-ai --mail-type=all --mail-user=$USER_EMAIL scripts/anvil/train_grpo__vllm_colocate.sh \
    recipes/accelerate_configs/zero1.yaml \
    recipes/Qwen/Qwen3-1.7B/grpo/config_4gpu__vllm_colocate.yaml
```
