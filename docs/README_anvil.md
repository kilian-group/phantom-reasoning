# Setup instructions for Anvil Purdue

We run all these commands from the root of this repository `./phantom-reasoning/` (and not `./docs/` for instance).

- The Anvil cluster provides shared conda installation, which we recommend over installing your own personal conda.

```bash
module load conda
./scripts/anvil/load_modules_cuda.sh
```

- Follow the instructions in [README.md] to install the repository in development mode and set the environment flags.
  NOTE as of 2025-08-11: flash-attn does not seem to work on Anvil because of old GLIBC version 2.28.

- Create a symlink to the data and runs directories.

```bash
# experiment runs in scratch, not shared
mkdir -p $RUN_BASE_DIR/runs
ln -s $RUN_BASE_DIR ./scratch

# shared data
ln -s /anvil/projects/x-$ANVIL_PROJECT_ID/phantom-reasoning/data .

# shared models and evals
ln -s /anvil/projects/x-$ANVIL_PROJECT_ID/phantom-reasoning ./share
```

- Run a GRPO experiment on Qwen3-1.7B model:

```bash
module load conda
conda activate $CONDA_ENV_NAME # to get ANVIL_PROJECT_ID variable

# Option 1: Interactive
salloc -A $ANVIL_PROJECT_ID-ai -p ai --gres=gpu:4 -n 16 -N 1 --mem=100GB -t 12:00:00 --mail-type=all --mail-user=$USER_EMAIL

# After getting an allocation:
module load conda
./scripts/anvil/load_modules_cuda.sh
conda activate $CONDA_ENV_NAME

./scripts/create_train_grpo__vllm_colocate.sh anvil

./scripts/train_grpo__vllm_colocate.sub \
    recipes/accelerate_configs/zero1.yaml \
    recipes/Qwen/Qwen3-1.7B/grpo/config_pw_4gpu.yaml

# Option 2: Batch job
./scripts/create_train_grpo__vlm_colocate.sh anvil
sbatch scripts/train_grpo__vllm_colocate.sub \
    recipes/accelerate_configs/zero1.yaml \
    recipes/Qwen/Qwen3-1.7B/grpo/config_pw_4gpu.yaml
```
