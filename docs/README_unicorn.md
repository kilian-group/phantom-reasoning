# Setup instructions for the Unicorn cluster

We run all these commands from the root of this repository `./phantom-reasoning/` (and not `./docs/` for instance).

- Follow the instructions in [README.md] to install the repository in development mode and set the environment flags.

- Create a symlink to the data and runs directories.

```bash
./scripts/setup_conda_env_vars.sh $CONDA_ENV_NAME unicorn

# shared data
ln -s /share/nikola/phantom-reasoning/data .

# shared models and evals
ln -s /share/aimi/phantom-reasoning ./share
```

- Run a GRPO experiment on Qwen3-1.7B model:

```bash
# Option 1: Interactive
salloc -p aimi --gres=gpu:2 -n 16 -N 1 --mem=100GB -t 12:00:00 --mail-type=all --mail-user=$USER_EMAIL

# After getting an allocation:
conda activate $CONDA_ENV_NAME

./scripts/create_train_grpo__vllm_colocate.sh unicorn

./scripts/train_grpo__vllm_colocate.sub \
    recipes/accelerate_configs/zero1.yaml \
    recipes/Qwen/Qwen3-1.7B/grpo/config_pw_2gpu.yaml

# Option 2: Batch job
./scripts/create_train_grpo__vllm_colocate.sh unicorn
sbatch scripts/train_grpo__vllm_colocate.sub \
    recipes/accelerate_configs/zero1.yaml \
    recipes/Qwen/Qwen3-1.7B/grpo/config_pw_2gpu.yaml
```
