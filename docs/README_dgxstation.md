# Setup instructions for NVIDIA DGX

We run all these commands from the root of this repository `./phantom-reasoning/` (and not `./docs/` for instance).

1. First, follow the instructions in [README.md] to install the repository in development mode.
For convenience, we write them down here.

```bash
export CONDA_ENV_NAME="phantom-reasoning" # or whatever the name of your conda environment is

conda create -n $CONDA_ENV_NAME
conda activate $CONDA_ENV_NAME

conda install conda-forge::swi-prolog
conda install python=3.12
pip install uv

# Attempt at installing package requirements, trl[vllm] from github source to allow for vllm
# trl[vllm] from git source
# vllm>=0.13.0 that supports CUDA 13
uv pip install -e .[dev]
uv pip install flash-attn --no-build-isolation
pre-commit install
```

2. Run a GRPO experiment on Qwen3-4B model:

```bash
./scripts/create_train_grpo__vllm_colocate.sh

./scripts/train_grpo__vllm_colocate.sub \
    recipes/accelerate_configs/zero1.yaml \
    recipes/Qwen/Qwen3-4B/grpo/config_pw_1gpu.yaml
```
