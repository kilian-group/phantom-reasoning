# Phantom Reasoning

## Setup instructions

Refer to cluster-specific instructions on [AIDA](docs/README_aida.md) and [Anvil](docs/README_anvil.md) and [Empire](docs/README_empire.md).

### Install `phantom-reasoning` in development mode

This repo uses external dependencies like SWI-Prolog.
From the root directory of this package:

```bash
# Assuming you are in ./phantom-reasoning git root repository

export CONDA_ENV_NAME="phantom-reasoning" # or whatever the name of your conda environment is

conda create -n $CONDA_ENV_NAME
conda activate $CONDA_ENV_NAME

# Install SWI-prolog
conda install conda-forge::swi-prolog
conda activate python=3.12
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

## PhantomWiki data

PhantomWiki paper used 3\*3 evaluation splits on Huggingface at `"kilian-group/phantom-wiki-v1"`: `depth_20_size_{50,500,5000}_seed_{1,2,3}`.

**NOTE**: For this project, we are using smaller universes, easy mode, no aggregation questions.
These are the easiest settings, due to small context length requirements for LLMs and easy questions, hence low GPU loads.

Concretely, we will use splits `depth_20_size_25_seed_*` created with `--easy-mode`.
We reserve seeds 1 through 10 for evaluation.
We will use seeds 11+ for training.

For these purposes, `depth_20_size_25_seed_{1,...,100}` are on G2 at `/share/nikola/phantom-wiki/data/wiki-v1-easy-depth_20_size_25.zip`.
We recommend copying them to `data/`:

```bash
mkdir -p data/
cp /share/nikola/phantom-wiki/data/wiki-v1-easy-depth_20_size_25.zip data/
# To transfer to another cluster: scp username@g2-login.coecis.cornell.edu:/share/nikola/phantom-wiki/data/wiki-v1-easy-depth_20_size_25.zip data/
cd data/
unzip wiki-v1-easy-depth_20_size_25.zip
cd ..
```

## Training on PhantomWiki data

> \[!NOTE\]
> If you are in multiple projects in the `mlcore` org, you will also need to set the `WANDB_PROJECT` environment variable. You can automatically load environment variables when your conda environment activates:
>
> ```bash
> conda env config vars set WANDB_ENTITY="mlcore"
> conda env config vars set WANDB_PROJECT="phantom-reasoning"
> ```

### GRPO settings

Recommendations for GRPO fine-tuning a Qwen3-1.7B model:

- 4 GPUs like A100s or H100s. 4 A6000s on G2 should also suffice, but you might need to adjust the batch size to avoid Out-Of-Memory errors.
  - `--gres=gpu:a100:4` on AIDA cluster.
  - `--gres=gpu:4` on Anvil cluster.
  - `--gres=gpu:a6000:4` on G2 cluster.
  - `--gres=gpu:4` on Empire cluster.
- `--mem=100GB` memory
- `-n 8` cores
- `-N 1` node
- `-t 24:00:00` hours

```bash
conda activate $CONDA_ENV_NAME

./scripts/create_train_grpo__vllm_colocate.sh <cluster_name>

./scripts/train_grpo__vllm_colocate.sub \
	/path/to/accelerate/config/file.yaml \
	/path/to/training/config/file.yaml
```

For example, running the following command full-finetunes a Qwen/Qwen3-1.7B model using GRPO.
Checkpoints are saved at `runs/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-1.7B/grpo/$USER/MMDD__<flags>/checkpoint-XX/`, and the final model is saved at `runs/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-1.7B/grpo/$USER/MMDD__<flags>/`

```bash
conda activate $CONDA_ENV_NAME

./scripts/create_train_grpo__vllm_colocate.sh anvil

./scripts/train_grpo__vllm_colocate.sub \
	recipes/accelerate_configs/zero1.yaml \
	recipes/Qwen/Qwen3-1.7B/grpo/config_4gpu__vllm_colocate.yaml
```

<details>

<summary>SFT settings (TODO)</summary>

### SFT on traces settings

From https://github.com/huggingface/open-r1?tab=readme-ov-file#sft:

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes 3 --config_file recipes/accelerate_configs/zero3.yaml \
	src/phantom_reasoner/sft_on_traces.py \
	--config recipes/Qwen/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
```

> \[!NOTE\]
> Add CLI arguments before `--config_file` to override the arguments in `zero3.yaml`

### SFT on docs settings

- Anmol's settings for full-finetuning https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct model:
  - `--gres=gpu:a100:4` on AIDA cluster. 4 A6000s on G2 should suffice. At bf16, accelerate-zero1, 1.5B model with the default settings uses ~70GB GPU memory.
  - `--mem=100GB` memory
  - `-n 8` cores

```bash
./scripts/train_sft_on_docs.sh \
	/path/to/accelerate/config/file.yaml \
	/path/to/training/config/file.yaml
```

For example, running the following command full-finetunes a 1.5B model using SFT on PhantomWiki documents with Zeroshot prompt.
The multi-gpu config distributes model and data across all GPUs.
Checkpoints are saved at `runs/Qwen/Qwen2.5-1.5B-Instruct/sft_on_docs/$USER/MMDD__<flags>/checkpoint-XX`.

```bash
./scripts/train_sft_on_docs.sh \
	recipes/accelerate_configs/multi_gpu.yaml \
	recipes/Qwen/Qwen2.5-1.5B-Instruct/sft_on_docs/config_base.yaml
```

</details>

## PhantomWiki evaluation

Since `phantom-wiki[eval]` is installed from github source, run the evaluation module like so:

```bash
CUDA_VISIBLE_DEVICES=0 python -m phantom_eval \
	--method cot \
	--server vllm \
	--inf_vllm_offline \
	--model_name /path/to/model/checkpoint \
	--dataset data/wiki-v1-easy-depth_20_size_25 \
	--split_list depth_20_size_25_seed_1 depth_20_size_25_seed_2 depth_20_size_25_seed_3 \
	--from_local \
	--inf_vllm_tensor_parallel_size 1 \
	--exclude_aggregation_questions \
	-od /path/to/output_for_preds/
```

Evaluating on just 1 GPU is faster than multiple GPUs due to communication overhead, so we can specify to only use the first GPU.

Then get numbers for the leaderboard:

```bash
python /path/to/phantom-wiki-installation/eval/format_leaderboard.py \
	-od /path/to/output_for_preds/ \
	--model_list /path/to/model/checkpoint \
	--size_list 25 \
	--method_list cot \
	--dataset data/wiki-v1-easy-depth_20_size_25 \
	--from_local
```

### GRPO training performance evolution

Evaluate all training checkpoints on an evaluation split of PhantomWiki with:

```bash
./scripts/pw-eval/evaluate_all_checkpoints.sh /path/to/checkpoint/parent
# for example, for this Qwen3-0.6B trained model:
./scripts/pw-eval/evaluate_all_checkpoints.sh runs/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-0.6B/grpo/$USER/MMDD__curr=random__prompt=cot/
```

Then we can produce how the model performance evolves as training progresses:

```bash
python scripts/plot_reasoning_during_training.py \
	-od runs/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-0.6B/grpo/$USER/MMDD__curr=random__prompt=cot/out \
	--model_list runs/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-0.6B/grpo/$USER/MMDD__curr=random__prompt=cot/ \
	--dataset data/wiki-v1-easy-depth_20_size_25 \
	--from_local
```

## Lighteval (GSM8k, ARC etc.)

```bash
python -m phantom_reasoner.utils.benchmarks \
	-cp /path/to/checkpoint \
	-t "leaderboard|arc:challenge|2|0,lighteval|arc:easy|2|0" \
	-od ./out-lighteval
```
