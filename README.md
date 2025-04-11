# Phantom Reasoning

## Setup instructions

### Install `phantom-reasoning` in development mode

This repo uses external dependencies like SWI-Prolog.
From the root directory of this package:

```bash
# Create new environment
conda create -n phantom-reasoning python=3.12
conda activate phantom-reasoning

# Install SWI-prolog. On linux:
conda install conda-forge::swi-prolog
# or on mac:
brew install swi-prolog

# Install phantom-wiki and phantom-reasoning in editable modes
git clone git@github.com:kilian-group/phantom-wiki.git
cd phantom-wiki
pip install -e ".[eval]"

cd ..
git clone git@github.com:anmolkabra/phantom-reasoning.git
cd phantom-reasoning
pip install -e ".[dev]"

pip install flash-attn --no-build-isolation

pre-commit install
```

> \[!NOTE\]
> The `flash-attn` dependency restricts the system requirements to Linux with CUDA/ROCm toolkit support.

For reference, see [example-environment.yml](./example-environment.yml) for exact package versions.

## PhantomWiki data

PhantomWiki paper used 3\*3 evaluation splits on Huggingface at `"kilian-group/phantom-wiki-v1"`: `depth_20_size_{50,500,5000}_seed_{1,2,3}`.

**NOTE**: For this project, we are using smaller universes, smaller question depth, easy mode, no aggregation questions.
These are the easiest settings, due to small context length requirements for LLMs and easy questions, hence low GPU loads.

Concretely, we will use splits `depth_10_size_25_seed_*` with created with `--easy-mode` and no aggregation questions.
We reserve seeds 1 through 10 for evaluation.
We will use seeds 11+ for training.

For these purposes, `depth_10_size_25_seed_{1,...,100}` are on G2 at `/share/nikola/phantom-wiki/data/wiki-v1-easy-no-agg.zip`.
We recommend copying them to `data/`:

```bash
mkdir -p data/
cp /share/nikola/phantom-wiki/data/wiki-v1-easy-no-agg.zip data/
# To transfer to another cluster: scp username@g2-login.coecis.cornell.edu:/share/nikola/phantom-wiki/data/wiki-v1-easy-no-agg.zip data/
cd data/
unzip wiki-v1-easy-no-agg.zip
cd ..
```

## Training on PhantomWiki data

> \[!NOTE\]
> If you are in multiple teams, you will also need to set the `WANDB_ENTITY` environment variable (e.g., `conda env config vars set WANDB_ENTITY=phantom-reasoner`)

### SFT on traces settings

From https://github.com/huggingface/open-r1?tab=readme-ov-file#sft:

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes 3 --config_file recipes/accelerate_configs/zero3.yaml \
	src/phantom_reasoner/sft_on_traces.py \
	--config recipes/qwen2.5-1.5b-instruct/sft/config_demo.yaml
```

> \[!NOTE\]
> Add CLI arguments before `--config_file` to override the arguments in `zero3.yaml`

### SFT on docs settings

- Anmol's settings for full-finetuning https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct model:
  - --gres=gpu:a100:4 on AIDA cluster. 4 A6000s on G2 should suffice. At bf16, accelerate-zero1, 1.5B model with the default settings uses ~70GB GPU memory.
  - --mem=100GB memory
  - -n 8 cores

```bash
./scripts/train_sft_on_docs.sh \
	/path/to/accelerate/config/file.yaml \
	/path/to/training/config/file.yaml \
	--output_dir /path/to/output_dir/
```

For example, running the following command full-finetunes a 1.5B model using SFT on PhantomWiki documents with Zeroshot prompt.
The multi-gpu config distributes model and data across all GPUs.
Checkpoints are saved at `runs/sft_on_docs/username/qwen1.5b__MMDD__flags/checkpoint-XX`.

```bash
./scripts/train_sft_on_docs.sh \
	recipes/accelerate_configs/multi_gpu.yaml \
	recipes/qwen2.5-1.5b-instruct/sft_on_docs/config_base.yaml \
	--output_dir runs/sft_on_docs/username/qwen1.5b__MMDD__flags
```

### GRPO settings

- Anmol's settings for full-finetuning https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct model:
  - --gres=gpu:a100:4 on AIDA cluster. 4 A6000s on G2 should suffice. At bf16, accelerate-zero1, 1.5B model with the default settings uses ~70GB GPU memory.
  - --mem=100GB memory
  - -n 8 cores

```bash
./scripts/train_grpo.sh \
	/path/to/accelerate/config/file.yaml \
	/path/to/training/config/file.yaml \
	--prompt_method cot \
	--output_dir /path/to/output_dir/
```

For example, running the following command full-finetunes a 1.5B model using GRPO.
The multi-gpu config distributes model and data across all but last GPU---the last GPU in your allocation is reserved for generating with vllm.
Checkpoints are saved at `runs/grpo/username/qwen1.5b__MMDD__flags/checkpoint-XX`.

```bash
./scripts/train_grpo.sh \
	recipes/accelerate_configs/zero1.yaml \
	recipes/qwen2.5-1.5b-instruct/grpo/config_base.yaml \
	--prompt_method cot \
	--output_dir runs/grpo/username/qwen1.5b__MMDD__flags
```

## PhantomWiki evaluation

Since `phantom-wiki[eval]` is installed from github source, run the evaluation module like so:

Evaluating on just 1 GPU is faster than multiple GPUs due to communication overhead, so we can specify to only use the first GPU.

```bash
CUDA_VISIBLE_DEVICES=0 python -m phantom_eval \
	--method cot \
	--server vllm \
	--model_name /path/to/model/checkpoint \
	--dataset data/wiki-v1-easy-no-agg \
	--split_list depth_10_size_25_seed_1 depth_10_size_25_seed_2 depth_10_size_25_seed_3 depth_10_size_25_seed_4 depth_10_size_25_seed_5 \
	--from_local \
	--inf_vllm_tensor_parallel_size 1 \
	-od /path/to/output_for_preds/
```

Then get numbers for the leaderboard:

```bash
python examples/phantomwiki_v1/format_leaderboard.py \
	-od /path/to/output_for_preds/ \
	--model_list /path/to/model/checkpoint \
	--size_list 25 \
	--method_list cot \
	--filter_by_depth 10 \
	--dataset data/wiki-v1-easy-no-agg --from_local
```
