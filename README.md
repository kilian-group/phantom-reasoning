# Phantom Reasoner

## Setup instructions

### Install special dependencies

<!-- TODO create a setup.py script to handle this automatically -->

```bash
pip install flash-attn --no-build-isolation
```

> \[!NOTE\]
> The `flash-attn` dependency restricts the system requirements to Linux with CUDA/ROCm toolkit support.

### Install `phantom-reasoner` in development mode

From the root directory of this package:

```bash
pip install -e .
```

## SFT settings

From https://github.com/huggingface/open-r1?tab=readme-ov-file#sft:

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes 3 --config_file recipes/accelerate_configs/zero3.yaml \
    src/phantom_reasoner/sft.py \
    --config recipes/qwen2.5-1.5b-instruct/sft/config_demo.yaml
```

> \[!NOTE\]
> Add CLI arguments before `--config_file` to override the arguments in `zero3.yaml`

## GRPO settings

On wandb.ai, create a new project, e.g. `phantom-reasoning`. Then run `wandb login`, and paste the API key given from the website.

```bash
python -m phantom_reasoner.grpo --dataset_name "kilian-group/phantom-wiki-v1" --use_vllm --model_name_or_path "Qwen/Qwen2.5-0.5B-Instruct" --output_dir out-0319-grpo-testrun
```
