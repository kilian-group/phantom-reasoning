# Phantom Reasoner

## Setup instructions

Create a conda environment: 
```bash
conda create -n rsn python=3.12
```
For reference, see [example-environment.yml](./example-environment.yml) for exact package versions.

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

On wandb.ai under the `mlcore` org, create your own project by setting the `WANDB_PROJECT` environment variable.
Then run `wandb login`, and paste the API key given from the website.

```bash
export WANDB_PROJECT="phantom-reasoner"
# or
conda env config vars set WANDB_PROJECT=phantom-reasoner
```

- Anmol's settings for full-finetuning https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct model:
  - --gres=gpu:a100:1 on AIDA cluster. 2 A600s on G2 should suffice. At bf16, no accelerate, 0.5B model with the default settings uses 55GB GPU memory.
  - --mem=100GB memory
  - -n 8 cores

```bash
./scripts/train_grpo.sh --prompt_method cot --output_dir /path/to/output_dir/
```
