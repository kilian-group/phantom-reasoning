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

There are 3\*3 evaluation splits on Huggingface at `"kilian-group/phantom-wiki-v1"`: `depth_20_size_{50,500,5000}_seed_{1,2,3}`.
We care about `depth_20_size_50_seed_{1,2,3}` as the bigger universe sizes do
not fit in 32K context length models.

We can train on 10 other seeds `depth_20_size_50_seed_{10,...,19}`, which are
saved on G2, as `/share/nikola/phantom-wiki/data/wiki-v1.zip` and
`/share/nikola/phantom-wiki/data/wiki-v1-easy.zip`.
We recommend copying them to `data/`:

```bash
mkdir -p data/
cp /share/nikola/phantom-wiki/data/wiki-v1.zip data/
# To transfer to another cluster: scp username@g2-login.coecis.cornell.edu:/share/nikola/phantom-wiki/data/wiki-v1.zip data/
cd data/
unzip wiki-v1.zip
cd ..
```

## SFT settings

From https://github.com/huggingface/open-r1?tab=readme-ov-file#sft:

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes 3 --config_file recipes/accelerate_configs/zero3.yaml \
	src/phantom_reasoner/sft_on_traces.py \
	--config recipes/qwen2.5-1.5b-instruct/sft/config_demo.yaml
```

> \[!NOTE\]
> Add CLI arguments before `--config_file` to override the arguments in `zero3.yaml`

## GRPO settings

On wandb.ai under the `mlcore` org, create your own project by setting the `WANDB_PROJECT` environment variable.
Then run `wandb login`, and paste the API key given from the website.

```bash
export WANDB_PROJECT="grpo"
# or
conda env config vars set WANDB_PROJECT=grpo
```

> \[!NOTE\]
> If you are in multiple teams, you will also need to set the `WANDB_ENTITY` environment variable (e.g., `conda env config vars set WANDB_ENTITY=phantom-reasoner`)

- Anmol's settings for full-finetuning https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct model:
  - --gres=gpu:a100:4 on AIDA cluster. 4 A600s on G2 should suffice. At bf16, no accelerate, 0.5B model with the default settings uses 55GB GPU memory.
  - --mem=100GB memory
  - -n 8 cores

```bash
./scripts/train_grpo.sh /path/to/training/config/file.yaml --prompt_method cot --output_dir /path/to/output_dir/
```

For example,

```bash
./scripts/train_grpo.sh recipes/qwen2.5-0.5b-instruct/grpo/config_base.yaml \
  --prompt_method cot \
  --output_dir runs/grpo/username/qwen0.5b__MMDD__flags
```

## PhantomWiki evaluation

Anmol: phantom-eval does not work with the latest `vllm==0.8.1` right now on
AIDA cluster. But it does work with `vllm==0.6.6`, which was the latest vllm
version when we released PhantomWiki paper.
Till we get phantom-eval working with the latest `vllm`, I recommend creating a
**separate** python environment with `pip install vllm==0.6.6` for running the
phantom-wiki evaluation.

Concretely, do this if you don't have a separate phantom-wiki environment:

```bash
conda create -n phantom-wiki python=3.12
conda activate phantom-wiki

conda install conda-forge::swi-prolog
cd phantom-wiki
pip install "vllm==0.6.6"
pip install -e ".[eval]"
```

Now, run phantom-eval in this new environment:

```bash
python -m phantom_eval \
	--method cot \
	--server vllm \
	--model_name /path/to/model/checkpoint/ \
	-od /path/to/output_for_preds/
```
