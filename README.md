# Phantom Reasoning

We train LLMs with GRPO on synthetic multi-hop reasoning datasets such as PhantomWiki and GSM-Infinite, and show performance improvement to real-world multi-hop datasets such as HotpotQA, 2Wiki, and Musique.

## Installation

The Anvil cluster provides shared conda installation, which we recommend over installing in your home directory.
This avoids python path conflicts and saves limited memory in `~/`.

1. Load the shared conda installation.

```bash
./scripts/anvil/load_modules_cuda.sh
```

2. Install SWI-prolog, python, uv package manager, phantom-wiki, phantom-reasoning, flash-attn, and pre-commit.

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
uv pip install flash-attn --no-build-isolation

pre-commit install
```

3. Set environment variables in the conda environment, so they are automatically loaded on activating the environment. We assume these flags are set later in the README.

```bash
conda env config vars set ANVIL_PROJECT_ID="nairr250102"
conda env config vars set RUN_BASE_DIR="$SCRATCH/phantom-reasoning"
conda env config vars set HF_HOME="$SCRATCH/huggingface"
conda env config vars set CONDA_ENV_NAME=$CONDA_ENV_NAME # so the env name is available automatically when activated

conda deactivate
conda activate $CONDA_ENV_NAME
```

## Dataset splits

We use several datasets for training and evaluating multi-hop reasoning:

- **PhantomWiki**: A synthetic, large-scale multi-hop QA dataset with variable universe size, depth, and seed. Used for both training and in-depth evaluation of reasoning skills.

- **GSM-Infinite**: An extension of GSM8K with high-complexity arithmetic and compositional reasoning questions. Used for both training and in-depth evaluation of arithmetic skills of models.

- **HotpotQA, 2Wiki, Musique**: Real-world Wikipedia-based datasets that require multi-hop reasoning over unstructured text. Used to evaluate real-world transfer and generalization to natural language settings.

All datasets are split into training and evaluation sets to test both in-domain and out-of-domain generalization.
The Anvil cluster contains these splits in shared storage, which we symlink under `data/`:

```bash
ln -s /anvil/projects/x-$ANVIL_PROJECT_ID/phantom-reasoning/data .
```

On clusters without data in shared storage, please copy dataset splits from the G2 cluster as described below.

<details>
  <summary><strong>PhantomWiki</strong></summary>

PhantomWiki paper used 3×3 evaluation splits available on Huggingface at `kilian-group/phantom-wiki-v1`: `depth_20_size_{50,500,5000}_seed_{1,2,3}`.

> \[!NOTE\]
> For this project, we use smaller universes, easy mode, and no aggregation questions. These are the easiest settings, due to small context length requirements for LLMs and easy questions, resulting in low GPU loads.

Specifically, we use splits `depth_20_size_25_seed_*` created with `--easy-mode`.

- **Seeds 1-10**: reserved for evaluation
- **Seeds 11+**: used for training

The datasets `depth_20_size_25_seed_{1,...,100}` are found on G2 at `/share/nikola/phantom-wiki/data/wiki-v1-easy-depth_20_size_25.zip`.

```bash
mkdir -p data/
scp username@g2-login.coecis.cornell.edu:/share/nikola/phantom-wiki/data/wiki-v1-easy-depth_20_size_25.zip data/

cd data/
unzip wiki-v1-easy-depth_20_size_25.zip
cd ..
```

</details>

<details>
  <summary><strong>GSM-Infinite</strong></summary>

We have generated GSM-infinite data and stored it on G2. See `gsm_realistic/README.md` for instructions to generate your own data.

```bash
mkdir -p data/
scp username@g2-login.coecis.cornell.edu:/share/nikola/phantom-reasoning/data/gsm-infinite-train.zip data/
scp username@g2-login.coecis.cornell.edu:/share/nikola/phantom-reasoning/data/gsm-infinite-eval.zip data/

cd data/
unzip gsm-infinite-train.zip
unzip gsm-infinite-eval.zip
cd ..
```

</details>

<details>
  <summary><strong>Real-world Wiki Datasets</strong></summary>

We collected HotpotQA (`hp`), 2Wiki (`2wiki`), and Musique (`msq`) datasets, and subsampled 500 questions from the evaluation dataset in `hp500, 2wiki500, msq500`.

```bash
mkdir -p data/
for d in "hp" "2wiki" "msq"; do scp username@g2-login.coecis.cornell.edu:/share/nikola/phantom-reasoning/data/${d}.zip data/; done
for d in "hp" "2wiki" "msq"; do scp username@g2-login.coecis.cornell.edu:/share/nikola/phantom-reasoning/data/${d}500.zip data/; done

cd data/
for d in "hp" "2wiki" "msq"; do unzip ${d}.zip; done
for d in "hp" "2wiki" "msq"; do unzip ${d}500.zip; done
cd ..
```

</details>

## LLM training instructions

We describe training instructions specific to the Anvil cluster, which contain all trained checkpoints.
To train models on other clusters---please refer to cluster-specific instructions on [AIDA](docs/README_aida.md) and [Empire](docs/README_empire.md).

1. Setup environment variables for model checkpoints and wandb. To set up wandb logging, run `wandb login` and paste the API key from your account's organization.

```bash
conda env config vars set USER_EMAIL="user@email.com" # for emailing when slurm allocations become available
conda env config vars set WANDB_ENTITY="organization"
conda env config vars set WANDB_PROJECT="phantom-reasoning"
```

2. Create a symlink to directories with training checkpoints.

```bash
# experiment runs in scratch, linked to ./scratch
mkdir -p $RUN_BASE_DIR/runs
ln -s $RUN_BASE_DIR ./scratch

# shared models and evals linked to ./share
ln -s /anvil/projects/x-$ANVIL_PROJECT_ID/phantom-reasoning share
```

> \[!NOTE\] We trained several LLMs on PhantomWiki and GSM-infinite data, and share all checkpoints and predictions in paths listed in `./scripts/final_plots/final_ckpts.yaml`.

3. Run a GRPO training experiment on Qwen3-1.7B model with PhantomWiki data. We provide a script `./scripts/create_train_grpo__vllm_colocate.sh <cluster_name>` to create a bash slurm submission file, adding default variables for the specified cluster. This script creates `./scripts/train_grpo__vllm_colocate.sub` that executes a training job on a configuration. For instance,

   1. Using `salloc`:

   ```bash
   salloc -A $ANVIL_PROJECT_ID-ai -p ai --gres=gpu:4 -n 16 -N 1 --mem=100GB -t 12:00:00 --mail-type=all --mail-user=$USER_EMAIL

   # After getting an allocation:
   ./scripts/anvil/load_modules_cuda.sh
   conda activate $CONDA_ENV_NAME

   ./scripts/create_train_grpo__vllm_colocate.sh anvil

   ./scripts/train_grpo__vllm_colocate.sub \
   	recipes/accelerate_configs/zero1.yaml \
   	recipes/Qwen/Qwen3-1.7B/grpo/config_pw_4gpu.yaml
   ```

   2. Using `sbatch`:

   ```bash
   ./scripts/create_train_grpo__vllm_colocate.sh anvil

   ./scripts/train_grpo__vllm_colocate.sub \
   	recipes/accelerate_configs/zero1.yaml \
   	recipes/Qwen/Qwen3-1.7B/grpo/config_pw_4gpu.yaml
   ```

   Checkpoints and final model are saved at `./scratch/runs/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-1.7B/grpo/$USER/MMDD__<flags>`.

> \[!NOTE\] YAML configurations for other datasets:
> TODO add reasoning gym
>
> - `recipes/Qwen/Qwen3-1.7B/grpo/config_gsminfinite_4gpu.yaml`
> - `recipes/Qwen/Qwen3-1.7B/grpo/config_hp_4gpu.yaml` (on HotpotQA training data splits)
> - `recipes/Qwen/Qwen3-1.7B/grpo/config_2wiki_4gpu.yaml` (on 2Wiki training data splits)
> - `recipes/Qwen/Qwen3-1.7B/grpo/config_msq_4gpu.yaml` (on Musique training data splits)

> \[!NOTE\] Similarly, YAML configurations for other LLMs:
>
> - `recipes/Qwen/Qwen3-0.6B/grpo/config_pw_4gpu.yaml`
> - `recipes/Qwen/Qwen2.5-1.5B-Instruct/grpo/config_pw_4gpu.yaml`
> - `recipes/microsoft/Phi-4-mini-reasoning/grpo/config_pw_4gpu.yaml`

## LLM evaluation instructions

We evaluate LLMs on evaluation splits of various datasets, such as 500 questions from evaluation splits of HotpotQA, 2wiki, and Musique that exist in `./data/hp500, ./data/2wiki500, ./data/msq500`.
We provide code to load these splits, get LLM predictions, and tabulate results in CSV files and output to the terminal.

### Real-world Wiki datasets

We can evaluate LLMs on real-world wiki datasets with the `scripts/eval/wiki_eval_grpo.sh` script:

```bash
# Assuming you have 1 GPU
# Replace hp500 with 2wiki500, msq500
MODEL_NAMES="Qwen/Qwen3-1.7B" bash scripts/eval/wiki_eval_grpo.sh \
	out__eval=wiki \
	hp500 \
	minidev
```

#### Evaluating trained checkpoints

We can also evaluate all trained checkpoints and their base models listed in `scripts/final_plots/final_ckpts.yaml`.

```bash
# Replace hp500 with 2wiki500, msq500
MODEL_NAMES=$(python3 scripts/final_plots/get_model_names_of_final_ckpts.py --final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml --dataset_name base) \
	bash scripts/eval/wiki_eval_grpo.sh out__train=base__eval=wiki hp500 minidev

MODEL_NAMES=$(python3 scripts/final_plots/get_model_names_of_final_ckpts.py --final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml --dataset_name pw) \
	bash scripts/eval/wiki_eval_grpo.sh out__train=pw__eval=wiki hp500 minidev

MODEL_NAMES=$(python3 scripts/final_plots/get_model_names_of_final_ckpts.py --final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml --dataset_name gsminf) \
	bash scripts/eval/wiki_eval_grpo.sh out__train=gsminf__eval=wiki hp500 minidev
```

For LLMs trained with PhantomWiki and GSM-Infinite, we provide a script to plot transfer performance to real-world wiki datasets:

```bash
python examples/wiki/create_bar_plots_performance_transfer.py \
	--final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml \
	--base_model_preds_dir "out__train=base__eval=wiki" \
	--pw_model_preds_dir "out__train=pw__eval=wiki" \
	--gsminf_model_preds_dir "out__train=gsminf__eval=wiki" \
	--figures_dir "scripts/final_plots/figures"
```

<p align="center">
  <img src="scripts/final_plots/figures/f1_transfer_performance_all.png" alt="Transfer performance bar plot"/>
</p>

TODO other plots

### PhantomWiki datasets

We can evaluate LLMs on PhantomWiki datasets with the `scripts/eval/pw_eval_grpo.sh` script:

```bash
TODO fix below
# Assuming you have 1 GPU
MODEL_NAMES="Qwen/Qwen3-1.7B" bash scripts/eval/pw_eval_grpo.sh \
	out__eval=wiki \
	hp500 \
	minidev
```

#### Evaluating trained checkpoints

We can also evaluate all trained checkpoints and their base models listed in `scripts/final_plots/final_ckpts.yaml`.

```bash
# Replace hp500 with 2wiki500, msq500
MODEL_NAMES=$(python3 scripts/final_plots/get_model_names_of_final_ckpts.py --final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml --dataset_name base) \
	bash scripts/eval/wiki_eval_grpo.sh out__train=base__eval=wiki hp500 minidev

MODEL_NAMES=$(python3 scripts/final_plots/get_model_names_of_final_ckpts.py --final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml --dataset_name pw) \
	bash scripts/eval/wiki_eval_grpo.sh out__train=pw__eval=wiki hp500 minidev

MODEL_NAMES=$(python3 scripts/final_plots/get_model_names_of_final_ckpts.py --final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml --dataset_name gsminf) \
	bash scripts/eval/wiki_eval_grpo.sh out__train=gsminf__eval=wiki hp500 minidev
```

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

Evaluate all training checkpoints on evaluation splits of PhantomWiki and plot how model performance evolves as a function of question difficulty, as training progresses.

```bash
./scripts/eval/pw_eval_all_ckpts.sh /path/to/checkpoint/parent <base_model_name> <training_dataset_name>
# for example, for this Qwen3-0.6B trained model:
./scripts/eval/pw_eval_all_ckpts.sh runs/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-0.6B/grpo/$USER/MMDD__curr=random__prompt=cot Qwen/Qwen3-0.6B pw
```

### Scaling plots for wiki datasets

Evaluate all training checkpoints on evaluation datasets of various wiki datasets (HP, 2Wiki, MSQ) and plot how model performance on wiki datasets evolves as training progresses.

```bash
./scripts/eval/wiki_eval_all_ckpts.sh /path/to/checkpoint/parent <dataset> <split> <base_model_name> <training_dataset_name>
# for example, for this Qwen3-0.6B trained model:
./scripts/eval/wiki_eval_all_ckpts.sh runs/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-0.6B/grpo/$USER/MMDD__curr=random__prompt=cot hp500 minidev Qwen/Qwen3-0.6B pw
```

### GSM-Infinite evaluation

We evaluate on the huggingface evaluation set of GSM-Infinite as follows:

```bash
./scripts/eval/gsminf_eval_grpo.py /path/to/output_for_preds/
```
