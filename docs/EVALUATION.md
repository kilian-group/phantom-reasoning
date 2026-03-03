# Evaluation

We evaluate LLMs on evaluation splits of various datasets. Predictions are written to output directories specified by `out__eval=<name>` or `out__train=<name>__eval=<name>`. Final checkpoint paths are listed in `scripts/final_plots/final_ckpts.yaml`.

## Real-world Wiki Datasets

Evaluate on 500-question splits of HotpotQA (`hp500`), 2Wiki (`2wiki500`), Musique (`msq500`), CofCA (`cofca500`), and SynthWorlds-RM (`synthrm500`):

```bash
# Replace hp500 with 2wiki500, msq500, cofca500, synthrm500
MODEL_NAMES="Qwen/Qwen3-1.7B" bash scripts/eval/wiki_eval_grpo.sh \
    out__eval=wiki \
    hp500 \
    minidev
```

<details>
<summary><strong>Evaluating trained checkpoints</strong></summary>

```bash
# Replace hp500 with 2wiki500, msq500, cofca500, synthrm500
MODEL_NAMES=$(python scripts/final_plots/get_model_names_of_final_ckpts.py --final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml --dataset_name base) \
    bash scripts/eval/wiki_eval_grpo.sh out__train=base__eval=wiki hp500 minidev

MODEL_NAMES=$(python scripts/final_plots/get_model_names_of_final_ckpts.py --final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml --dataset_name pw) \
    bash scripts/eval/wiki_eval_grpo.sh out__train=pw__eval=wiki hp500 minidev

MODEL_NAMES=$(python scripts/final_plots/get_model_names_of_final_ckpts.py --final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml --dataset_name gsminf) \
    bash scripts/eval/wiki_eval_grpo.sh out__train=gsminf__eval=wiki hp500 minidev

MODEL_NAMES=$(python scripts/final_plots/get_model_names_of_final_ckpts.py --final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml --dataset_name rg-family_relationships) \
    bash scripts/eval/wiki_eval_grpo.sh out__train=rg-family_relationships__eval=wiki hp500 minidev
```

Plot transfer performance to real-world wiki datasets:

```bash
python examples/wiki/create_bar_plots_performance_transfer.py \
    --final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml \
    --base_model_preds_dir "out__train=base__eval=wiki" \
    --pw_model_preds_dir "out__train=pw__eval=wiki" \
    --gsminf_model_preds_dir "out__train=gsminf__eval=wiki" \
    --figures_dir "scripts/final_plots/figures"
```

Evaluate all intermediate training checkpoints to track overfitting:

```bash
# Replace hp500 with 2wiki500, msq500, cofca500, synthrm500
bash scripts/eval/wiki_eval_all_ckpts.sh \
    ./scratch/runs/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-1.7B/grpo/$USER/MMDD__<flags> \
    hp500 \
    minidev \
    Qwen/Qwen3-1.7B \
    pw

bash scripts/eval/wiki_eval_all_ckpts.sh \
    ./scratch/runs/data/gsm-infinite-train/zero_context/realistic/Qwen/Qwen3-1.7B/grpo/$USER/MMDD__<flags> \
    hp500 \
    minidev \
    Qwen/Qwen3-1.7B \
    gsminf
```

Plot transfer performance as a function of training steps:

```bash
python examples/wiki/plot_all_wiki_scaling_final_ckpts.py \
    --final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml \
    --model_name "Qwen/Qwen3-1.7B" \
    --data_dir data \
    --dataset hp500 \
    --split minidev \
    --figures_dir "scripts/final_plots/figures"
```

</details>

## Reasoning Evolution Plots

Evaluate all intermediate training checkpoints on synthetic evaluation splits and plot how model performance evolves as a function of question difficulty as training progresses:

```bash
# TODO: add command for evaluating all ckpts on pw and gsminf
export DATASET="data/wiki-v1-easy-depth_30_size_25"
export PW_SPLITS="depth_30_size_25_seed_1 depth_30_size_25_seed_2 depth_30_size_25_seed_3"
bash scripts/eval/pw_eval_all_ckpts.sh \
    ./scratch/runs/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-1.7B/grpo/$USER/MMDD__<flags> \
    Qwen/Qwen3-1.7B \
    pw

bash scripts/eval/gsminf_eval_all_ckpts.sh \
    ./scratch/runs/data/gsm-infinite-train/zero_context/realistic/Qwen/Qwen3-1.7B/grpo/$USER/MMDD__<flags> \
    Qwen/Qwen3-1.7B \
    gsminf
```

<details>
<summary><strong>Plotting reasoning evolution for trained checkpoints</strong></summary>

```bash
python scripts/final_plots/create_reasoning_evolution.py \
    --final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml \
    --base_model_names_to_plot "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" \
    --figures_dir "scripts/final_plots/figures"
```

To visualize how frequently model chain-of-thoughts mention intermediate hop answers on `msq500`/`cofca500`:

```bash
# Replace msq500 with cofca500
# 1. Create CSV
python examples/wiki/create_reasoning_evolution.csv \
    --final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml \
    --train_dataset pw \
    --eval_dataset msq500 \
    --figures_dir "scripts/final_plots/figures"

# 2. Plot
python scripts/final_plots/create_reasoning_evolution_realworld.py \
    --train_dataset pw \
    --eval_dataset msq500 \
    --base_model_names_to_plot "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" \
    --figures_dir "scripts/final_plots/figures"
```

</details>
