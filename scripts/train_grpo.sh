#!/usr/bin/env bash
#SBATCH --job-name=grpo
#SBATCH --output=logs/grpo-%j.out
#SBATCH --error=logs/grpo-%j.err
#SBATCH -p full
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=24:00:00

python -m phantom_reasoner.grpo \
    --dataset_name "kilian-group/phantom-wiki-v1" \
    --split_name "depth_20_size_50_seed_1" \
    --model_name_or_path "Qwen/Qwen2.5-0.5B-Instruct" \
    --num_train_epochs 10 \
    --log_level "info" \
    --logging_strategy "steps" \
    --logging_first_step \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --log_completions \
    --bf16 \
    $@
    # --use_vllm \

# NOTE: Using vllm goes OOM even for 0.5B model on 160GB GPU memory (2 A100s on AIDA)
