#!/usr/bin/env bash
#SBATCH --job-name=grpo
#SBATCH --output=logs/grpo-%j.out
#SBATCH --error=logs/grpo-%j.err
#SBATCH -p full
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=100GB
#SBATCH --time=24:00:00

# NUM_PROCESSES=NUM_GPUS
NUM_PROCESSES=2

# TODO: accelerate is actually really slow...
# 20 hours for 0.5B model vs 2 hours without accelerate
# Anmol: I think it's because of zero3 config that's offloading
# matrices to the CPU. Might want to try zero1 or zero2.

ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes=$NUM_PROCESSES \
    --config_file recipes/accelerate_configs/zero3.yaml \
	src/phantom_reasoner/grpo.py \
	--config recipes/qwen2.5-0.5b-instruct/grpo/config_base.yaml \
    $@


# python -m phantom_reasoner.grpo \
#     --dataset_name "kilian-group/phantom-wiki-v1" \
#     --split_name "depth_20_size_50_seed_1" \
#     --model_name_or_path "Qwen/Qwen2.5-0.5B-Instruct" \
#     --num_train_epochs 10 \
#     --log_level "info" \
#     --logging_strategy "steps" \
#     --logging_first_step \
#     --logging_steps 10 \
#     --save_strategy "steps" \
#     --save_steps 100 \
#     --save_total_limit 5 \
#     --log_completions \
#     --bf16 \
#     $@
#     # --use_vllm \

# NOTE: Using vllm goes OOM even for 0.5B model on 160GB GPU memory (2 A100s on AIDA)
