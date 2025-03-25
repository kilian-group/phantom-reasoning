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
# --use_vllm reserves 1 GPU for generation, so then set NUM_PROCESSES=NUM_GPUS-1
NUM_PROCESSES=3

# Start a vllm server in a separate terminal
# CUDA_VISIBLE_DEVICES=3 trl vllm-serve --model "Qwen/Qwen2.5-3B-Instruct" &

# 3B model on 4 A100s (320GB GPU memory) works with zero1
# Start the training script on the first 3 GPUs
CUDA_VISIBLE_DEVICES=0,1,2 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --num_processes=$NUM_PROCESSES \
    --config_file recipes/accelerate_configs/zero1.yaml \
	src/phantom_reasoner/grpo.py \
	--config recipes/qwen2.5-3b-instruct/grpo/config_vllm.yaml \
    $@

# python -m phantom_reasoner.grpo \
#     --dataset_name "kilian-group/phantom-wiki-v1" \
#     --split_name "depth_20_size_50_seed_1" \
#     --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
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
#     --per_device_train_batch_size 4 \
#     --num_generations 4 \
#     $@
#     # --use_vllm \

# NOTE: Using vllm goes OOM even for 0.5B model on 160GB GPU memory (2 A100s on AIDA)
