#!/usr/bin/env bash
#SBATCH --job-name=grpo
#SBATCH --output=logs/grpo-%j.out
#SBATCH --error=logs/grpo-%j.err
#SBATCH -p full
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=100GB
#SBATCH --time=24:00:00

# Get NUM_GPUS from nvidia-smi. It repeats the number of GPUs a few times, take the first one.
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n 1)

# --use_vllm reserves 1 GPU for generation, so then set NUM_PROCESSES=NUM_GPUS-1
NUM_PROCESSES=$((NUM_GPUS - 1))

# TODO: Figure out why vllm server keeps crashing.
# Till then we train without vllm generations.
# Start a vllm server in a separate terminal
# CUDA_VISIBLE_DEVICES=3 trl vllm-serve --model "Qwen/Qwen2.5-3B-Instruct" &

# Get CUDA visible devices as 0,...,NUM_GPUS-2 (0 indexing, and last one is reserved for vllm)
CUDA_DEVICES=$(seq -s, 0 $((NUM_GPUS - 2)))

# 3B model on 4 A100s (320GB GPU memory) works with zero1 with lora
# Start the training script on the GPUs
export WANDB_PROJECT="grpo"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES ACCELERATE_LOG_LEVEL=info accelerate launch \
    --num_processes=$NUM_PROCESSES \
    --config_file recipes/accelerate_configs/zero1.yaml \
	src/phantom_reasoner/grpo.py \
	--config recipes/qwen2.5-3b-instruct/grpo/config_lora.yaml \
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
