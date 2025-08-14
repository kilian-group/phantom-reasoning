#!/bin/bash
# NOTE: hacked together
# Script to run the Qwen3 family of models on the Wiki datasets (HP, 2Wiki, MSQ)
# Usage: ./eval_grpo_on_other_datasets.sh <output_dir> <split>
# Split can be minidev (500 examples) or dev (~11K examples)

OUTPUT_DIR=$1
SPLIT=$2

DATASET_LIST=(
    "hp500"
    "2wiki500"
    "msq500"
)
MODEL_NAMES=(
    "Qwen/Qwen3-0.6B"
    "runs__bs=1/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-0.6B/grpo/ak2426/0804__curr=random__prompt=cot/"
    "runs__bs=1/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-0.6B/grpo/ak2426/0804__curr=difficulty_asc__prompt=cot/"
    "Qwen/Qwen3-1.7B"
    "runs__bs=1/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-1.7B/grpo/ak2426/0804__curr=random__prompt=cot/"
    "runs__bs=1/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-1.7B/grpo/ak2426/0804__curr=difficulty_asc__prompt=cot/"
)

for dataset in ${DATASET_LIST[@]}; do
    for model_name in ${MODEL_NAMES[@]}; do
        # NOTE: thinking mode is on by default
        CUDA_VISIBLE_DEVICES=0 python examples/wiki/pred.py \
          --data_dir data/ \
          --dataset $dataset \
          --split $SPLIT \
          --method cot \
          --server vllm \
          --model_name $model_name \
          --inf_temperature 0.6 \
          --inf_top_p 0.95 \
          --inf_top_k 20 \
          -od $OUTPUT_DIR \
          --inf_vllm_tensor_parallel_size 1
    done
done
