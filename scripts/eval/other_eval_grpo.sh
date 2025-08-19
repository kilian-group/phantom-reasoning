#!/bin/bash
# NOTE: hacked together
# Script to run the Qwen3 family of models on the Wiki datasets (HP, 2Wiki, MSQ)
# Usage: ./eval/other_eval_grpo.sh <output_dir> <dataset> <split>
# Split can be minidev (500 examples) or dev (~11K examples)

OUTPUT_DIR=$1
dataset=$2
SPLIT=$3

DATASET_LIST=(
    "hp500"
    "2wiki500"
    "msq500"
)

# If dataset not in DATASET_LIST, complain
if [[ ! " ${DATASET_LIST[@]} " =~ " ${dataset} " ]]; then
    echo "Dataset $dataset not in DATASET_LIST"
    exit 1
fi

MODEL_NAMES=(
    "Qwen/Qwen3-0.6B"
    "runs__answertag/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-0.6B/grpo/ak2426/0813__curr=random__prompt=cot"
    "runs__answertag/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-0.6B/grpo/ak2426/0814__curr=difficulty_asc__prompt=cot"
    "Qwen/Qwen3-1.7B"
    "runs__answertag/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-1.7B/grpo/ak2426/0814__curr=random__prompt=cot"
    "runs__answertag/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-1.7B/grpo/ak2426/0814__curr=difficulty_asc__prompt=cot"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "runs__answertag/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen2.5-1.5B-Instruct/grpo/ak2426/0814__curr=random__prompt=cot"
    "runs__answertag/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen2.5-1.5B-Instruct/grpo/ak2426/0815__curr=difficulty_asc__prompt=cot"
    "google/gemma-3-1b-it"
    "runs__answertag/data/wiki-v1-easy-depth_20_size_25/google/gemma-3-1b-it/grpo/ak2426/0815__curr=random__prompt=cot"
    "runs__answertag/data/wiki-v1-easy-depth_20_size_25/google/gemma-3-1b-it/grpo/ak2426/0815__curr=difficulty_asc__prompt=cot"
    "meta-llama/Llama-3.2-3B-Instruct"
    "runs__answertag/data/wiki-v1-easy-depth_20_size_25/meta-llama/Llama-3.2-3B-Instruct/grpo/ak2426/0816__curr=difficulty_asc__prompt=cot"
)

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
