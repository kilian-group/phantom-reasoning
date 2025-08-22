#!/usr/bin/env bash
# Script to run all checkpoints of the specified directory on the Wiki datasets (HP, 2Wiki, MSQ)
# Usage: ./scripts/eval/pw_eval_all_ckpts.sh <path_to_checkpoint_parent_dir> <dataset> <split>

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_checkpoint_parent_dir> <dataset> <split> <base_model_name>"
    exit 1
fi

CHECKPOINT_PARENT_DIR=$1
dataset=$2
split=$3
base_model_name=$4

DATASET_LIST=(
    "hp500"
    "2wiki500"
    "msq500"
)

shift 4
cmd_args=$@

# If dataset not in DATASET_LIST, complain
if [[ ! " ${DATASET_LIST[@]} " =~ " ${dataset} " ]]; then
    echo "Dataset $dataset not in DATASET_LIST"
    exit 1
fi

# checkpoints are in the format "CHECKPOINT_PARENT_DIR/checkpoint-<number>"
# Go over all checkpoints, and run evaluation script on them
OUT_DIR="$CHECKPOINT_PARENT_DIR/out"

# Evaluate the base model
python examples/wiki/pred.py \
    --data_dir data/ \
    --dataset $dataset \
    --split $split \
    --method cot \
    --server vllm \
    --model_name "$base_model_name" \
    --inf_temperature 0.6 \
    --inf_top_p 0.95 \
    --inf_top_k 20 \
    -od "$OUT_DIR" \
    --inf_vllm_tensor_parallel_size 1 \
    $cmd_args

for ckpt in $CHECKPOINT_PARENT_DIR/checkpoint-*
do
    if [ -d "$ckpt" ]; then
        echo "Evaluating checkpoint: $ckpt"
        CUDA_VISIBLE_DEVICES=0 python examples/wiki/pred.py \
            --data_dir data/ \
            --dataset $dataset \
            --split $split \
            --method cot \
            --server vllm \
            --model_name "$ckpt" \
            --inf_temperature 0.6 \
            --inf_top_p 0.95 \
            --inf_top_k 20 \
            -od "$OUT_DIR" \
            --inf_vllm_tensor_parallel_size 1 \
            $cmd_args
    fi
done
