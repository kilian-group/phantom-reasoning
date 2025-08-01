#!/usr/bin/env bash
# Script to run PW evaluation on all checkpoints of the specified directory
# Usage: ./scripts/pw-eval/evaluate_all_ckpts.sh <path_to_checkpoint_parent_dir>

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_checkpoint_parent_dir>"
    exit 1
fi

CHECKPOINT_PARENT_DIR=$1

# checkpoints are in the format "CHECKPOINT_PARENT_DIR/checkpoint-<number>"
# Go over all checkpoints, and run evaluation script on them
OUT_DIR="$CHECKPOINT_PARENT_DIR/out"

# NOTE: we run on depth_20_size_25_seed_2 split
DATASET="data/wiki-v1-easy-depth_20_size_25"
SPLITS="depth_20_size_25_seed_2"

for ckpt in $CHECKPOINT_PARENT_DIR/checkpoint-*
do
    if [ -d "$ckpt" ]; then
        echo "Evaluating checkpoint: $ckpt"
        # Run the evaluation script, assuming it is named evaluate.py and takes the checkpoint path as an argument
        CUDA_VISIBLE_DEVICES=1 python -m phantom_eval \
            --method cot \
            --server vllm \
            --inf_vllm_offline \
            --model_name "$ckpt" \
            --dataset "$DATASET" \
            --split_list "$SPLITS" \
            --from_local \
            --inf_vllm_tensor_parallel_size 1 \
            --exclude_aggregation_questions \
            -od "$OUT_DIR"

    fi
done
