#!/usr/bin/env bash
# Script to run PW evaluation on all checkpoints of the specified directory
# Usage: ./scripts/eval/pw_eval_all_ckpts.sh <path_to_checkpoint_parent_dir> <base_model_name> <training_dataset_name>

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_checkpoint_parent_dir> <base_model_name> <training_dataset_name>"
    exit 1
fi

CHECKPOINT_PARENT_DIR=$1
BASE_MODEL_NAME=$2
TRAINING_DATASET_NAME=$3

shift 3
cmd_args=$@

# checkpoints are in the format "CHECKPOINT_PARENT_DIR/checkpoint-<number>"
# Go over all checkpoints, and run evaluation script on them
OUT_DIR="$CHECKPOINT_PARENT_DIR/out-pw"

DATASET="data/wiki-v1-easy-depth_20_size_25"
SPLITS="depth_20_size_25_seed_1 depth_20_size_25_seed_2 depth_20_size_25_seed_3"

# Evaluate the base model
python -m phantom_eval \
    --method cot \
    --server vllm \
    --inf_vllm_offline \
    --model_name "$BASE_MODEL_NAME" \
    --dataset "$DATASET" \
    --split_list $SPLITS \
    --from_local \
    --inf_vllm_tensor_parallel_size 1 \
    --exclude_aggregation_questions \
    -od "$OUT_DIR" \
    $cmd_args

for ckpt in $CHECKPOINT_PARENT_DIR/checkpoint-*
do
    if [ -d "$ckpt" ]; then
        echo "Evaluating checkpoint: $ckpt"
        # Run the evaluation script, assuming it is named evaluate.py and takes the checkpoint path as an argument
        python -m phantom_eval \
            --method cot \
            --server vllm \
            --inf_vllm_offline \
            --model_name "$ckpt" \
            --dataset "$DATASET" \
            --split_list $SPLITS \
            --from_local \
            --inf_vllm_tensor_parallel_size 1 \
            --exclude_aggregation_questions \
            -od "$OUT_DIR" \
            $cmd_args

    fi
done

# Evaluate the final model
python -m phantom_eval \
    --method cot \
    --server vllm \
    --inf_vllm_offline \
    --model_name "$CHECKPOINT_PARENT_DIR" \
    --dataset "$DATASET" \
    --split_list $SPLITS \
    --from_local \
    --inf_vllm_tensor_parallel_size 1 \
    --exclude_aggregation_questions \
    -od "$OUT_DIR" \
    $cmd_args

python scripts/plot_reasoning_during_training.py \
    -od "$OUT_DIR" \
    --model_list "$CHECKPOINT_PARENT_DIR" \
    --dataset "$DATASET" \
    --base_model_name "$BASE_MODEL_NAME" \
    --from_local

python scripts/plot_pw_scaling_all_ckpts.py \
    -od "$OUT_DIR" \
    --model_list "$CHECKPOINT_PARENT_DIR" \
    --dataset "$DATASET" \
    --from_local \
    --method cot \
    --base_model_name "$BASE_MODEL_NAME" \
    --training_dataset_name "$TRAINING_DATASET_NAME"
