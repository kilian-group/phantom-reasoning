#!/usr/bin/env bash
# Script to run all checkpoints of the specified directory on the Wiki datasets (HP, 2Wiki, MSQ)
# Usage: ./scripts/eval/other_eval_all_ckpts.sh <path_to_checkpoint_parent_dir> <dataset> <split> <base_model_name> <training_dataset_name>

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_checkpoint_parent_dir> <dataset> <split> <base_model_name> <training_dataset_name>"
    exit 1
fi

CHECKPOINT_PARENT_DIR=$1
DATASET=$2
SPLIT=$3
BASE_MODEL_NAME=$4
TRAINING_DATASET_NAME=$5

DATASET_LIST=(
    "hp500"
    "2wiki500"
    "msq500"
)

shift 5
cmd_args=$@

# If dataset not in DATASET_LIST, complain
if [[ ! " ${DATASET_LIST[@]} " =~ " ${DATASET} " ]]; then
    echo "Dataset $DATASET not in DATASET_LIST"
    exit 1
fi

# checkpoints are in the format "CHECKPOINT_PARENT_DIR/checkpoint-<number>"
# Go over all checkpoints, and run evaluation script on them
OUT_DIR="$CHECKPOINT_PARENT_DIR/out-${DATASET}"

# Evaluate the base model
python examples/wiki/pred.py \
    --data_dir data/ \
    --dataset $DATASET \
    --split $SPLIT \
    --method cot \
    --server vllm \
    --model_name "$BASE_MODEL_NAME" \
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
            --dataset $DATASET \
            --split $SPLIT \
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

# Evaluate the final model
python examples/wiki/pred.py \
    --data_dir data/ \
    --dataset $DATASET \
    --split $SPLIT \
    --method cot \
    --server vllm \
    --model_name "$CHECKPOINT_PARENT_DIR" \
    --inf_temperature 0.6 \
    --inf_top_p 0.95 \
    --inf_top_k 20 \
    -od "$OUT_DIR" \
    --inf_vllm_tensor_parallel_size 1 \
    $cmd_args

python examples/wiki/plot_scaling_all_ckpts.py \
	-dd data/ \
	-od "$OUT_DIR" \
	--split "$SPLIT" \
	--dataset "$DATASET" \
	--method cot \
	--base_model_name "$BASE_MODEL_NAME" \
    --model_list "$CHECKPOINT_PARENT_DIR" \
	--training_dataset_name "$TRAINING_DATASET_NAME"
