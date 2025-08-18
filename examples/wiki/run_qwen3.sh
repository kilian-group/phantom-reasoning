#!/usr/bin/env bash
# Script to run the Qwen3 family of models on the Wiki datasets (HP, 2Wiki, MSQ)
# Usage: ./run_qwen3.sh <output_dir> <split>
# Split can be minidev (500 examples) or dev (~11K examples)

OUTPUT_DIR=$1
SPLIT=$2

DATASET_LIST=(
    "hp500"
    "2wiki500"
    "msq500"
)

MODELS_LIST=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
)

for dataset in ${DATASET_LIST[@]}; do
    for model in ${MODELS_LIST[@]}; do
        CMD="python pred.py --dataset $dataset --split $SPLIT --method cot --server vllm -m $model --inf_temperature 0.6 --inf_top_p 0.95 --inf_top_k 20 -od $OUTPUT_DIR"
        echo $CMD
        $CMD
    done
done
