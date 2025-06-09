#!/bin/bash
# Script to run the Qwen3 family of models on the Wiki datasets (HP, 2Wiki, MSQ)
# Usage: ./run_qwen3.sh <output_dir> <split>
# Split can be minidev (500 examples) or dev (~11K examples)

OUTPUT_DIR=$1
SPLIT=$2

DATASET_LIST=(
    # "hp500"
    # "2wiki500"
    "msq500"
)
PARAMS_LIST=(
    "1.7b"
    # "4b"
    # "8b"
    # "14b"
    # "32b"
)

for dataset in ${DATASET_LIST[@]}; do
    for params in ${PARAMS_LIST[@]}; do
        # NOTE: thinking mode is on by default
        python pred.py --dataset $dataset --split $SPLIT --method cot --server vllm -m qwen/qwen3-$params --inf_temperature 0.6 --inf_top_p 0.95 --inf_top_k 20 -od $OUTPUT_DIR
    done
done
