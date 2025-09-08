#!/bin/bash
# NOTE: hacked together
# Script to run the Qwen3 family of models on the Wiki datasets (HP, 2Wiki, MSQ)
# Usage: ./scripts/eval/other_eval_grpo.sh <output_dir> <dataset> <split>
# Split can be minidev (500 examples) or dev (~11K examples)

OUTPUT_DIR=$1
DATASET=$2
SPLIT=$3

shift 3
cmd_args=$@

DATASET_LIST=(
    "hp500"
    "2wiki500"
    "msq500"
)

# If dataset not in DATASET_LIST, complain
if [[ ! " ${DATASET_LIST[@]} " =~ " ${DATASET} " ]]; then
    echo "Dataset $DATASET not in DATASET_LIST"
    exit 1
fi

MODEL_NAMES=(
    "google/gemma-3-1b-it"
    "runs__bs=8/data/wiki-v1-easy-depth_20_size_25/google/gemma-3-1b-it/grpo/x-anmolkab/0905__curr=random__training_seed=2"
    "meta-llama/Llama-3.2-3B-Instruct"
    "runs__bs=8/data/wiki-v1-easy-depth_20_size_25/meta-llama/Llama-3.2-3B-Instruct/grpo/x-anmolkab/0905__curr=random__training_seed=1"
)

for model_name in ${MODEL_NAMES[@]}; do
    CUDA_VISIBLE_DEVICES=0 python examples/wiki/pred.py \
        --data_dir data/ \
        --dataset $DATASET \
        --split $SPLIT \
        --method cot \
        --server vllm \
        --model_name $model_name \
        --inf_temperature 0.6 \
        --inf_top_p 0.95 \
        --inf_top_k 20 \
        -od $OUTPUT_DIR \
        --inf_vllm_tensor_parallel_size 1 \
        $cmd_args
done

rm -r cachedir/

python examples/wiki/format_split_accuracy.py \
    -dd data/ \
    -od $OUTPUT_DIR \
    --split $SPLIT \
    --dataset $DATASET \
    --method cot
