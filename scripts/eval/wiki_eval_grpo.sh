#!/usr/bin/env bash
# Script to evaluate LLMs on the Wiki datasets (HP, 2Wiki, MSQ, CofCA, SynthWorlds-RM)

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <output_dir> <dataset> <split>"
    echo "<dataset> can be hp500, 2wiki500, msq500, cofca500, synthrm500"
    echo "<split> should be minidev (500 examples), others not supported yet"
    echo "Set MODEL_NAMES env variable to a space-separated list of model names to evaluate"
    exit 1
fi

OUTPUT_DIR=$1
DATASET=$2
SPLIT=$3

shift 3
cmd_args=$@

DATASET_LIST=(
    "hp500"
    "2wiki500"
    "msq500"
    "cofca500"
    "synthrm500"
)

# If dataset not in DATASET_LIST, complain
if [[ ! " ${DATASET_LIST[@]} " =~ " ${DATASET} " ]]; then
    echo "Dataset $DATASET not in DATASET_LIST"
    exit 1
fi

# If MODEL_NAMES is not set, use the default list of models
if [ -z "$MODEL_NAMES" ]; then
    MODEL_NAMES=(
        "Qwen/Qwen3-0.6B"
        "Qwen/Qwen3-1.7B"
        "Qwen/Qwen2.5-1.5B-Instruct"
        "microsoft/Phi-4-mini-reasoning"
    )
    echo "Using default model list: ${MODEL_NAMES[*]}"
else
    # MODEL_NAMES is a space-separated list of model names
    MODEL_NAMES=($(echo $MODEL_NAMES | tr ' ' '\n'))
    echo "Using model list from env variable: ${MODEL_NAMES[*]}"
fi

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
