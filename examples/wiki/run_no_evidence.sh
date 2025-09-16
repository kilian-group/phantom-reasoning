#!/bin/bash
# Script to test internal knowledge of the model

OUTPUT_DIR=$1
TEMPERATURE=$2
TOP_P=$3
DATASET=$4
DATA_DIR="~/phantom/src/phantom-reasoning/data"

MODEL_LIST=(
    "Qwen/Qwen3-1.7B"
    "/anvil/projects/x-nairr250102/phantom-reasoning/runs/data/hp/Qwen/Qwen3-1.7B/grpo/x-anmolkab/0912__curr\=random__training_seed\=1"
)

# With evidence
for model in ${MODEL_LIST[@]}; do
    CMD="python pred.py -od ${OUTPUT_DIR} --dataset ${DATASET} --split minidev --method cot --server vllm -m ${model} --inf_temperature ${TEMPERATURE} --inf_top_p ${TOP_P} -dd ${DATA_DIR}"
    echo $CMD
    eval $CMD
done

# Without evidence
for model in ${MODEL_LIST[@]}; do
    CMD="python pred.py -od ${OUTPUT_DIR}-no-evidence --dataset ${DATASET} --split minidev --method cot --server vllm -m ${model} --no_evidence --inf_temperature ${TEMPERATURE} --inf_top_p ${TOP_P} -dd ${DATA_DIR}"
    echo $CMD
    eval $CMD
done
