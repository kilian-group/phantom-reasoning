#!/usr/bin/env bash
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
    "runs__pw_then_wiki/data/hp/Qwen/Qwen2.5-1.5B-Instruct/grpo/x-anmolkab/after=pw__curr=random__training_seed=1"
    "runs__pw_then_wiki/data/2wiki/Qwen/Qwen2.5-1.5B-Instruct/grpo/x-anmolkab/after=pw__curr=random__training_seed=1"
    "runs__pw_then_wiki/data/msq/Qwen/Qwen2.5-1.5B-Instruct/grpo/x-anmolkab/after=pw__curr=random__training_seed=1"
    "runs__pw_then_wiki/data/hp/Qwen/Qwen3-1.7B/grpo/x-anmolkab/after=pw__curr=random__training_seed=1"
    "runs__pw_then_wiki/data/2wiki/Qwen/Qwen3-1.7B/grpo/x-anmolkab/after=pw__curr=random__training_seed=1"
    # "runs__pw_then_wiki/data/msq/Qwen/Qwen3-1.7B/grpo/x-anmolkab/after=pw__curr=random__training_seed=1"
    # "runs/data/hp/Qwen/Qwen3-1.7B/grpo/x-anmolkab/0912__curr=random__training_seed=1/checkpoint-2500"
    # "runs/data/2wiki/Qwen/Qwen3-1.7B/grpo/x-anmolkab/0912__curr=random__training_seed=1/checkpoint-2500"
    # "runs/data/msq/Qwen/Qwen3-1.7B/grpo/x-anmolkab/0912__curr=random__training_seed=1/checkpoint-2500"
    # "runs/data/hp/Qwen/Qwen2.5-1.5B-Instruct/grpo/x-anmolkab/0911__curr=random__training_seed=1/checkpoint-2500"
    # "runs/data/2wiki/Qwen/Qwen2.5-1.5B-Instruct/grpo/x-anmolkab/0911__curr=random__training_seed=1/checkpoint-2500"
    # "runs/data/msq/Qwen/Qwen2.5-1.5B-Instruct/grpo/x-anmolkab/0911__curr=random__training_seed=1/checkpoint-2500"
    # "runs__wiki_then_pw/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen2.5-1.5B-Instruct/grpo/x-anmolkab/after=hp__curr=random__training_seed=1"
    # "runs__wiki_then_pw/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen2.5-1.5B-Instruct/grpo/x-anmolkab/after=2wiki__curr=random__training_seed=1"
    # "runs__wiki_then_pw/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen2.5-1.5B-Instruct/grpo/x-anmolkab/after=msq__curr=random__training_seed=1"
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
