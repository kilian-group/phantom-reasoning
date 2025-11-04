#!/usr/bin/env bash
# Script to evaluate LLMs on the PhantomWiki datasets

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <output_dir>"
    exit 1
fi

OUTPUT_DIR=$1

shift 1
cmd_args=$@

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

PW_SPLITS="depth_20_size_25_seed_1 depth_20_size_25_seed_2 depth_20_size_25_seed_3"
for model_name in ${MODEL_NAMES[@]}
do
    CUDA_VISIBLE_DEVICES=0 python -m phantom_eval \
      --method cot \
      --server vllm \
      --inf_vllm_offline \
      --model_name "${model_name}" \
      --dataset data/wiki-v1-easy-depth_20_size_25 \
      --split_list $PW_SPLITS \
      --from_local \
      --exclude_aggregation_questions \
      --inf_temperature 1 \
      -od "${OUTPUT_DIR}" \
      --inf_vllm_tensor_parallel_size 1 \
      $cmd_args
done

rm -r cachedir/

python ../phantom-wiki/eval/format_leaderboard.py \
    -od "${OUTPUT_DIR}" \
    --model_list ${MODEL_NAMES[*]} \
    --size_list 25 \
    --method_list cot \
    --dataset data/wiki-v1-easy-depth_20_size_25 \
    --from_local
