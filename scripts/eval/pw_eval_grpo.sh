#!/usr/bin/env bash
# NOTE: hacked together
# Script to run the Qwen3 family of models on the PhantomWiki datasets
# Usage: ./scripts/eval/pw_eval_grpo.sh <output_dir>

OUTPUT_DIR=$1

MODEL_NAMES=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "microsoft/Phi-4-mini-reasoning"
)

for model_name in ${MODEL_NAMES[@]}
do
  python -m phantom_eval \
    --method cot \
    --server vllm \
    --inf_vllm_offline \
    --model_name "${model_name}" \
    --dataset data/wiki-v1-easy-depth_20_size_25 \
    --split_list depth_20_size_25_seed_1 depth_20_size_25_seed_2 depth_20_size_25_seed_3 \
    --from_local \
    --exclude_aggregation_questions \
    --inf_temperature 1 \
    --inf_vllm_tensor_parallel_size 1 \
    -od "${OUTPUT_DIR}"
done

echo "${MODEL_NAMES[*]}"
rm -r cachedir/

python ~/work/phantom-wiki/eval/format_leaderboard.py \
  -od "${OUTPUT_DIR}" \
  --model_list ${MODEL_NAMES[*]} \
  --size_list 25 \
  --method_list cot \
  --dataset data/wiki-v1-easy-depth_20_size_25 \
  --from_local
