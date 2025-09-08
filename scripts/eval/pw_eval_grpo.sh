#!/usr/bin/env bash
# NOTE: hacked together
# Script to run the Qwen3 family of models on the PhantomWiki datasets
# Usage: ./scripts/eval/pw_eval_grpo.sh <output_dir>

OUTPUT_DIR=$1

MODEL_NAMES=(
    "google/gemma-3-1b-it"
    "runs__bs=8/data/wiki-v1-easy-depth_20_size_25/google/gemma-3-1b-it/grpo/x-anmolkab/0905__curr=random__training_seed=2"
    "meta-llama/Llama-3.2-3B-Instruct"
    "runs__bs=8/data/wiki-v1-easy-depth_20_size_25/meta-llama/Llama-3.2-3B-Instruct/grpo/x-anmolkab/0905__curr=random__training_seed=1"
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
