#!/usr/bin/env bash
# NOTE: hacked together
# Script to run the Qwen3 family of models on the PhantomWiki datasets
# Usage: ./scripts/eval/pw_eval_grpo.sh <output_dir>

OUTPUT_DIR="out-0804"

MODEL_NAMES=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "runs__bs=1/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-0.6B/grpo/ak2426/0804__curr=random__prompt=cot/"
    "runs__bs=1/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-0.6B/grpo/ak2426/0804__curr=difficulty_asc__prompt=cot/"
    "runs__bs=1/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-1.7B/grpo/ak2426/0804__curr=random__prompt=cot/"
    "runs__bs=1/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-1.7B/grpo/ak2426/0804__curr=difficulty_asc__prompt=cot/"
)

# for MODEL_NAME in ${MODEL_NAMES[@]}
# do
#   # Create a temporary script file for this model
#   SCRIPT_FILE="scripts/run_phantom_eval_$(basename "${MODEL_NAME//\//_}").sh"

#   cat > "$SCRIPT_FILE" << 'EOF'
# #!/usr/bin/env bash
# #SBATCH --get-user-env

# MODEL_NAME=$1
# OUTPUT_DIR=$2

# echo $MODEL_NAME
# echo $OUTPUT_DIR

# python -m phantom_eval \
#   --method cot \
#   --server vllm \
#   --inf_vllm_offline \
#   --model_name "${MODEL_NAME}" \
#   --dataset data/wiki-v1-easy-depth_20_size_25 \
#   --split_list depth_20_size_25_seed_1 depth_20_size_25_seed_2 depth_20_size_25_seed_3 \
#   --from_local \
#   --exclude_aggregation_questions \
#   --inf_is_deepseek_r1_model \
#   --inf_vllm_tensor_parallel_size 1 \
#   -od "${OUTPUT_DIR}"
# EOF

#   # Make the script executable
#   chmod +x "$SCRIPT_FILE"

#   # Submit the job with the model name as an environment variable
#   sbatch -p full -n 8 -t 2:00:00 --gres=gpu:a100:1 --mem=50GB "$SCRIPT_FILE" "$MODEL_NAME" "$OUTPUT_DIR"
# done

# for MODEL_NAME in ${MODEL_NAMES[@]}
# do
#   # Create a temporary script file for this model
#   SCRIPT_FILE="scripts/run_phantom_eval_temp1_$(basename "${MODEL_NAME//\//_}").sh"

#   cat > "$SCRIPT_FILE" << 'EOF'
# #!/usr/bin/env bash
# #SBATCH --get-user-env

# MODEL_NAME=$1
# OUTPUT_DIR=$2

# python -m phantom_eval \
#   --method cot \
#   --server vllm \
#   --inf_vllm_offline \
#   --model_name "${MODEL_NAME}" \
#   --dataset data/wiki-v1-easy-depth_20_size_25 \
#   --split_list depth_20_size_25_seed_1 depth_20_size_25_seed_2 depth_20_size_25_seed_3 \
#   --from_local \
#   --exclude_aggregation_questions \
#   --inf_is_deepseek_r1_model \
#   --inf_temperature 1 \
#   --inf_vllm_tensor_parallel_size 1 \
#   -od "${OUTPUT_DIR}__temp=1.0"
# EOF

#   # Make the script executable
#   chmod +x "$SCRIPT_FILE"

#   # Submit the job with the model name as an environment variable
#   sbatch -p full -n 8 -t 2:00:00 --gres=gpu:a100:1 --mem=50GB "$SCRIPT_FILE" "$MODEL_NAME" "$OUTPUT_DIR"
# done

echo "${MODEL_NAMES[*]}"

python ~/work/phantom-wiki/eval/format_leaderboard.py \
  -od $OUTPUT_DIR \
  --model_list "${MODEL_NAMES[*]}" \
  --size_list 25 \
  --method_list cot \
  --dataset data/wiki-v1-easy-depth_20_size_25 \
  --from_local

python ~/work/phantom-wiki/eval/format_leaderboard.py \
  -od "${OUTPUT_DIR}__temp=1.0" \
  --model_list "${MODEL_NAMES[*]}" \
  --size_list 25 \
  --method_list cot \
  --dataset data/wiki-v1-easy-depth_20_size_25 \
  --from_local
