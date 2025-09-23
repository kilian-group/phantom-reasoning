#!/usr/bin/env bash
# Script to run gsm infinite evaluation on all checkpoints of the specified directory
# Usage: ./scripts/eval/gsminf_eval_all_ckpts.sh <path_to_checkpoint_parent_dir> <base_model_name> <training_dataset_name>

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_checkpoint_parent_dir> <base_model_name> <training_dataset_name>"
    exit 1
fi

CHECKPOINT_PARENT_DIR=$1
BASE_MODEL_NAME=$2
TRAINING_DATASET_NAME=$3

shift 3
cmd_args=$@

# checkpoints are in the format "CHECKPOINT_PARENT_DIR/checkpoint-<number>"
# Go over all checkpoints, and run evaluation script on them
OUT_DIR="$CHECKPOINT_PARENT_DIR/out-gsminf"

DATASET="data/gsm-infinite-eval"
ops_start=2
ops_end=30
ops_stride=1
numbers=$(seq "$ops_start" "$ops_stride" "$ops_end")
ops=$(echo "$numbers" | paste -s -d, -)

mkdir -p logs
PORT=8001
VLLM_BASE_URL="http://localhost:${PORT}/v1"
export OPENAI_BASE_URL="${VLLM_BASE_URL}"
export OPENAI_API_KEY="EMPTY"

eval_model_on_gsm_infinite() {
    model_name=$1

    # Kill any processes running on port ${PORT}
    # Get the PID of the process running on port ${PORT}
    PID=$(lsof -i :${PORT} | awk 'NR>1 {print $2}')
    if [ -n "${PID}" ]; then
        kill -9 "${PID}"
    fi

    # Launch the model in vLLM
    log_file="vllm_${model_name}.log"
    log_file=$(echo "${log_file}" | sed 's/\//--/g') # replace slash in model_name with --

    python -m vllm.entrypoints.openai.api_server \
        --model "${model_name}" \
        --tokenizer "${model_name}" \
        --tensor-parallel-size 1 \
        --port "${PORT}" \
        --served-model-name "${model_name}" \
        &> "logs/${log_file}" &

    echo "Waiting for vLLM server to start..."
    until curl -s "${VLLM_BASE_URL}/models" | grep -q "${model_name}"; do
        echo "Still waiting..."
        sleep 5
    done

    echo "Running $model_name with length: 0, dataset: $DATASET, save-dataset: medium"

    save_name="$(echo "$model_name" | sed 's|/|--|g' | sed 's|^-*||')"
    python examples/gsm_infinite/pred/pred.py \
        --output-dir "$OUT_DIR" \
        --dataset-name "$DATASET" \
        --model-name "$model_name" \
        --save-dataset "medium" \
        --save-name="$save_name" \
        --backend-type "openai" \
        --num-samples 1 \
        --temperature 1 \
        --max-tokens 4096 \
        --length "0" \
        --op-range "$ops" \
        --batch-size 200 \
        --limit 200

    echo "Calculating accuracy..."
    python examples/gsm_infinite/pred/eval_realistic.py \
        --output-dir "$OUT_DIR" \
        --save-dataset "medium" \
        --model-name "$model_name"

    echo "Killing vLLM server..."
    pkill -f "vllm.entrypoints.openai.api_server"
}

# Evaluate the base model
eval_model_on_gsm_infinite "$BASE_MODEL_NAME"

# Evaluate all models in the checkpoint parent directory
for ckpt in $CHECKPOINT_PARENT_DIR/checkpoint-*
do
    if [ -d "$ckpt" ]; then
        echo "Evaluating checkpoint: $ckpt"
        eval_model_on_gsm_infinite "$ckpt"
    fi
done

# Evaluate the final model
eval_model_on_gsm_infinite "$CHECKPOINT_PARENT_DIR"

# Print the overall accuracy per model
python examples/gsm_infinite/format_model_accuracy.py \
    --output-dir "$OUT_DIR"
