#!/usr/bin/env bash
# NOTE: hacked together
# Script to run the Qwen3 family of models on the GSM Infinite datasets
# Usage: ./scripts/eval/gsminf_eval_grpo.sh <output_dir>

OUTPUT_DIR=$1

MODEL_NAMES=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "microsoft/Phi-4-mini-reasoning"
)

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

    dataset_name="data/gsm-infinite-eval"
    ops_start=2
    ops_end=30
    ops_stride=1
    numbers=$(seq "$ops_start" "$ops_stride" "$ops_end")
    ops=$(echo "$numbers" | paste -s -d, -)

    echo "Running $model_name with length: 0, dataset: $dataset_name, save-dataset: medium"

    save_name="$(echo "$model_name" | sed 's|/|--|g' | sed 's|^-*||')"
    python examples/gsm_infinite/pred/pred.py \
        --output-dir "$OUTPUT_DIR" \
        --dataset-name "$dataset_name" \
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
        --output-dir "$OUTPUT_DIR" \
        --save-dataset "medium" \
        --model-name "$model_name"

    echo "Killing vLLM server..."
    pkill -f "vllm.entrypoints.openai.api_server"
}

for model_name in ${MODEL_NAMES[@]}
do
    eval_model_on_gsm_infinite "$model_name"
done

# Print the overall accuracy per model
python examples/gsm_infinite/format_model_accuracy.py \
    --output-dir "$OUTPUT_DIR"
