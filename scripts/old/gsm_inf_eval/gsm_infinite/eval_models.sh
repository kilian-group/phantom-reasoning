#!/usr/bin/env bash
#SBATCH --job-name=evalqwen3-0.6b-0727
#SBATCH --output=logs/evalqwen3-0.6b-0727-%j.out
#SBATCH --error=logs/evalqwen3-0.6b-0727-%j.err
#SBATCH -p kilian
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:a6000:3
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --mail-user=yy958@cornell.edu
#SBATCH --mail-type=BEGIN,END,FAIL

conda activate gsm-env

export OPENAI_API_KEY="EMPTY"

mkdir -p logs

# activate vLLM server
CUDA_VISIBLE_DEVICES=0 \
# python -m vllm.entrypoints.openai.api_server \
#     --model Qwen/Qwen3-0.6B \
#     --tokenizer Qwen/Qwen3-0.6B \
#     --tensor-parallel-size 1 \
#     --port 8001 \
#     --served-model-name Qwen/Qwen3-0.6B \
#     &> logs/vllm_qwen3-0.6b.log &

# CUDA_VISIBLE_DEVICES=1 \
# python -m vllm.entrypoints.openai.api_server \
#     --model Qwen/Qwen2.5-7B-Instruct \
#     --tokenizer Qwen/Qwen2.5-7B-Instruct \
#     --tensor-parallel-size 1 \
#     --port 8002 \
#     --served-model-name Qwen2.5-7B-Instruct \
#     &> logs/vllm_qwen2.5_7b.log &

echo "Waiting for vLLM server to start..."
# until curl -s http://localhost:8001/v1/models | grep -q Qwen/Qwen3-0.6B; do
#     echo "Still waiting..."
#     sleep 5
# done
# echo "vLLM is ready."

# until curl -s http://localhost:8002/v1/models | grep -q Qwen2.5-7B-Instruct; do
#     echo "Still waiting..."
#     sleep 5
# done
# echo "vLLM is ready."

# evaluation
bash run.sh
