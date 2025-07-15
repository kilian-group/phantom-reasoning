#!/usr/bin/env bash
#SBATCH --job-name=evalQwen3-8b
#SBATCH --output=logs/evalQwen3-8b-%j.out
#SBATCH --error=logs/evalQwen3-8b-%j.err
#SBATCH -p kilian
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=100GB
#SBATCH --time=24:00:00
#SBATCH --mail-user=yy958@cornell.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate phantom-reasoning

# 设置环境变量
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"

# 启动 vLLM server
nohup python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen3-8B \
  --tokenizer ./models/Qwen3-8B \
  --tensor-parallel-size 1 \
  --port 8000 \
  --served-model-name Qwen3-8B &



echo "Waiting for vLLM server to start..."
until curl -s http://localhost:8000/v1/models | grep -q Qwen3-8B; do
    echo "Still waiting..."
    sleep 5
done
echo "vLLM is ready."



# 启动评估
CUDA_VISIBLE_DEVICES=0 python -m phantom_eval \
  --method cot \
  --server vllm \
  --model_name Qwen3-8B \
  --dataset data/wiki-v1-easy-no-agg \
  --split_list depth_10_size_25_seed_1 depth_10_size_25_seed_2 depth_10_size_25_seed_3 \
  --from_local \
  --inf_vllm_tensor_parallel_size 1 \
  -od ./evals/Qwen3-8B

CUDA_VISIBLE_DEVICES=0 python -m phantom_eval \
  --method zeroshot \
  --server vllm \
  --model_name Qwen3-8B \
  --dataset data/wiki-v1-easy-no-agg \
  --split_list depth_10_size_25_seed_1 depth_10_size_25_seed_2 depth_10_size_25_seed_3 \
  --from_local \
  --inf_vllm_tensor_parallel_size 1 \
  -od ./evals/Qwen3-8B
