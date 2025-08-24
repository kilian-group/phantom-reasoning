#!/bin/bash

# Configure your API Keys and URLs, Leave it blank if not necessary
# You can use environment variables here as well
# If you choose openai backend compatible backend, please fill SAMPLER_OPENAI_BASE_URL and SAMPLER_OPENAI_API_KEY with openai base url and api key of the backend
backend_type='openai' # can be 'openai', 'gemini' and 'anthropic'
# SAMPLER_OPENAI_BASE_URL=$OPENAI_BASE_URL
# SAMPLER_OPENAI_API_KEY=$OPENAI_API_KEY
# GEMINI_API_KEY=$GEMINI_API_KEY
# ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY

# To evaluate symbolic subset, you should first launch an openai compatible backend.
# We use Qwen/Qwen2.5-7B-Instruct as our parser to extract the answer.
# Fill EVAL_OPENAI_BASE_URL and EVAL_OPENAI_API_KEY with openai base url and api key of the backend
# EVAL_OPENAI_BASE_URL=$OPENAI_BASE_URL
# EVAL_OPENAI_API_KEY=$OPENAI_API_KEY

backend_type='openai'
SAMPLER_OPENAI_BASE_URL="http://localhost:8001/v1"
SAMPLER_OPENAI_API_KEY="EMPTY"

EVAL_OPENAI_BASE_URL="http://localhost:8002/v1"
EVAL_OPENAI_API_KEY="EMPTY"

# Control sampling and evaluation (can be set from command line)
run_sampling=true  # Set to "true" to run sampling, "false" to skip
run_evaluation=true # Set to "true" to run evaluation, "false" to skip
run_symbolic_evaluation=false # Set to "true" to ONLY run symbolic evaluation
run_realistic_evaluation=false # Set to "true" to ONLY run realistic evaluation

# Model and Dataset Configuration
model_name='qwen3-0.6b-0727' # SAMPLER API model name
dataset_base='InfiniAILab/gsm_infinite' # Base name for the dataset 'InfiniAILab/gsm_infinite'
save_name='qwen3-0.6b-0727' # Model name for saving the results

# Sampling Settings
num_samples=1
temperature_symbolic=1.0 # Temperature for symbolic
temperature_realistic=0.0 # Temperature for realistic
max_tokens=4096

# Batch size and example limit per op
batch_size=200
limit_symbolic=100 # Limit for symbolic
limit_realistic=200 # Limit for realistic


# Lengths to process (can be numbers or strings like '8k')
lengths=(
    "0"
)

# Dataset suffixes
dataset_suffixes=(
    "medium"
    "hard"
)

# Operation Range Configuration (Per length and suffix). if empty, the subset will be skipped.
declare -A ops_config
# Example configurations(Change the 'start's and 'end's as you wish):

ops_config["0_medium"]='{"start": 2, "end": 30, "stride": 1}'

ops_config["0_hard"]='{"start": 2, "end": 30, "stride": 1}'


# Filter Configuration (JSON string, only used for realistic)
filter_config='[
    {"percentage": 0.4, "template": "crazy_zootopia", "mode": "normalforward"},
    {"percentage": 0.05, "template": "movie_festival_awards", "mode": "normalforward"},
    {"percentage": 0.05, "template": "teachers_in_school", "mode": "normalforward"},
    {"percentage": 0.4, "template": "crazy_zootopia", "mode": "forwardreverse"},
    {"percentage": 0.05, "template": "movie_festival_awards", "mode": "forwardreverse"},
    {"percentage": 0.05, "template": "teachers_in_school", "mode": "forwardreverse"}
]'
