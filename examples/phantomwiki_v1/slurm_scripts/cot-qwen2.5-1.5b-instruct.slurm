#!/bin/bash
#SBATCH -J cot-qwen2.5-1.5b-instruct                              # Job name
#SBATCH -o logs/cot-qwen2.5-1.5b-instruct_%j.out                 # output file (%j expands to jobID)
#SBATCH -e logs/cot-qwen2.5-1.5b-instruct_%j.err                 # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 8                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=100GB                         # server memory (MBs) requested (per node)
#SBATCH -t infinite                          # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:a6000:1                   # Number of GPUs requested
#SBATCH --partition=kilian                   # Request partition

source constants.sh

# The first argument is expected to be the output directory
OUTPUT_DIR=$1
echo "Output directory: $OUTPUT_DIR"
echo "##################################################"

# Parse the arguments to get extra arguments after the first 2
shift 1
cmd_args=$@
echo "Extra arguments: $cmd_args"
echo "##################################################"

# Base command
cmd="python -m phantom_eval     --method cot     -od $OUTPUT_DIR     --server vllm     -m qwen/qwen2.5-1.5b-instruct"

# concatenate the extra arguments to the base command
cmd+=" $cmd_args"

# check if input contains --split_list if not add the default SPLIT_LIST
if [[ ! "$@" =~ "--split_list" ]]; then
    cmd+=" --split_list $SPLIT_LIST"
fi

# if the model is deepseek, add the temperature and top_p and inf_is_deepseek_r1_model flag
if [[ "" =~ ^[Yy]$ ]]; then
    TEMPERATURE=0.6
    TOP_P=0.95
    cmd+=" --inf_temperature $TEMPERATURE            --inf_top_p $TOP_P            --inf_is_deepseek_r1_model"
else
    TEMPERATURE=0
    cmd+=" --inf_temperature $TEMPERATURE"
fi

cmd+=" --inf_seed_list $(get_inf_seed_list $TEMPERATURE)"

# Function to check if the vLLM server is up
check_server() {
    local model_name=$1
    local port=$2
    # echo http://0.0.0.0:8001/v1/chat/completions
    response=$(curl -o /dev/null -s -w "%{http_code}" http://0.0.0.0:$port/v1/chat/completions                     -X POST                     -H "Content-Type: application/json"                     -H "Authorization: Bearer token-abc123"                     -d "{\"model\": \"$model_name\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}")

    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

spinup_server(){
    # Check for the next available port not in use with vllm
    PORT=8000
    while lsof -Pi :$PORT -sTCP:LISTEN -t; do
        PORT=$((PORT + 1))
    done

    cmd+="         --inf_vllm_port $PORT"
    echo "Using port: $PORT"

    echo "Starting vLLM server..."
    vllm_cmd="vllm serve qwen/qwen2.5-1.5b-instruct --api-key token-abc123 --tensor_parallel_size 1 --host 0.0.0.0 --port $PORT" #nohup launches this in the background
    echo $vllm_cmd
    nohup $vllm_cmd &

    # Wait for the server to start
    echo "Waiting for vLLM server to start..."
    SLEEP=60
    while ! check_server qwen/qwen2.5-1.5b-instruct $PORT; do
        echo "Server is not up yet. Checking again in $SLEEP seconds..."
        sleep $SLEEP
    done

    echo "vLLM server is up and running."
}

# If the method is react or act, we need to start vLLM server
if [[ "cot" = act ]] || [[ "cot" = react ]]; then
    # print the method
    echo "Method: cot"
    spinup_server
fi

if [[ "cot" =~ "rag" ]]; then
    cmd+="         --retriever_method whereisai/uae-large-v1"
fi
echo $cmd
echo "##################################################"
eval $cmd

# if the method is react or act, we need to stop the vLLM server
if [[ "cot" = act ]] || [[ "cot" = react ]]; then
    # Stop the vLLM server using pkill
    echo "Stopping vLLM server..."
    pkill -e -f vllm
    echo "vLLM server stopped."
fi
