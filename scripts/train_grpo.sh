#!/usr/bin/env bash

python -m phantom_reasoner.grpo \
    --dataset_name "kilian-group/phantom-wiki-v1" \
    --model_name_or_path "Qwen/Qwen2.5-0.5B-Instruct" \
    --num_train_epochs 3 \
    --log_level "info" \
    --logging_strategy "steps" \
    --logging_first_step \
    --logging_steps 100 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    $@
    # --use_vllm \

# NOTE: Using vllm goes OOM even for 0.5B model on 160GB GPU memory (2 A100s on AIDA)
