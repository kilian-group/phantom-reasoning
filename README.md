# phantom-reasoning

**Setup instructions:**
```bash
conda create -n rsn python=3.12
# NOTE: vllm automatically installs pytorch
pip install vllm
pip install flash-attn --no-build-isolation
pip install deepspeed trl wandb accelerate datasets "huggingface_hub[cli]"
```

Install `phantom_reasoner` in development mode:
```bash
pip install -e .
```

## SFT
From https://github.com/huggingface/open-r1?tab=readme-ov-file#sft:
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes 3 --config_file recipes/accelerate_configs/zero3.yaml \
    src/phantom_reasoner/sft.py \
    --config recipes/qwen2.5-1.5b-instruct/sft/config_demo.yaml
```
NOTE: add CLI arguments before `--config_file` to override the arguments in `zero3.yaml`