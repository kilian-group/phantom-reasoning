# LightEval

To understand the lighteval CLI, please run:

```bash
lighteval vllm --help
```

**Relevant vLLM parameters:**

- `--output-dir`: Output directory for evaluation results. \[default: results\]
- `--save-details`: save a parquet file containing sample-by-sample details.

## GSM8K

To evaluate Qwen3 on GSM8K with 4 few-shot examples, please use the following command:

```bash
# 0.6B model
lighteval vllm model_configs/Qwen3-0.6B.yaml "lighteval|gsm8k|4|0" --use-chat-template --save-details
# 1.7B model
lighteval vllm model_configs/Qwen3-1.7B.yaml "lighteval|gsm8k|4|0" --use-chat-template --save-details
```

> \[!IMPORTANT\]
> The `--use-chat-template` flag is essential for eliciting normal behavior from instruction-tuned models.

> \[!IMPORTANT\]
> When using a local model with vLLM, "model_name" in the yaml config must be an absolute path.

> \[!TIP\]
> Use `null` instead of -1 to disable vLLM parameters. For example, `top_k: null` considers all tokens when sampling.
