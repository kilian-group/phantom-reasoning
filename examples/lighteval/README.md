# LightEval

To understand the lighteval CLI, please run:

```bash
lighteval vllm --help
```

## GSM8K

To run GSM8K with 4 few-shot examples, please use the following command:

```bash
lighteval vllm model_configs/qwen3.yaml "lighteval|gsm8k|4|0" --save-details --use-chat-template
```

> \[!IMPORTANT\]
> The `--use-chat-template` flag is essential for eliciting normal behavior from instruction-tuned models.

The `--save-details` flag tells lighteval to save a parquet file containing sample-by-sample details.
