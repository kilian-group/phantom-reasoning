"""This script generates predictions on the PhantomWiki dataset using the phantom_eval package.

Example usage:
```bash
python -m phantom_eval --method zeroshot --server vllm --model_name PATH_TO_CHECKPOINT --split_list depth_20_size_50_seed_1 --od out
```
"""

