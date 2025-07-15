import json
import os
from glob import glob

MODEL_RAW_NAME = "Qwen3-1_7B"
MODEL_CLEAN_NAME = "Qwen3-1point7B"
SEED_LIST = [1, 2, 3, 4, 5]
DIR = "evals/out-0605-qwen3/preds/cot"

for seed in SEED_LIST:
    split_name = f"depth_10_size_25_seed_{seed}"
    pattern = os.path.join(
        DIR, f"split={split_name}__model_name=.--models--{MODEL_RAW_NAME}__bs=10__bn=*.json"
    )
    output_path = os.path.join(DIR, f"{MODEL_CLEAN_NAME}-{split_name}.json")

    files = sorted(glob(pattern))
    if not files:
        print(f"[!] No files found for seed {seed}, skipping...")
        continue

    merged = []
    for file in files:
        with open(file) as f:
            data = json.load(f)
            merged.extend(data)

    with open(output_path, "w") as out_file:
        json.dump(merged, out_file, indent=2)
    print(f"[✓] Saved merged file: {output_path}")
