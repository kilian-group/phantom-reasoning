# run with: python gsm_realistic/split_datasets.py \
# --data_dir /home/yy958/phantom-wiki/phantom-reasoning/data --seed 42

import argparse
import os
import random
from pathlib import Path

SPLIT_PROB = 0.2  # keep as constant as requested


def process_file(src_file: Path, train_file: Path, eval_file: Path, split_prob: float) -> None:
    os.makedirs(train_file.parent, exist_ok=True)
    os.makedirs(eval_file.parent, exist_ok=True)

    train_lines, eval_lines = [], []
    with src_file.open("r", encoding="utf-8") as f:
        for line in f:
            if random.random() < split_prob:
                eval_lines.append(line)
            else:
                train_lines.append(line)

    if train_lines:
        with train_file.open("w", encoding="utf-8") as f_train:
            f_train.writelines(train_lines)

    if eval_lines:
        with eval_file.open("w", encoding="utf-8") as f_eval:
            f_eval.writelines(eval_lines)


def walk_and_split(src_root: Path, train_root: Path, eval_root: Path, split_prob: float) -> None:
    for src_path in src_root.rglob("*.jsonl"):
        relative_path = src_path.relative_to(src_root)
        train_path = train_root / relative_path
        eval_path = eval_root / relative_path
        process_file(src_path, train_path, eval_path, split_prob)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministically split GSM dataset into train/eval.")
    parser.add_argument(
        "--data_dir", default="data/", type=str, help="Base directory containing gsm-infinite."
    )
    parser.add_argument("--seed", default=1, type=int, help="Random seed for deterministic splitting.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    base = Path(args.data_dir)
    SRC_DIR = base / "gsm-infinite"
    TRAIN_DIR = base / "gsm-infinite-train"
    EVAL_DIR = base / "gsm-infinite-eval"

    print(f"Splitting dataset with eval probability = {SPLIT_PROB}, seed = {args.seed}")
    walk_and_split(SRC_DIR, TRAIN_DIR, EVAL_DIR, SPLIT_PROB)
    print("Finished splitting.")
