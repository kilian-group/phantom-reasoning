#!/usr/bin/env python3
import argparse
import json
import os
import random

import reasoning_gym


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description="Generate and split reasoning_gym dataset to JSONL.")
    p.add_argument("--dataset", required=True, help="dataset type, e.g.'leg_counting'")
    p.add_argument("--size", type=int, required=True, help="total number of generation per dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    if not (0.0 < args.train_frac < 1.0):
        raise ValueError("--train-frac must be in (0,1)")

    data = reasoning_gym.create_dataset(args.dataset, size=args.size, seed=args.seed)
    entries = []
    for x in data:
        entries.append({"question": x["question"], "answer": x["answer"], "metadata": x.get("metadata", {})})

    rng = random.Random(args.seed)
    idx = list(range(len(entries)))
    rng.shuffle(idx)
    cut = int(len(idx) * args.train_frac)
    train_idx = idx[:cut]
    eval_idx = idx[cut:]

    train = [entries[i] for i in train_idx]
    eval_ = [entries[i] for i in eval_idx]

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.jsonl")
    eval_path = os.path.join(out_dir, "eval.jsonl")
    write_jsonl(train_path, train)
    write_jsonl(eval_path, eval_)

    print(f"Dataset: {args.dataset}")
    print(f"Total: {len(entries)}, train: {len(train)}, eval: {len(eval_)}")
    print(f"Saved: {train_path}")
    print(f"Saved: {eval_path}")


if __name__ == "__main__":
    main()
