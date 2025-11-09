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


rg_task2config = {
    "family_relationships": {
        # family_size is the proxy difficulty of the task
        "min_family_size": 3,
        "max_family_size": 20,
    },
    "knights_knaves": {
        # NOTE: what is the difficulty parameter for this task?
        "n_people": 2,
        # TODO what is the depth and width of the statement?
    },
}


def main():
    parser = argparse.ArgumentParser(description="Generate and split reasoning_gym dataset to JSONL.")
    parser.add_argument(
        "--dataset", required=True, help="dataset type, e.g.'family_relationships', 'knights_knaves', etc."
    )
    parser.add_argument("--size", type=int, required=True, help="total number of generation per dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--out_dir", "-od", type=str, required=True)
    args = parser.parse_args()

    if not (0.0 < args.train_frac < 1.0):
        raise ValueError("--train_frac must be in (0,1)")

    optional_config = rg_task2config.get(args.dataset, {})
    data = reasoning_gym.create_dataset(args.dataset, size=args.size, seed=args.seed, **optional_config)
    entries = []
    for x in data:
        entries.append({"question": x["question"], "answer": x["answer"], "metadata": x.get("metadata", {})})

    # Shuffle entries list randomly with seed, then split into train and eval
    rng = random.Random(args.seed)
    rng.shuffle(entries)
    cut = int(len(entries) * args.train_frac)
    train_entries = entries[:cut]
    eval_entries = entries[cut:]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.jsonl")
    eval_path = os.path.join(args.out_dir, "eval.jsonl")
    write_jsonl(train_path, train_entries)
    write_jsonl(eval_path, eval_entries)

    print(f"Dataset: {args.dataset}, seed: {args.seed}, train_frac: {args.train_frac}")
    print(f"Total: {len(entries)}, train: {len(train_entries)}, eval: {len(eval_entries)}")
    print(f"Saved: {train_path}")
    print(f"Saved: {eval_path}")


if __name__ == "__main__":
    main()
