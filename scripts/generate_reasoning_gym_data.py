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
    "knights_knaves": {"n_people": [2, 3, 4, 5, 6]},
}


def generate_knights_knaves_data(
    size: int, seed: int, train_frac: float, **optional_config
) -> tuple[list[dict], list[dict]]:
    """
    Generate knights_knaves data going over all possible values of n_people.
    This generates a dataset with different difficulty levels.
    """
    train_entries = []
    eval_entries = []
    difficulty_levels = rg_task2config["knights_knaves"]["n_people"]
    size_per_difficulty_level = size // len(difficulty_levels)
    for n_people in difficulty_levels:
        print(f"Generating knights_knaves data for {n_people} people")
        optional_config = {
            "n_people": n_people,
        }
        data = reasoning_gym.create_dataset(
            "knights_knaves", size=size_per_difficulty_level, seed=seed, **optional_config
        )
        entries = []
        for x in data:
            entries.append(
                {"question": x["question"], "answer": x["answer"], "metadata": x.get("metadata", {})}
            )

        # Shuffle entries list randomly with seed, then split into train and eval
        rng = random.Random(seed)
        rng.shuffle(entries)
        cut = int(len(entries) * train_frac)
        train_entries.extend(entries[:cut])
        eval_entries.extend(entries[cut:])

    return train_entries, eval_entries


def generate_data(dataset: str, size: int, seed: int, train_frac: float) -> tuple[list[dict], list[dict]]:
    optional_config = rg_task2config.get(dataset, {})
    data = reasoning_gym.create_dataset(dataset, size=size, seed=seed, **optional_config)
    entries = []
    for x in data:
        entries.append({"question": x["question"], "answer": x["answer"], "metadata": x.get("metadata", {})})

    # Shuffle entries list randomly with seed, then split into train and eval
    rng = random.Random(seed)
    rng.shuffle(entries)
    cut = int(len(entries) * train_frac)
    train_entries = entries[:cut]
    eval_entries = entries[cut:]
    return train_entries, eval_entries


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

    match args.dataset:
        case "knights_knaves":
            train_entries, eval_entries = generate_knights_knaves_data(args.size, args.seed, args.train_frac)
        case _:
            train_entries, eval_entries = generate_data(args.dataset, args.size, args.seed, args.train_frac)

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.jsonl")
    eval_path = os.path.join(args.out_dir, "eval.jsonl")
    write_jsonl(train_path, train_entries)
    write_jsonl(eval_path, eval_entries)

    print(f"Dataset: {args.dataset}, seed: {args.seed}, train_frac: {args.train_frac}")
    print(
        f"Total: {len(train_entries) + len(eval_entries)}, "
        f"train: {len(train_entries)}, eval: {len(eval_entries)}"
    )
    print(f"Saved: {train_path}")
    print(f"Saved: {eval_path}")


if __name__ == "__main__":
    main()
