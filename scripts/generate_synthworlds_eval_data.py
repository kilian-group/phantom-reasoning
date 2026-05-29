"""Generate a 500-sample evaluation set from SynthWorlds-RM/SM QA data."""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate SynthRM500/SM500 evaluation set by sampling from SynthWorlds QA-RM/SM data"
    )
    parser.add_argument(
        "--synth_type",
        type=str,
        required=True,
        choices=["rm", "sm"],
        help="Type of SynthWorlds dataset to use (rm: real-world, sm: synthetic)",
    )
    parser.add_argument(
        "--destination_path",
        type=str,
        default=None,
        help="Path to save the sampled evaluation set "
        "(default: data/synth<synth_type><num_samples>/minidev.json)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for shuffling")
    parser.add_argument(
        "--num_samples", type=int, default=500, help="Number of samples to include in evaluation set"
    )

    args = parser.parse_args()

    if args.destination_path is None:
        args.destination_path = f"data/synth{args.synth_type}{args.num_samples}/minidev.json"

    # Set random seed
    random.seed(args.seed)

    # Load dataset from Hugging Face
    # Login using e.g. `huggingface-cli login` to access this dataset
    print(f"Loading SynthWorlds qa-{args.synth_type} dataset (split=test) from Huggingface...")
    ds = load_dataset("kenqgu/SynthWorlds", f"qa-{args.synth_type}", split="test")
    print(f"Loaded {len(ds)} entries")

    # Convert to list of dictionaries
    all_data = []
    for i, example in enumerate(ds):
        # Keep the original structure from the dataset
        entry = dict(example)
        all_data.append(entry)

    # Shuffle the data
    random.shuffle(all_data)
    print(f"Shuffled {len(all_data)} entries with seed={args.seed}")

    # Sample the requested number of entries
    if len(all_data) < args.num_samples:
        print(f"Warning: Requested {args.num_samples} samples but only {len(all_data)} available")
        sampled_data = all_data
    else:
        sampled_data = all_data[: args.num_samples]

    print(f"Sampled {len(sampled_data)} entries")

    # Save to destination
    destination_path = Path(args.destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with open(destination_path, "w") as f:
        json.dump(sampled_data, f, indent=2)

    print(f"Saved evaluation set to {destination_path}")


if __name__ == "__main__":
    main()
