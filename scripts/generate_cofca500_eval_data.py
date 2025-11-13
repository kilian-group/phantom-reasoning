"""Generate a 500-sample evaluation set from COFCA multi-hop QA data."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

# Global dictionary to track the number of occurrences of each ID+type combination.
# There are question IDs common even with _id_<type> information, so we will use this
# dict to avoid duplicates IDs.
ID_TYPE2COUNTS: dict[str, int] = {}


def update_id_with_hop_number(entry: dict) -> dict:
    """Update entry["_id"] with type data. If the new id already exists, add a number to the end."""
    if isinstance(entry["type"], list):
        type_str = "-".join(entry["type"])
    else:
        type_str = entry["type"]

    id_type_str = f"{entry['_id']}__{type_str}"

    if id_type_str in ID_TYPE2COUNTS:
        ID_TYPE2COUNTS[id_type_str] += 1
    else:
        ID_TYPE2COUNTS[id_type_str] = 1

    entry["_id"] = f"{id_type_str}__{ID_TYPE2COUNTS[id_type_str]}"
    return entry


def load_json_files(source_dir: Path) -> list[dict[str, Any]]:
    """Load all JSON files from the source directory.

    Args:
        source_dir: Directory containing JSON files

    Returns:
        Combined list of all entries from all JSON files
    """
    all_data = []
    json_files = sorted(source_dir.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {source_dir}")

    for json_file in json_files:
        print(f"Loading {json_file.name}...")
        with open(json_file) as f:
            data = json.load(f)
            data = [update_id_with_hop_number(entry) for entry in data]
            all_data.extend(data)

    print(f"Loaded {len(all_data)} total entries from {len(json_files)} files")
    return all_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate COFCA500 evaluation set by sampling from multi-hop QA data"
    )
    parser.add_argument(
        "--source_data_dir", type=str, default="data/cofca/", help="Directory containing source JSON files"
    )
    parser.add_argument(
        "--destination_path",
        type=str,
        default="data/cofca500/minidev.json",
        help="Path to save the sampled evaluation set",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for shuffling")
    parser.add_argument(
        "--num_samples", type=int, default=500, help="Number of samples to include in evaluation set"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load and combine data
    source_dir = Path(args.source_data_dir)
    if not source_dir.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")

    all_data = load_json_files(source_dir)

    # Shuffle the combined data
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
