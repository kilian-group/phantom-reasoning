"""
Script to extract model names from final_ckpts.yaml depending on the dataset name
(base, pw, gsminf, rg-family_relationships, hp, 2wiki, msq).

Optionally also filter by model name.

Example usage:
```bash
python scripts/final_plots/get_model_names_of_final_ckpts.py \
    --final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml \
    --dataset_name pw \
    --model_names "Qwen/Qwen3-0.6B" "Qwen/Qwen2.5-7B-Instruct"
```

Outputs space-separated model names to stdout, which can be piped into bash scripts.
"""

import argparse
from pathlib import Path

import yaml


def main(args: argparse.Namespace):
    # Load config and extract model names
    with open(Path(args.final_ckpts_yaml_path)) as f:
        config = yaml.safe_load(f)

    model_names: list[str] = []
    match args.dataset_name:
        case "base":
            # For base dataset, we use the model names from the synthetic_train_ckpts for the pw dataset
            # Search through all synthetic_train_ckpts for pw dataset
            for synthetic_train_ckpts in config["synthetic_train_ckpts"]:
                if synthetic_train_ckpts["dataset_name"] == "pw":
                    # ckpt["model"] will be like "Qwen/Qwen3-0.6B", so add to list
                    model_names = [ckpt["model"] for ckpt in synthetic_train_ckpts["ckpts"]]
                    break
            # Filter the model names to only include the ones in the list
            if args.model_names:
                model_names = [name for name in model_names if name in args.model_names]
        case "pw" | "gsminf" | "rg-family_relationships" | "rg-knights_knaves":
            # Search through all synthetic_train_ckpts and append the paths
            for synthetic_train_ckpts in config["synthetic_train_ckpts"]:
                if synthetic_train_ckpts["dataset_name"] == args.dataset_name:
                    for ckpt in synthetic_train_ckpts["ckpts"]:
                        # ckpt["paths"] will be like ["path1", "path2"] so add to list
                        if args.model_names and ckpt["model"] not in args.model_names:
                            continue
                        model_names.extend(ckpt["paths"])
                    break
        case "hp" | "2wiki" | "msq":
            for real_train_ckpts in config["real_train_ckpts"]:
                if real_train_ckpts["dataset_name"] == args.dataset_name:
                    for ckpt in real_train_ckpts["ckpts"]:
                        # ckpt["paths"] will be like ["path1", "path2"] so add to list
                        if args.model_names and ckpt["model"] not in args.model_names:
                            continue
                        model_names.extend(ckpt["paths"])
                    break
        case _:
            raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    # Output space-separated model names
    print(" ".join(model_names))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract model names from final_ckpts.yaml depending on the dataset name"
    )
    parser.add_argument(
        "--final_ckpts_yaml_path",
        type=str,
        default="scripts/final_plots/final_ckpts.yaml",
        help="Path to the final_ckpts.yaml file",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=[
            "base",
            "pw",
            "gsminf",
            "rg-family_relationships",
            "rg-knights_knaves",
            "hp",
            "2wiki",
            "msq",
        ],
        help="Name of the dataset to extract model names for",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=[],
        help="Model names to filter by (space-separated list, optional)",
    )
    args = parser.parse_args()

    main(args)
