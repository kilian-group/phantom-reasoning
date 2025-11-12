"""
Script to extract model names from final_ckpts.yaml depending on the dataset name
(base, pw, gsminf),

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
        case "pw" | "gsminf" | "rg-family_relationships":
            # Search through all synthetic_train_ckpts and append the paths
            for synthetic_train_ckpts in config["synthetic_train_ckpts"]:
                if synthetic_train_ckpts["dataset_name"] == args.dataset_name:
                    for ckpt in synthetic_train_ckpts["ckpts"]:
                        # ckpt["paths"] will be like ["path1", "path2"] so add to list
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
        choices=["base", "pw", "gsminf", "rg-family_relationships"],
        help="Name of the dataset to extract model names for",
    )
    args = parser.parse_args()

    main(args)
