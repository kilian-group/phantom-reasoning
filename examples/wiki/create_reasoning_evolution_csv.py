#!/usr/bin/env python3
"""Reasoning Evolution Analysis Script

Analyzes training checkpoint predictions to track how models learn to find intermediate answers
in multi-step reasoning questions. Generates a CSV with metrics on intermediate answer discovery
across different checkpoints and question complexities.

Usage:
    python examples/wiki/create_reasoning_evolution_csv.py [OPTIONS]

    # Run with defaults (pw train, msq500 eval, training_seed=1)
    python examples/wiki/create_reasoning_evolution_csv.py

    # Run on hp dataset with all seeds
    python examples/wiki/create_reasoning_evolution_csv.py \
        --train-dataset hp \
        --eval-dataset hp500 \
        --training-seed -1

    # Specify custom output
    python examples/wiki/create_reasoning_evolution_csv.py \
        --output-filename my_results.csv

Command Line Arguments:
    --yaml-path: Path to YAML file containing checkpoint configurations
    --data-dir: Base directory containing datasets
    --train-dataset: Training dataset to filter from YAML (e.g. 'pw', 'hp', '2wiki', 'msq')
    --eval-dataset: Evaluation dataset to load ground truth from (e.g. 'msq500', 'hp500')
    --split: Dataset split to use (e.g., 'minidev')
    --preds-subdir: Subdirectory path to prediction JSON files
    --output-dir: Directory for output CSV
    --output-filename: Name of output CSV file
    --training-seed: Training seed to filter experiments (1, 2, etc., or -1 for all seeds)

Output:
    CSV file with columns:
    - experiment: Experiment directory name
    - checkpoint: Training checkpoint number (0=base, max_checkpoint+500=final)
    - kth_intermediate_answer: Position of intermediate answer (1-4)
    - proportion_found: Proportion of questions where answer was found
    - found_count: Count of questions where answer was found
    - total_questions: Total questions at this complexity
    - complexity: Question complexity (number of reasoning steps)
"""

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import yaml
from utils.data_utils import load_data

# Configuration constants
COMPLEXITY_RANGE = [2, 3, 4]  # MSQ has 2-4 hops https://arxiv.org/pdf/2108.00573


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze training checkpoint predictions to track reasoning evolution"
    )
    parser.add_argument(
        "--yaml-path",
        type=str,
        default="scripts/final_plots/final_ckpts.yaml",
        help="Path to YAML file containing checkpoint configurations "
        "(default: scripts/final_plots/final_ckpts.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        # default="/anvil/projects/x-nairr250102/phantom-reasoning/data",
        default="/share/nikola/phantom-reasoning/data",
        help="Base directory containing datasets " "(default: /share/nikola/phantom-reasoning/data)",
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        default="pw",
        help="Training dataset name to filter experiments from YAML, "
        "e.g. 'pw', 'hp', '2wiki', 'msq' (default: pw)",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default="msq500",
        help="Evaluation dataset to load ground truth from, "
        "e.g. 'msq500', 'hp500', '2wiki500' (default: msq500)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="minidev",
        help="Dataset split to use (default: minidev)",
    )
    parser.add_argument(
        "--preds-subdir",
        type=str,
        default="out-msq500/preds/cot",
        help="Subdirectory path to prediction JSON files " "(default: out-msq500/preds/cot)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scripts/plots",
        help="Directory for output CSV (default: scripts/plots)",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="reasoning_evolution.csv",
        help="Name of output CSV file (default: reasoning_evolution.csv)",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=1,
        help="Training seed to filter experiments (default: 1, use -1 for all seeds)",
    )
    return parser.parse_args()


def extract_checkpoint_number(model_name):
    """Extract checkpoint number from model name.

    Returns:
        int or None: Checkpoint number (0 for base models, None for grpo/final, or actual step number)
    """
    checkpoint_match = re.search(r"checkpoint-(\d+)", model_name)
    if checkpoint_match:
        return int(checkpoint_match.group(1))
    elif "grpo" in model_name:
        return None  # Will be set to max + 100 later
    else:
        return 0  # Base model


def load_experiment_data(experiment_dir, preds_subdir):
    """Load all training checkpoint predictions for one experiment."""
    preds_dir = experiment_dir / preds_subdir
    if not preds_dir.exists():
        print(f"Warning: {preds_dir} does not exist")
        return {}

    checkpoint_data = {}

    for pred_file in preds_dir.glob("*.json"):
        # Extract model name from filename (format: ...model_name=<NAME>__bs=...)
        try:
            model_name = pred_file.stem.split("model_name=")[1].split("__bs=")[0]
            checkpoint_num = extract_checkpoint_number(model_name)

            with open(pred_file) as f:
                predictions = json.load(f)
            checkpoint_data[checkpoint_num] = {"predictions": predictions}
        except (IndexError, ValueError):
            print(f"Warning: Could not parse model name from {pred_file.name}")
            continue

    # Replace None (grpo/final) with max_checkpoint + 100
    if None in checkpoint_data:
        final_data = checkpoint_data.pop(None)
        # Find max checkpoint number (excluding 0 and None)
        numeric_checkpoints = [k for k in checkpoint_data.keys() if k is not None and k > 0]
        if numeric_checkpoints:
            max_checkpoint = max(numeric_checkpoints)
            final_checkpoint = max_checkpoint + 500
        checkpoint_data[final_checkpoint] = final_data

    return OrderedDict(sorted(checkpoint_data.items()))


def load_ground_truth(data_dir, dataset_name, split):
    """Load ground truth data preserving all fields including question_decomposition.

    Args:
        data_dir: Base directory containing datasets
        dataset_name: Name of dataset (e.g., 'msq500', 'hp500', '2wiki500')
        split: Dataset split (e.g., 'minidev')

    Returns:
        Dict mapping question_id -> full question data including decomposition info
    """
    # Use data_utils with intermediate_answers=True
    # This preserves question_decomposition and returns a dict keyed by question ID
    data = load_data(
        data_dir=data_dir,
        dataset=dataset_name,
        split=split,
        intermediate_answers=True,
    )

    return data["qa_pairs"]


def analyze_msq_decomposition_for_checkpoint(predictions: dict, ground_truth: dict) -> dict:
    """Analyze MuSiQue question decomposition for a single checkpoint.

    Args:
        predictions: Dict mapping question_id -> prediction data
            with "interaction" field
        ground_truth: Dict mapping question_id -> ground truth
            with "question_decomposition" field

    Returns:
        Dict mapping complexity level -> {"total": int, "found": list[int]}
        Example: {2: {"total": 100, "found": [80, 60]},
                  3: {"total": 50, "found": [40, 35, 25]}}
    """
    breakdown = {}

    # Track by number of decomposition steps
    for question_id, pred_data in predictions.items():
        if question_id not in ground_truth:
            continue

        gt_item = ground_truth[question_id]
        if "question_decomposition" not in gt_item:
            continue

        decomp_steps = gt_item["question_decomposition"]
        num_steps = len(decomp_steps)

        # Initialize breakdown for this complexity
        if num_steps not in breakdown:
            breakdown[num_steps] = {"total": 0, "found": [0] * num_steps}

        breakdown[num_steps]["total"] += 1

        # Extract reasoning text from interaction messages
        interaction = pred_data.get("interaction", {})
        reasoning_text = ""

        if isinstance(interaction, dict) and "messages" in interaction:
            for message in interaction["messages"]:
                if message.get("role") == "assistant":
                    content = message.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                reasoning_text += item.get("text", "") + " "
                    elif isinstance(content, str):
                        reasoning_text += content + " "

        reasoning_text = reasoning_text.lower()

        # Check each step
        for step_idx, step_info in enumerate(decomp_steps):
            step_answer = step_info.get("answer", "").lower().strip()

            # if (
            #     step_answer and len(step_answer) >= 3
            # ):  # avoid matching "to", "in", "is", but can be removed
            if step_answer in reasoning_text:
                breakdown[num_steps]["found"][step_idx] += 1

    return breakdown


def create_csv(all_experiment_results, output_dir, output_filename):
    """Create CSV file with breakdown data."""
    csv_data = []

    for experiment_name, experiment_results in all_experiment_results.items():
        for checkpoint_step, results in experiment_results.items():
            breakdown = results["breakdown"]

            # For each complexity and position, add a row
            for complexity in COMPLEXITY_RANGE:
                if complexity in breakdown:
                    total_questions = breakdown[complexity]["total"]
                    found_counts = breakdown[complexity]["found"]

                    # Iterate through each intermediate answer position (1-indexed)
                    # For complexity=3 questions, check positions 1, 2, 3
                    for position in range(1, complexity + 1):
                        # Convert to 0-indexed
                        found_count = found_counts[position - 1]
                        if total_questions > 0:
                            proportion = found_count / total_questions
                        else:
                            proportion = 0.0

                        csv_data.append(
                            {
                                "experiment": experiment_name,
                                "checkpoint": checkpoint_step,
                                "kth_intermediate_answer": position,
                                "proportion_found": proportion,
                                "found_count": found_count,
                                "total_questions": total_questions,
                                "complexity": complexity,
                            }
                        )

    df = pd.DataFrame(csv_data)
    csv_file = output_dir / output_filename
    df.to_csv(csv_file, index=False)

    print(f"CSV saved to {csv_file}")


def main():
    """Main function to analyze multiple training experiments and generate CSV."""
    # Parse command line arguments
    args = parse_args()

    # Convert args to config dictionary with Path objects
    config = {
        "yaml_path": Path(args.yaml_path),
        "data_dir": args.data_dir,
        "train_dataset": args.train_dataset,
        "eval_dataset": args.eval_dataset,
        "split": args.split,
        "preds_subdir": args.preds_subdir,
        "output_dir": Path(args.output_dir),
        "output_filename": args.output_filename,
        "training_seed": None if args.training_seed == -1 else args.training_seed,
    }

    # Create output directory
    config["output_dir"].mkdir(exist_ok=True)

    # Load YAML configuration
    with open(config["yaml_path"]) as f:
        yaml_config = yaml.safe_load(f)

    # Load ground truth data (use eval_dataset)
    ground_truth = load_ground_truth(
        data_dir=config["data_dir"],
        dataset_name=config["eval_dataset"],
        split=config["split"],
    )

    all_experiment_results = {}

    # Process all checkpoint groups in YAML (real_train_ckpts, synthetic_train_ckpts, etc.)
    for group_name, datasets in yaml_config.items():
        if not isinstance(datasets, list):
            continue

        for dataset_config in datasets:
            dataset_name = dataset_config.get("dataset_name", "")

            # Filter by train_dataset if specified
            if config["train_dataset"] and dataset_name != config["train_dataset"]:
                continue

            ckpts = dataset_config.get("ckpts", [])

            for ckpt_group in ckpts:
                model_name = ckpt_group.get("model", "unknown")
                paths = ckpt_group.get("paths", [])

                for exp_path in paths:
                    exp_dir = Path(exp_path)
                    # Include model name in experiment name (extract part after "/")
                    model_short = model_name.split("/")[-1]
                    exp_name = f"{model_short}_{exp_dir.name}"

                    # Filter by training seed if specified
                    if config["training_seed"] is not None:
                        seed_pattern = f"training_seed={config['training_seed']}"
                        if seed_pattern not in str(exp_dir):
                            continue

                    checkpoint_data = load_experiment_data(exp_dir, config["preds_subdir"])
                    if not checkpoint_data:
                        continue

                    experiment_results = {}
                    for checkpoint_step, data in checkpoint_data.items():
                        breakdown = analyze_msq_decomposition_for_checkpoint(
                            data["predictions"], ground_truth
                        )
                        experiment_results[checkpoint_step] = {"breakdown": breakdown}

                    all_experiment_results[exp_name] = experiment_results

    create_csv(all_experiment_results, config["output_dir"], config["output_filename"])


if __name__ == "__main__":
    main()
