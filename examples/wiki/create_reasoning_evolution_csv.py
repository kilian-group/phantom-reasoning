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
        --train_dataset hp \
        --eval_dataset hp500 \
        --training_seed -1

Command Line Arguments:
    --final_ckpts_yaml_path: Path to YAML file containing checkpoint configurations
    --data_dir: Base directory containing datasets
    --train_dataset: Training dataset to filter from YAML (e.g. 'pw', 'gsminf', 'hp' etc.)
    --eval_dataset: Evaluation dataset to load ground truth from (e.g. 'msq500', 'cofca500')
    --split: Dataset split to use (e.g., 'minidev')
    --figures_dir: Directory for output CSV
    --training_seed: Training seed to filter experiments (1, 2, etc., or -1 for all seeds, default: 1)

Output:
    CSV file with columns at location:
      "<figures_dir>/reasoning_evolution__train=<train_dataset>__eval=<eval_dataset>.csv"
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
EVAL_DATASET2COMPLEXITY_RANGE = {
    "msq500": [2, 3, 4],  # MSQ has 2-4 hops https://arxiv.org/abs/2108.00573
    "cofca500": [2, 3, 4],  # CofCA has 2-4 hops https://arxiv.org/abs/2402.11924v5
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze training checkpoint predictions to track reasoning evolution"
    )
    parser.add_argument(
        "--final_ckpts_yaml_path",
        type=str,
        required=True,
        help="Path to YAML file containing checkpoint configurations",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base directory containing datasets (default: data)",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="pw",
        help="Training dataset name to filter experiments from YAML, "
        "e.g. 'pw', 'hp', '2wiki', 'msq' etc. (default: pw)",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="msq500",
        choices=["msq500", "cofca500"],
        help="Evaluation dataset to load ground truth from, " "e.g. 'msq500', 'cofca500' (default: msq500)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="minidev",
        help="Dataset split to use (default: minidev)",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="scripts/final_plots/figures",
        help="Directory for output CSV (default: scripts/final_plots/figures)",
    )
    parser.add_argument(
        "--training_seed",
        type=int,
        default=1,
        help="Training seed to filter experiments (default: 1, use -1 for all seeds)",
    )
    return parser.parse_args()


def extract_checkpoint_number(model_name: str) -> int | None:
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


def load_experiment_data(experiment_dir: Path, eval_dataset: str) -> dict:
    """Load all training checkpoint predictions for one experiment."""
    preds_dir = experiment_dir / f"out-{eval_dataset}" / "preds" / "cot"
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


def load_gt_qa_pairs(data_dir: Path, dataset_name: str, split: str) -> list:
    """Load ground truth data preserving all fields including question_decomposition.

    Args:
        data_dir: Base directory containing datasets
        dataset_name: Name of dataset (e.g., 'msq500', 'hp500', '2wiki500')
        split: Dataset split (e.g., 'minidev')

    Returns:
        List of QA pairs with "id", "question", "answer", "question_decomposition"
    """
    data = load_data(
        data_dir=data_dir,
        dataset=dataset_name,
        split=split,
        intermediate_answers=True,
    )

    return data["qa_pairs"]


def analyze_msq_decomposition_for_checkpoint(predictions: dict, gt_qa_pairs: list) -> dict:
    """Analyze MuSiQue question decomposition for a single checkpoint.

    Args:
        predictions: Dict mapping question_id -> prediction data
            with "interaction" field
        gt_qa_pairs: list of QA pairs with "id", "question_decomposition"
            in ground truth evaluation dataset

    Returns:
        Dict mapping complexity level -> {"total": int, "found": list[int]}
        Example: {2: {"total": 100, "found": [80, 60]},
                  3: {"total": 50, "found": [40, 35, 25]}}
    """
    breakdown = {}

    # Track by number of decomposition steps
    gt_question_ids = {q["id"] for q in gt_qa_pairs}
    gt_question_ids_to_decomp = {q["id"]: q["question_decomposition"] for q in gt_qa_pairs}
    for question_id, pred_data in predictions.items():
        if question_id not in gt_question_ids:
            continue

        decomp_steps = gt_question_ids_to_decomp[question_id]
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


def create_csv(all_experiment_results: dict, eval_dataset: str, output_csv_path: Path) -> None:
    """Create CSV file with breakdown data."""
    csv_data = []

    for experiment_name, experiment_results in all_experiment_results.items():
        for checkpoint_step, results in experiment_results.items():
            breakdown = results["breakdown"]

            # For each complexity and position, add a row
            for complexity in EVAL_DATASET2COMPLEXITY_RANGE[eval_dataset]:
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
    df.to_csv(output_csv_path, index=False)
    print(f"CSV saved to {output_csv_path}")


def main():
    """Main function to analyze multiple training experiments and generate CSV."""
    # Parse command line arguments
    args = parse_args()
    args.final_ckpts_yaml_path = Path(args.final_ckpts_yaml_path)
    args.data_dir = Path(args.data_dir)
    args.figures_dir = Path(args.figures_dir)
    args.output_csv_filename = (
        f"reasoning_evolution__train={args.train_dataset}__eval={args.eval_dataset}.csv"
    )
    args.output_csv_path = args.figures_dir / args.output_csv_filename

    # Create output directory
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # Load YAML configuration
    with open(args.final_ckpts_yaml_path) as f:
        yaml_config = yaml.safe_load(f)

    # Load ground truth data (use eval_dataset)
    ground_truth = load_gt_qa_pairs(
        data_dir=args.data_dir,
        dataset_name=args.eval_dataset,
        split=args.split,
    )

    all_experiment_results = {}

    # Process all checkpoint groups in YAML (real_train_ckpts, synthetic_train_ckpts, etc.)
    for _, datasets in yaml_config.items():
        if not isinstance(datasets, list):
            continue

        for dataset_config in datasets:
            # Only process the specified train dataset
            if args.train_dataset != dataset_config["dataset_name"]:
                continue

            for ckpt_group in dataset_config["ckpts"]:
                model_name = ckpt_group.get("model", "unknown")
                paths = ckpt_group.get("paths", [])

                for exp_path in paths:
                    exp_dir = Path(exp_path)
                    exp_name = f"{model_name}/{exp_dir.name}"

                    # Filter by training seed if specified
                    if args.training_seed != -1:
                        seed_pattern = f"training_seed={args.training_seed}"
                        if seed_pattern not in str(exp_dir):
                            continue

                    checkpoint_data = load_experiment_data(exp_dir, args.eval_dataset)
                    if not checkpoint_data:
                        continue

                    experiment_results = {}
                    for checkpoint_step, data in checkpoint_data.items():
                        breakdown = analyze_msq_decomposition_for_checkpoint(
                            data["predictions"], ground_truth
                        )
                        experiment_results[checkpoint_step] = {"breakdown": breakdown}

                    all_experiment_results[exp_name] = experiment_results

    create_csv(all_experiment_results, args.eval_dataset, args.output_csv_path)


if __name__ == "__main__":
    main()
