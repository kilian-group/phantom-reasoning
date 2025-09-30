#!/usr/bin/env python3

import json
import re
from collections import OrderedDict
from pathlib import Path

import pandas as pd

# Configuration constants
FINAL_CHECKPOINT_THRESHOLD = 10000
COMPLEXITY_RANGE = [2, 3, 4]  # Question complexity levels to analyze
MAX_POSITIONS = 4  # Maximum intermediate answer positions


def parse_checkpoint_info(model_name):
    """Extract checkpoint information from model name."""
    # Extract checkpoint number if present
    checkpoint_match = re.search(r"checkpoint-(\d+)", model_name)
    if checkpoint_match:
        step = int(checkpoint_match.group(1))
        return {"type": "checkpoint", "checkpoint": step, "name": f"{step} training steps"}

    # Everything else is either base or final model
    if "grpo" in model_name:
        return {"type": "final", "checkpoint": FINAL_CHECKPOINT_THRESHOLD, "name": "Final Model"}
    else:
        return {"type": "base", "checkpoint": 0, "name": "Base Model"}


def extract_model_size(model_name):
    """Extract model size from model name."""
    # Extract size pattern like "0.6B", "1.7B", "2.5B", etc.
    match = re.search(r"(\d+\.?\d*B)", model_name)
    return match.group(1) if match else model_name


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
            checkpoint_info = parse_checkpoint_info(model_name)
        except (IndexError, ValueError):
            print(f"Warning: Could not parse model name from {pred_file.name}")
            continue

        if checkpoint_info:
            model_size = extract_model_size(model_name)
            key = (checkpoint_info["checkpoint"], model_size)

            with open(pred_file) as f:
                predictions = json.load(f)
            checkpoint_data[key] = {
                "predictions": predictions,
                "info": checkpoint_info,
                "model_size": model_size,
            }

    return OrderedDict(sorted(checkpoint_data.items()))


def load_ground_truth_msq(gt_file_path):
    """Load MuSiQue ground truth data."""
    gt_file = Path(gt_file_path)

    ground_truth = {}
    with open(gt_file) as f:
        for line in f:
            item = json.loads(line.strip())
            question_id = item["id"]
            ground_truth[question_id] = item

    return ground_truth


def analyze_msq_decomposition_for_checkpoint(predictions, ground_truth):
    """Analyze MuSiQue question decomposition for a single checkpoint."""
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

            if step_answer and len(step_answer) >= 3:
                if step_answer in reasoning_text:
                    breakdown[num_steps]["found"][step_idx] += 1

    return breakdown


def create_csv(all_experiment_results, output_dir):
    """Create CSV file with breakdown data."""
    csv_data = []

    for experiment_name, experiment_results in all_experiment_results.items():
        for (checkpoint_step, model_size), results in experiment_results.items():
            breakdown = results["breakdown"]

            # For each complexity and position, add a row
            for complexity in COMPLEXITY_RANGE:
                if complexity in breakdown:
                    total_questions = breakdown[complexity]["total"]
                    found_counts = breakdown[complexity]["found"]

                    for pos in range(1, min(complexity + 1, MAX_POSITIONS + 1)):
                        if pos <= len(found_counts):
                            found_count = found_counts[pos - 1]  # Convert to 0-indexed
                            proportion = found_count / total_questions if total_questions > 0 else 0.0

                            csv_data.append(
                                {
                                    "experiment": experiment_name,
                                    "checkpoint": checkpoint_step,
                                    "kth_intermediate_answer": pos,
                                    "proportion_found": proportion,
                                    "found_count": found_count,
                                    "total_questions": total_questions,
                                    "complexity": complexity,
                                }
                            )

    df = pd.DataFrame(csv_data)
    csv_file = output_dir / "reasoning_evolution.csv"  # Keep hardcoded for now
    df.to_csv(csv_file, index=False)

    print(f"CSV saved to {csv_file}")

    return df


def main():
    """Main function to analyze multiple training experiments and generate CSV."""
    # Configuration - modify these paths as needed
    config = {
        "checkpoints_base_dir": Path("eval/training_checkpoints"),
        "ground_truth_file": "eval/datasets/msq500/musique_ans_v1.0_minidev.jsonl",
        "preds_subdir": "out-msq500/preds/cot",
        "output_dir": Path("scripts/plots"),
        "output_filename": "reasoning_evolution.csv",
    }
    # Create output directory
    config["output_dir"].mkdir(exist_ok=True)

    # Find all experiment directories
    experiment_dirs = [
        d
        for d in config["checkpoints_base_dir"].iterdir()
        if d.is_dir() and (d / config["preds_subdir"]).exists()
    ]
    experiment_dirs.sort()

    ground_truth = load_ground_truth_msq(config["ground_truth_file"])
    all_experiment_results = {}

    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        checkpoint_data = load_experiment_data(exp_dir, config["preds_subdir"])
        if not checkpoint_data:
            continue

        experiment_results = {}
        for (checkpoint_step, model_size), data in checkpoint_data.items():
            breakdown = analyze_msq_decomposition_for_checkpoint(data["predictions"], ground_truth)
            experiment_results[(checkpoint_step, model_size)] = {
                "breakdown": breakdown,
                "info": data["info"],
                "model_size": model_size,
            }

        all_experiment_results[exp_name] = experiment_results

    create_csv(all_experiment_results, config["output_dir"])


if __name__ == "__main__":
    main()
