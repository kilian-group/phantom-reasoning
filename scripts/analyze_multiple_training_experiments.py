#!/usr/bin/env python3

import json
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_checkpoint_info(model_name):
    """Extract checkpoint information from model name."""
    if "Qwen--Qwen3-0.6B" in model_name and "checkpoint" not in model_name and "grpo" not in model_name:
        return {"type": "base", "checkpoint": 0, "name": "Base Model"}
    elif "Qwen--Qwen3-1.7B" in model_name and "checkpoint" not in model_name and "grpo" not in model_name:
        return {"type": "base", "checkpoint": 0, "name": "Base Model"}
    elif "checkpoint-" in model_name:
        match = re.search(r"checkpoint-(\d+)", model_name)
        if match:
            step = int(match.group(1))
            return {"type": "checkpoint", "checkpoint": step, "name": f"{step} training steps"}
    elif "grpo" in model_name and "checkpoint" not in model_name:
        return {"type": "final", "checkpoint": 10000, "name": "Final Model"}

    return None


def extract_model_size(model_name):
    """Extract model size from model name."""
    if "Qwen3-0.6B" in model_name or "0.6B" in model_name:
        return "0.6B"
    elif "Qwen3-1.7B" in model_name or "1.7B" in model_name:
        return "1.7B"
    return "0.6B"  # Default to 0.6B if unclear


def load_experiment_data(experiment_dir):
    """Load all training checkpoint predictions for one experiment."""
    preds_dir = experiment_dir / "out-msq500" / "preds" / "cot"
    if not preds_dir.exists():
        print(f"Warning: {preds_dir} does not exist")
        return {}

    checkpoint_data = {}

    for pred_file in preds_dir.glob("*.json"):
        model_name = pred_file.stem.split("dataset=msq500__split=minidev__model_name=")[1].split("__bs=")[0]
        checkpoint_info = parse_checkpoint_info(model_name)

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


def load_ground_truth_msq():
    """Load MuSiQue ground truth data."""
    gt_file = Path("eval/datasets/msq500/musique_ans_v1.0_minidev.jsonl")

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


def create_experiment_subplots(all_experiment_results, output_dir):
    """Create subplots showing training progression for each experiment."""
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Training Progression Across Multiple Experiments", fontsize=16, fontweight="bold")

    for idx, (experiment_name, experiment_results) in enumerate(all_experiment_results.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        # Define positions outside the loops
        positions = list(range(1, 5))  # Positions 1-4

        # Separate by model size and plot both
        model_sizes = ["0.6B", "1.7B"]
        colors_by_size = {"0.6B": plt.cm.Blues, "1.7B": plt.cm.Oranges}

        for model_size in model_sizes:
            # Filter data for this model size
            size_filtered = {k: v for k, v in experiment_results.items() if k[1] == model_size}
            if not size_filtered:
                continue

            # Get checkpoint order and create colormap with more obvious color gradient
            checkpoints = [k[0] for k in sorted(size_filtered.keys())]
            colors = colors_by_size[model_size](np.linspace(0.2, 1.0, len(checkpoints)))

            # For each checkpoint, calculate average success rate across all complexities for each position
            for i, (checkpoint_key, results) in enumerate(sorted(size_filtered.items())):
                breakdown = results["breakdown"]
                checkpoint_info = results["info"]
                color = colors[i]

                # Calculate success rate for each position (1-4) averaged across complexities
                position_rates = []

                for pos in positions:
                    total_found = 0
                    total_questions = 0

                    # Aggregate across all complexities that have this position
                    for complexity in [2, 3, 4]:
                        if complexity in breakdown and pos <= complexity:
                            complexity_total = breakdown[complexity]["total"]
                            if pos <= len(breakdown[complexity]["found"]):
                                complexity_found = breakdown[complexity]["found"][
                                    pos - 1
                                ]  # Convert to 0-indexed
                                total_found += complexity_found
                                total_questions += complexity_total

                    if total_questions > 0:
                        position_rates.append(total_found / total_questions)
                    else:
                        position_rates.append(0.0)

                # Plot this checkpoint's line with simplified legend (just training steps)
                label = checkpoint_info["name"]
                ax.plot(
                    positions,
                    position_rates,
                    "o-",
                    color=color,
                    label=label,
                    linewidth=2,
                    markersize=6,
                    alpha=0.8,
                )

        # Create a cleaner experiment title using model + size format
        if "Qwen3-0.6B" in experiment_name:
            title = "Qwen3-0.6B"
        elif "Qwen3-1.7B" in experiment_name:
            title = "Qwen3-1.7B"
        elif "Qwen2.5-1.5B" in experiment_name:
            title = "Qwen2.5-1.5B"
        elif "Phi4" in experiment_name:
            title = "Phi4-mini"
        else:
            title = experiment_name

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Intermediate answers found")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1.1)
        ax.set_xlim(0.5, 4.5)
        ax.set_xticks(positions)
        ax.grid(True, alpha=0.3)

        # Add legend to each subplot to show training steps
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/multiple_training_experiments.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Multi-experiment plot saved to {output_dir}/multiple_training_experiments.png")


def create_single_training_progression_plot(all_experiment_results: dict, output_dir: Path):
    """Create a single plot showing training progression for Qwen3-0.6B experiment."""
    # Set font to Fira Code and increase all text sizes
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.monospace"] = ["Fira Code", "Courier New", "monospace"]
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = 14

    # Create figure with exact 3:2 aspect ratio
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Find Qwen3-0.6B experiment
    qwen_experiment = None
    for exp_name, exp_results in all_experiment_results.items():
        if "Qwen3-0.6B" in exp_name:
            qwen_experiment = exp_results
            break

    if qwen_experiment is None:
        print("Warning: No Qwen3-0.6B experiment found for single plot")
        return

    # Get only 0.6B model data and create colormap
    size_filtered = {k: v for k, v in qwen_experiment.items() if k[1] == "0.6B"}
    checkpoints = [k[0] for k in sorted(size_filtered.keys())]
    colors = plt.cm.viridis(np.linspace(0.2, 1.0, len(checkpoints)))

    # Define positions once (1-4)
    positions = list(range(1, 5))

    # For each checkpoint, plot the training progression
    for i, (checkpoint_key, results) in enumerate(sorted(size_filtered.items())):
        breakdown = results["breakdown"]
        checkpoint_info = results["info"]
        color = colors[i]

        # Calculate success rate for each position (1-4) averaged across complexities
        position_rates = []

        for pos in positions:
            total_found = 0
            total_questions = 0

            # Aggregate across all complexities that have this position
            for complexity in [2, 3, 4]:
                if complexity in breakdown and pos <= complexity:
                    complexity_total = breakdown[complexity]["total"]
                    if pos <= len(breakdown[complexity]["found"]):
                        complexity_found = breakdown[complexity]["found"][pos - 1]  # Convert to 0-indexed
                        total_found += complexity_found
                        total_questions += complexity_total

            if total_questions > 0:
                position_rates.append(total_found / total_questions)
            else:
                position_rates.append(0.0)

        # Plot this checkpoint's line
        ax.plot(
            positions,
            position_rates,
            "o-",
            color=color,
            label=checkpoint_info["name"],
            linewidth=2.5,
            markersize=8,
            alpha=0.8,
        )

    ax.set_xlabel("Intermediate answers found")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0.5, 4.5)
    ax.set_xticks(positions)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_progression_msq.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Reset rcParams to default
    plt.rcParams.update(plt.rcParamsDefault)

    print(f"Single training progression plot saved to {output_dir}/training_progression_msq.png")


def extract_model_name_from_experiment(experiment_name: str) -> str:
    """Extract clean model name from experiment directory name."""
    if "Qwen3-0.6B" in experiment_name:
        return "Qwen3-0.6B"
    elif "Qwen3-1.7B" in experiment_name:
        return "Qwen3-1.7B"
    elif "Qwen2.5-1.5B" in experiment_name:
        return "Qwen2.5-1.5B"
    elif "Phi4" in experiment_name:
        return "Phi4-mini"
    else:
        return experiment_name


def create_detailed_csv(all_experiment_results, output_dir):
    """Create CSV file with detailed breakdown data."""
    csv_data = []

    for experiment_name, experiment_results in all_experiment_results.items():
        for (checkpoint_step, model_size), results in experiment_results.items():
            breakdown = results["breakdown"]

            # For each complexity and position, add a row
            for complexity in [2, 3, 4]:
                if complexity in breakdown:
                    total_questions = breakdown[complexity]["total"]
                    found_counts = breakdown[complexity]["found"]

                    for pos in range(1, min(complexity + 1, 5)):  # Positions 1-4
                        if pos <= len(found_counts):
                            found_count = found_counts[pos - 1]  # Convert to 0-indexed
                            proportion = found_count / total_questions if total_questions > 0 else 0.0

                            csv_data.append(
                                {
                                    "model": extract_model_name_from_experiment(experiment_name),
                                    "checkpoint": checkpoint_step,
                                    "kth_intermediate_answer": pos,
                                    "proportion_found": proportion,
                                    "experiment": experiment_name,
                                    "found_count": found_count,
                                    "total_questions": total_questions,
                                    "complexity": complexity,
                                }
                            )

    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    csv_file = output_dir / "training_progression_detailed_data.csv"
    df.to_csv(csv_file, index=False)

    print(f"Detailed CSV saved to {csv_file}")
    print(f"CSV contains {len(df)} rows with detailed breakdown data")

    return df


def main():
    """Main function to analyze multiple training experiments."""
    checkpoints_base_dir = Path("eval/training_checkpoints")
    output_dir = Path("scripts/plots")
    output_dir.mkdir(exist_ok=True)

    # Find all experiment directories
    experiment_dirs = [
        d for d in checkpoints_base_dir.iterdir() if d.is_dir() and (d / "out-msq500").exists()
    ]
    experiment_dirs.sort()

    print("Loading ground truth data...")
    ground_truth = load_ground_truth_msq()

    print("Analyzing training experiments...")
    all_experiment_results = {}

    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        print(f"  Processing experiment: {exp_name}")

        checkpoint_data = load_experiment_data(exp_dir)
        if not checkpoint_data:
            print(f"    Warning: No data found for {exp_name}")
            continue

        experiment_results = {}
        for (checkpoint_step, model_size), data in checkpoint_data.items():
            print(f"    Analyzing {model_size} model at step {checkpoint_step}...")
            breakdown = analyze_msq_decomposition_for_checkpoint(data["predictions"], ground_truth)
            experiment_results[(checkpoint_step, model_size)] = {
                "breakdown": breakdown,
                "info": data["info"],
                "model_size": model_size,
            }

        all_experiment_results[exp_name] = experiment_results

    print("Creating subplot visualization...")
    create_experiment_subplots(all_experiment_results, output_dir)

    # Create single training progression plot for Qwen3-0.6B
    print("Creating single training progression plot...")
    create_single_training_progression_plot(all_experiment_results, output_dir)

    print("Creating detailed CSV file...")
    create_detailed_csv(all_experiment_results, output_dir)

    print("\n=== Analysis Complete ===")
    print(f"Analyzed {len(all_experiment_results)} experiments:")
    for exp_name, results in all_experiment_results.items():
        print(f"- {exp_name}: {len(results)} model+checkpoint combinations")
    print("Done!")


if __name__ == "__main__":
    main()
