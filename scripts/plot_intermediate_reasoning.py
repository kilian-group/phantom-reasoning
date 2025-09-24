#!/usr/bin/env python3
"""
Plot intermediate reasoning analysis results.
Color coding:
- Blue: out__train=pw__eval=wiki (trained on PW, eval on wiki)
- Orange: out__train=wiki__eval=wiki (trained on wiki, eval on wiki)
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


def load_analysis_results(json_path: str) -> dict[str, Any]:
    """Load the analysis results from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def parse_model_info(model_name: str) -> dict[str, str]:
    """Parse model name to extract training info and model details."""
    # Extract key information from model path
    model_info = {
        "full_name": model_name,
        "train_data": "unknown",
        "eval_data": "wiki",  # Both eval on wiki
        "model_size": "unknown",
        "color": "gray",
        "label": "Unknown",
    }

    # Now we have proper PW_TRAIN vs WIKI_TRAIN vs NOTRAIN prefixes from the new analysis
    if "PW_TRAIN::" in model_name:
        model_info["train_data"] = "pw"
        model_info["color"] = "blue"
        model_info["label"] = "Train=PW, Eval=Wiki"
    elif "WIKI_TRAIN::" in model_name:
        model_info["train_data"] = "wiki"
        model_info["color"] = "orange"
        model_info["label"] = "Train=Wiki, Eval=Wiki"
    elif "NOTRAIN::" in model_name:
        model_info["train_data"] = "notrain"
        model_info["color"] = "green"
        model_info["label"] = "NoTrain/Base Models"

    # Extract model size
    if "Qwen3-0.6B" in model_name:
        model_info["model_size"] = "0.6B"
    elif "Qwen3-1.7B" in model_name:
        model_info["model_size"] = "1.7B"
    elif "Qwen2.5-1.5B" in model_name:
        model_info["model_size"] = "1.5B"
    else:
        model_info["model_size"] = "Unknown"

    # Create display name
    model_info["display_name"] = f"Qwen-{model_info['model_size']} ({model_info['label']})"

    return model_info


def create_main_metrics_plot(data: dict[str, Any], output_dir: str):
    """Create main metrics comparison plots for all datasets."""
    datasets = ["2wiki", "hp", "msq"]

    # Define metrics for each dataset
    metrics_config = {
        "2wiki": {
            "metrics": [
                "partial_evidence_accuracy",
                "full_evidence_accuracy",
                "final_answer_in_pred_accuracy",
            ],
            "labels": ["Partial\nEvidence", "Full\nEvidence", "Correct Answer\nSuccessfully Identified"],
        },
        "hp": {
            "metrics": [
                "partial_supporting_fact_accuracy",
                "full_supporting_fact_accuracy",
                "final_answer_in_pred_accuracy",
            ],
            "labels": [
                "Partial\nSupport Facts",
                "Full\nSupport Facts",
                "Correct Answer\nSuccessfully Identified",
            ],
        },
        "msq": {
            "metrics": ["partial_accuracy", "full_decomposition_accuracy", "final_answer_in_pred_accuracy"],
            "labels": ["Partial\nDecomp", "Full\nDecomp", "Correct Answer\nSuccessfully Identified"],
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Intermediate Reasoning Analysis - Main Metrics Comparison", fontsize=16, fontweight="bold")

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        if dataset not in data["detailed_results"]:
            continue

        config = metrics_config[dataset]
        models_data = data["detailed_results"][dataset]

        # Prepare data for plotting
        plot_data = []

        for model_name, results in models_data.items():
            model_info = parse_model_info(model_name)

            for metric in config["metrics"]:
                if metric in results:
                    plot_data.append(
                        {
                            "Model": model_info["display_name"],
                            "Metric": metric,
                            "Accuracy": results[metric],
                            "Color": model_info["color"],
                            "Train_Data": model_info["train_data"],
                        }
                    )

        df = pd.DataFrame(plot_data)

        if not df.empty:
            # Create grouped bar plot
            metric_positions = np.arange(len(config["metrics"]))
            width = 0.35

            # Group by training data
            pw_data = df[df["Train_Data"] == "pw"]
            wiki_data = df[df["Train_Data"] == "wiki"]
            notrain_data = df[df["Train_Data"] == "notrain"]

            # Get accuracy values for each metric
            pw_accuracies = []
            wiki_accuracies = []
            notrain_accuracies = []

            for metric in config["metrics"]:
                pw_vals = pw_data[pw_data["Metric"] == metric]["Accuracy"].values
                wiki_vals = wiki_data[wiki_data["Metric"] == metric]["Accuracy"].values
                notrain_vals = notrain_data[notrain_data["Metric"] == metric]["Accuracy"].values

                pw_accuracies.append(np.mean(pw_vals) if len(pw_vals) > 0 else 0)
                wiki_accuracies.append(np.mean(wiki_vals) if len(wiki_vals) > 0 else 0)
                notrain_accuracies.append(np.mean(notrain_vals) if len(notrain_vals) > 0 else 0)

            # Plot bars with adjusted width for 3 groups
            width = 0.25
            bars1 = ax.bar(
                metric_positions - width,
                pw_accuracies,
                width,
                color="blue",
                alpha=0.7,
                label="Train=PW, Eval=Wiki",
            )
            bars2 = ax.bar(
                metric_positions,
                wiki_accuracies,
                width,
                color="orange",
                alpha=0.7,
                label="Train=Wiki, Eval=Wiki",
            )
            bars3 = ax.bar(
                metric_positions + width,
                notrain_accuracies,
                width,
                color="green",
                alpha=0.7,
                label="NoTrain/Base Models",
            )

            # Customize plot
            ax.set_title(f"{dataset.upper()} Dataset", fontweight="bold")
            ax.set_ylabel("Accuracy")
            ax.set_xticks(metric_positions)
            ax.set_xticklabels(config["labels"], rotation=45, ha="right")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)
            ax.legend()

            # Add value labels on bars
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:  # Only annotate non-zero bars
                        ax.annotate(
                            f"{height:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/main_metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_reasoning_vs_prediction_plot(data: dict[str, Any], output_dir: str):
    """Create plot showing reasoning vs prediction gap."""
    fig, ax = plt.subplots(figsize=(12, 8))

    datasets = ["2wiki", "hp", "msq"]
    plot_data = []

    for dataset in datasets:
        if dataset not in data["detailed_results"]:
            continue

        models_data = data["detailed_results"][dataset]

        for model_name, results in models_data.items():
            model_info = parse_model_info(model_name)

            reasoning_acc = results.get("final_answer_in_reasoning_accuracy", 0)
            pred_acc = results.get("final_answer_in_pred_accuracy", 0)
            gap = reasoning_acc - pred_acc

            plot_data.append(
                {
                    "Dataset": dataset.upper(),
                    "Model": model_info["display_name"],
                    "Reasoning_Accuracy": reasoning_acc,
                    "Prediction_Accuracy": pred_acc,
                    "Gap": gap,
                    "Color": model_info["color"],
                    "Train_Data": model_info["train_data"],
                }
            )

    df = pd.DataFrame(plot_data)

    if not df.empty:
        # Create scatter plot
        for dataset in datasets:
            dataset_data = df[df["Dataset"] == dataset.upper()]

            for train_data in ["pw", "wiki", "notrain"]:
                subset = dataset_data[dataset_data["Train_Data"] == train_data]
                if not subset.empty:
                    if train_data == "pw":
                        color = "blue"
                    elif train_data == "wiki":
                        color = "orange"
                    else:  # notrain
                        color = "green"

                    label = f"{dataset.upper()} - {train_data.upper()}"

                    ax.scatter(
                        subset["Prediction_Accuracy"],
                        subset["Reasoning_Accuracy"],
                        color=color,
                        alpha=0.7,
                        s=100,
                        label=label,
                    )

        # Add diagonal line (perfect agreement)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Agreement")

        ax.set_xlabel("Final Answer in Prediction Accuracy")
        ax.set_ylabel("Final Answer in Reasoning Accuracy")
        ax.set_title("Reasoning vs Prediction Gap Analysis", fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add text annotation about the gap
        ax.text(
            0.05,
            0.95,
            "Points above diagonal:\nModels understand but\nstruggle with synthesis",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/reasoning_vs_prediction_gap.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_complexity_breakdown_plot(data: dict[str, Any], output_dir: str):
    """Create plots showing performance by question complexity."""
    datasets = ["2wiki", "hp", "msq"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Performance by Question Complexity", fontsize=16, fontweight="bold")

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        if dataset not in data["detailed_results"]:
            continue

        models_data = data["detailed_results"][dataset]
        complexity_data = []

        for model_name, results in models_data.items():
            model_info = parse_model_info(model_name)
            breakdown = results.get("partial_breakdown", {})

            # Extract complexity information
            if dataset == "2wiki":
                for key, value in breakdown.items():
                    if "evidence_questions" in key:
                        complexity = key.split("_")[0]
                        total_questions = value

                        # Get full completion rate using new positional format
                        full_key = f"{complexity}_evidence_{complexity}_found"
                        if full_key in breakdown:
                            full_value = breakdown[full_key]
                            if isinstance(full_value, str) and "/" in full_value:
                                completed, total = full_value.split("/")
                                completion_rate = int(completed) / int(total)
                            else:
                                # Handle case where it's just a count
                                questions_key = f"{complexity}_evidence_questions"
                                total_questions = breakdown.get(questions_key, 1)
                                completion_rate = (
                                    (full_value if isinstance(full_value, int) else 0) / total_questions
                                    if total_questions > 0
                                    else 0
                                )

                            complexity_data.append(
                                {
                                    "Complexity": f"{complexity} Evidence",
                                    "Completion_Rate": completion_rate,
                                    "Total_Questions": total_questions,
                                    "Train_Data": model_info["train_data"],
                                    "Color": model_info["color"],
                                }
                            )

            elif dataset == "hp":
                for key, value in breakdown.items():
                    if "fact_questions" in key:
                        complexity = key.split("_")[0]
                        total_questions = value

                        # Get full completion rate using new positional format
                        full_key = f"{complexity}_fact_{complexity}_found"
                        if full_key in breakdown:
                            full_value = breakdown[full_key]
                            if isinstance(full_value, str) and "/" in full_value:
                                completed, total = full_value.split("/")
                                completion_rate = int(completed) / int(total)
                            else:
                                # Handle case where it's just a count
                                questions_key = f"{complexity}_fact_questions"
                                total_questions = breakdown.get(questions_key, 1)
                                completion_rate = (
                                    (full_value if isinstance(full_value, int) else 0) / total_questions
                                    if total_questions > 0
                                    else 0
                                )

                            complexity_data.append(
                                {
                                    "Complexity": f"{complexity} Facts",
                                    "Completion_Rate": completion_rate,
                                    "Total_Questions": total_questions,
                                    "Train_Data": model_info["train_data"],
                                    "Color": model_info["color"],
                                }
                            )

            elif dataset == "msq":
                for key, value in breakdown.items():
                    if "step_questions" in key:
                        complexity = key.split("_")[0]
                        total_questions = value

                        # Get full completion rate using new positional format
                        full_key = f"{complexity}_step_{complexity}_found"
                        if full_key in breakdown:
                            full_value = breakdown[full_key]
                            if isinstance(full_value, str) and "/" in full_value:
                                completed, total = full_value.split("/")
                                completion_rate = int(completed) / int(total)
                            else:
                                # Handle case where it's just a count
                                questions_key = f"{complexity}_step_questions"
                                total_questions = breakdown.get(questions_key, 1)
                                completion_rate = (
                                    (full_value if isinstance(full_value, int) else 0) / total_questions
                                    if total_questions > 0
                                    else 0
                                )

                            complexity_data.append(
                                {
                                    "Complexity": f"{complexity} Steps",
                                    "Completion_Rate": completion_rate,
                                    "Total_Questions": total_questions,
                                    "Train_Data": model_info["train_data"],
                                    "Color": model_info["color"],
                                }
                            )

        df = pd.DataFrame(complexity_data)

        if not df.empty:
            # Group by complexity and training data
            complexities = sorted(df["Complexity"].unique())
            x_pos = np.arange(len(complexities))
            width = 0.35

            pw_data = df[df["Train_Data"] == "pw"].groupby("Complexity")["Completion_Rate"].mean()
            wiki_data = df[df["Train_Data"] == "wiki"].groupby("Complexity")["Completion_Rate"].mean()
            notrain_data = df[df["Train_Data"] == "notrain"].groupby("Complexity")["Completion_Rate"].mean()

            pw_rates = [pw_data.get(c, 0) for c in complexities]
            wiki_rates = [wiki_data.get(c, 0) for c in complexities]
            notrain_rates = [notrain_data.get(c, 0) for c in complexities]

            width = 0.25
            bars1 = ax.bar(x_pos - width, pw_rates, width, color="blue", alpha=0.7, label="Train=PW")
            bars2 = ax.bar(x_pos, wiki_rates, width, color="orange", alpha=0.7, label="Train=Wiki")
            bars3 = ax.bar(x_pos + width, notrain_rates, width, color="green", alpha=0.7, label="NoTrain")

            ax.set_title(f"{dataset.upper()} - Full Completion by Complexity")
            ax.set_ylabel("Completion Rate")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(complexities, rotation=45, ha="right")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)
            ax.legend()

            # Add value labels
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(
                            f"{height:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/complexity_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_model_comparison_heatmap(data: dict[str, Any], output_dir: str):
    """Create heatmap comparing all models across all metrics."""
    # Collect all data
    all_data = []

    for dataset in ["2wiki", "hp", "msq"]:
        if dataset not in data["detailed_results"]:
            continue

        models_data = data["detailed_results"][dataset]

        for model_name, results in models_data.items():
            model_info = parse_model_info(model_name)

            # Get key metrics
            if dataset == "2wiki":
                metrics = {
                    "Partial_Evidence": results.get("partial_evidence_accuracy", 0),
                    "Full_Evidence": results.get("full_evidence_accuracy", 0),
                    "Final_in_Reasoning": results.get("final_answer_in_reasoning_accuracy", 0),
                    "Final_in_Pred": results.get("final_answer_in_pred_accuracy", 0),
                    "Sequential": results.get("sequential_accuracy", 0),
                }
            elif dataset == "hp":
                metrics = {
                    "Partial_Support": results.get("partial_supporting_fact_accuracy", 0),
                    "Full_Support": results.get("full_supporting_fact_accuracy", 0),
                    "Final_in_Reasoning": results.get("final_answer_in_reasoning_accuracy", 0),
                    "Final_in_Pred": results.get("final_answer_in_pred_accuracy", 0),
                    "Sequential": results.get("sequential_accuracy", 0),
                }
            elif dataset == "msq":
                metrics = {
                    "Partial_Decomp": results.get("partial_accuracy", 0),
                    "Full_Decomp": results.get("full_decomposition_accuracy", 0),
                    "Final_in_Reasoning": results.get("final_answer_in_reasoning_accuracy", 0),
                    "Final_in_Pred": results.get("final_answer_in_pred_accuracy", 0),
                    "Sequential": results.get("sequential_accuracy", 0),
                }

            for metric_name, value in metrics.items():
                all_data.append(
                    {
                        "Model": model_info["display_name"],
                        "Dataset": dataset.upper(),
                        "Metric": metric_name,
                        "Accuracy": value,
                        "Train_Data": model_info["train_data"],
                    }
                )

    df = pd.DataFrame(all_data)

    if not df.empty:
        # Create pivot table
        pivot_df = df.pivot_table(
            index=["Model", "Train_Data"], columns=["Dataset", "Metric"], values="Accuracy"
        )

        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 8))

        # Create custom colormap
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".3f",
            cmap="RdYlBu_r",
            center=0.5,
            ax=ax,
            cbar_kws={"label": "Accuracy"},
        )

        ax.set_title(
            "Model Performance Heatmap Across All Datasets and Metrics", fontweight="bold", fontsize=14
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Color code y-axis labels by training data
        y_labels = ax.get_yticklabels()
        for i, label in enumerate(y_labels):
            if "(Train=PW" in label.get_text():
                label.set_color("blue")
            elif "(Train=Wiki" in label.get_text():
                label.set_color("orange")
            elif "(NoTrain" in label.get_text():
                label.set_color("green")

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main function to generate all plots."""
    # Load data from the new combined analysis
    json_path = "scripts/intermediate_reasoning_analysis_results.json"
    output_dir = "scripts/plots"

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    print("Loading analysis results...")
    data = load_analysis_results(json_path)

    print("Creating main metrics comparison plot...")
    create_main_metrics_plot(data, output_dir)

    print("Creating reasoning vs prediction gap plot...")
    create_reasoning_vs_prediction_plot(data, output_dir)

    print("Creating positional breakdown plot...")
    create_complexity_breakdown_plot(data, output_dir)

    print("Creating model comparison heatmap...")
    create_model_comparison_heatmap(data, output_dir)

    print(f"All plots saved to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
