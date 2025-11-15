import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Patch
from phantom_eval.evaluate_utils import mean, std
from tabulate import tabulate
from utils.evaluate_utils import get_preds

from phantom_reasoner.utils import plotting_utils

parser = argparse.ArgumentParser()
parser.add_argument("--final_ckpts_yaml_path", type=str, required=True)
parser.add_argument(
    "--no_evidence",
    action="store_true",
    help="Evaluate scores for evals without any evidence (empty text_corpus)",
)
parser.add_argument("--base_model_preds_dir", type=str, default="out__train=base__eval=wiki")
parser.add_argument("--pw_model_preds_dir", type=str, default="out__train=pw__eval=wiki")
parser.add_argument("--wiki_model_preds_dir", type=str, default="out__train=wiki__eval=wiki")
parser.add_argument("--figures_dir", type=str, default="scripts/final_plots/figures")
args = parser.parse_args()

figures_dir = Path(args.figures_dir)
figures_dir.mkdir(parents=True, exist_ok=True)

if args.no_evidence:
    print("*** No evidence: scoring evals that used empty text_corpus ***")
    no_evidence_suffix = "__no_evidence"
    # Update dirs to use no_evidence suffix
    args.base_model_preds_dir = args.base_model_preds_dir + no_evidence_suffix
    args.pw_model_preds_dir = args.pw_model_preds_dir + no_evidence_suffix
    args.wiki_model_preds_dir = args.wiki_model_preds_dir + no_evidence_suffix
    print(
        f"*** Updated preds dirs: {args.base_model_preds_dir}, "
        f"{args.pw_model_preds_dir}, {args.wiki_model_preds_dir} ***"
    )
else:
    no_evidence_suffix = ""

with open(args.final_ckpts_yaml_path) as f:
    final_ckpts_yaml = yaml.safe_load(f)
    real_train_ckpts = final_ckpts_yaml["real_train_ckpts"]

models_in_order = [
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen2.5-1.5B-Instruct",
]
train_dataset_names = ["base", "pw", "hp", "2wiki", "msq"]
train_dataset_alias2name = {
    "base": plotting_utils.TRAIN_DATASET_ALIAS2NAME["base"],
    "pw": plotting_utils.TRAIN_DATASET_ALIAS2NAME["pw"],
    "hp": "HotpotQA",
    "2wiki": "2Wiki",
    "msq": "MuSiQue",
}
eval_dataset_names = plotting_utils.EVAL_DATASET_NAMES
eval_dataset_alias2name = {
    "hp500": "HotpotQA",
    "2wiki500": "2Wiki",
    "msq500": "MuSiQue",
    "cofca500": "CofCA",
    "synthrm500": "SynthWorlds-RM",
}

LINE_WIDTH = 1

# Define colors for real-world training datasets
# Use base and pw colors from plotting_utils
# For hp, 2wiki, msq: use pastel colors with hatches
TRAIN_DATASET_COLORS = {
    "base": plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR["base"]],  # myYellow
    "pw": plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR["pw"]],  # myOrange
    "hp": "#B8E6B8",  # Pastel green
    "2wiki": "#B8D8E6",  # Pastel blue
    "msq": "#FFD8B8",  # Pastel orange
}

TRAIN_DATASET_HATCHES = {
    "base": "",  # no hatch
    "pw": "",  # no hatch
    "hp": "//",  # vertical hatch
    "2wiki": "\\\\",  # vertical hatch
    "msq": "xx",  # cross hatch
}

LABEL_FONT_SIZE = plotting_utils.LABEL_FONT_SIZE + 2
TICK_FONT_SIZE = plotting_utils.TICK_FONT_SIZE + 2
LEGEND_FONT_SIZE = plotting_utils.LEGEND_FONT_SIZE + 2


def create_bar_plot_for_model(
    mean_data: dict,
    std_data: dict,
    model: str,
    yticks: list[float],
    axes: list[plt.Axes],
    show_yticks: bool = True,
    show_eval_titles: bool = True,
    x_left: float = 0.0,
    x_right: float = 0.0,
):
    # Bar settings
    bar_width = 0.85
    x_pos = np.arange(len(train_dataset_names))
    EDGE_COLOR = "black"

    # Create bars for each subplot
    for j, dataset in enumerate(eval_dataset_names):
        ax = axes[j]

        # Get values for this model-dataset combination
        values = [mean_data[model][dataset][train_dataset_name] for train_dataset_name in train_dataset_names]
        errors = [std_data[model][dataset][train_dataset_name] for train_dataset_name in train_dataset_names]

        # Create bars with error bars
        colors = [TRAIN_DATASET_COLORS[label] for label in train_dataset_names]
        hatches = [TRAIN_DATASET_HATCHES[label] for label in train_dataset_names]

        bars = ax.bar(
            x_pos,
            values,
            bar_width,
            yerr=errors,
            color=colors,
            edgecolor=EDGE_COLOR,
            linewidth=LINE_WIDTH,
            error_kw={"linewidth": LINE_WIDTH, "ecolor": "black", "capsize": 5},
        )

        # Apply hatches to bars
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        # Customize subplot
        ax.set_ylim(0, max(yticks))
        ax.set_yticks(yticks)
        # Only show y-axis tick labels for the left most subplots
        if show_yticks and j == 0:
            ax.set_yticklabels(list(map(str, yticks)), fontsize=TICK_FONT_SIZE)
            ax.set_ylabel("F1", fontsize=LABEL_FONT_SIZE)
        else:
            ax.set_yticklabels([])

        # Don't show x-axis ticks and labels
        ax.set_xticks([])
        ax.set_xticklabels([])

        # Add horizontal grid lines
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=LINE_WIDTH)
        ax.set_axisbelow(True)

        # Style the spines
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(LINE_WIDTH)

        # Set titles for top row (eval datasets)
        if show_eval_titles:
            ax.set_title(dataset, fontsize=LABEL_FONT_SIZE, fontweight="bold")

    # Get the positions of the first and last subplot in this row
    LEFT_OFFSET = 0.0
    RIGHT_OFFSET = 0.0
    BRACKET_Y_OFFSET = 0.05

    # Calculate positions
    ax_left = axes[0].get_position()
    y_pos = ax_left.y0 - BRACKET_Y_OFFSET
    BRACKET_HEIGHT = 0.015

    # Draw left vertical line
    fig.add_artist(
        plt.Line2D(
            [x_left - LEFT_OFFSET, x_left - LEFT_OFFSET],
            [y_pos, y_pos + BRACKET_HEIGHT],
            transform=fig.transFigure,
            color="black",
            linewidth=LINE_WIDTH,
        )
    )

    # Draw horizontal line
    fig.add_artist(
        plt.Line2D(
            [x_left - LEFT_OFFSET, x_right + RIGHT_OFFSET],
            [y_pos, y_pos],
            transform=fig.transFigure,
            color="black",
            linewidth=LINE_WIDTH,
        )
    )

    # Draw right vertical line
    fig.add_artist(
        plt.Line2D(
            [x_right + RIGHT_OFFSET, x_right + RIGHT_OFFSET],
            [y_pos, y_pos + BRACKET_HEIGHT],
            transform=fig.transFigure,
            color="black",
            linewidth=LINE_WIDTH,
        )
    )

    # Add the model name centered below the bracket
    x_center = (x_left + x_right) / 2
    fig.text(
        x_center,
        y_pos + BRACKET_HEIGHT,
        model,
        fontsize=LABEL_FONT_SIZE,
        fontweight="bold",
        verticalalignment="top",
        horizontalalignment="center",
        transform=fig.transFigure,
        bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="black", linewidth=LINE_WIDTH),
    )


def load_data(
    models: list[str],
    base_model_preds_dir: str,
    pw_model_preds_dir: str,
    wiki_model_preds_dir: str,
    no_evidence: bool,
) -> tuple[dict, dict]:
    mean_data = {
        model: {
            dataset: {train_dataset_name: 0.0 for train_dataset_name in train_dataset_names}
            for dataset in eval_dataset_names
        }
        for model in models
    }
    std_data = {
        model: {
            dataset: {train_dataset_name: 0.0 for train_dataset_name in train_dataset_names}
            for dataset in eval_dataset_names
        }
        for model in models
    }

    # Get base model data
    for alias, eval_name in eval_dataset_alias2name.items():
        df_preds, _ = get_preds(args.base_model_preds_dir, "data", alias, "minidev", "cot")
        df_preds["_model"] = df_preds["_model"].map(plotting_utils.MODEL_NAME2ALIAS)
        acc = df_preds.groupby(["_model"])[["f1"]].agg([mean, std])
        for model in models:
            mean_data[model][eval_name]["base"] = acc.loc[model, ("f1", "mean")]
            std_data[model][eval_name]["base"] = acc.loc[model, ("f1", "std")]

    # Get pw model data from synthetic_train_ckpts
    synthetic_train_ckpts = final_ckpts_yaml["synthetic_train_ckpts"]
    for alias, eval_name in eval_dataset_alias2name.items():
        df_preds, _ = get_preds(args.pw_model_preds_dir, "data", alias, "minidev", "cot")

        # Get pw train dataset dict
        pw_train_dataset_dict = None
        for train_dataset_dict in synthetic_train_ckpts:
            if train_dataset_dict["dataset_name"] == "pw":
                pw_train_dataset_dict = train_dataset_dict
                break

        # Get ckpt paths of model
        for model in models:
            ckpt_paths_of_model = None
            for ckpt in pw_train_dataset_dict["ckpts"]:
                if plotting_utils.MODEL_NAME2ALIAS[ckpt["model"]] == model:
                    ckpt_paths_of_model = ckpt["paths"]
                    break

            if ckpt_paths_of_model is None:
                continue

            # Collect data for all checkpoints in a single df
            dfs_of_ckpt_paths = []
            for ckpt_path in ckpt_paths_of_model:
                dfs_of_ckpt_paths.append(df_preds[df_preds["_model"] == ckpt_path])

            dfs_of_ckpt_paths = pd.concat(dfs_of_ckpt_paths)
            mean_data[model][eval_name]["pw"] = mean(dfs_of_ckpt_paths["f1"])
            std_data[model][eval_name]["pw"] = std(dfs_of_ckpt_paths["f1"])

    # Get real-world training data (hp, 2wiki, msq)
    real_dataset_names = ["hp", "2wiki", "msq"]
    for real_dataset_name in real_dataset_names:
        for alias, eval_name in eval_dataset_alias2name.items():
            df_preds, _ = get_preds(args.wiki_model_preds_dir, "data", alias, "minidev", "cot")

            # Get train dataset dict for this real dataset
            train_dataset_dict = None
            for td_dict in real_train_ckpts:
                if td_dict["dataset_name"] == real_dataset_name:
                    train_dataset_dict = td_dict
                    break

            if train_dataset_dict is None:
                continue

            # Get ckpt paths of model
            for model in models:
                ckpt_paths_of_model = None
                for ckpt in train_dataset_dict["ckpts"]:
                    if plotting_utils.MODEL_NAME2ALIAS[ckpt["model"]] == model:
                        ckpt_paths_of_model = ckpt["paths"]
                        break

                if ckpt_paths_of_model is None:
                    continue

                # Collect data for all checkpoints in a single df
                dfs_of_ckpt_paths = []
                for ckpt_path in ckpt_paths_of_model:
                    dfs_of_ckpt_paths.append(df_preds[df_preds["_model"] == ckpt_path])

                if len(dfs_of_ckpt_paths) > 0:
                    dfs_of_ckpt_paths = pd.concat(dfs_of_ckpt_paths)
                    mean_data[model][eval_name][real_dataset_name] = mean(dfs_of_ckpt_paths["f1"])
                    std_data[model][eval_name][real_dataset_name] = std(dfs_of_ckpt_paths["f1"])
    return mean_data, std_data


if __name__ == "__main__":
    mean_data, std_data = load_data(
        models_in_order,
        args.base_model_preds_dir,
        args.pw_model_preds_dir,
        args.wiki_model_preds_dir,
        args.no_evidence,
    )

    # Also save the data to a json file
    save_path = Path(args.figures_dir) / f"f1_realworld_performance_all{no_evidence_suffix}.json"
    with open(save_path, "w") as f:
        json.dump({"mean_data": mean_data, "std_data": std_data}, f, indent=4)
        f.write("\n")

    # Create a num_models x num_eval_datasets subplot figure
    fig, axes = plt.subplots(len(models_in_order), len(eval_dataset_names), figsize=(8, 10))
    for i, model in enumerate(models_in_order):
        yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        show_eval_titles = i == 0  # Only show eval titles for the first model (top row)
        axes_slice = axes[i, :]
        x_left = axes_slice[0].get_position().x0
        x_right = axes_slice[-1].get_position().x1
        create_bar_plot_for_model(
            mean_data,
            std_data,
            model,
            yticks,
            axes_slice,
            show_yticks=True,
            show_eval_titles=show_eval_titles,
            x_left=x_left,
            x_right=x_right,
        )

    # Create legend with better styling
    legend_elements = [
        Patch(
            facecolor=TRAIN_DATASET_COLORS[train_dataset_name],
            edgecolor="black",
            hatch=TRAIN_DATASET_HATCHES[train_dataset_name],
            label=train_dataset_alias2name[train_dataset_name],
        )
        for train_dataset_name in train_dataset_names
    ]

    # Position legend at the top center
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        fontsize=LABEL_FONT_SIZE,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        bbox_to_anchor=(0.53, 0.94),
        ncol=len(train_dataset_names),
    )

    # Adjust layout
    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.85,
        bottom=0.1,
        hspace=0.35,
        wspace=0.05,
    )

    save_path = Path(args.figures_dir) / f"f1_realworld_performance_all{no_evidence_suffix}.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved bar plot to {save_path} and {save_path.with_suffix('.png')}")
    plt.close()

    if args.no_evidence:
        # Generate a table (for each model) of delta scores like so:
        # Columns: eval_dataset_names
        # Rows: train_dataset_names
        # Value: mean_data[model][eval_dataset_name][train_dataset_name] -
        #   mean_data[model][eval_dataset_name]["base"]
        # Save the table to a csv file
        for model in models_in_order:
            delta_scores_records = []
            for eval_dataset_name in eval_dataset_names:
                # delta_scores_dict[train_dataset_name] = {}
                for train_dataset_name in train_dataset_names:
                    delta_score = (
                        mean_data[model][eval_dataset_name][train_dataset_name]
                        - mean_data[model][eval_dataset_name]["base"]
                    )
                    delta_scores_records.append(
                        {
                            "train_dataset_name": train_dataset_name,
                            "eval_dataset_name": eval_dataset_name,
                            "delta_score": delta_score,
                        }
                    )
            delta_scores_df = pd.DataFrame(delta_scores_records)

            # Some formatting now...
            # Pivot the dataframe so that the columns are the eval_dataset_names and
            # the rows are the train_dataset_names
            # Ensure that columns are in the order of eval_dataset_names
            # Ensure that the rows are in the order of train_dataset_names - "base"
            delta_scores_df = delta_scores_df.pivot(
                index="train_dataset_name", columns="eval_dataset_name", values="delta_score"
            )
            delta_scores_df = delta_scores_df[eval_dataset_names]
            delta_scores_df = delta_scores_df.reindex([t for t in train_dataset_names if t != "base"])
            print(f"Delta scores table for {model}, saved to {save_path}")
            print(tabulate(delta_scores_df, headers="keys", tablefmt="github", floatfmt=".3f"))
            save_path = Path(args.figures_dir) / f"delta_scores_{model}{no_evidence_suffix}.csv"
            delta_scores_df.to_csv(save_path)
