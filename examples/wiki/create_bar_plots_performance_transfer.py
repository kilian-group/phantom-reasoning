import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Patch
from phantom_eval.evaluate_utils import mean, std
from utils.evaluate_utils import get_preds

from phantom_reasoner.utils import plotting_utils

parser = argparse.ArgumentParser()
parser.add_argument("--final_ckpts_yaml_path", type=str, required=True)
parser.add_argument("--base_model_preds_dir", type=str, default="out__train=base__eval=wiki")
parser.add_argument("--pw_model_preds_dir", type=str, default="out__train=pw__eval=wiki")
parser.add_argument("--gsminf_model_preds_dir", type=str, default="out__train=gsminf__eval=wiki")
parser.add_argument(
    "--rg_family_relationships_model_preds_dir",
    type=str,
    default="out__train=rg-family_relationships__eval=wiki",
)
parser.add_argument(
    "--rg_knights_knaves_model_preds_dir",
    type=str,
    default="out__train=rg-knights_knaves__eval=wiki",
)
parser.add_argument("--figures_dir", type=str, default="scripts/final_plots/figures")
args = parser.parse_args()

figures_dir = Path(args.figures_dir)
figures_dir.mkdir(parents=True, exist_ok=True)

with open(args.final_ckpts_yaml_path) as f:
    final_ckpts_yaml = yaml.safe_load(f)
    synthetic_train_ckpts = final_ckpts_yaml["synthetic_train_ckpts"]

models_in_order = [
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen2.5-1.5B-Instruct",
    "Phi-4-Mini-Reasoning",
    "Qwen3-4B",
    "Qwen2.5-7B-Instruct",
]
models_in_main_text = [
    "Qwen3-0.6B",
    "Qwen2.5-1.5B-Instruct",
    "Phi-4-Mini-Reasoning",
    "Qwen2.5-7B-Instruct",
]
models_in_appendix = [
    "Qwen3-1.7B",
    "Qwen3-4B",
]
train_dataset_names = ["base", "rg-family_relationships", "rg-knights_knaves", "gsminf", "pw"]
eval_dataset_names = plotting_utils.EVAL_DATASET_NAMES
eval_dataset_alias2name = {
    "hp500": "HotpotQA",
    "2wiki500": "2Wiki",
    "msq500": "MuSiQue",
    "cofca500": "CofCA",
    "synthrm500": "SynthWorlds-RM",
}

LINE_WIDTH = 1

# Data from the table
# data = {
#     "Qwen3-0.6B": {
#         "HotpotQA": {"base": 0.3654, "format": 0.3780, "gsminf": 0.4787, "pw": 0.5905},
#         "2Wiki": {"base": 0.3691, "format": 0.3319, "gsminf": 0.4940, "pw": 0.6013},
#         "MuSiQue": {"base": 0.1415, "format": 0.1337, "gsminf": 0.1983, "pw": 0.3283},
#     },
#     "Qwen3-1.7B": {
#         "HotpotQA": {"base": 0.5958, "format": 0.6407, "gsminf": 0.6437, "pw": 0.6581},
#         "2Wiki": {"base": 0.6354, "format": 0.6665, "gsminf": 0.7278, "pw": 0.7502},
#         "MuSiQue": {"base": 0.3411, "format": 0.3449, "gsminf": 0.3968, "pw": 0.4029},
#     },
#     "Qwen2.5-1.5B-Instruct": {
#         "HotpotQA": {"base": 0.0199, "format": 0.4333, "gsminf": 0.3322, "pw": 0.5359},
#         "2Wiki": {"base": 0.1422, "format": 0.2957, "gsminf": 0.3561, "pw": 0.4526},
#         "MuSiQue": {"base": 0.0402, "format": 0.1983, "gsminf": 0.1629, "pw": 0.2878},
#     },
#     "Phi-4-Mini-Reasoning": {
#         "HotpotQA": {"base": 0.4871, "format": 1, "gsminf": 0.5431, "pw": 0.6210},
#         "2Wiki": {"base": 0.6663, "format": 1, "gsminf": 0.6739, "pw": 0.7586},
#         "MuSiQue": {"base": 0.2923, "format": 1, "gsminf": 0.3250, "pw": 0.4469},
#     },
# }

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
    # for i, model in enumerate(models):
    for j, dataset in enumerate(eval_dataset_names):
        ax = axes[j]

        # Get values for this model-dataset combination
        values = [mean_data[model][dataset][train_dataset_name] for train_dataset_name in train_dataset_names]
        errors = [std_data[model][dataset][train_dataset_name] for train_dataset_name in train_dataset_names]

        # Create bars with error bars (using small errors for visual effect)
        colors = [
            plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR[label]]
            for label in train_dataset_names
        ]
        _ = ax.bar(
            x_pos,
            values,
            bar_width,
            yerr=errors,
            color=colors,
            edgecolor=EDGE_COLOR,
            linewidth=LINE_WIDTH,
            error_kw={"linewidth": LINE_WIDTH, "ecolor": "black", "capsize": 5},
        )

        # Customize subplot
        ax.set_ylim(0, max(yticks))
        ax.set_yticks(yticks)
        # Only show y-axis tick labels for the left most subplots
        # So no y-axis tick labels for the other subplots
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
    LEFT_OFFSET = -0.025
    RIGHT_OFFSET = 0.01
    BRACKET_Y_OFFSET = 0.02

    # Calculate positions
    ax_left = axes[0].get_position()
    y_pos = ax_left.y0 - BRACKET_Y_OFFSET
    BRACKET_HEIGHT = 0.01

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
    x_center = (x_left - LEFT_OFFSET + x_right + RIGHT_OFFSET) / 2
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
    gsminf_model_preds_dir: str,
    rg_family_relationships_model_preds_dir: str,
    rg_knights_knaves_model_preds_dir: str,
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

    # Get pw model data
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

            # Collect data for the all checkpoints in a single df
            dfs_of_ckpt_paths = []
            for ckpt_path in ckpt_paths_of_model:
                dfs_of_ckpt_paths.append(df_preds[df_preds["_model"] == ckpt_path])

            dfs_of_ckpt_paths = pd.concat(dfs_of_ckpt_paths)
            mean_data[model][eval_name]["pw"] = mean(dfs_of_ckpt_paths["f1"])
            std_data[model][eval_name]["pw"] = std(dfs_of_ckpt_paths["f1"])

    # Get gsminf model data
    for alias, eval_name in eval_dataset_alias2name.items():
        df_preds, _ = get_preds(args.gsminf_model_preds_dir, "data", alias, "minidev", "cot")

        # Get pw train dataset dict
        pw_train_dataset_dict = None
        for train_dataset_dict in synthetic_train_ckpts:
            if train_dataset_dict["dataset_name"] == "gsminf":
                pw_train_dataset_dict = train_dataset_dict
                break

        # Get ckpt paths of model
        for model in models:
            ckpt_paths_of_model = None
            for ckpt in pw_train_dataset_dict["ckpts"]:
                if plotting_utils.MODEL_NAME2ALIAS[ckpt["model"]] == model:
                    ckpt_paths_of_model = ckpt["paths"]
                    break

            # Collect data for the all checkpoints in a single df
            dfs_of_ckpt_paths = []
            for ckpt_path in ckpt_paths_of_model:
                dfs_of_ckpt_paths.append(df_preds[df_preds["_model"] == ckpt_path])

            dfs_of_ckpt_paths = pd.concat(dfs_of_ckpt_paths)
            mean_data[model][eval_name]["gsminf"] = mean(dfs_of_ckpt_paths["f1"])
            std_data[model][eval_name]["gsminf"] = std(dfs_of_ckpt_paths["f1"])

    # Get rg-family_relationships model data
    for alias, eval_name in eval_dataset_alias2name.items():
        df_preds, _ = get_preds(args.rg_family_relationships_model_preds_dir, "data", alias, "minidev", "cot")

        # Get pw train dataset dict
        pw_train_dataset_dict = None
        for train_dataset_dict in synthetic_train_ckpts:
            if train_dataset_dict["dataset_name"] == "rg-family_relationships":
                pw_train_dataset_dict = train_dataset_dict
                break

        # Get ckpt paths of model
        for model in models:
            ckpt_paths_of_model = None
            for ckpt in pw_train_dataset_dict["ckpts"]:
                if plotting_utils.MODEL_NAME2ALIAS[ckpt["model"]] == model:
                    ckpt_paths_of_model = ckpt["paths"]
                    break

            # Collect data for the all checkpoints in a single df
            dfs_of_ckpt_paths = []
            for ckpt_path in ckpt_paths_of_model:
                dfs_of_ckpt_paths.append(df_preds[df_preds["_model"] == ckpt_path])

            dfs_of_ckpt_paths = pd.concat(dfs_of_ckpt_paths)
            mean_data[model][eval_name]["rg-family_relationships"] = mean(dfs_of_ckpt_paths["f1"])
            std_data[model][eval_name]["rg-family_relationships"] = std(dfs_of_ckpt_paths["f1"])

    # Get rg-knights_knaves model data
    for alias, eval_name in eval_dataset_alias2name.items():
        df_preds, _ = get_preds(args.rg_knights_knaves_model_preds_dir, "data", alias, "minidev", "cot")

        # Get pw train dataset dict
        pw_train_dataset_dict = None
        for train_dataset_dict in synthetic_train_ckpts:
            if train_dataset_dict["dataset_name"] == "rg-knights_knaves":
                pw_train_dataset_dict = train_dataset_dict
                break

        # Get ckpt paths of model
        for model in models:
            ckpt_paths_of_model = None
            for ckpt in pw_train_dataset_dict["ckpts"]:
                if plotting_utils.MODEL_NAME2ALIAS[ckpt["model"]] == model:
                    ckpt_paths_of_model = ckpt["paths"]
                    break

            # Collect data for the all checkpoints in a single df
            dfs_of_ckpt_paths = []
            for ckpt_path in ckpt_paths_of_model:
                dfs_of_ckpt_paths.append(df_preds[df_preds["_model"] == ckpt_path])

            dfs_of_ckpt_paths = pd.concat(dfs_of_ckpt_paths)
            mean_data[model][eval_name]["rg-knights_knaves"] = mean(dfs_of_ckpt_paths["f1"])
            std_data[model][eval_name]["rg-knights_knaves"] = std(dfs_of_ckpt_paths["f1"])

    return mean_data, std_data


if __name__ == "__main__":
    save_path = Path(args.figures_dir) / "f1_transfer_performance_all.json"
    if not save_path.exists():
        mean_data, std_data = load_data(
            models_in_order,
            args.base_model_preds_dir,
            args.pw_model_preds_dir,
            args.gsminf_model_preds_dir,
            args.rg_family_relationships_model_preds_dir,
            args.rg_knights_knaves_model_preds_dir,
        )

        # Also save the data to a json file
        with open(save_path, "w") as f:
            json.dump({"mean_data": mean_data, "std_data": std_data}, f, indent=4)
            f.write("\n")

    with open(save_path) as f:
        data = json.load(f)
        mean_data = data["mean_data"]
        std_data = data["std_data"]

    # Create a num_models x num_eval_datasets subplot figure
    # First main text figure
    num_models = len(models_in_main_text)
    fig, axes = plt.subplots(num_models, len(eval_dataset_names), figsize=(12, 4 * num_models))
    for i, model in enumerate(models_in_main_text):
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
            facecolor=plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR[label]],
            edgecolor="black",
            label=plotting_utils.TRAIN_DATASET_ALIAS2NAME[label],
        )
        for label in train_dataset_names
    ]

    # Position legend on the right side
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        fontsize=LABEL_FONT_SIZE,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        bbox_to_anchor=(0.53, 0.93),  # Move just below the title at the top center
        ncol=len(train_dataset_names),
    )

    # Adjust layout
    plt.subplots_adjust(
        left=0.08,  # where the left subplot y-labels are, increase to move them away from left figure edge
        right=0.98,  # where the right subplot edges are, increase to move them closer to right figure edge
        # top=0.85,  # where the top subplot edges are, increase to move them closer to top figure edge
        # bottom=0.1,  # where the bottom subplot edges are, increase to move them closer to bottom figure
        # hspace=0.25,  # horizontal space between subplots, increase to move them away
        wspace=0.05,  # vertical space between subplots, increase to move them away
    )

    save_path = Path(args.figures_dir) / "f1_transfer_performance_main_text.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved bar plot to {save_path} and {save_path.with_suffix('.png')}")
    plt.close()

    # Second appendix figure
    num_models = len(models_in_appendix)
    fig, axes = plt.subplots(num_models, len(eval_dataset_names), figsize=(12, 4 * num_models))
    for i, model in enumerate(models_in_appendix):
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
            facecolor=plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR[label]],
            edgecolor="black",
            label=plotting_utils.TRAIN_DATASET_ALIAS2NAME[label],
        )
        for label in train_dataset_names
    ]

    # Position legend on the right side
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        fontsize=LABEL_FONT_SIZE,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        bbox_to_anchor=(0.53, 0.93),  # Move just below the title at the top center
        ncol=len(train_dataset_names),
    )

    # Adjust layout
    plt.subplots_adjust(
        left=0.08,  # where the left subplot y-labels are, increase to move them away from left figure edge
        right=0.98,  # where the right subplot edges are, increase to move them closer to right figure edge
        # top=0.85,  # where the top subplot edges are, increase to move them closer to top figure edge
        # bottom=0.1,  # where the bottom subplot edges are, increase to move them closer to bottom figure
        # hspace=0.25,  # horizontal space between subplots, increase to move them away
        wspace=0.05,  # vertical space between subplots, increase to move them away
    )

    save_path = Path(args.figures_dir) / "f1_transfer_performance_appendix.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved bar plot to {save_path} and {save_path.with_suffix('.png')}")
    plt.close()
