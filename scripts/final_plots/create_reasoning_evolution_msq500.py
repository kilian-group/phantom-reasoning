"""
Create reasoning evolution plots for all base models in the csv file.
A subplot is created for each base model, showing performance vs kth intermediate answer as the training evolves.

Example usage:
```bash
python scripts/final_plots/create_reasoning_evolution_msq500.py \
    --csv_path scripts/final_plots/figures/reasoning_evolution_msq500.csv \
    --base_model_names_to_plot "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" \
    --figures_dir "scripts/final_plots/figures"
```
"""  # noqa: E501

import os
from pathlib import Path

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from phantom_eval import get_parser
from phantom_eval.utils import setup_logging

from phantom_reasoner.utils import plotting_utils

setup_logging("INFO")


parser = get_parser()
parser.add_argument(
    "--csv_path", type=str, default="reasoning_evolution_msq500.csv", help="Path to the csv file"
)
parser.add_argument(
    "--base_model_names_to_plot",
    nargs="+",
    default=["Qwen3-0.6B", "Qwen3-1.7B"],
    help="Base model names to plot",
)
parser.add_argument("--figures_dir", type=str, default="scripts/final_plots/figures")
args = parser.parse_args()

train_dataset_names2xticks = {
    "pw": [1, 2, 3, 4],
}
train_dataset_names2metric = {
    "pw": "proportion_found",
}
train_dataset_names2xlabel = {
    "pw": "Nth intermediate answer",
}
train_dataset_names2max_difficulty = {
    "pw": 4,
}

DIFFICULTY = "complexity"

# Increase font sizes for better readability, since we plot two models
LABEL_FONT_SIZE = plotting_utils.LABEL_FONT_SIZE
TICK_FONT_SIZE = plotting_utils.TICK_FONT_SIZE
LEGEND_FONT_SIZE = plotting_utils.LEGEND_FONT_SIZE


def get_colormap(training_dataset_name):
    COLORMAP_LAST_HEX = plotting_utils.COLORS2HEX[
        plotting_utils.TRAIN_DATASET_ALIAS2COLOR[training_dataset_name]
    ]
    colors = [(0, "white"), (1, COLORMAP_LAST_HEX)]
    COLORMAP = LinearSegmentedColormap.from_list(
        "WhiteToHex", colors, N=256
    )  # N is the number of colors in the map
    return COLORMAP


def plot_training_evolution(base_model_names: list[str], csv_path: str, save_path: Path):
    """
    Plots a 1 x (num_base_model_names) subplot figure
    for training evolution for each training dataset.
    """
    num_subplots = len(base_model_names)
    fig_width = 4 * num_subplots
    fig, axs = plt.subplots(1, num_subplots, figsize=(fig_width, 4), layout="constrained")

    # Load the csv file
    df = pd.read_csv(csv_path)
    for i, base_model_name in enumerate(base_model_names):
        ax = axs[i] if len(base_model_names) > 1 else axs

        # Get the data for the base model
        df_base_model = df[df["experiment"].str.contains(base_model_name)]
        df_base_model = df_base_model.sort_values(by="checkpoint")
        ckpts = df_base_model["checkpoint"].unique()
        max_ckpt = 5000

        train_dataset_name = "pw"
        metric = train_dataset_names2metric[train_dataset_name]
        colormap = get_colormap(train_dataset_name)
        max_difficulty = train_dataset_names2max_difficulty[train_dataset_name]

        # Group by checkpoint and kth_intermediate_answer and take the mean
        # i.e. average over complexity of all questions
        df_base_model = (
            df_base_model.groupby(["checkpoint", "kth_intermediate_answer"])[metric].mean().reset_index()
        )
        for ckpt in ckpts:
            if ckpt == 0:
                color = plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR["base"]]
            else:
                # Get the gradient color based on the checkpoint number, indexed into a colormap
                cmap = plt.get_cmap(colormap)
                color = cmap(ckpt / max_ckpt)

            linewidth = plotting_utils.LINE_WIDTH
            if ckpt in [0, max_ckpt]:
                linewidth = 1.5

            # Plot the data for the current checkpoint, with x values sorted by kth_intermediate_answer
            df_ckpt = df_base_model[df_base_model["checkpoint"] == ckpt].sort_values(
                by="kth_intermediate_answer"
            )
            x = df_ckpt["kth_intermediate_answer"]
            y = df_ckpt[metric]
            ax.plot(
                x,
                y,
                color=color,
                linestyle="solid",
                linewidth=linewidth,
                alpha=plotting_utils.LINE_ALPHA,
            )

        # format x-axis
        xticks = train_dataset_names2xticks[train_dataset_name]
        ax.set_xlim(xticks[0], max_difficulty)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=TICK_FONT_SIZE)
        ax.set_xlabel(train_dataset_names2xlabel[train_dataset_name], fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis="x", which="major")
        ax.tick_params(axis="x", which="minor")

        # format y-axis
        ax.set_ylim(0, 1)
        if i == 0:
            yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks, fontsize=TICK_FONT_SIZE)
            ax.set_ylabel("Fraction present", fontsize=LABEL_FONT_SIZE)
        else:
            ax.set_yticklabels([])

        ax.set_title(
            base_model_name,
            fontsize=LABEL_FONT_SIZE,
            fontweight="bold",
        )

        if i == 1:
            # Add a colorbar below the subplots for the training dataset
            norm = Normalize(vmin=0, vmax=max_ckpt)
            sm = ScalarMappable(cmap=plt.get_cmap(colormap), norm=norm)
            sm.set_array([])  # Only needed for older versions of matplotlib
            cbar = fig.colorbar(
                sm,
                ax=ax,
                orientation="vertical",
            )
            cbar_label = f"{plotting_utils.TRAIN_DATASET_ALIAS2NAME[train_dataset_name]} training steps"
            cbar.set_label(cbar_label, fontsize=LABEL_FONT_SIZE)
            cbar.ax.tick_params(
                labelsize=TICK_FONT_SIZE,
                labelcolor="black",
            )
            cbar.ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{int(x // 1000):d}" + "K")
            )
            cbar.outline.set_visible(False)  # Remove bounding box

    # Add legend entry for the training dataset name
    # First add a yellow line for the base model
    legend_handles = [
        lines.Line2D(
            [0],
            [0],
            color=plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR["base"]],
            label=plotting_utils.TRAIN_DATASET_ALIAS2NAME["base"],
            linewidth=1.5,
        )
    ]
    fig.legend(
        handles=legend_handles,
        fontsize=LEGEND_FONT_SIZE,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        bbox_to_anchor=(0.7, 0.4),  # Move in the middle of the plots
    )

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.savefig(save_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"Saved reasoning evolution plot to {save_path} and {save_path.with_suffix('.png')}")


if __name__ == "__main__":
    str_for_model_names = "__".join([m.replace("/", "--") for m in args.base_model_names_to_plot])
    save_path = Path(args.figures_dir) / f"reasoning_evolution_msq500_{str_for_model_names}.pdf"
    os.makedirs(args.figures_dir, exist_ok=True)
    plot_training_evolution(args.base_model_names_to_plot, args.csv_path, save_path)
