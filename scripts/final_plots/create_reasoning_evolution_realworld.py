"""
Create reasoning evolution plots for all base models in the csv file.
A subplot is created for each base model, showing performance vs kth intermediate answer as the training evolves.

Example usage:
```bash
python scripts/final_plots/create_reasoning_evolution_realworld.py \
    --figures_dir "scripts/final_plots/figures"
```
"""  # noqa: E501

from pathlib import Path

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from phantom_eval import get_parser
from phantom_eval.utils import setup_logging

from phantom_reasoner.utils import plotting_utils

setup_logging("INFO")


parser = get_parser()
parser.add_argument("--figures_dir", type=str, default="scripts/final_plots/figures")
args = parser.parse_args()
args.figures_dir = Path(args.figures_dir)
args.figures_dir.mkdir(parents=True, exist_ok=True)

# Create a plot for each model in models_in_order
models_in_order = [
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen2.5-1.5B-Instruct",
    "Phi-4-mini-reasoning",
    "Qwen3-4B",
    "Qwen2.5-7B-Instruct",
]
train_dataset_names = ["pw", "gsminf"]
eval_dataset_names = ["msq500", "cofca500"]
eval_dataset_alias2name = {
    "msq500": "MuSiQue",
    "cofca500": "CofCA",
}
# Both datasets have upto 4 hops
eval_dataset_names2max_difficulty = {
    "msq500": 4,
    "cofca500": 4,
}

METRIC = "proportion_found"
XLABEL = "Nth intermediate answer"
DIFFICULTY = "complexity"

# Increase font sizes for better readability, since we plot two models
LABEL_FONT_SIZE = plotting_utils.LABEL_FONT_SIZE + 5
TICK_FONT_SIZE = plotting_utils.TICK_FONT_SIZE + 5
LEGEND_FONT_SIZE = plotting_utils.LEGEND_FONT_SIZE + 5

BASE_COLOR = "darkgray"
BASE_LINE_STYLE = "dashed"


def get_colormap(training_dataset_name):
    COLORMAP_LAST_HEX = plotting_utils.COLORS2HEX[
        plotting_utils.TRAIN_DATASET_ALIAS2COLOR[training_dataset_name]
    ]
    colors = [(0, "white"), (1, COLORMAP_LAST_HEX)]
    COLORMAP = LinearSegmentedColormap.from_list(
        "WhiteToHex", colors, N=256
    )  # N is the number of colors in the map
    return COLORMAP


def plot_training_evolution(base_model_name: str, figures_dir: Path, save_path: Path):
    """
    Plots a 1 x (num_train_dataset_names*num_eval_dataset_names) subplot figure
    for training evolution of given base model.
    """
    num_subplots = len(train_dataset_names) * len(eval_dataset_names)
    fig_width = 4 * num_subplots
    fig, axs = plt.subplots(1, num_subplots, figsize=(fig_width, 4), layout="constrained")

    # For each train dataset, plot num_eval_dataset_names subplots
    for i, train_dataset_name in enumerate(train_dataset_names):
        for j, eval_dataset_name in enumerate(eval_dataset_names):
            ax = axs[i * len(train_dataset_names) + j]
            # Load the csv file for the train x eval dataset
            csv_path = figures_dir / (
                f"reasoning_evolution__train={train_dataset_name}__eval={eval_dataset_name}.csv"
            )
            df = pd.read_csv(csv_path)

            # Get the data for the base model
            df_base_model = df[df["experiment"].str.contains(base_model_name)]
            df_base_model = df_base_model.sort_values(by="checkpoint")
            ckpts = df_base_model["checkpoint"].unique()
            max_ckpt = 5000

            colormap = get_colormap(train_dataset_name)

            # Group by checkpoint and kth_intermediate_answer and take the mean
            # i.e. average over complexity of all questions
            df_base_model = (
                df_base_model.groupby(["checkpoint", "kth_intermediate_answer"])[METRIC].mean().reset_index()
            )
            # Select every 1000th checkpoint
            for ckpt in ckpts:
                if ckpt == 0:
                    color = BASE_COLOR
                    linestyle = BASE_LINE_STYLE
                else:
                    # Get the gradient color based on the checkpoint number, indexed into a colormap
                    cmap = plt.get_cmap(colormap)
                    color = cmap(ckpt / max_ckpt)
                    linestyle = "solid"
                linewidth = plotting_utils.LINE_WIDTH
                if ckpt in [0, max_ckpt]:
                    linewidth = 1.5

                # Plot the data for the current checkpoint, with x values sorted by kth_intermediate_answer
                df_ckpt = df_base_model[df_base_model["checkpoint"] == ckpt].sort_values(
                    by="kth_intermediate_answer"
                )
                x = df_ckpt["kth_intermediate_answer"]
                y = df_ckpt[METRIC]
                ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=plotting_utils.LINE_ALPHA,
                )

            # format x-axis
            max_difficulty = eval_dataset_names2max_difficulty[eval_dataset_name]
            xticks = np.arange(1, max_difficulty + 1)
            ax.set_xlim(xticks[0], max_difficulty)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontsize=TICK_FONT_SIZE)
            ax.set_xlabel(XLABEL, fontsize=LABEL_FONT_SIZE)
            ax.tick_params(axis="x", which="major")
            ax.tick_params(axis="x", which="minor")

            # format y-axis
            ax.set_ylim(0, 1)
            if i == 0 and j == 0:
                yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticks, fontsize=TICK_FONT_SIZE)
            else:
                ax.set_yticklabels([])

            ax.set_title(
                eval_dataset_alias2name[eval_dataset_name],
                fontsize=LABEL_FONT_SIZE,
                fontweight="bold",
            )

        # Add a colorbar below the subplots for the training dataset
        norm = Normalize(vmin=0, vmax=max_ckpt)
        sm = ScalarMappable(cmap=plt.get_cmap(colormap), norm=norm)
        sm.set_array([])  # Only needed for older versions of matplotlib
        cbar = fig.colorbar(
            sm,
            ax=axs[i * len(train_dataset_names) : (i + 1) * len(train_dataset_names)],
            orientation="horizontal",
            shrink=0.6,
            location="bottom",  # pad=0.7
        )
        cbar_label = f"{plotting_utils.TRAIN_DATASET_ALIAS2NAME[train_dataset_name]} training steps"
        cbar.set_label(cbar_label, fontsize=LABEL_FONT_SIZE)
        cbar.ax.tick_params(
            labelsize=TICK_FONT_SIZE,
            labelcolor="black",
        )
        cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x // 1000):d}" + "K"))
        cbar.outline.set_visible(False)  # Remove bounding box

    fig.suptitle(
        f"Fraction of {base_model_name} generations with ground-truth intermediate answer",
        fontsize=LABEL_FONT_SIZE + 2,
        fontweight="bold",
    )
    # Add legend entry for the training dataset name
    # First add a yellow line for the base model
    legend_handles = [
        lines.Line2D(
            [0],
            [0],
            color=BASE_COLOR,
            linestyle=BASE_LINE_STYLE,
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
        bbox_to_anchor=(0.515, 0.2),  # Move below the plots
    )

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.savefig(save_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"Saved reasoning evolution plot to {save_path} and {save_path.with_suffix('.png')}")


if __name__ == "__main__":
    for base_model_name in models_in_order:
        save_filename = f"reasoning_evolution__realworld__{base_model_name.replace('/', '--')}.pdf"
        save_path = Path(args.figures_dir) / save_filename
        plot_training_evolution(base_model_name, args.figures_dir, save_path)
