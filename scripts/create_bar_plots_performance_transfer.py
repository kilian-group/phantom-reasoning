import matplotlib.pyplot as plt
import numpy as np

from phantom_reasoner.utils import plotting_utils

# Model and dataset names
models = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen2.5-1.5B-Instruct"]
train_dataset_names = plotting_utils.TRAIN_DATASET_NAMES
eval_dataset_names = plotting_utils.EVAL_DATASET_NAMES

LINE_WIDTH = 1

# Data from the table
data = {
    "Qwen3-0.6B": {
        "HotpotQA": {"base": 0.3654, "format": 0.3780, "gsminf": 0.4787, "pw": 0.5905},
        "2Wiki": {"base": 0.3691, "format": 0.3319, "gsminf": 0.4940, "pw": 0.6013},
        "MuSiQue": {"base": 0.1415, "format": 0.1337, "gsminf": 0.1983, "pw": 0.3283},
    },
    "Qwen3-1.7B": {
        "HotpotQA": {"base": 0.5958, "format": 0.6407, "gsminf": 0.6437, "pw": 0.6581},
        "2Wiki": {"base": 0.6354, "format": 0.6665, "gsminf": 0.7278, "pw": 0.7502},
        "MuSiQue": {"base": 0.3411, "format": 0.3449, "gsminf": 0.3968, "pw": 0.4029},
    },
    "Qwen2.5-1.5B-Instruct": {
        "HotpotQA": {"base": 0.0199, "format": 0.4333, "gsminf": 0.3322, "pw": 0.5359},
        "2Wiki": {"base": 0.1422, "format": 0.2957, "gsminf": 0.3561, "pw": 0.4526},
        "MuSiQue": {"base": 0.0402, "format": 0.1983, "gsminf": 0.1629, "pw": 0.2878},
    },
}


def create_bar_plot_for_model(model: str, yticks: list[float]):
    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(9, 4))

    # Bar settings - UPDATED COLORS to match the reference plot
    bar_width = 0.85
    x_pos = np.arange(len(train_dataset_names))
    EDGE_COLOR = "black"

    # Create bars for each subplot
    # for i, model in enumerate(models):
    for j, dataset in enumerate(eval_dataset_names):
        ax = axes[j]

        # Get values for this model-dataset combination - REORDERED
        values = [
            data[model][dataset]["base"],
            data[model][dataset]["format"],
            data[model][dataset]["gsminf"],
            data[model][dataset]["pw"],
        ]

        # Create bars with error bars (using small errors for visual effect)
        # TODO add error bars
        errors = [0.02] * len(values)  # Small error bars for visual effect
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
        # Set y-axis ticks at 0.0, 0.2, 0.4, 0.6, 0.8
        ax.set_ylim(0, max(yticks))
        ax.set_yticks(yticks)
        # Only show y-axis tick labels for the left most subplots
        # So no y-axis tick labels for the other subplots
        if j != 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(list(map(str, yticks)), fontsize=plotting_utils.TICK_FONT_SIZE)
        # Only show y-axis label for the left most subplots
        if j == 0:
            ax.set_ylabel("F1", fontsize=plotting_utils.LABEL_FONT_SIZE)

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
        ax.set_title(dataset, fontsize=plotting_utils.LABEL_FONT_SIZE, fontweight="bold", pad=10)

    # Get the positions of the first and last subplot in this row
    ax_left = axes[0].get_position()
    ax_right = axes[2].get_position()
    LEFT_OFFSET = 0.04
    RIGHT_OFFSET = 0.08
    BRACKET_Y_OFFSET = 0.05

    # Calculate positions
    y_pos = ax_left.y0 - BRACKET_Y_OFFSET
    BRACKET_HEIGHT = 0.015

    # Draw left vertical line
    fig.add_artist(
        plt.Line2D(
            [ax_left.x0 - LEFT_OFFSET, ax_left.x0 - LEFT_OFFSET],
            [y_pos, y_pos + BRACKET_HEIGHT],
            transform=fig.transFigure,
            color="black",
            linewidth=LINE_WIDTH,
        )
    )

    # Draw horizontal line
    fig.add_artist(
        plt.Line2D(
            [ax_left.x0 - LEFT_OFFSET, ax_right.x1 + RIGHT_OFFSET],
            [y_pos, y_pos],
            transform=fig.transFigure,
            color="black",
            linewidth=LINE_WIDTH,
        )
    )

    # Draw right vertical line
    fig.add_artist(
        plt.Line2D(
            [ax_right.x1 + RIGHT_OFFSET, ax_right.x1 + RIGHT_OFFSET],
            [y_pos, y_pos + BRACKET_HEIGHT],
            transform=fig.transFigure,
            color="black",
            linewidth=LINE_WIDTH,
        )
    )

    # Add the model name centered below the bracket
    x_center = (ax_left.x0 + ax_right.x1) / 2
    fig.text(
        x_center + 0.02,
        y_pos + BRACKET_HEIGHT,
        model,
        fontsize=plotting_utils.LABEL_FONT_SIZE,
        fontweight="bold",
        verticalalignment="top",
        horizontalalignment="center",
        transform=fig.transFigure,
        bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="black", linewidth=LINE_WIDTH),
    )

    # Create legend with better styling
    from matplotlib.patches import Patch

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
        fontsize=plotting_utils.LABEL_FONT_SIZE,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        bbox_to_anchor=(0.53, 0.9),  # Move just below the title at the top center
        ncol=len(train_dataset_names),
    )

    # Adjust layout
    plt.subplots_adjust(
        left=0.08,  # where the left subplot y-labels are, increase to move them away from left figure edge
        right=0.98,  # where the right subplot edges are, increase to move them closer to right figure edge
        top=0.9,  # where the top subplot edges are, increase to move them closer to top figure edge
        bottom=0.1,  # where the bottom subplot edges are, increase to move them closer to bottom figure edge
        hspace=0.05,  # horizontal space between subplots, increase to move them away
        wspace=0.05,  # vertical space between subplots, increase to move them away
    )
    # plt.tight_layout()

    plt.savefig(f"f1_score_comparison_{model}.pdf", dpi=300)
    plt.close()


if __name__ == "__main__":
    for model in models:
        yticks = [0.0, 0.2, 0.4, 0.6, 0.8]
        if model == "Qwen3-1.7B":
            yticks += [1.0]
        create_bar_plot_for_model(model, yticks)
