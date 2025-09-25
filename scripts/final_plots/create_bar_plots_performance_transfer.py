import matplotlib.pyplot as plt
import numpy as np

from phantom_reasoner.utils import plotting_utils

# Model and dataset names
models_in_matrix = [["Qwen3-0.6B", "Phi-4-mini-reasoning"], ["Qwen3-1.7B", "Qwen2.5-1.5B-Instruct"]]
train_dataset_names = ["base", "gsminf", "pw"]
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
    "Phi-4-mini-reasoning": {
        "HotpotQA": {"base": 0.4871, "format": 1, "gsminf": 0.5431, "pw": 0.6210},
        "2Wiki": {"base": 0.6663, "format": 1, "gsminf": 0.6739, "pw": 0.7586},
        "MuSiQue": {"base": 0.2923, "format": 1, "gsminf": 0.3250, "pw": 0.4469},
    },
}

LABEL_FONT_SIZE = plotting_utils.LABEL_FONT_SIZE + 2
TICK_FONT_SIZE = plotting_utils.TICK_FONT_SIZE + 2
LEGEND_FONT_SIZE = plotting_utils.LEGEND_FONT_SIZE + 2


def create_bar_plot_for_model(
    model: str, yticks: list[float], axes: list[plt.Axes], show_yticks: bool, x_left: float, x_right: float
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
        values = [data[model][dataset][train_dataset_name] for train_dataset_name in train_dataset_names]

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


if __name__ == "__main__":
    # Create a 2 x 6 subplot figure, where the first row is for Qwen3-0.6B and Phi-4-mini-reasoning,
    # and the second row is for Qwen3-1.7B and Qwen2.5-1.5B-Instruct
    # Select the axes
    fig, axes = plt.subplots(2, 6, figsize=(12, 8))
    for i, model_row in enumerate(models_in_matrix):
        for j, model in enumerate(model_row):
            yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            show_yticks = j == 0
            # i is the row index, and j*3:(j+1)*3 is the column index
            axes_slice = axes[i, j * 3 : (j + 1) * 3]
            x_left = axes_slice[0].get_position().x0
            x_right = axes_slice[-1].get_position().x1
            if j == 0:
                # Move the x a bit left
                x_left -= 0.01
                x_right -= 0.01
            elif j == 1:
                # Move the x a bit right
                x_left += 0.045
                x_right += 0.045
            create_bar_plot_for_model(model, yticks, axes_slice, show_yticks, x_left, x_right)

            # if j == 1, move all axes a bit right
            if j == 1:
                for ax in axes_slice:
                    ax.set_position(
                        ax.get_position().x0 + 0.05,
                        ax.get_position().y0,
                        ax.get_position().width,
                        ax.get_position().height,
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
        top=0.82,  # where the top subplot edges are, increase to move them closer to top figure edge
        bottom=0.1,  # where the bottom subplot edges are, increase to move them closer to bottom figure edge
        hspace=0.35,  # horizontal space between subplots, increase to move them away
        wspace=0.05,  # vertical space between subplots, increase to move them away
    )

    plt.savefig("f1_transfer_performance_all.pdf", dpi=300, bbox_inches="tight")
    plt.close()
