import matplotlib.pyplot as plt
import numpy as np

# TODO: use plotting_utils
# Change font to Fira Code
plt.rcParams["font.family"] = "Fira Code"

BASE_NAME = "base"
FORMAT_NAME = "format"
GSM_NAME = "GSM-$\\infty$"
PW_NAME = "PhantomWiki"

# Model and dataset names
models = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen2.5-1.5B-Instruct"]
train_dataset_names = [BASE_NAME, FORMAT_NAME, GSM_NAME, PW_NAME]
eval_dataset_names = ["HotpotQA", "2Wiki", "MuSiQue"]

LINE_WIDTH = 1

# Data from the table
data = {
    "Qwen3-0.6B": {
        "HotpotQA": {BASE_NAME: 0.3654, FORMAT_NAME: 0.3780, GSM_NAME: 0.4787, PW_NAME: 0.5905},
        "2Wiki": {BASE_NAME: 0.3691, FORMAT_NAME: 0.3319, GSM_NAME: 0.4940, PW_NAME: 0.6013},
        "MuSiQue": {BASE_NAME: 0.1415, FORMAT_NAME: 0.1337, GSM_NAME: 0.1983, PW_NAME: 0.3283},
    },
    "Qwen3-1.7B": {
        "HotpotQA": {BASE_NAME: 0.5958, FORMAT_NAME: 0.6407, GSM_NAME: 0.6437, PW_NAME: 0.6581},
        "2Wiki": {BASE_NAME: 0.6354, FORMAT_NAME: 0.6665, GSM_NAME: 0.7278, PW_NAME: 0.7502},
        "MuSiQue": {BASE_NAME: 0.3411, FORMAT_NAME: 0.3449, GSM_NAME: 0.3968, PW_NAME: 0.4029},
    },
    "Qwen2.5-1.5B-Instruct": {
        "HotpotQA": {BASE_NAME: 0.0199, FORMAT_NAME: 0.4333, GSM_NAME: 0.3322, PW_NAME: 0.5359},
        "2Wiki": {BASE_NAME: 0.1422, FORMAT_NAME: 0.2957, GSM_NAME: 0.3561, PW_NAME: 0.4526},
        "MuSiQue": {BASE_NAME: 0.0402, FORMAT_NAME: 0.1983, GSM_NAME: 0.1629, PW_NAME: 0.2878},
    },
}


def create_bar_plot_for_model(model: str, yticks: list[float]):
    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(9, 4))

    # Bar settings - UPDATED COLORS to match the reference plot
    bar_width = 0.85
    x_pos = np.arange(len(train_dataset_names))
    label2color = {
        BASE_NAME: "#E8A93C",  # gold/yellow
        FORMAT_NAME: "#D96831",  # orange
        GSM_NAME: "#3B9B7B",  # teal/green
        PW_NAME: "#3B8FBF",  # blue
    }
    EDGE_COLOR = "black"
    TICK_FONT_SIZE = 10
    LABEL_FONT_SIZE = 13

    # Create bars for each subplot
    # for i, model in enumerate(models):
    for j, dataset in enumerate(eval_dataset_names):
        ax = axes[j]

        # Get values for this model-dataset combination - REORDERED
        values = [
            data[model][dataset][BASE_NAME],
            data[model][dataset][FORMAT_NAME],
            data[model][dataset][GSM_NAME],
            data[model][dataset][PW_NAME],
        ]

        # Create bars with error bars (using small errors for visual effect)
        # TODO add error bars
        errors = [0.02] * len(values)  # Small error bars for visual effect
        colors = [label2color[label] for label in [BASE_NAME, FORMAT_NAME, GSM_NAME, PW_NAME]]
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
            ax.set_yticklabels(list(map(str, yticks)), fontsize=TICK_FONT_SIZE)
        # Only show y-axis label for the left most subplots
        if j == 0:
            ax.set_ylabel("F1", fontsize=LABEL_FONT_SIZE)

        # Don't show x-axis ticks and labels
        ax.set_xticks([])
        ax.set_xticklabels([])
        # Show x-axis label for the middle subplot
        # if j == 1:
        #     ax.set_xlabel(model, fontsize=LABEL_FONT_SIZE, fontweight='bold')

        # Add horizontal grid lines
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=LINE_WIDTH)
        ax.set_axisbelow(True)

        # Style the spines
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(LINE_WIDTH)

        # Set titles for top row (eval datasets)
        # if i == 0:
        ax.set_title(dataset, fontsize=LABEL_FONT_SIZE, fontweight="bold", pad=10)

        # # Set row labels on the left
        # if j == 0:
        #     # Add text outside the plot area
        #     axes[i, j].text(-1.8, 0.5, model,
        #             fontsize=LABEL_FONT_SIZE, fontweight='bold', rotation=90,
        #             verticalalignment='center', horizontalalignment='center')

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
    # axes[1].set_xlabel(model, fontsize=LABEL_FONT_SIZE, fontweight='bold')
    fig.text(
        x_center + 0.02,
        y_pos + BRACKET_HEIGHT,
        model,
        fontsize=LABEL_FONT_SIZE,
        fontweight="bold",
        verticalalignment="top",
        horizontalalignment="center",
        transform=fig.transFigure,
        bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="black", linewidth=LINE_WIDTH),
    )

    # Create legend with better styling
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=label2color[label], edgecolor="black", label=label)
        for label in [BASE_NAME, FORMAT_NAME, GSM_NAME, PW_NAME]
    ]

    # Position legend on the right side
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        fontsize=LABEL_FONT_SIZE,
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
