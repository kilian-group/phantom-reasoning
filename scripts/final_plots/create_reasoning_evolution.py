"""
Create reasoning evolution plots for all base models in the final ckpts yaml file.
A subplot is created for each training dataset, showing performance vs difficulty as the training evolves.

Example usage:
```bash
python scripts/final_plots/create_reasoning_evolution.py \
    --final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml
```
"""  # noqa: E501

import json
import os
import re
from pathlib import Path

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from phantom_eval import get_parser
from phantom_eval.evaluate_utils import get_evaluation_data, mean, pivot_mean_std, std
from phantom_eval.utils import setup_logging

from phantom_reasoner.utils import plotting_utils

setup_logging("INFO")


parser = get_parser()
parser.add_argument(
    "--final_ckpts_yaml_path", type=str, required=True, help="Path to the final ckpts yaml file"
)
args = parser.parse_args()

train_dataset_names2xticks = {
    "pw": [1, 5, 9],
    "gsminf": [2, 5, 10, 15, 20],  # TODO change to what it uses
}
train_dataset_names2metric = {
    "pw": "f1",
    "gsminf": "accuracy",
}
train_dataset_names2xlabel = {
    "pw": "Question difficulty",
    "gsminf": "Arithmetic operations",
}
train_dataset_names2max_difficulty = {
    "pw": 9,
    "gsminf": 20,  # TODO change to what it uses
}

# Load the yaml file
with open(args.final_ckpts_yaml_path) as f:
    final_ckpts_yaml = yaml.safe_load(f)
    synthetic_train_ckpts = final_ckpts_yaml["synthetic_train_ckpts"]

base_model_names = {
    ckpt["model"] for train_dataset in synthetic_train_ckpts for ckpt in train_dataset["ckpts"]
}
DIFFICULTY = "difficulty"


def get_colormap(training_dataset_name):
    COLORMAP_LAST_HEX = plotting_utils.COLORS2HEX[
        plotting_utils.TRAIN_DATASET_ALIAS2COLOR[training_dataset_name]
    ]
    colors = [(0, "white"), (1, COLORMAP_LAST_HEX)]
    COLORMAP = LinearSegmentedColormap.from_list(
        "WhiteToHex", colors, N=256
    )  # N is the number of colors in the map
    return COLORMAP


def get_color(ckpt_name, base_model_name, ckpt_parent_dir, max_ckpt=0, colormap=None):
    ckpt = get_ckpt_number(ckpt_name, base_model_name, ckpt_parent_dir)
    # Get the color based on the checkpoint number, indexed into a colormap from matplotlib
    if max_ckpt > 0:
        # Use a colormap to get a color based on the checkpoint number
        cmap = plt.get_cmap(colormap)
        color = cmap(ckpt / max_ckpt)
        return color
    else:
        raise ValueError("max_ckpt must be greater than 0 to get a color based on the checkpoint number.")


def get_ckpt_number(model_name: str, base_model_name: str, ckpt_parent_dir: str) -> int:
    """Extract the checkpoint number from the model name."""
    if model_name == base_model_name:
        # Base model is checkpoint-0
        return 0
    elif model_name == ckpt_parent_dir:
        # Final model is checkpoint-<global_step> where <global_step> is from the json file
        with open(os.path.join(ckpt_parent_dir, "trainer_state.json")) as f:
            trainer_state = json.load(f)
            global_step = int(trainer_state["global_step"])
            return global_step
    else:
        return int(re.search(r"checkpoint-(\d+)", model_name).group(1))


def get_acc_mean_std(
    train_dataset_name: str,
    preds_output_dir: str,
    method: str,
    metrics: list[str],
    dataset_path: str = "",
    from_local: bool = False,
) -> pd.DataFrame:
    max_difficulty = train_dataset_names2max_difficulty[train_dataset_name]
    match train_dataset_name:
        case "pw":
            # get evaluation data from the specified output directory and method subdirectory
            df = get_evaluation_data(preds_output_dir, method, dataset_path, from_local)
            if df.empty:
                print(f"No data found for {method}")
                return pd.DataFrame()
            df = df[df[DIFFICULTY] <= max_difficulty]

            # get accuracies by model, split, difficulty, seed
            COLS = ["_model", "_size", "_data_seed", "_seed", DIFFICULTY]
            acc_by_type = df.groupby(COLS)[metrics].mean()

            # get the mean and std of the accuracy for each model, split, and difficulty across seeds
            # first compute the mean across inference generation seeds
            acc_mean_std = acc_by_type.groupby(["_model", "_size", "_data_seed", DIFFICULTY]).agg("mean")
            # second compute the mean and standard error across data generation seeds
            acc_mean_std = acc_mean_std.groupby(["_model", "_size", DIFFICULTY]).agg([mean, std])
            acc_mean_std = acc_mean_std.reset_index()
            return acc_mean_std
        case "gsminf":
            df = []
            scores_path = Path(preds_output_dir) / Path("scores")
            for csv_file_path in scores_path.glob("*.csv"):
                df.append(pd.read_csv(csv_file_path))
            df = pd.concat(df)
            # df = get_gsminf_evaluation_data(preds_output_dir)
            if df.empty:
                print(f"No data found for {method}")
                return pd.DataFrame()
            df = df[df[DIFFICULTY] <= max_difficulty]
            df["_size"] = 0  # placeholder for size (corresponding to length in GSM infinite)
            df["_model"] = df["model_name"]

            # get the mean and std of the accuracy for each model, and difficulty across sizes
            # first compute the mean across inference generation seeds
            acc_mean_std = df.groupby(["_model", "_size", DIFFICULTY])[metrics].agg([mean, std])
            # # second compute the mean and standard error across data generation seeds
            # acc_mean_std = acc_mean_std.groupby(["_model", "_size", DIFFICULTY]).agg([mean, std])
            acc_mean_std = acc_mean_std.reset_index()
            return acc_mean_std
        case _:
            raise ValueError(f"Invalid train dataset name: {train_dataset_name}")


def plot_training_evolution(base_model_name: str, synthetic_train_ckpts: list[dict], save_path: str):
    """
    Plots a 1 x num_train_dataset_names subplot figure for training evolution for each training dataset.
    """
    train_dataset_names = [train_ckpts_dict["dataset_name"] for train_ckpts_dict in synthetic_train_ckpts]
    fig_width = 4 * len(train_dataset_names)
    fig, axs = plt.subplots(1, len(train_dataset_names), figsize=(fig_width, 4))

    for i, train_ckpts_dict in enumerate(synthetic_train_ckpts):
        train_dataset_name = train_ckpts_dict["dataset_name"]
        metric = train_dataset_names2metric[train_dataset_name]
        colormap = get_colormap(train_dataset_name)
        method = "cot"
        max_difficulty = train_dataset_names2max_difficulty[train_dataset_name]
        ax = axs[i] if len(train_dataset_names) > 1 else axs

        ckpt_dict_to_use = None
        for ckpt in train_ckpts_dict["ckpts"]:
            if ckpt["model"] == base_model_name:
                ckpt_dict_to_use = ckpt
                break
        # Take the first path only
        ckpt_parent_dir = ckpt_dict_to_use["paths"][0]
        print(f"{train_dataset_name} trained ckpt parent dir: {ckpt_parent_dir}")

        # Load from the preds output directory
        preds_output_dir = os.path.join(ckpt_parent_dir, f"out-{train_dataset_name}")
        acc_mean_std_df = get_acc_mean_std(
            train_dataset_name,
            preds_output_dir,
            method,
            [metric],
            dataset_path=train_ckpts_dict["dataset_path"],
            from_local=True,
        )

        # Get sorted list of universe sizes, only plot the minimum size
        sizes_in_preds = sorted(acc_mean_std_df["_size"].unique().tolist())
        acc_mean_std_size = acc_mean_std_df[acc_mean_std_df["_size"].astype(int) == min(sizes_in_preds)]
        df_mean, _ = pivot_mean_std(
            acc_mean_std_size, metric, independent_variable=DIFFICULTY, enforce_order=False
        )
        x = df_mean.columns
        ckpts = [
            get_ckpt_number(model_name, base_model_name, ckpt_parent_dir) for model_name in df_mean.index
        ]
        max_ckpt = max(ckpts) if ckpts else 0

        # Sort df_mean by the checkpoint number
        df_mean = df_mean.loc[
            df_mean.index.sort_values(
                key=lambda x: [get_ckpt_number(name, base_model_name, ckpt_parent_dir) for name in x]
            )
        ]

        for ckpt_name, row in df_mean.iterrows():
            y = row
            # If the ckpt_number is 0, use the base model color, else use the gradient color
            ckpt_number = get_ckpt_number(ckpt_name, base_model_name, ckpt_parent_dir)
            if ckpt_number == 0:
                color = plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR["base"]]
            else:
                color = get_color(
                    ckpt_name, base_model_name, ckpt_parent_dir, max_ckpt=max_ckpt, colormap=colormap
                )

            linewidth = plotting_utils.LINE_WIDTH
            if ckpt_number in [0, max_ckpt]:
                linewidth = 1.5
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
        ax.set_xticklabels(xticks, fontsize=plotting_utils.TICK_FONT_SIZE)
        ax.set_xlabel(train_dataset_names2xlabel[train_dataset_name], fontsize=plotting_utils.LABEL_FONT_SIZE)
        ax.tick_params(axis="x", which="major")
        ax.tick_params(axis="x", which="minor")

        # format y-axis
        ax.set_ylim(0, 1)
        yticks = [0, 0.25, 0.5, 0.75, 1]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=plotting_utils.TICK_FONT_SIZE)
        ax.set_ylabel(metric.capitalize(), fontsize=plotting_utils.LABEL_FONT_SIZE)

        # On the right of the figure, add a vertical colorbar of colormap
        # based on the checkpoint numbers
        norm = Normalize(vmin=0, vmax=max_ckpt)
        sm = ScalarMappable(cmap=plt.get_cmap(colormap), norm=norm)
        sm.set_array([])  # Only needed for older versions of matplotlib
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical")
        cbar.set_label("Training steps", fontsize=plotting_utils.TICK_FONT_SIZE)
        cbar.ax.tick_params(labelsize=plotting_utils.TICK_FONT_SIZE)
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
    for train_dataset_name in train_dataset_names:
        legend_handles.append(
            lines.Line2D(
                [0],
                [0],
                color=plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR[train_dataset_name]],
                label=plotting_utils.TRAIN_DATASET_ALIAS2NAME[train_dataset_name],
                linewidth=1.5,
            )
        )
    fig.legend(
        handles=legend_handles,
        fontsize=plotting_utils.LEGEND_FONT_SIZE,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        bbox_to_anchor=(0.5, 1.02),  # Move above the plots
    )

    # Add model name to the bottom of the figure
    ax_left = axs[0].get_position()
    ax_right = axs[-1].get_position()
    LEFT_OFFSET = 0.0
    RIGHT_OFFSET = 0.0

    # Calculate positions
    y_pos = ax_left.y0 - 0.15
    BRACKET_HEIGHT = 0.015

    # Draw left vertical line
    fig.add_artist(
        plt.Line2D(
            [ax_left.x0 - LEFT_OFFSET, ax_left.x0 - LEFT_OFFSET],
            [y_pos, y_pos + BRACKET_HEIGHT],
            transform=fig.transFigure,
            color="black",
            linewidth=plotting_utils.LINE_WIDTH,
        )
    )

    # Draw horizontal line
    fig.add_artist(
        plt.Line2D(
            [ax_left.x0 - LEFT_OFFSET, ax_right.x1 + RIGHT_OFFSET],
            [y_pos, y_pos],
            transform=fig.transFigure,
            color="black",
            linewidth=plotting_utils.LINE_WIDTH,
        )
    )

    # Draw right vertical line
    fig.add_artist(
        plt.Line2D(
            [ax_right.x1 + RIGHT_OFFSET, ax_right.x1 + RIGHT_OFFSET],
            [y_pos, y_pos + BRACKET_HEIGHT],
            transform=fig.transFigure,
            color="black",
            linewidth=plotting_utils.LINE_WIDTH,
        )
    )
    # Add the model name centered below the bracket
    x_center = (ax_left.x0 + ax_right.x1) / 2
    fig.text(
        x_center + 0.02,
        y_pos + BRACKET_HEIGHT,
        plotting_utils.MODEL_NAME2ALIAS[base_model_name],
        fontsize=plotting_utils.LABEL_FONT_SIZE,
        fontweight="bold",
        verticalalignment="top",
        horizontalalignment="center",
        transform=fig.transFigure,
        bbox=dict(
            boxstyle="square,pad=0.3",
            facecolor="white",
            edgecolor="black",
            linewidth=plotting_utils.LINE_WIDTH,
        ),
    )

    # Adjust layout
    plt.subplots_adjust(
        left=0.08,  # where the left subplot y-labels are, increase to move them away from left figure edge
        right=0.98,  # where the right subplot edges are, increase to move them closer to right figure edge
        top=0.85,  # where the top subplot edges are, increase to move them closer to top figure edge
        bottom=0.15,  # where the bottom subplot edges are, increase to move them closer to bottom figure edge
        # hspace=0.4,  # horizontal space between subplots, increase to move them away
        wspace=0.4,  # vertical space between subplots, increase to move them away
    )
    # plt.tight_layout()

    print(f"Saving to {save_path}")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    for base_model_name in base_model_names:
        save_path = f"reasoning_evolution_{base_model_name.replace('/', '--')}.pdf"
        print("--------------------------------")
        print(f"Plotting reasoning evolution for {base_model_name}")
        print("--------------------------------")
        plot_training_evolution(base_model_name, synthetic_train_ckpts, save_path)
