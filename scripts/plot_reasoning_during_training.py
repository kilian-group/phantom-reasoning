"""Script for plotting the accuracy of the models versus the difficulty as the training evolves.
We use the phantom_eval saved predictions for checkpoints saved during training.

Generates a plot for each metric (EM, precision, recall, f1) with the difficulty on the x-axis and the metric
on the y-axis.
Saves the plots to the figures directory of the output directory.

Example usage:
```bash
python scripts/plot_reasoning_during_training.py \
    --model_list runs/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-1.7B/grpo/x-anmolkab/0626__curr=random__prompt=cot/ \
    --dataset data/wiki-v1-easy-depth_20_size_25 --from_local \
    --base_model_name Qwen/Qwen3-1.7B \
    --training_dataset_names pw gsminf
```
"""  # noqa: E501

import json
import logging

# %%
import os
import random
import re

import matplotlib.lines as lines
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from phantom_eval import get_parser
from phantom_eval.evaluate_utils import get_evaluation_data, mean, pivot_mean_std, std
from phantom_eval.utils import setup_logging

from phantom_reasoner.utils import plotting_utils

setup_logging("INFO")


parser = get_parser()
parser.add_argument("--filter_by_depth", type=int, default=20, help="Depth to plot accuracies for")
parser.add_argument(
    "--filter_by_num_solutions", type=int, default=None, help="Number of solutions to filter by"
)
parser.add_argument("--model_list", nargs="+", default=[], help="List of models to plot")
parser.add_argument(
    "--training_dataset_names",
    nargs="+",
    choices=["pw", "gsminf"],
    default=["pw", "gsminf"],
    help="Training dataset name to plot",
)
parser.add_argument("--base_model_name", type=str, default=None, help="Base model name to plot")
parser.add_argument("--seed", type=int, default=42, help="Random seed for color generation")
args = parser.parse_args()
# output_dir = args.output_dir
model_list = args.model_list
dataset = args.dataset
filter_by_depth = args.filter_by_depth
from_local = args.from_local
seed = args.seed
base_model_name = args.base_model_name

assert len(model_list) == 1, "Please provide a single model to plot the accuracies for."

train_dataset_names = args.training_dataset_names
train_dataset_names2xticks = {
    "pw": [1, 5, 9],
    "gsminf": [1, 5, 9],  # TODO change to what it uses
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
    "gsminf": 9,  # TODO change to what it uses
}

# Output directory is out-pw or out-gsminf or out-pw-gsminf etc.
output_dir = os.path.join(model_list[0], f"out-{'-'.join(train_dataset_names)}")
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)
METRICS = [
    # 'EM',
    # 'precision',
    # 'recall',
    "f1",
]
# Difficulty can either be 'difficulty' (i.e., reasoning steps) or 'solutions' (i.e., number of solutions to
# the questions)
DIFFICULTY = "difficulty"

random.seed(seed)


def random_rgb_color():
    """Generate a random RGB color tuple."""
    return (random.random(), random.random(), random.random())


def get_colormap(training_dataset_name):
    COLORMAP_LAST_HEX = plotting_utils.COLORS2HEX[
        plotting_utils.TRAIN_DATASET_ALIAS2COLOR[training_dataset_name]
    ]
    colors = [(0, "white"), (1, COLORMAP_LAST_HEX)]
    COLORMAP = LinearSegmentedColormap.from_list(
        "WhiteToHex", colors, N=256
    )  # N is the number of colors in the map
    return COLORMAP


def get_color(ckpt_name, method, max_ckpt=0, colormap=None):
    ckpt = get_ckpt_number(ckpt_name)
    # num_ckpts is the total number of checkpoints, and ckpt is the current checkpoint number
    # Get the color based on the checkpoint number, indexed into a colormap from matplotlib
    if max_ckpt > 0:
        # Use a colormap to get a color based on the checkpoint number
        cmap = plt.get_cmap(colormap)
        color = cmap(ckpt / max_ckpt)
        return color
    else:
        raise ValueError("max_ckpt must be greater than 0 to get a color based on the checkpoint number.")


def get_ckpt_number(model_name: str) -> int:
    """Extract the checkpoint number from the model name."""
    if model_name == base_model_name:
        # Base model is checkpoint-0
        return 0
    elif model_name == model_list[0]:
        # Final model is checkpoint-<global_step> where <global_step> is from the json file
        with open(os.path.join(model_list[0], "trainer_state.json")) as f:
            trainer_state = json.load(f)
            global_step = int(trainer_state["global_step"])
            return global_step
    else:
        return int(re.search(r"checkpoint-(\d+)", model_name).group(1))


fig_width = 4 * len(train_dataset_names)
fig, axs = plt.subplots(1, len(train_dataset_names), figsize=(fig_width, 4))

for i, train_dataset_name in enumerate(train_dataset_names):
    metric = train_dataset_names2metric[train_dataset_name]
    colormap = get_colormap(train_dataset_name)
    method = "cot"
    max_difficulty = train_dataset_names2max_difficulty[train_dataset_name]
    ax = axs[i] if len(train_dataset_names) > 1 else axs
    output_dir = os.path.join(model_list[0], f"out-{train_dataset_name}")

    print(f"Plotting for {metric} and {train_dataset_name}")
    # get evaluation data from the specified output directory and method subdirectory
    df = get_evaluation_data(output_dir, method, dataset, from_local)
    if df.empty:
        print(f"No data found for {method}")
        continue

    # ignore difficulty beyond 15
    df = df[df[DIFFICULTY] <= max_difficulty]

    if args.filter_by_num_solutions is not None:
        logging.warning(f"Filtering out {method} with more than 1 solution")
        df = df[df["solutions"] <= args.filter_by_num_solutions]

    # filter by depth
    df = df[(df["_depth"] == filter_by_depth)]

    # get accuracies by model, split, difficulty, seed
    COLS = ["_model", "_size", "_data_seed", "_seed", DIFFICULTY]
    # TODO METRICS for gsm inf
    acc_by_type = df.groupby(COLS)[METRICS].mean()

    # get the mean and std of the accuracy for each model, split, and difficulty across seeds
    # first compute the mean across inference generation seeds
    acc_mean_std = acc_by_type.groupby(["_model", "_size", "_data_seed", DIFFICULTY]).agg("mean")
    # second compute the mean and standard error across data generation seeds
    acc_mean_std = acc_mean_std.groupby(["_model", "_size", DIFFICULTY]).agg([mean, std])
    acc_mean_std = acc_mean_std.reset_index()

    # Get sorted list of universe sizes
    sizes_in_preds = sorted(acc_mean_std["_size"].unique().tolist())
    # only plot the minimum size
    sizes_in_preds = [min(sizes_in_preds)]

    for size in sizes_in_preds:
        acc_mean_std_size = acc_mean_std[acc_mean_std["_size"].astype(int) == size]
        df_mean, df_std = pivot_mean_std(
            acc_mean_std_size, metric, independent_variable=DIFFICULTY, enforce_order=False
        )
        x = df_mean.columns
        ckpts = [get_ckpt_number(model_name) for model_name in df_mean.index]
        max_ckpt = max(ckpts) if ckpts else 0
        num_ckpts = len(ckpts)
        ckpt2color = {}

        # Sort df_mean by the checkpoint number
        df_mean = df_mean.loc[df_mean.index.sort_values(key=lambda x: [get_ckpt_number(name) for name in x])]

        for ckpt_name, row in df_mean.iterrows():
            y = row
            # If the ckpt_number is 0, use the base model color, else use the gradient color
            if get_ckpt_number(ckpt_name) == 0:
                color = plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR["base"]]
            else:
                color = get_color(ckpt_name, method, max_ckpt=max_ckpt, colormap=colormap)
            ckpt2color[ckpt_name] = color

            ax.plot(
                x,
                y,
                color=color,
                linestyle="solid",
                linewidth=1,
                alpha=plotting_utils.LINE_ALPHA,
            )

    # format x-axis
    ax.set_xlim(1, max_difficulty)
    xticks = train_dataset_names2xticks[train_dataset_name]
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
    ax.set_ylabel(metric.upper(), fontsize=plotting_utils.LABEL_FONT_SIZE)

    # Get model name between args.dataset/ and /grpo in the string using re
    model_name = re.search(f"{args.dataset}/(.*)/grpo", model_list[0]).group(1)
    ax.set_title(
        model_name,
        fontsize=plotting_utils.LABEL_FONT_SIZE,
        fontweight="bold",
    )

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
legend_handles = [
    lines.Line2D(
        [0],
        [0],
        color=plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR["base"]],
        label=plotting_utils.TRAIN_DATASET_ALIAS2NAME["base"],
        linewidth=1,
    )
]
for train_dataset_name in train_dataset_names:
    legend_handles.append(
        lines.Line2D(
            [0],
            [0],
            color=plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR[train_dataset_name]],
            label=plotting_utils.TRAIN_DATASET_ALIAS2NAME[train_dataset_name],
            linewidth=1,
        )
    )
fig.legend(
    handles=legend_handles,
    fontsize=plotting_utils.LEGEND_FONT_SIZE,
    loc="upper center",
    ncol=len(legend_handles),
    frameon=True,
    bbox_to_anchor=(0.5, 1.05),
)
plt.tight_layout()

fig_path = os.path.join(figures_dir, f"{DIFFICULTY}-{metric}.pdf")
print(f"Saving to {os.path.abspath(fig_path)}")
plt.savefig(fig_path, bbox_inches="tight", dpi=300)

# %%
