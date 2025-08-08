"""Script for plotting the accuracy of the models versus the difficulty as the training evolves.
We use the phantom_eval saved predictions for checkpoints saved during training.

Generates a plot for each metric (EM, precision, recall, f1) with the difficulty on the x-axis and the metric
on the y-axis.
Saves the plots to the figures directory of the output directory.

Example usage:
```bash
python scripts/plot_reasoning_during_training.py \
    -od runs/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-1.7B/grpo/x-anmolkab/0626__curr=random__prompt=cot/out \
    --model_list runs/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-1.7B/grpo/x-anmolkab/0626__curr=random__prompt=cot/ \
    --dataset data/wiki-v1-easy-depth_20_size_25 --from_local
```
"""

import logging

# %%
import os
import random
import re

import matplotlib.lines as lines
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from phantom_eval import get_parser, plotting_utils
from phantom_eval.evaluate_utils import get_evaluation_data, mean, pivot_mean_std, std
from phantom_eval.utils import setup_logging

setup_logging("INFO")

# utils for plotting
# plt.rcParams.update(
#     {
#         "font.family": "serif",
#         "font.serif": ["Times New Roman"],
#         "axes.spines.top": False,
#         "axes.spines.right": False,
#     }
# )


parser = get_parser()
parser.add_argument("--filter_by_depth", type=int, default=20, help="Depth to plot accuracies for")
parser.add_argument(
    "--filter_by_num_solutions", type=int, default=None, help="Number of solutions to filter by"
)
parser.add_argument(
    "--model_list", nargs="+", default=plotting_utils.DEFAULT_MODEL_LIST, help="List of models to plot"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed for color generation")
args = parser.parse_args()
output_dir = args.output_dir
model_list = args.model_list
dataset = args.dataset
filter_by_depth = args.filter_by_depth
from_local = args.from_local
seed = args.seed

assert len(model_list) == 1, "Please provide a single model to plot the accuracies for."

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
MAX_DIFFICULTY = 15
DIFFICULTY = "difficulty"

random.seed(seed)


def random_rgb_color():
    """Generate a random RGB color tuple."""
    return (random.random(), random.random(), random.random())


COLORMAP = "coolwarm"  # Default colormap to use for colors


def get_color(ckpt_name, method, by_model=True, max_ckpt=0):
    if by_model:
        ckpt = get_ckpt_number(ckpt_name)
        # num_ckpts is the total number of checkpoints, and ckpt is the current checkpoint number
        # Get the color based on the checkpoint number, indexed into a colormap from matplotlib
        if max_ckpt > 0:
            # Use a colormap to get a color based on the checkpoint number
            cmap = plt.get_cmap(COLORMAP)
            color = cmap(ckpt / max_ckpt)
            return color
        else:
            raise ValueError("max_ckpt must be greater than 0 to get a color based on the checkpoint number.")
    else:
        match method.lower():
            case "selfask":
                color = "tab:blue"
            case "ircot":
                color = "tab:green"
            case "sft":
                color = "tab:blue"
            case "grpo":
                color = "tab:green"
            case "zeroshot":
                color = "tab:red"
            case "cot":
                color = "tab:red"
            case _:
                color = "black"
        return color


def get_ckpt_number(model_name: str) -> int:
    """Extract the checkpoint number from the model name."""
    return int(re.search(r"checkpoint-(\d+)", model_name).group(1))


METHOD_LIST = [
    ("In-Context", plotting_utils.INCONTEXT_METHODS),
    # ("RAG", plotting_utils.RAG_METHODS),
    # ("Agentic", plotting_utils.AGENTIC_METHODS),
]

for metric in METRICS:
    # fig = plt.figure(figsize=(3.25, 2.75)) # exact dimensions of ICML single column width
    # replace this with a subplot figure with 1 rows and 3 columns
    fig_width = max(2.25 * len(METHOD_LIST), 3.25)
    fig, axs = plt.subplots(1, len(METHOD_LIST), figsize=(fig_width, 2.5))

    for i, (name, methods) in enumerate(METHOD_LIST):
        method_handles = {}
        ax = axs[i] if len(METHOD_LIST) > 1 else axs
        for method in methods:
            print(f"Plotting {method} for {metric}")
            # get evaluation data from the specified output directory and method subdirectory
            df = get_evaluation_data(output_dir, method, dataset, from_local)
            if df.empty:
                print(f"No data found for {method}")
                continue

            # ignore difficulty beyond 15
            df = df[df[DIFFICULTY] <= MAX_DIFFICULTY]

            if args.filter_by_num_solutions is not None:
                logging.warning(f"Filtering out {method} with more than 1 solution")
                df = df[df["solutions"] <= args.filter_by_num_solutions]

            # filter by depth
            df = df[(df["_depth"] == filter_by_depth)]

            # get accuracies by model, split, difficulty, seed
            COLS = ["_model", "_size", "_data_seed", "_seed", DIFFICULTY]
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
                df_mean = df_mean.loc[
                    df_mean.index.sort_values(key=lambda x: [get_ckpt_number(name) for name in x])
                ]

                for ckpt_name, row in df_mean.iterrows():
                    y = row
                    color = get_color(ckpt_name, method, max_ckpt=max_ckpt)
                    ckpt2color[ckpt_name] = color

                    ax.plot(
                        x,
                        y,
                        color=color,
                        # NOTE: determine the linestyle using the method
                        linestyle=plotting_utils.METHOD_LINESTYLES.get(method.lower(), "solid"),
                        linewidth=1,
                        alpha=plotting_utils.LINE_ALPHA,
                    )
                    # Add scatter plot
                    # ax.scatter(
                    #     x,
                    #     y,
                    #     color=color,
                    #     s=plotting_utils.MARKER_SIZE,  # marker size
                    #     alpha=plotting_utils.MARKER_ALPHA,
                    #     clip_on=False,
                    # )

                    # # Add error bars
                    # yerr = df_std.loc[ckpt_name]
                    # # Change color intensity for fill to be between 0 and 0.25
                    # color_intensity_for_fill = 0.1
                    # ax.fill_between(
                    #     x,
                    #     y - yerr,
                    #     y + yerr,
                    #     alpha=color_intensity_for_fill,
                    #     color=color,
                    # )

            # Add method to legend
            key = f"{plotting_utils.METHOD_ALIASES.get(method.lower(), method)}"
            if key not in method_handles:
                method_handles[key] = lines.Line2D(
                    [0],
                    [0],
                    color=get_color(None, method, by_model=False),
                    label=key,
                    linestyle=plotting_utils.METHOD_LINESTYLES[method],
                    linewidth=1,
                )
        ax.legend(
            handles=[v for _, v in method_handles.items()],
            fontsize=plotting_utils.LEGEND_FONT_SIZE,
            loc="upper right",
            ncol=1,
            handlelength=2,
            frameon=True,
        )

        ax.spines["bottom"].set_position(("outward", plotting_utils.OUTWARD))  # Move x-axis outward
        ax.spines["left"].set_position(("outward", plotting_utils.OUTWARD))  # Move y-axis outward

        # format x-axis
        ax.set_xlim(1, MAX_DIFFICULTY)
        LABELS = {
            "difficulty": "Reasoning steps",
            "solutions": "Solutions",
        }
        ax.set_xlabel(LABELS[DIFFICULTY], fontsize=plotting_utils.LABEL_FONT_SIZE)
        xticks = [1, 5, 10, 15]
        ax.set_xticks(xticks)
        ax.set_xticks(range(1, MAX_DIFFICULTY + 1), minor=True)
        ax.tick_params(axis="x", which="major")
        ax.tick_params(axis="x", which="minor")
        ax.set_xticklabels(xticks, fontsize=plotting_utils.TICK_FONT_SIZE)
        ax.set_xlim(1, MAX_DIFFICULTY)
        if i == 0:
            ax.set_ylabel(metric.upper(), fontsize=plotting_utils.LABEL_FONT_SIZE)
        # set ylim
        ax.set_ylim(0, 1)
        yticks = [0, 0.25, 0.5, 0.75, 1]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=plotting_utils.TICK_FONT_SIZE)
        # set title if there are more than panel
        if len(METHOD_LIST) > 1:
            ax.set_title(name, fontsize=plotting_utils.LABEL_FONT_SIZE)

    # On the right of the figure, add a vertical colorbar of COLORMAP
    # based on the checkpoint numbers
    norm = Normalize(vmin=0, vmax=max_ckpt)
    sm = ScalarMappable(cmap=plt.get_cmap(COLORMAP), norm=norm)
    sm.set_array([])  # Only needed for older versions of matplotlib
    cbar = fig.colorbar(sm, ax=axs, orientation="vertical")
    cbar.set_label("Training steps", fontsize=plotting_utils.TICK_FONT_SIZE)
    cbar.ax.tick_params(labelsize=plotting_utils.TICK_FONT_SIZE)
    cbar.outline.set_visible(False)  # Remove bounding box

    # Add a title with the model name
    # Get the slug after .../<dataset>/
    model_name = model_list[0]
    model_slug = model_name[model_name.rfind(args.dataset) + len(args.dataset) + 1 :]
    model_slug = model_slug[:-1] if model_slug.endswith("/") else model_slug
    plt.title(model_slug, fontsize=plotting_utils.LABEL_FONT_SIZE)

    plt.tight_layout()
    if len(METHOD_LIST) == 1:
        plt.subplots_adjust(
            left=0.2, right=0.95, top=0.9, bottom=0.2, wspace=0.3
        )  # Adjust horizontal space between subplots and reduce padding to the left and right
    else:
        plt.subplots_adjust(
            left=0.1, right=0.95, top=0.9, bottom=0.3, wspace=0.3
        )  # Adjust horizontal space between subplots and reduce padding to the left and right

    fig_path = os.path.join(figures_dir, f"{DIFFICULTY}-{metric}.pdf")
    print(f"Saving to {os.path.abspath(fig_path)}")
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)

# %%
