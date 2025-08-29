"""Script to plot the number of completion tokens per model, split, and seed.

Example usage:
```bash
python plot_completion_tokens.py -od OUTPUT_DIR --method METHOD --split SPLIT --dataset DATASET
```
"""

import os
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from utils.evaluate_utils import get_preds

from phantom_reasoner.utils.data_utils import get_parser

# utils for plotting
plt.rcParams.update(
    {
        # "font.family": "serif",
        # "font.serif": ["Times New Roman"],
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = {
    "none": "#0072B2",
    "difficulty_asc": "#009E73",
    "random": "#F0E442",
}


def get_model_curriculum(model):
    """
    Get the curriculum from the model name

    Args:
        model (str): The model name

    Returns:
        str: The curriculum
    """
    if "qwen3-0.6b" in model.lower():
        base = "qwen3-0.6b"
    elif "qwen3-1.7b" in model.lower():
        base = "qwen3-1.7b"
    else:
        raise ValueError(f"Base model could not be inferred from {model}")
    if match := re.search(r"__curr=(.*)__", model):
        curr = match.group(1)
    else:
        curr = "none"
    return base, curr


parser = get_parser()
args = parser.parse_args()
output_dir = args.output_dir
data_dir = args.data_dir
method = args.method
dataset = args.dataset
split = args.split

figure_dir = os.path.join(output_dir, "figures")
os.makedirs(figure_dir, exist_ok=True)

df, metrics = get_preds(output_dir, data_dir, dataset, split, method)
df["completion_tokens"] = df["usage"].apply(lambda x: x["completion_tokens"])

assert df["_dataset"].nunique() == 1, "Only one dataset is supported"
assert df["_split"].nunique() == 1, "Only one split is supported"

df["base"], df["curr"] = zip(*df["_model"].apply(get_model_curriculum))
df_grouped = df.groupby(["base", "_model"])["completion_tokens"].apply(list).reset_index()

ys = []
positions = []
colors = []
hatches = []
alphas = []
# Model handles at the bottom of the figure
model_handles = {}
xticks = []
base_positions = {}

# Plotting logic
current_pos = 1
for base_name, base_group in df_grouped.groupby("base"):
    base_start_pos = current_pos
    for _, row in base_group.iterrows():
        model = row["_model"]
        base, curr = get_model_curriculum(model)
        color = COLORS[curr]

        ys.append(row["completion_tokens"])
        positions.append(current_pos)
        colors.append(color)
        hatches.append("")
        alphas.append(0.5)

        xticks.append(current_pos)

        if curr not in model_handles:
            model_handles[curr] = Patch(facecolor=color, label=curr, alpha=0.5)

        current_pos += 1

    base_end_pos = current_pos - 1
    base_positions[base_name] = (base_start_pos + base_end_pos) / 2
    current_pos += 1  # Add a gap between base model groups

fig, ax = plt.subplots(figsize=(10, 6))

bplot = ax.boxplot(
    ys,
    patch_artist=True,  # fill with color
    positions=positions,
    widths=0.8,
    # don't show outliers
    showfliers=False,
    showmeans=True,
    meanline=True,
    meanprops=dict(
        color="black",  # set mean line color
    ),
    medianprops=dict(color="black"),  # set median line color
)
# fill with colors
for patch, color, hatch, alpha in zip(bplot["boxes"], colors, hatches, alphas):
    patch.set_facecolor(color)
    patch.set_hatch(hatch)
    patch.set_alpha(alpha)

ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.25)

# Set the x-axis labels for curricula
# ax.set_xticks(xticks)
# ax.set_xticklabels([])
# Turn off the x-axis
ax.set_xticks([])

# Add overarching labels for base models
for base_name, pos in base_positions.items():
    ax.text(
        pos,
        -0.05,
        base_name,
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontsize=12,
    )

ax.set_ylabel("Number of Completion Tokens")
ax.set_title(dataset, fontsize=20)
# add dashed vertical lines
# for logn in logn_list:
#     ax.axvline((logn+0.)*len(all_names), color='black', linestyle='--', linewidth=0.5, dashes=(5, 5))

plt.tight_layout()
fig.legend(
    handles=list(model_handles.values()),
    fontsize=20,
    loc="lower center",
    ncol=len(model_handles),
    handlelength=2,
    frameon=False,  # Remove bounding box around legend
    bbox_to_anchor=(0.5, 0.0),
)
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2, wspace=0.3)

save_path = os.path.join(figure_dir, f"completion_tokens_{dataset}_{split}.pdf")
plt.savefig(save_path)
print(f"Saved figure to {save_path}")
