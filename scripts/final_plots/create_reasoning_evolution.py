"""
Create reasoning evolution plots for all base models in the final ckpts yaml file.
A subplot is created for each training dataset, showing performance vs difficulty as the training evolves.

Example usage:
```bash
python scripts/final_plots/create_reasoning_evolution.py \
    --final_ckpts_yaml_path scripts/final_plots/final_ckpts.yaml \
    --base_model_names_to_plot "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" \
    --figures_dir "scripts/final_plots/figures"
```
"""  # noqa: E501

import json
import os
import re
from pathlib import Path

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
parser.add_argument("--base_model_names_to_plot", nargs="+", default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"])
parser.add_argument("--figures_dir", type=str, default="scripts/final_plots/figures")
args = parser.parse_args()

assert len(args.base_model_names_to_plot) == 2, "Only two base models together are supported for this script"

train_dataset_names2xticks = {
    "pw": [1, 5, 9],
    "gsminf": [2, 5, 10, 15, 20],
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
    "gsminf": 20,
}

# Load the yaml file
with open(args.final_ckpts_yaml_path) as f:
    final_ckpts_yaml = yaml.safe_load(f)
    synthetic_train_ckpts = final_ckpts_yaml["synthetic_train_ckpts"]

DIFFICULTY = "difficulty"

# Increase font sizes for better readability, since we plot two models
LABEL_FONT_SIZE = plotting_utils.LABEL_FONT_SIZE + 5
TICK_FONT_SIZE = plotting_utils.TICK_FONT_SIZE + 5
LEGEND_FONT_SIZE = plotting_utils.LEGEND_FONT_SIZE + 5


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


def plot_training_evolution(base_model_names: list[str], synthetic_train_ckpts: list[dict], save_path: Path):
    """
    Plots a 1 x (num_train_dataset_names * num_base_model_names) subplot figure
    for training evolution for each training dataset.
    """
    train_dataset_names = [train_ckpts_dict["dataset_name"] for train_ckpts_dict in synthetic_train_ckpts]
    # Remove rg-family_relationships from the list
    if "rg-family_relationships" in train_dataset_names:
        train_dataset_names.pop(train_dataset_names.index("rg-family_relationships"))
    if "rg-knights_knaves" in train_dataset_names:
        train_dataset_names.pop(train_dataset_names.index("rg-knights_knaves"))
    num_subplots = len(train_dataset_names) * len(base_model_names)
    # To maintain colors, plot all models for a training dataset,
    # then move to the next training dataset
    fig_width = 4 * num_subplots
    fig, axs = plt.subplots(1, num_subplots, figsize=(fig_width, 4), layout="constrained")

    for i, train_ckpts_dict in enumerate(synthetic_train_ckpts):
        train_dataset_name = train_ckpts_dict["dataset_name"]
        if train_dataset_name == "rg-family_relationships":
            # Not supported for this script
            continue
        if train_dataset_name == "rg-knights_knaves":
            # TODO: not supported for this script
            continue
        metric = train_dataset_names2metric[train_dataset_name]
        colormap = get_colormap(train_dataset_name)
        method = "cot"
        max_difficulty = train_dataset_names2max_difficulty[train_dataset_name]
        for j, base_model_name in enumerate(base_model_names):
            ax = axs[i * len(train_dataset_names) + j]

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

            for ckpt_name, y in df_mean.iterrows():
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
            ax.set_xticklabels(xticks, fontsize=TICK_FONT_SIZE)
            ax.set_xlabel(train_dataset_names2xlabel[train_dataset_name], fontsize=LABEL_FONT_SIZE)
            ax.tick_params(axis="x", which="major")
            ax.tick_params(axis="x", which="minor")

            # format y-axis
            ax.set_ylim(0, 1)
            if j == 0:
                yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticks, fontsize=TICK_FONT_SIZE)
                ax.set_ylabel(metric.capitalize(), fontsize=LABEL_FONT_SIZE)
            else:
                ax.set_yticklabels([])

            ax.set_title(
                plotting_utils.MODEL_NAME2ALIAS[base_model_name],
                fontsize=LABEL_FONT_SIZE,
                fontweight="bold",
            )

        # Add a colorbar below the subplots for the training dataset
        norm = Normalize(vmin=0, vmax=max_ckpt)
        sm = ScalarMappable(cmap=plt.get_cmap(colormap), norm=norm)
        sm.set_array([])  # Only needed for older versions of matplotlib
        training_dataset_axes = axs[i * len(base_model_names) : (i + 1) * len(base_model_names)]
        cbar = fig.colorbar(
            sm,
            ax=training_dataset_axes,
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
        bbox_to_anchor=(0.515, 0.2),  # Move below the plots
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved reasoning evolution plot to {save_path} and {save_path.with_suffix('.png')}")


if __name__ == "__main__":
    str_for_model_names = "__".join([m.replace("/", "--") for m in args.base_model_names_to_plot])
    save_path = Path(args.figures_dir) / f"reasoning_evolution_{str_for_model_names}.pdf"
    os.makedirs(args.figures_dir, exist_ok=True)
    plot_training_evolution(args.base_model_names_to_plot, synthetic_train_ckpts, save_path)
