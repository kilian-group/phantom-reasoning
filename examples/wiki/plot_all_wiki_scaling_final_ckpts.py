"""Script that formats the split accuracy.

The output is a markdown table with columns corresponding to the metrics and
rows corresponding to the model, split, and seed.

Example usage:
NOTE: The script plots for all wiki, so dataset and split are ignored even though they are required.
```bash
python examples/wiki/plot_all_wiki_scaling_final_ckpts.py \
    --final_ckpts_yaml_path final_ckpts.yaml \
    --data_dir data \
    --dataset hp500 \
    --split minidev
```
By default, --data_dir is `/share/nikola/phantom-reasoning/data`.
"""

import json
import os

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import yaml
from utils.data_utils import get_parser
from utils.evaluate_utils import get_preds

from phantom_reasoner.utils import plotting_utils

parser = get_parser()
parser.add_argument("--final_ckpts_yaml_path", type=str, required=True)
# parser.add_argument("--training_dataset_names", nargs="+", required=True)
# parser.add_argument("--base_model_name", type=str, required=True)
# parser.add_argument("--training_dataset_name", type=str, required=True)
args = parser.parse_args()
# output_dir = args.output_dir
# method = args.method
# dataset = args.dataset
# split = args.split
# base_model_name = args.base_model_name
# training_dataset_names = args.training_dataset_names

eval_datasets = ["hp500", "2wiki500", "msq500"]
eval_dataset2name = {
    "hp500": "HotpotQA",
    "2wiki500": "2Wiki",
    "msq500": "MuSiQue",
}
eval_dataset2plot_metric = {
    "hp500": "f1",
    "2wiki500": "f1",
    "msq500": "f1",
}
eval_dataset2ylims = {
    "hp500": (0.25, 0.75),
    "2wiki500": (0.25, 0.75),
    "msq500": (0.0, 0.5),
}
eval_dataset2yticks = {
    "hp500": [0.25, 0.5, 0.75],
    "2wiki500": [0.25, 0.5, 0.75],
    "msq500": [0.0, 0.25, 0.5],
}

# Load the yaml file
with open(args.final_ckpts_yaml_path) as f:
    final_ckpts_yaml = yaml.safe_load(f)
    synthetic_train_ckpts = final_ckpts_yaml["synthetic_train_ckpts"]

training_dataset_names: list[str] = [train_dataset["dataset_name"] for train_dataset in synthetic_train_ckpts]
base_model_names: set[str] = set()

fig, axes = plt.subplots(1, len(eval_datasets), figsize=(len(eval_datasets) * 4, 4))

# Create a subfigure for each dataset
for i, eval_dataset in enumerate(eval_datasets):
    split = "minidev"
    method = "cot"
    ax = axes[i]

    max_ckpt_number = 0

    # Load the predictions and metrics for a training checkpoint from each synthetic dataset
    for train_ckpts_dict in synthetic_train_ckpts:
        train_dataset_name = train_ckpts_dict["dataset_name"]

        for ckpt in train_ckpts_dict["ckpts"]:
            base_model_name = ckpt["model"]
            if base_model_name == "Qwen/Qwen2.5-1.5B-Instruct":
                # Do not plot Qwen2.5-1.5B-Instruct because it starts at 0, so the plot will be weird
                continue
            base_model_names.add(base_model_name)
            ckpt_path = ckpt["paths"][0]  # NOTE: Take the first model path only
            output_dir = os.path.join(ckpt_path, f"out-{eval_dataset}")

            # Get predictions and metrics, and assign checkpoint number
            df_preds, metrics = get_preds(output_dir, args.data_dir, eval_dataset, split, method)
            df_preds["checkpoint_number"] = df_preds["_model"].str.extract(r"checkpoint-(\d+)")
            # base_model_name will not have a checkpoint number extracted, so manually
            # assign checkpoint number 0 to the base model, then convert to int
            df_preds.loc[df_preds["_model"] == base_model_name, "checkpoint_number"] = 0
            # ckpt_path will not have a checkpoint number, so manually assign from trainer_state.json
            with open(os.path.join(ckpt_path, "trainer_state.json")) as f:
                trainer_state = json.load(f)
                global_step = int(trainer_state["global_step"])
                df_preds.loc[df_preds["_model"] == ckpt_path, "checkpoint_number"] = global_step
            df_preds["checkpoint_number"] = df_preds["checkpoint_number"].astype(int)

            df_preds["completion_tokens"] = df_preds["usage"].apply(lambda x: x["completion_tokens"])

            # Define aggregation functions
            agg_dict = {
                **{metric: "mean" for metric in metrics},
                "completion_tokens": [
                    "mean",
                    lambda x: x.quantile(0.5),
                ],
            }

            acc = df_preds.groupby(["_model", "_split", "_seed", "checkpoint_number"]).agg(agg_dict)

            # Flatten column names
            acc.columns = metrics + [
                "completion_tokens_mean",
                "completion_tokens_median",
            ]

            # print(tabulate(acc, headers="keys", tablefmt="github"))
            # save to csv
            # scores_dir = os.path.join(output_dir, "scores")
            # os.makedirs(scores_dir, exist_ok=True)
            # scores_path = os.path.join(scores_dir, f"{eval_dataset}_{split}_{method}.csv")
            # acc.to_csv(scores_path)
            # print(f"Saved scores to {scores_path}")

            # Plot a line chart of metric vs checkpoint number
            # acc is a multi-index dataframe, with index as (_model, _split, _seed, checkpoint_number)
            metric = eval_dataset2plot_metric[eval_dataset]
            metric_data = acc[metric].sort_index(level="checkpoint_number")
            marker = plotting_utils.MODEL_NAME2MARKER[base_model_name]  # based on model name
            color = plotting_utils.COLORS2HEX[
                plotting_utils.TRAIN_DATASET_ALIAS2COLOR[train_dataset_name]
            ]  # based on pw or gsminf
            ax.plot(
                metric_data.index.get_level_values("checkpoint_number"),
                metric_data,
                marker=marker,
                color=color,
                linestyle="solid",
                linewidth=plotting_utils.LINE_WIDTH,
            )
            max_ckpt_number = max(
                max_ckpt_number, max(metric_data.index.get_level_values("checkpoint_number"))
            )

    ax.set_xlabel("Training steps", fontsize=plotting_utils.LABEL_FONT_SIZE)
    if i == 0:
        ax.set_ylabel(metric.upper(), fontsize=plotting_utils.LABEL_FONT_SIZE)
    ax.set_title(eval_dataset2name[eval_dataset], fontsize=plotting_utils.LABEL_FONT_SIZE, fontweight="bold")
    ax.set_xlim(1, max_ckpt_number)
    ax.set_ylim(eval_dataset2ylims[eval_dataset])
    ax.set_yticks(eval_dataset2yticks[eval_dataset])
    # Add horizontal grid lines
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=plotting_utils.LINE_WIDTH)
    ax.set_axisbelow(True)

handles = []
for training_dataset_name in training_dataset_names:
    for base_model_name in base_model_names:
        label = (
            f"{plotting_utils.MODEL_NAME2ALIAS[base_model_name]} x "
            f"{plotting_utils.TRAIN_DATASET_ALIAS2NAME[training_dataset_name]}"
        )
        handles.append(
            lines.Line2D(
                [0],
                [0],
                color=plotting_utils.COLORS2HEX[
                    plotting_utils.TRAIN_DATASET_ALIAS2COLOR[training_dataset_name]
                ],
                marker=plotting_utils.MODEL_NAME2MARKER[base_model_name],
                label=label,
                linewidth=plotting_utils.LINE_WIDTH,
            )
        )
fig.legend(
    handles=handles,
    fontsize=plotting_utils.LEGEND_FONT_SIZE,
    loc="lower center",
    ncol=len(training_dataset_names),
    frameon=False,
    bbox_to_anchor=(0.5, -0.15),
    bbox_transform=fig.transFigure,
)
# train_dataset_handles = [
#     lines.Line2D(
#         [0],
#         [0],
#         color=plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR[training_dataset_name]],
#         label=plotting_utils.TRAIN_DATASET_ALIAS2NAME[training_dataset_name],
#     ) for training_dataset_name in training_dataset_names
# ]
# dataset_legend = fig.legend(
#     handles=train_dataset_handles,
#     fontsize=plotting_utils.LEGEND_FONT_SIZE,
#     loc="lower center",
#     ncol=len(training_dataset_names),
#     frameon=False,
#     bbox_to_anchor=(0.5, -0.1),
#     bbox_transform=fig.transFigure,
# )
# fig.add_artist(dataset_legend)
# model_handles = [
#     lines.Line2D(
#         [0],
#         [0],
#         color='black',
#         marker=plotting_utils.MODEL_NAME2MARKER[base_model_name],
#         label=plotting_utils.MODEL_NAME2ALIAS[base_model_name],
#     ) for base_model_name in base_model_names
# ]
# model_legend = fig.legend(
#     handles=model_handles,
#     fontsize=plotting_utils.LEGEND_FONT_SIZE,
#     loc="lower center",
#     ncol=len(base_model_names),
#     frameon=False,
#     bbox_to_anchor=(0.5, -0.3),
#     bbox_transform=fig.transFigure,
# )
plt.tight_layout()

save_path = os.path.join("f1_v_training_steps.pdf")
# os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, bbox_inches="tight", dpi=300)
print(f"Saved scaling plot to {save_path}")

plt.close()
# # Plot a line chart of metric vs checkpoint number
# for metric in metrics + ["completion_tokens_median"]:
#     plt.figure()
#     metric_data = acc[metric].sort_index(level="checkpoint_number")
#     plt.plot(metric_data.index.get_level_values("checkpoint_number"), metric_data, label=metric, marker="o")
#     plt.title(f"Train {base_model_name} on {training_dataset_name} -> eval on {dataset}:{split}")
#     plt.ylabel(metric)
#     plt.xlabel("Training steps")

#     save_path = os.path.join(
#         output_dir, "figures", f"train={training_dataset_name}__eval={dataset}_{split}-{metric}.pdf"
#     )
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path)
#     print(f"Saved scaling plot to {save_path}")

#     plt.close()
