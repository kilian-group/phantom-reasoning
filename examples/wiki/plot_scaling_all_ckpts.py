"""Script that formats the split accuracy.

The output is a markdown table with columns corresponding to the metrics and
rows corresponding to the model, split, and seed.

Example usage:
```bash
python plot_scaling_all_ckpts.py \
    -dd DATA_DIR \
    -od OUTPUT_DIR \
    --split SPLIT \
    --dataset DATASET \
    --method METHOD \
    --base_model_name BASE_MODEL_NAME \
    --training_dataset_name TRAINING_DATASET_NAME
```
By default, DATA_DIR is `/share/nikola/phantom-reasoning/data`.
"""

import json
import os

import matplotlib.pyplot as plt
from tabulate import tabulate
from utils.data_utils import get_parser
from utils.evaluate_utils import get_preds

parser = get_parser()
parser.add_argument("--base_model_name", type=str, required=True)
parser.add_argument("--training_dataset_name", type=str, required=True)
parser.add_argument("--model_list", nargs="+", required=True)
args = parser.parse_args()
output_dir = args.output_dir
data_dir = args.data_dir
method = args.method
dataset = args.dataset
split = args.split
base_model_name = args.base_model_name
training_dataset_name = args.training_dataset_name
model_list = args.model_list

df_preds, metrics = get_preds(output_dir, data_dir, dataset, split, method)
df_preds["checkpoint_number"] = df_preds["_model"].str.extract(r"checkpoint-(\d+)")
# base_model_name will not have a checkpoint number extracted, so manually
# assign checkpoint number 0 to the base model, then convert to int
df_preds.loc[df_preds["_model"] == base_model_name, "checkpoint_number"] = 0
# ckpt_path will not have a checkpoint number, so manually assign from trainer_state.json
with open(os.path.join(model_list[0], "trainer_state.json")) as f:
    trainer_state = json.load(f)
    global_step = int(trainer_state["global_step"])
    df_preds.loc[df_preds["_model"] == model_list[0], "checkpoint_number"] = global_step
df_preds["checkpoint_number"] = df_preds["checkpoint_number"].astype(int)

df_preds["completion_tokens"] = df_preds["usage"].apply(lambda x: x["completion_tokens"])

# Define aggregation functions
agg_dict = {
    **{metric: "mean" for metric in metrics},
    "completion_tokens": [
        "mean",
        lambda x: x.quantile(0.5),
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        lambda x: x.quantile(0.95),
        lambda x: x.quantile(0.99),
    ],
}

acc = df_preds.groupby(["_model", "_split", "_seed", "checkpoint_number"]).agg(agg_dict)

# Flatten column names
acc.columns = metrics + [
    "completion_tokens_mean",
    "completion_tokens_median",
    "completion_tokens_75",
    "completion_tokens_90",
    "completion_tokens_95",
    "completion_tokens_99",
]

print(tabulate(acc, headers="keys", tablefmt="github"))
# save to csv
scores_dir = os.path.join(output_dir, "scores")
os.makedirs(scores_dir, exist_ok=True)
scores_path = os.path.join(scores_dir, f"{dataset}_{split}_{method}.csv")
acc.to_csv(scores_path)
print(f"Saved scores to {scores_path}")

# Plot a line chart of metric vs checkpoint number
# acc is a multi-index dataframe, with index as (_model, _split, _seed, checkpoint_number)

# Plot a line chart of metric vs checkpoint number
for metric in metrics + ["completion_tokens_median"]:
    plt.figure()
    metric_data = acc[metric].sort_index(level="checkpoint_number")
    plt.plot(metric_data.index.get_level_values("checkpoint_number"), metric_data, label=metric, marker="o")
    plt.title(f"Train {base_model_name} on {training_dataset_name} -> eval on {dataset}:{split}")
    plt.ylabel(metric)
    plt.xlabel("Training steps")

    save_path = os.path.join(
        output_dir, "figures", f"train={training_dataset_name}__eval={dataset}_{split}-{metric}.pdf"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved scaling plot to {save_path}")

    plt.close()
