"""Script that formats the split accuracy.

The output is a markdown table with columns corresponding to the metrics and
rows corresponding to the model, split, and seed.

Example usage:
```bash
python format_split_accuracy.py -dd DATA_DIR -od OUTPUT_DIR --split SPLIT --dataset DATASET --method METHOD
```
By default, DATA_DIR is `/share/nikola/phantom-reasoning/data`.
"""

import os

from tabulate import tabulate
from utils.data_utils import get_parser

parser = get_parser()
args = parser.parse_args()
output_dir = args.output_dir
method = args.method
dataset = args.dataset
split = args.split

match dataset:
    case "hp" | "hp500":
        from utils.evaluate_utils.hp import get_preds

        df_preds = get_preds(
            output_dir,
            args.data_dir,
            dataset,
            split,
            "distractor",
            method,
        )
        metrics = ["em", "f1", "prec", "recall"]
    case "2wiki" | "2wiki500":
        from utils.evaluate_utils.evaluate_2wiki import get_preds

        df_preds = get_preds(
            output_dir,
            args.data_dir,
            dataset,
            split,
            method,
        )
        metrics = ["em", "f1", "prec", "recall"]
    case "msq" | "msq500":
        from utils.evaluate_utils.msq import get_preds

        df_preds = get_preds(
            output_dir,
            args.data_dir,
            dataset,
            split,
            False,
            method,
        )
        metrics = ["em", "f1"]
    case _:
        raise ValueError(f"Invalid dataset: {args.dataset}")

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

acc = df_preds.groupby(["_model", "_split", "_seed"]).agg(agg_dict)

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
