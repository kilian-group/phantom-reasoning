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
from utils.evaluate_utils import get_preds

parser = get_parser()
args = parser.parse_args()
output_dir = args.output_dir
data_dir = args.data_dir
method = args.method
dataset = args.dataset
split = args.split

df_preds, metrics = get_preds(output_dir, data_dir, dataset, split, method)
df_preds["completion_tokens"] = df_preds["usage"].apply(lambda x: x["completion_tokens"])

# Define aggregation functions
# Show the mean +- sem
metrics = ["f1"]
agg_dict = {
    **{metric: ["mean", "sem"] for metric in metrics},
}

# Want to group by ignoring the MMDD__curr=random__training_seed=... suffix  pattern
df_preds["_model"] = df_preds["_model"].str.replace(r"\d+__curr=random__training_seed=\d+", "", regex=True)
acc = df_preds.groupby(["_model"]).agg(agg_dict)

print(tabulate(acc, headers="keys", tablefmt="github"))
# save to csv
scores_dir = os.path.join(output_dir, "scores")
os.makedirs(scores_dir, exist_ok=True)
scores_path = os.path.join(scores_dir, f"{dataset}_{split}_{method}.csv")
acc.to_csv(scores_path)
print(f"Saved scores to {scores_path}")
