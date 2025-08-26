"""Script to format the accuracy of the models on the splits.

Generates a table with rows for each model, split, and seed combination.
Saves to a csv file called scores.csv in the scores directory of the output directory.

Example:
    python eval/format_split_accuracy.py -od out --method zeroshot
"""

import logging
import os

from phantom_eval import get_parser
from phantom_eval.evaluate_utils import get_evaluation_data
from phantom_eval.utils import setup_logging

logger = logging.getLogger(__name__)

metrics = ["EM", "precision", "recall", "f1"]

parser = get_parser()
args = parser.parse_args()
output_dir = args.output_dir
method = args.method
dataset = args.dataset
from_local = args.from_local
log_level = args.log_level
setup_logging(log_level)
# get evaluation data from the specified output directory and method subdirectory
df = get_evaluation_data(output_dir, method, dataset, from_local=from_local)
if False:
    logger.warning(f"Filtering out {method} with more than 1 solution")
    df = df[df["solutions"] <= 1]

df["completion_tokens"] = df["usage"].apply(lambda x: x["completion_tokens"])
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
grouped = df.groupby(["_model", "_depth", "_size", "_data_seed", "_seed"])
acc = grouped.agg(agg_dict)
# Flatten column names
acc.columns = metrics + [
    "completion_tokens_mean",
    "completion_tokens_median",
    "completion_tokens_75",
    "completion_tokens_90",
    "completion_tokens_95",
    "completion_tokens_99",
]
# add a column that counts the number of elements in the group
acc["count"] = grouped.size()
# add a column at the end for the method
acc["method"] = method
# print as markdown
print(acc.to_markdown())

# save to a csv file
scores_dir = os.path.join(output_dir, "scores", method)
os.makedirs(scores_dir, exist_ok=True)
save_path = os.path.join(scores_dir, "scores.csv")
print(f"Saving scores to {save_path}")
acc.to_csv(save_path)
