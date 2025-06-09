"""Script that formats the split accuracy.

The output is a markdown table with columns corresponding to the metrics and
rows corresponding to the model, split, and seed.

Example usage:
```bash
python format_split_accuracy.py -dd DATA_DIR -od OUTPUT_DIR --split SPLIT --dataset DATASET --method METHOD
```
By default, DATA_DIR is `/share/nikola/phantom-reasoning/data`.
"""

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
        acc = df_preds.groupby(["_model", "_split", "_seed"])[["em", "f1", "prec", "recall"]].agg("mean")
    case "2wiki" | "2wiki500":
        from utils.evaluate_utils.evaluate_2wiki import get_preds

        df_preds = get_preds(
            output_dir,
            args.data_dir,
            dataset,
            split,
            method,
        )
        acc = df_preds.groupby(["_model", "_split", "_seed"])[["em", "f1", "prec", "recall"]].agg("mean")
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
        acc = df_preds.groupby(["_model", "_split", "_seed"])[["em", "f1"]].agg("mean")
    case _:
        raise ValueError(f"Invalid dataset: {args.dataset}")

print(tabulate(acc, headers="keys", tablefmt="github"))
