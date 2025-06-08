"""Script that formats the split accuracy.

Example usage:
```bash
python format_split_accuracy.py -dd DATA_DIR -od OUTPUT_DIR --split SPLIT
```
By default, DATA_DIR is `data/`.
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
    case "hp":
        from utils.evaluate_utils.hp import get_preds

        df_preds = get_preds(output_dir, os.path.join(args.data_dir, "hotpotqa"), split, "distractor", method)
        acc = df_preds.groupby(["_model", "_split", "_seed"])[["em", "f1", "prec", "recall"]].agg("mean")
    case "2wiki":
        from utils.evaluate_utils.evaluate_2wiki import get_preds

        df_preds = get_preds(output_dir, os.path.join(args.data_dir, "2wikimultihopqa"), split, method)
        acc = df_preds.groupby(["_model", "_split", "_seed"])[["em", "f1", "prec", "recall"]].agg("mean")
    case "msq":
        from utils.evaluate_utils.msq import get_preds

        df_preds = get_preds(output_dir, os.path.join(args.data_dir, "musique"), split, False, method)
        acc = df_preds.groupby(["_model", "_split", "_seed"])[["em", "f1"]].agg("mean")
    case _:
        raise ValueError(f"Invalid dataset: {args.dataset}")

print(tabulate(acc, headers="keys", tablefmt="github"))
