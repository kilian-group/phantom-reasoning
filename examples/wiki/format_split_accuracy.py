"""Script that formats the split accuracy.

Example usage:
```bash
python format_split_accuracy.py -dd DATA_DIR -od OUTPUT_DIR --split SPLIT
```
By default, DATA_DIR is `data/`.
"""
import json
import os

from data_utils import get_hp_data_path, get_parser
from phantom_eval.evaluate_utils import _get_preds

parser = get_parser()
args = parser.parse_args()
output_dir = args.output_dir
method = args.method
dataset = args.dataset
split = args.split

df_preds = _get_preds(output_dir, method)
df_preds = df_preds[df_preds["_dataset"] == dataset]

match dataset:
    case "hp":
        from evaluate_utils.hp import eval

        prediction = {
            "answer": {},
            "sp": {},
        }
        for row in df_preds.itertuples():
            qid = row.id
            prediction["answer"][qid] = row.pred
            prediction["sp"][qid] = []

        # save preds to a temporary json file
        gold_path = get_hp_data_path(os.path.join(args.data_dir, "hotpotqa"), split, "distractor")
        print(f"Loading answers from {gold_path}...")
        with open(gold_path) as f:
            gold = json.load(f)

        # filter gold to only include rows with no error
        gold_filtered = []
        for dp in gold:
            if dp["_id"] in prediction["answer"]:
                gold_filtered.append(dp)

        eval(prediction, gold_filtered)
    case "2wiki":
        from evaluate_utils.evaluate_2wiki import eval

        # convert to a dict with id as key and pred as value
        prediction = {
            "answer": {},
            "sp": {},
            "evidence": {},
        }
        for row in df_preds.itertuples():
            qid = row.id
            prediction["answer"][qid] = row.pred
            prediction["sp"][qid] = []
            prediction["evidence"][qid] = {}

        # save preds to a temporary json file
        gold_path = os.path.join(args.data_dir, "2wikimultihopqa", f"{split}.json")
        print(f"Loading answers from {gold_path}...")
        with open(gold_path) as f:
            gold = json.load(f)

        # filter gold to only include rows with no error
        gold_filtered = []
        for dp in gold:
            if dp["_id"] in prediction["answer"]:
                gold_filtered.append(dp)

        alias_path = os.path.join(args.data_dir, "2wikimultihopqa", "id_aliases.json")
        eval(prediction, gold_filtered, alias_path)
    case "msq":
        from evaluate_utils.msq import evaluate, read_jsonl

        # convert to a list of dicts
        prediction_instances = []
        prediction_ids = []
        for row in df_preds.itertuples():
            prediction_instances.append(
                {
                    "id": row.id,
                    "predicted_answer": row.pred,
                    "predicted_support_idxs": [],
                    "predicted_answerable": 0,  # 0 = unanswerable, 1 = answerable
                }
            )
            prediction_ids.append(row.id)

        # save preds to a temporary json file
        gold_path = os.path.join(args.data_dir, "musique", f"musique_ans_v1.0_{split}.jsonl")
        print(f"Loading answers from {gold_path}...")
        ground_truth_instances = read_jsonl(gold_path)

        filtered_ground_truth_instances = []
        for instance in ground_truth_instances:
            if instance["id"] in prediction_ids:
                filtered_ground_truth_instances.append(instance)

        print(json.dumps(evaluate(prediction_instances, filtered_ground_truth_instances), indent=4))
    case _:
        raise ValueError(f"Invalid dataset: {args.dataset}")
