"""Evaluation utils for SynthWorlds-RM. Uses the same evaluation code as MuSiQue."""

import json
from pathlib import Path

import pandas as pd
from phantom_eval.evaluate_utils import _get_preds

from phantom_reasoner.utils.msq.evaluate_utils import score_pred as score_pred_msq


def get_preds(output_dir: str, data_dir: str, dataset: str, split: str, method: str) -> pd.DataFrame:
    """
    Get predictions for a given output directory and method.
    """
    # Load gold data
    gold_path = Path(data_dir) / dataset / f"{split}.json"
    print(f"Loading answers from {gold_path}...")
    with open(gold_path) as f:
        gold = json.load(f)

    # Extract relevant fields, evaluation is done with MuSiQue code so the answer must be a list of strings
    # i.e. use item["gold_answers"] instead of item["gold_answers"][0]
    processed_gold = []
    for item in gold:
        processed_item = {
            "_id": item["instance_id"],
            "answer": item["gold_answers"],
            "question": item["query"],
            "type": item["question_graph_type"],
        }
        processed_gold.append(processed_item)

    df_gold = pd.DataFrame(processed_gold)

    print(f"Loading predictions from {output_dir}...")
    df_preds = _get_preds(output_dir, method)
    df_preds = df_preds[df_preds["_dataset"] == dataset]

    df_preds = df_preds.merge(df_gold, left_on="id", right_on="_id", how="left")

    df_preds["scores"] = df_preds.apply(lambda row: score_pred_msq(row), axis=1)

    # explode the score into separate columns
    df1 = df_preds.drop("scores", axis=1).reset_index(drop=True)
    df2 = pd.json_normalize(df_preds["scores"]).reset_index(drop=True)
    df_preds = pd.concat([df1, df2], axis=1)

    return df_preds
