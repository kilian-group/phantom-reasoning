"""Evaluation utils for COFCA."""

import json
from pathlib import Path

import pandas as pd
from phantom_eval.evaluate_utils import _get_preds

from .hotpot_evaluate_v1 import exact_match_score, f1_score


def get_preds(output_dir: str, data_dir: str, dataset: str, split: str, method: str) -> pd.DataFrame:
    """
    Get predictions for a given output directory and method.
    """
    # Load gold data
    gold_path = Path(data_dir) / dataset / f"{split}.json"
    print(f"Loading answers from {gold_path}...")
    with open(gold_path) as f:
        gold = json.load(f)

    df_gold = pd.DataFrame(gold)

    print(f"Loading predictions from {output_dir}...")
    df_preds = _get_preds(output_dir, method)
    df_preds = df_preds[df_preds["_dataset"] == dataset]

    df_preds = df_preds.merge(df_gold, left_on="id", right_on="_id", how="left")

    # Score the predictions
    def score_pred(row):
        em = exact_match_score(row["pred"], row["answer"])
        f1, prec, recall = f1_score(row["pred"], row["answer"])
        return {"em": em, "f1": f1, "prec": prec, "recall": recall}

    df_preds["scores"] = df_preds.apply(lambda row: score_pred(row), axis=1)

    # Explode the score into separate columns
    df1 = df_preds.drop("scores", axis=1).reset_index(drop=True)
    df2 = pd.json_normalize(df_preds["scores"]).reset_index(drop=True)
    df_preds = pd.concat([df1, df2], axis=1)

    return df_preds
