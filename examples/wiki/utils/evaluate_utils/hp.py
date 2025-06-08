"""Evaluation utils for HotpotQA."""

import json

import pandas as pd
from phantom_eval.evaluate_utils import _get_preds

from ..data_utils import get_hp_data_path
from .hotpot_evaluate_v1 import exact_match_score, f1_score, update_answer, update_sp


def eval(prediction, gold):
    metrics = {
        "em": 0,
        "f1": 0,
        "prec": 0,
        "recall": 0,
        "sp_em": 0,
        "sp_f1": 0,
        "sp_prec": 0,
        "sp_recall": 0,
        "joint_em": 0,
        "joint_f1": 0,
        "joint_prec": 0,
        "joint_recall": 0,
    }
    for dp in gold:
        cur_id = dp["_id"]
        can_eval_joint = True
        if cur_id not in prediction["answer"]:
            print(f"missing answer {cur_id}")
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(metrics, prediction["answer"][cur_id], dp["answer"])
        if cur_id not in prediction["sp"]:
            print(f"missing sp fact {cur_id}")
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(metrics, prediction["sp"][cur_id], dp["supporting_facts"])

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.0
            joint_em = em * sp_em

            metrics["joint_em"] += joint_em
            metrics["joint_f1"] += joint_f1
            metrics["joint_prec"] += joint_prec
            metrics["joint_recall"] += joint_recall

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    return metrics


def get_preds(output_dir: str, data_dir: str, split: str, setting: str, method: str) -> pd.DataFrame:
    """
    Get predictions for a given output directory and method.
    """
    # save preds to a temporary json file
    gold_path = get_hp_data_path(data_dir, split, setting)
    print(f"Loading answers from {gold_path}...")
    with open(gold_path) as f:
        gold = json.load(f)

    df_gold = pd.DataFrame(gold)

    print(f"Loading predictions from {output_dir}...")
    df_preds = _get_preds(output_dir, method)
    df_preds = df_preds[df_preds["_dataset"] == "hp"]

    df_preds = df_preds.merge(df_gold, left_on="id", right_on="_id", how="left")

    # score the preds
    def score_pred(row):
        em = exact_match_score(row["pred"], row["answer"])
        f1, prec, recall = f1_score(row["pred"], row["answer"])
        return {"em": em, "f1": f1, "prec": prec, "recall": recall}

    df_preds["scores"] = df_preds.apply(lambda row: score_pred(row), axis=1)

    # explode the score into separate columns
    df1 = df_preds.drop("scores", axis=1).reset_index(drop=True)
    df2 = pd.json_normalize(df_preds["scores"]).reset_index(drop=True)
    df_preds = pd.concat([df1, df2], axis=1)

    return df_preds
