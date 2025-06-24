import argparse
import json
import os
from collections import Counter

import pandas as pd
from phantom_eval.evaluate_utils import _get_preds

from .msq_metrics.answer import (
    AnswerMetric,
    compute_exact,
    compute_f1,
    metric_max_over_ground_truths,
)
from .msq_metrics.group_answer_sufficiency import GroupAnswerSufficiencyMetric
from .msq_metrics.group_support_sufficiency import GroupSupportSufficiencyMetric
from .msq_metrics.support import SupportMetric


def read_jsonl(file_path: str) -> list[dict]:
    with open(file_path) as file:
        instances = [json.loads(line.strip()) for line in file if line.strip()]
    return instances


def evaluate(prediction_instances: list[dict], ground_truth_instances: list[dict]) -> dict:
    # prediction_instances = read_jsonl(filepath_with_predictions)
    # ground_truth_instances = read_jsonl(filepath_with_ground_truths)

    do_sufficiency_eval = False
    answer_metric = AnswerMetric()
    support_metric = SupportMetric()
    group_answer_sufficiency_metric = GroupAnswerSufficiencyMetric()
    group_support_sufficiency_metric = GroupSupportSufficiencyMetric()

    assert len(prediction_instances) == len(
        ground_truth_instances
    ), "The number of lines in the two files are not the same."

    for ground_truth_instance, prediction_instance in zip(ground_truth_instances, prediction_instances):
        assert (
            ground_truth_instance["id"] == prediction_instance["id"]
        ), "The instances (ids) in prediction and gold filepath jsonl should be in same order."

        question_id = ground_truth_instance["id"]

        predicted_answer = prediction_instance["predicted_answer"]
        ground_truth_answers = [ground_truth_instance["answer"]] + ground_truth_instance["answer_aliases"]

        predicted_support_indices = prediction_instance["predicted_support_idxs"]
        ground_truth_support_indices = [
            paragraph["idx"]
            for paragraph in ground_truth_instance["paragraphs"]
            if paragraph["is_supporting"]
        ]

        predicted_sufficiency = prediction_instance["predicted_answerable"]
        ground_truth_sufficiency = ground_truth_instance["answerable"]

        if ground_truth_sufficiency:
            answer_metric(predicted_answer, ground_truth_answers)
            support_metric(predicted_support_indices, ground_truth_support_indices)

        group_answer_sufficiency_metric(
            predicted_answer,
            ground_truth_answers,
            predicted_sufficiency,
            ground_truth_sufficiency,
            question_id,
        )
        group_support_sufficiency_metric(
            predicted_support_indices,
            ground_truth_support_indices,
            predicted_sufficiency,
            ground_truth_sufficiency,
            question_id,
        )

        # If there's any instance with ground truth of unanswerable, we'll assume
        # it's full version of the dataset and not only the answerable version.
        if not ground_truth_sufficiency:
            do_sufficiency_eval = True

    metrics = {}
    metrics["answer_f1"] = round(answer_metric.get_metric()[1], 3)
    metrics["answer_em"] = round(answer_metric.get_metric()[0], 3)
    metrics["support_f1"] = round(support_metric.get_metric()[1], 3)

    if do_sufficiency_eval:
        assert set(Counter([e["id"] for e in prediction_instances]).values()) == {
            2
        }, "For sufficiency evaluation, there should two instances for each question."

        metrics["group_answer_sufficiency_f1"] = round(group_answer_sufficiency_metric.get_metric()["f1"], 3)
        metrics["group_support_sufficiency_f1"] = round(
            group_support_sufficiency_metric.get_metric()["f1"], 3
        )
    return metrics


def get_preds(
    output_dir: str, data_dir: str, dataset: str, split: str, use_musique_full: bool, method: str
) -> pd.DataFrame:
    """
    Get predictions for a given output directory and method.
    """
    suffix = "full" if use_musique_full else "ans"
    gold_path = os.path.join(data_dir, dataset, f"musique_{suffix}_v1.0_{split}.jsonl")
    print(f"Loading answers from {gold_path}...")
    ground_truth_instances = read_jsonl(gold_path)

    df_gold = pd.DataFrame(ground_truth_instances)

    def update_gold_with_aliases(row):
        return [row["answer"]] + row["answer_aliases"]

    df_gold["answer"] = df_gold.apply(update_gold_with_aliases, axis=1)

    df_preds = _get_preds(output_dir, method)
    df_preds = df_preds[df_preds["_dataset"] == dataset]

    df_preds = df_preds.merge(df_gold, on="id", how="left")

    # score the preds
    def score_pred(row):
        max_em = metric_max_over_ground_truths(compute_exact, row["pred"], row["answer"])
        max_f1 = metric_max_over_ground_truths(compute_f1, row["pred"], row["answer"])
        return {"em": int(max_em), "f1": max_f1}

    df_preds["scores"] = df_preds.apply(lambda row: score_pred(row), axis=1)

    # explode the score into separate columns
    df1 = df_preds.drop("scores", axis=1).reset_index(drop=True)
    df2 = pd.json_normalize(df_preds["scores"]).reset_index(drop=True)
    df_preds = pd.concat([df1, df2], axis=1)

    return df_preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate MuSiQue predictions.")
    parser.add_argument(
        "filepath_with_predictions",
        type=str,
        help="jsonl filepath to predicted instances.",
    )
    parser.add_argument(
        "filepath_with_ground_truths",
        type=str,
        help="jsonl filepath to data instances.",
    )
    parser.add_argument("--output_filepath", type=str, help="(optional) filepath to save output metrics.")
    args = parser.parse_args()

    metrics = evaluate(args.filepath_with_predictions, args.filepath_with_ground_truths)

    if args.output_filepath:
        print(f"Writing metrics output in: {args.output_filepath}")
        with open(args.output_filepath, "w") as file:
            json.dump(metrics, file, indent=4)
    else:
        print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
