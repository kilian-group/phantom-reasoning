import pandas as pd

# NOTE: We don't use CofCA or SynthWorlds-RM for training, so the evaluation code is in utils.evaluate_utils
# instead of phantom_reasoner.utils
from utils.evaluate_utils.cofca import get_preds as get_preds_cofca
from utils.evaluate_utils.synthrm import get_preds as get_preds_synthrm

from phantom_reasoner.utils.hp import get_preds as get_preds_hp
from phantom_reasoner.utils.msq.evaluate_utils import get_preds as get_preds_msq
from phantom_reasoner.utils.twowiki.evaluate_2wiki import get_preds as get_preds_2wiki


def get_preds(output_dir, data_dir, dataset, split, method) -> tuple[pd.DataFrame, list[str]]:
    """Get the predictions for the different datasets.

    Args:
        output_dir (str): The output directory.
        data_dir (str): The data directory.
        dataset (str): The dataset name.
        split (str): The split name.
        method (str): The method name.

    Returns:
        tuple[pd.DataFrame, list[str]]: The predictions and the metrics.
    """
    match dataset:
        case "hp" | "hp500":
            df_preds = get_preds_hp(
                output_dir,
                data_dir,
                dataset,
                split,
                "distractor",
                method,
            )
            metrics = ["em", "f1", "prec", "recall"]
        case "2wiki" | "2wiki500":
            df_preds = get_preds_2wiki(
                output_dir,
                data_dir,
                dataset,
                split,
                method,
            )
            metrics = ["em", "f1", "prec", "recall"]
        case "msq" | "msq500":
            df_preds = get_preds_msq(
                output_dir,
                data_dir,
                dataset,
                split,
                False,
                method,
            )
            metrics = ["em", "f1"]
        case "cofca500":
            df_preds = get_preds_cofca(
                output_dir,
                data_dir,
                dataset,
                split,
                method,
            )
            metrics = ["em", "f1", "prec", "recall"]
        case "synthrm500":
            df_preds = get_preds_synthrm(
                output_dir,
                data_dir,
                dataset,
                split,
                method,
            )
            metrics = ["em", "f1"]
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")
    return df_preds, metrics
