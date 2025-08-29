"""Utility functions for HotpotQA, 2WikiMultiHopQA, and MuSiQue."""

import logging
import os
from argparse import ArgumentParser

from phantom_eval import get_parser as get_base_parser

from phantom_reasoner.utils.hp.data_utils import load_hp_data
from phantom_reasoner.utils.msq.data_utils import load_msq_data
from phantom_reasoner.utils.twowiki.data_utils import load_2wiki_data

logger = logging.getLogger(__name__)


def load_data(data_dir: str, dataset: str, split: str) -> dict:
    """Load data from disk.

    Args:
        data_dir: Path to the dataset directory
        dataset: Name of the dataset
        split: Dataset split (e.g., 'dev', 'train')

    Returns:
        dict: Dictionary containing:
            - qa_pairs: List of QA pairs with metadata
            - text: List of context paragraphs with metadata
    """
    match dataset:
        case "hp" | "hp500":
            return load_hp_data(os.path.join(data_dir, dataset), split, setting="distractor")
        case "2wiki" | "2wiki500":
            return load_2wiki_data(os.path.join(data_dir, dataset), split)
        case "msq" | "msq500":
            return load_msq_data(os.path.join(data_dir, dataset), split, answerable_only=True)
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")


def get_parser():
    parser = ArgumentParser(parents=[get_base_parser()], conflict_handler="resolve")
    parser.add_argument(
        "--data_dir",
        "-dd",
        type=str,
        default="/share/nikola/phantom-reasoning/data",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="The split to evaluate on.",
        choices=["dev", "train", "minitrain", "minidev"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to evaluate on.",
        choices=["hp", "hp500", "2wiki", "2wiki500", "msq", "msq500"],
    )
    return parser
