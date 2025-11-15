"""Utility functions for HotpotQA, 2WikiMultiHopQA, and MuSiQue."""

import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path

from phantom_eval import get_parser as get_base_parser

logger = logging.getLogger(__name__)


def load_data(data_dir: str, dataset: str, split: str, **kwargs) -> dict:
    """Load data from disk.

    Args:
        data_dir: Path to the dataset directory
        dataset: Name of the dataset
        split: Dataset split (e.g., 'dev', 'train')
        **kwargs: Additional arguments passed to dataset-specific loaders (for MuSiQue: intermediate_answers)

    Returns:
        dict: Dictionary containing:
            - qa_pairs: List of QA pairs with metadata
            - text: List of context paragraphs with metadata
    """
    match dataset:
        case "hp" | "hp500":
            return load_hp_data(os.path.join(data_dir, dataset), split, setting="distractor", **kwargs)
        case "2wiki" | "2wiki500":
            return load_2wiki_data(os.path.join(data_dir, dataset), split, **kwargs)
        case "msq" | "msq500":
            return load_msq_data(os.path.join(data_dir, dataset), split, answerable_only=True, **kwargs)
        case "cofca" | "cofca500":
            return load_cofca_data(os.path.join(data_dir, dataset), split, **kwargs)
        case "synthrm" | "synthrm500":
            return load_synthrm_data(os.path.join(data_dir, dataset), split, **kwargs)
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")


# ------------------------------------------------------------------------------------------------
# HotpotQA
# ------------------------------------------------------------------------------------------------
def get_hp_data_path(data_path: str, split: str, setting: str) -> str:
    """Get the path to the HotpotQA dataset

    Args:
        data_path: Path to the dataset directory
        split: Dataset split ('dev', 'train', 'minitrain', 'minidev')
        setting: Either 'distractor' or 'fullwiki'

    Returns:
        str: Path to the dataset file
    """
    if split in ["train", "minitrain"]:
        return Path(data_path) / f"hotpot_{split}_v1.1.json"
    else:
        return Path(data_path) / f"hotpot_{split}_{setting}_v1.json"


def load_hp_data(data_path: str, split: str, setting: str) -> dict:
    """Load HotpotQA dataset from disk.

    Args:
        data_path: Path to the dataset directory
        split: Dataset split ('dev', 'train', 'minitrain', 'minidev')
        setting: Either 'distractor' or 'fullwiki'

    Returns:
        dict: Dictionary containing:
            - qa_pairs: List of QA pairs with metadata
            - text: List of context paragraphs with metadata
    """
    file_path = get_hp_data_path(data_path, split, setting)
    logger.info(f"Loading HotpotQA dataset from {file_path}")

    with open(file_path) as f:
        data = json.load(f)

    # Convert to format similar to phantom-wiki
    qa_pairs = []
    text_corpus = []

    articles = {}
    for group in data:
        # Process articles
        titles, text = [], []
        for title, article in group["context"]:
            if title in articles:
                if articles[title] != article:
                    logger.warning(f"Article with {title=} already exists with different content")
            else:
                articles[title] = article
            titles.append(title)
            text.append("\n".join(article))
        text_corpus.append(
            {
                "title": titles,
                # NOTE: article is a list of sentences , which we convert to a single string
                "article": text,
                "id": group["_id"],
            }
        )

        qa_pairs.append(
            {
                "id": group["_id"],
                "question": group["question"],
                "answer": group["answer"],
                "type": group["type"],
            }
        )

    # Log final statistics
    logger.info(f"Loaded {len(qa_pairs)} questions " f"and {len(text_corpus)} articles")

    return {
        "qa_pairs": qa_pairs,
        "text": text_corpus,
    }


# ------------------------------------------------------------------------------------------------
# 2WikiMultiHopQA
# ------------------------------------------------------------------------------------------------
def load_2wiki_data(data_path: str, split: str) -> dict:
    """Load 2WikiMultiHopQA dataset from disk.

    Args:
        data_path: Path to the dataset directory
        split: Dataset split (e.g., 'train', 'dev')

    Returns:
        dict: Dictionary containing:
            - qa_pairs: List of QA pairs with metadata
            - text: List of context paragraphs with metadata
    """

    file_path = Path(data_path) / f"{split}.json"
    logger.info(f"Loading 2WikiMultiHopQA dataset from {file_path}")

    with open(file_path) as f:
        data = json.load(f)

    # Convert to format similar to phantom-wiki
    qa_pairs = []
    text_corpus = []

    articles = {}
    for group in data:
        # Process articles
        titles, text = [], []
        for title, article in group["context"]:
            if title in articles:
                assert articles[title] == article, "Article with same title has different content"
            else:
                articles[title] = article
            titles.append(title)
            text.append("\n".join(article))
        text_corpus.append(
            {
                "title": titles,
                # NOTE: article is a list of sentences , which we convert to a single string
                "article": text,
                "id": group["_id"],
            }
        )

        qa_pairs.append(
            {
                "id": group["_id"],
                "question": group["question"],
                "answer": group["answer"],
                "type": group["type"],
            }
        )

    # Log final statistics
    logger.info(f"Loaded {len(qa_pairs)} questions " f"and {len(text_corpus)} articles")

    return {
        "qa_pairs": qa_pairs,
        "text": text_corpus,
    }


# ------------------------------------------------------------------------------------------------
# COFCA
# ------------------------------------------------------------------------------------------------
def load_cofca_data(data_path: str, split: str) -> dict:
    """Load COFCA dataset from disk.

    Args:
        data_path: Path to the dataset directory
        split: Dataset split (e.g., 'train', 'dev', 'minidev')

    Returns:
        dict: Dictionary containing:
            - qa_pairs: List of QA pairs with metadata
            - text: List of context paragraphs with metadata
    """
    file_path = Path(data_path) / f"{split}.json"
    logger.info(f"Loading COFCA dataset from {file_path}")

    with open(file_path) as f:
        data = json.load(f)

    # Convert to format similar to phantom-wiki
    qa_pairs = []
    text_corpus = []

    all_ids = []
    for group in data:
        if group["_id"] in all_ids:
            logger.warning(f"Article with {group['_id']=} already exists")
        else:
            all_ids.append(group["_id"])
        # Process articles, create a paragraph for each article
        titles, text = [], []
        for title, article in group["context"]:
            titles.append(title)
            text.append("\n".join(article))

        text_corpus.append(
            {
                "title": titles,
                # NOTE: article is a list of sentences, which we convert to a single string
                "article": text,
                "id": group["_id"],
            }
        )

        qa_pairs.append(
            {
                "id": group["_id"],
                "question": group["question"],
                "answer": group["answer"],
                "type": group["type"],
            }
        )

    # Log final statistics
    logger.info(f"Loaded {len(qa_pairs)} questions " f"and {len(text_corpus)} articles")

    return {
        "qa_pairs": qa_pairs,
        "text": text_corpus,
    }


# ------------------------------------------------------------------------------------------------
# SynthWorlds-RM
# ------------------------------------------------------------------------------------------------
def load_synthrm_data(data_path: str, split: str) -> dict:
    """Load SynthWorlds-RM dataset from disk.

    Args:
        data_path: Path to the dataset directory
        split: Dataset split (e.g., 'train', 'dev', 'minidev')

    Returns:
        dict: Dictionary containing:
            - qa_pairs: List of QA pairs with metadata
            - text: List of context paragraphs with metadata
    """
    file_path = Path(data_path) / f"{split}.json"
    logger.info(f"Loading SynthWorlds-RM dataset from {file_path}")

    with open(file_path) as f:
        data = json.load(f)

    # Convert to format similar to phantom-wiki
    qa_pairs = []
    text_corpus = []

    all_ids = []
    for group in data:
        if group["instance_id"] in all_ids:
            logger.warning(f"Article with {group['instance_id']=} already exists")
        else:
            all_ids.append(group["instance_id"])

        # Process articles - gold_docs is a list of article texts
        # Use first 20 characters of each article as the title
        titles = [doc[:20] for doc in group["gold_docs"]]
        articles = group["gold_docs"]

        text_corpus.append(
            {
                "title": titles,
                "article": articles,
                "id": group["instance_id"],
            }
        )

        # gold_answers is a list, take the first element
        if len(group["gold_answers"]) > 1:
            logger.warning(
                f"Question {group['instance_id']} has {len(group['gold_answers'])} answers. "
                f"Taking first answer. (SynthWorlds-RM should only have 1 answer per question)"
            )
        answer = group["gold_answers"][0] if group["gold_answers"] else ""

        qa_pairs.append(
            {
                "id": group["instance_id"],
                "question": group["query"],
                "answer": answer,
                "type": group["question_graph_type"],
            }
        )

    # Log final statistics
    logger.info(f"Loaded {len(qa_pairs)} questions and {len(text_corpus)} articles")

    return {
        "qa_pairs": qa_pairs,
        "text": text_corpus,
    }


# ------------------------------------------------------------------------------------------------
# MuSiQue
# ------------------------------------------------------------------------------------------------
def load_msq_data(
    data_path: str, split: str, answerable_only: bool = True, intermediate_answers: bool = False
) -> dict:
    """Load MuSiQue dataset from disk.

    Args:
        data_path: Path to the dataset directory
        split: Dataset split (e.g., 'train', 'dev')
        answerable_only: if True, use MuSiQue-Ans, otherwise use MuSiQue-Full
        intermediate_answers: if True, preserve question_decomposition field and return dict keyed by id
        # TODO: returning dict by id is not ideal, we should return a list of QA pairs

    Returns:
        dict: Dictionary containing:
            - qa_pairs: List or Dict (if intermediate_answers=True) of QA pairs with metadata
            - text: List of context paragraphs with metadata
    """
    if answerable_only:
        file_path = Path(data_path) / f"musique_ans_v1.0_{split}.jsonl"
    else:
        file_path = Path(data_path) / f"musique_full_v1.0_{split}.jsonl"
    logger.info(f"Loading MuSiQue dataset from {file_path}")

    # Convert to format similar to phantom-wiki
    qa_pairs = {} if intermediate_answers else []
    text_corpus = []

    with open(file_path) as f:
        for line in f:
            group = json.loads(line)
            # Process articles
            titles, text = [], []
            for paragraph in group["paragraphs"]:
                titles.append(paragraph["title"])
                text.append(paragraph["paragraph_text"])
            text_corpus.append(
                {
                    "title": titles,
                    "article": text,
                    "id": group["id"],
                }
            )

            if intermediate_answers:
                # Keep all fields from original data (including question_decomposition)
                qa_data = group
                qa_pairs[group["id"]] = qa_data
            else:
                # Extract only specific fields
                qa_data = {
                    "id": group["id"],
                    "question": group["question"],
                    "answer": group["answer"],
                    "type": None,  # TODO: add type
                }
                qa_pairs.append(qa_data)

    # Log final statistics
    logger.info(f"Loaded {len(qa_pairs)} questions " f"and {len(text_corpus)} articles")

    return {
        "qa_pairs": qa_pairs,
        "text": text_corpus,
    }


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
        choices=["hp", "hp500", "2wiki", "2wiki500", "msq", "msq500", "cofca500", "synthrm500"],
    )
    return parser
