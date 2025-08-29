"""Data utils for HotpotQA."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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
