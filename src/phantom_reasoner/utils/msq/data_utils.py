"""Data utils for MuSiQue."""

import json
import logging
from pathlib import Path

from phantom_reasoner.utils.msq.evaluate_utils import update_gold_with_aliases

logger = logging.getLogger(__name__)


def load_msq_data(data_path: str, split: str, answerable_only: bool = True) -> dict:
    """Load MuSiQue dataset from disk.

    Args:
        data_path: Path to the dataset directory
        split: Dataset split (e.g., 'train', 'dev')
        answerable_only: if True, use MuSiQue-Ans, otherwise use MuSiQue-Full

    Returns:
        dict: Dictionary containing:
            - qa_pairs: List of QA pairs with metadata
            - text: List of context paragraphs with metadata
    """
    if answerable_only:
        file_path = Path(data_path) / f"musique_ans_v1.0_{split}.jsonl"
    else:
        file_path = Path(data_path) / f"musique_full_v1.0_{split}.jsonl"
    logger.info(f"Loading MuSiQue dataset from {file_path}")

    # Convert to format similar to phantom-wiki
    qa_pairs = []
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

            qa_pairs.append(
                {
                    "id": group["id"],
                    "question": group["question"],
                    "answer": update_gold_with_aliases(group),
                    "type": None,  # TODO: add type
                }
            )

    # Log final statistics
    logger.info(f"Loaded {len(qa_pairs)} questions " f"and {len(text_corpus)} articles")

    return {
        "qa_pairs": qa_pairs,
        "text": text_corpus,
    }
