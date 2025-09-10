"""Data utils for 2WikiMultiHopQA."""

import json
import logging
import os
from pathlib import Path

from phantom_reasoner.utils.twowiki.evaluate_2wiki import update_gold_with_aliases

logger = logging.getLogger(__name__)


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

    # Load aliases
    alias_path = os.path.join(data_path, "id_aliases.json")
    with open(alias_path) as f:
        aliases = {}
        for json_line in map(json.loads, f):
            aliases[json_line["Q_id"]] = {"aliases": set(json_line["aliases"] + json_line["demonyms"])}

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
                "answer": update_gold_with_aliases(group, aliases),
                "type": group["type"],
            }
        )

    # Log final statistics
    logger.info(f"Loaded {len(qa_pairs)} questions " f"and {len(text_corpus)} articles")

    return {
        "qa_pairs": qa_pairs,
        "text": text_corpus,
    }
