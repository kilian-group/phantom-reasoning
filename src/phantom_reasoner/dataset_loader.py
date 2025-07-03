"""
Dataset loader functions for different datasets used in training.

This module provides functions to load and format various datasets
to be compatible with the GRPO training pipeline.
"""

import logging
import re
from typing import Any

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


def extract_text_between_markers(text: str, start_marker: str, end_marker: str) -> str:
    """
    Extract text between two markers in a string.
    
    Args:
        text: The text to search in
        start_marker: The starting marker
        end_marker: The ending marker
        
    Returns:
        The extracted text, or empty string if markers not found
    """
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return ""
    
    start_idx += len(start_marker)
    end_idx = text.find(end_marker, start_idx)
    
    if end_idx == -1:
        return ""
    
    return text[start_idx:end_idx].strip()


def get_openthoughts_dataset(skip_null_answers: bool = True) -> Dataset:
    """
    Load and format the OpenThoughts dataset to match the format expected by GRPO training.
    
    Args:
        skip_null_answers: If True, skip samples that don't have answers. If False, include all samples.
    
    Returns:
        Dataset with fields: prompt, answer, prompt_method, difficulty
    """
    logger.info("Loading OpenThoughts dataset...")
    ds = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
    
    def format_sample(sample: dict[str, Any]) -> dict[str, Any] | None:
        """Format a single sample from the OpenThoughts dataset."""
        conversation = sample["conversations"]
        
        # Extract the answer (everything between "**Final Answer**" and "</think>")
        answer = extract_text_between_markers(conversation, "**Final Answer**", "</think>")
        
        # Skip samples where no answer is found (if skip_null_answers is True)
        if skip_null_answers and not answer:
            return None
        # https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M/discussions/3
        
        # Extract the prompt (everything between "human:" and "gpt:")
        prompt_text = extract_text_between_markers(conversation, "human:", "gpt:")
        
        # Extract the response (everything after "gpt:")
        gpt_idx = conversation.find("gpt:")
        if gpt_idx != -1:
            response = conversation[gpt_idx + 4:].strip()
        else:
            response = ""
        
        # Format as conversational messages
        prompt = [
            {
                "role": "user",
                "content": prompt_text,
            }
        ]
        
        return {
            "prompt": prompt,
            "answer": [answer],  # Wrap in list to match get_pw_dataset format
            "prompt_method": "zershot",  # placeholder
            "difficulty": sample["difficulty"],
            "response": response,
        }
    
    # Apply formatting to all samples
    formatted_dataset = ds.map(format_sample)
    
    # Filter out None values only if skip_null_answers is True
    if skip_null_answers:
        formatted_dataset = formatted_dataset.filter(lambda x: x is not None)
    
    logger.info(f"Loaded OpenThoughts dataset with {len(formatted_dataset)} samples")
    
    return formatted_dataset 