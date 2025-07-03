"""
Dataset loader functions for different datasets used in training.

This module provides functions to load and format various datasets
to be compatible with the GRPO training pipeline.
"""

import logging
import os
import pickle
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


def get_openthoughts_dataset(skip_null_answers: bool = True, cache_dir: str = "cache") -> Dataset:
    """
    Load and format the OpenThoughts dataset to match the format expected by GRPO training.
    
    Args:
        skip_null_answers: If True, skip samples that don't have answers. If False, include all samples.
        cache_dir: Directory to store cached datasets. If None, no caching is used.
    
    Returns:
        Dataset with fields: prompt, answer, prompt_method, difficulty
    """
    # Create cache directory if it doesn't exist
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"openthoughts_skip_null_{skip_null_answers}.pkl")
        
        # Try to load from cache first
        if os.path.exists(cache_file):
            logger.info(f"Loading OpenThoughts dataset from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}. Rebuilding dataset...")
    
    logger.info("Loading OpenThoughts dataset from source...")
    ds = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
    
    def format_sample(sample: dict[str, Any]) -> dict[str, Any]:
        """Format a single sample from the OpenThoughts dataset."""
        conversations = sample["conversations"]
        
        # Convert conversations list to a single string
        conversation_text = ""
        for conv in conversations:
            if isinstance(conv, dict):
                # Handle dict format: {"from": "human", "value": "..."}
                role = conv.get("from", "unknown")
                value = conv.get("value", "")
                conversation_text += f"{role}: {value}\n"
            elif isinstance(conv, str):
                # Handle string format directly
                conversation_text += conv + "\n"
        
        # Extract the answer (everything between "**Final Answer**" and "</think>")
        answer = extract_text_between_markers(conversation_text, "**Final Answer**", "</think>")
        
        # Extract the prompt (everything between "human:" and "gpt:")
        prompt_text = extract_text_between_markers(conversation_text, "human:", "gpt:")
        
        # Extract the response (everything after "gpt:")
        gpt_idx = conversation_text.find("gpt:")
        if gpt_idx != -1:
            response = conversation_text[gpt_idx + 4:].strip()
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
            "has_answer": bool(answer),  # Flag to indicate if answer was found
        }
    
    # Apply formatting to all samples
    formatted_dataset = ds.map(format_sample)
    
    # Filter out samples without answers only if skip_null_answers is True
    if skip_null_answers:
        formatted_dataset = formatted_dataset.filter(lambda x: x["has_answer"], desc="Filtering samples without answers")
        # Remove the has_answer field since it's no longer needed
        formatted_dataset = formatted_dataset.remove_columns(["has_answer"])
    
    logger.info(f"Loaded OpenThoughts dataset with {len(formatted_dataset)} samples")
    
    # Save to cache if cache_dir is specified
    if cache_dir:
        logger.info(f"Saving OpenThoughts dataset to cache: {cache_file}")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(formatted_dataset, f)
            logger.info("Dataset cached successfully")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    return formatted_dataset 