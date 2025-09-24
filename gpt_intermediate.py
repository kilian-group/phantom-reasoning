#!/usr/bin/env python3
"""
GPT-based Intermediate Step Evaluator for Phantom Wiki Dataset

This script uses GPT to evaluate whether the model correctly solved intermediate steps
in question decomposition and reasoning. It takes a question, question_decomposition,
and ai_reasoning as input and validates the intermediate reasoning process.
"""

import json
import argparse
import openai
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Data class to store evaluation results for a single example."""
    id: str
    question: str
    question_decomposition: List[Dict[str, Any]]
    ai_reasoning: str
    intermediate_step_evaluations: List[Dict[str, Any]]


class GPTIntermediateEvaluator:
    """GPT-based evaluator for intermediate reasoning steps."""
    
    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = 0.1):
        """
        Initialize the GPT evaluator.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use (default: gpt-4)
            temperature: Temperature for generation (default: 0.1 for more deterministic output)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        
    def create_evaluation_prompt(self, question: str, question_decomposition: List[Dict], 
                               ai_reasoning: str) -> str:
        """
        Create a detailed prompt for GPT to evaluate intermediate steps.
        
        Args:
            question: The main question to be answered
            question_decomposition: List of intermediate questions and answers
            ai_reasoning: The AI's reasoning process
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are an expert evaluator tasked with assessing the quality of intermediate reasoning steps in a multi-hop question answering system.

TASK: Evaluate whether the model correctly formulates each intermediate question and answers each intermediate answer.

INPUT DATA:
Main Question: {question}

Question Decomposition (Intermediate Steps):
"""
        
        for i, step in enumerate(question_decomposition, 1):
            prompt += f"Step {i}: {step['question']}\n"
            prompt += f"Expected Answer: {step['answer']}\n"
        
        prompt += f"""
AI Reasoning Process:
{ai_reasoning}

EVALUATION CRITERIA:
1. For each intermediate step, determine if the AI correctly formulated the intermediate question
2. For each intermediate step, determine if the AI correctly answered the intermediate question

Please provide your evaluation in the following JSON format:
{{
    "intermediate_step_evaluations": [
        {{
            "step_number": 1,
            "question_formulation_correct": true/false,
            "answer_correct": true/false,
        }},
        ...
    ]
}}

Be thorough and specific in your evaluation. Focus on the correctness of question formulation and answer accuracy for each intermediate step.
"""
        return prompt
    
    def evaluate_single_example(self, id: str, question: str, question_decomposition: List[Dict], 
                              ai_reasoning: str) -> EvaluationResult:
        """
        Evaluate a single example using GPT.
        
        Args:
            id: The unique identifier for this example
            question: The main question
            question_decomposition: List of intermediate questions and answers
            ai_reasoning: The AI's reasoning process
            
        Returns:
            EvaluationResult object
        """
        prompt = self.create_evaluation_prompt(question, question_decomposition, ai_reasoning)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of reasoning processes in question answering systems. Provide detailed, objective evaluations in the requested JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000
            )
            
            # Parse the JSON response
            evaluation_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            try:
                # Find JSON in the response (in case there's extra text)
                start_idx = evaluation_text.find('{')
                end_idx = evaluation_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_text = evaluation_text[start_idx:end_idx]
                    evaluation_data = json.loads(json_text)
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                # Create a fallback evaluation
                evaluation_data = {
                    "intermediate_step_evaluations": [],
                }
            
            # Create EvaluationResult object
            result = EvaluationResult(
                id=id,
                question=question,
                question_decomposition=question_decomposition,
                ai_reasoning=ai_reasoning,
                intermediate_step_evaluations=evaluation_data.get("intermediate_step_evaluations", []),
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating example: {e}")
            # Return a default evaluation result
            return EvaluationResult(
                id=id,
                question=question,
                question_decomposition=question_decomposition,
                ai_reasoning=ai_reasoning,
                intermediate_step_evaluations=[],
            )
    
    def evaluate_dataset(self, data: List[Dict[str, Any]], 
                        output_file: Optional[str] = None) -> List[EvaluationResult]:
        """
        Evaluate a dataset of examples.
        
        Args:
            data: List of examples to evaluate
            output_file: Optional file to save results
            
        Returns:
            List of EvaluationResult objects
        """
        results = []
        
        for i, example in enumerate(data):
            logger.info(f"Evaluating example {i+1}/{len(data)}")
            
            # Extract required fields
            id = example.get('id', f'key_error_idx_{i}')  # Use provided id or generate one
            question = example.get('question', '')
            question_decomposition = example.get('question_decomposition', [])
            ai_reasoning = example.get('ai_reasoning', '')
            
            # Evaluate the example
            result = self.evaluate_single_example(id, question, question_decomposition, ai_reasoning)
            results.append(result)
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Save results if output file is specified
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def save_results(self, results: List[EvaluationResult], output_file: str):
        """Save evaluation results to a JSON file."""
        results_data = []
        
        for result in results:
            result_dict = {
                "id": result.id,
                "question": result.question,
                "question_decomposition": list(result.question_decomposition),
                "ai_reasoning": result.ai_reasoning,
                "intermediate_step_evaluations": result.intermediate_step_evaluations,
            }
            results_data.append(result_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")



def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different data formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'data' in data:
        return data['data']
    else:
        # Assume it's a single example
        return [data]


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate intermediate reasoning steps using GPT')
    parser.add_argument('--input-file', type=str, required=True,
                       help='Path to input JSON file containing examples')
    parser.add_argument('--output-file', type=str, default='evaluation_results.json',
                       help='Path to save evaluation results')
    parser.add_argument('--api-key', type=str, required=True,
                       help='OpenAI API key')
    parser.add_argument('--model', type=str, default='gpt-4',
                       help='GPT model to use (default: gpt-4)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for generation (default: 0.1)')
    parser.add_argument('--max-examples', type=int, default=None,
                       help='Maximum number of examples to evaluate (for testing)')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input_file}")
    data = load_data(args.input_file)
    
    if args.max_examples:
        data = data[:args.max_examples]
        logger.info(f"Limited to {len(data)} examples for evaluation")
    
    # Initialize evaluator
    evaluator = GPTIntermediateEvaluator(
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature
    )
    
    # Run evaluation
    logger.info(f"Starting evaluation of {len(data)} examples")
    results = evaluator.evaluate_dataset(data, args.output_file)

if __name__ == "__main__":
    main()

