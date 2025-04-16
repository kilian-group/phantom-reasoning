"""
Model Evaluation Script for Reasoning Benchmarks using LightEval.

This script evaluates a given language model checkpoint on standard reasoning benchmarks
such as ARC and CommonSenseQA. It leverages the LightEval evaluation framework and supports
evaluating HuggingFace-compatible checkpoints with multi-GPU parallelism via Accelerate.

Example Usage:
    python -m phantom_reasoner.utils.benchmarks \
        -cp /path/to/checkpoint \
        -t "leaderboard|arc:challenge|2|0,lighteval|arc:easy|2|0" \
        -od ./out-lighteval

Arguments:
    -cp / --checkpoint-path : Path to the model checkpoint directory.
    -t  / --tasks           : Comma-separated task specification string (see below).
    -od / --output-dir      : Directory to save results (default: "./out").
    -bs / --batch-size      : Inference batch size (default: 16).

Task Format:
    Each task should be specified in the format:
        {suite}|{task_name}|{num_few_shot}|{truncate_flag}
    - suite: Either `leaderboard` or `lighteval`
    - task_name: Task identifier (e.g., arc:challenge, gsm8k)
    - num_few_shot: Number of few-shot examples (e.g., 5)
    - truncate_flag: 0 for strict adherence to few-shot count, 1 to allow truncation
    More details: https://huggingface.co/docs/lighteval/en/quicktour

Available tasks list: https://huggingface.co/docs/lighteval/en/available-tasks
"""

import argparse
from datetime import timedelta
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.utils import EnvConfig
from lighteval.utils.imports import is_accelerate_available

def run_evaluation(checkpoint_path: str, tasks: list, batch_size: int, output_dir: str):
    """
    Runs evaluation on a model checkpoint using LightEval on the specified tasks.

    Args:
        checkpoint_path (str): Path to the model checkpoint (HuggingFace format).
        tasks (str): A string of task(s) in LightEval-compatible format separated by commas.
        batch_size (int): Batch size used during evaluation.
        output_dir (str): Directory where evaluation results are saved.

    Returns:
        dict: A dictionary containing configuration, results, and task metadata.
              Example keys include:
                  - 'config_general': General run info and timing
                  - 'results': Per-task and aggregate scores
                  - 'summary_tasks': Prompt/input stats per task
                  - 'config_tasks': Task-specific metadata and metrics used
    """
    # Initialize Accelerator if available for distributed evaluation
    if is_accelerate_available():
        from accelerate import Accelerator, InitProcessGroupKwargs
        accelerator = Accelerator(
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))]
        )
    else:
        accelerator = None

    # Set up result tracking and logging
    evaluation_tracker = EvaluationTracker(output_dir=output_dir)

    # Define evaluation pipeline parameters
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=EnvConfig(cache_dir="tmp/"),
        override_batch_size=batch_size,
    )

    # Set up model configuration (HuggingFace transformer-based)
    model_config = TransformersModelConfig(
        pretrained=checkpoint_path,
        accelerator=accelerator,
        batch_size=batch_size,
    )

    # Initialize the LightEval pipeline with the model and tasks
    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )
    # Run the evaluation and store results
    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

    # Return raw results dictionary for further processing or analysis
    return pipeline.get_results()

def main():
    parser = argparse.ArgumentParser(description="Evaluate an LLM with lighteval")
    parser.add_argument(
        "--checkpoint-path", "-cp", type=str, required=True,
        help="Path to the HuggingFace-compatible model checkpoint."
    )
    parser.add_argument(
        "--output-dir", "-od", type=str, default="./out-lighteval",
        help="Directory to store output results and logs."
    )
    parser.add_argument(
        "--tasks", "-t", type=str,
        default="lighteval|gsm8k|5|0,leaderboard|arc:challenge|10|0,lighteval|arc:easy|10|0",
        help="""Comma-separated list of tasks to evaluate. Format: suite|task|fewshot|truncate_flag.
See: https://huggingface.co/docs/lighteval/en/quicktour for details."""
    )
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=16,
        help="Batch size to use during inference."
    )
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint_path,
        tasks=args.tasks,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()
