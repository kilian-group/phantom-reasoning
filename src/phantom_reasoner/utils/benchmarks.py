"""
Model Evaluation Script for Reasoning Benchmarks

This script evaluates a model checkpoint on standard reasoning benchmarks like ARC, 
CommonSenseQA, etc. using LightEval.

Usage:
    python -m phantom_reasoner.utils.benchmarks -cp /share/nikola/phantom-reasoning/runs/grpo/ak2426/qwen3b__method=cot__cur=random/runs/grpo/ak2426/qwen3b__method=cot__curr=random/checkpoint-900  
"""
import lighteval
import argparse
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.utils import EnvConfig
from lighteval.utils.imports import is_accelerate_available
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import timedelta
import os

def run_evaluation(checkpoint_path: str, tasks: list, batch_size: int, output_dir: str):
    # Enable accelerate if possible
    if is_accelerate_available():
        from accelerate import Accelerator, InitProcessGroupKwargs
        accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
    else:
        accelerator = None

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=EnvConfig(cache_dir="tmp/"),
        override_batch_size = batch_size,
    )

    # Refere to this link: https://huggingface.co/docs/lighteval/package_reference/models
    # We can use LightevalModel (not sure what it does), TransformersModel (Huggingface),
    # AdapterModel, DeltaModel, Endpoints-based Models, Nanotron Models, and VLLMModel (
    # example use in the provided link). For now, using TransformersModels since we want 
    # to load from a huggingface checkpoint.
    model_config = TransformersModelConfig(
        pretrained = checkpoint_path,
        accelerator = accelerator,
        batch_size = batch_size,
    )

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

def main():
    parser = argparse.ArgumentParser(description="Evaluate an LLM with lighteval")
    parser.add_argument("--checkpoint-path", "-cp", type=str, help="Path to the HuggingFace checkpoint.")
    parser.add_argument("--output-dir", "-or", type=str, default="./out", help="Path to saving the outputs.")
    parser.add_argument("--tasks", "-t", type=str, default="lighteval|gsm8k|5|0,leaderboard|arc:challenge|10|0,lighteval|arc:easy|10|0",
                        help="""List of tasks to evaluate the model on. A single task needs to follow
                        a certain format type: {suite}|{task}|{num_few_shot}|{0 for strict `num_few_shots`, or 1 to allow 
                        a truncation if context size is too small}. When including multiple tasks, separated them with commas.
                        More on the formatting here https://huggingface.co/docs/lighteval/en/quicktour). To see a comprehensive 
                        list of available tasks go to this link: https://huggingface.co/docs/lighteval/en/available-tasks. Look
                        at default value for example.""")
    parser.add_argument("--batch-size", "-bs", default=32, type=int, help="Batch size to be used during inference.")
    args = parser.parse_args()

    run_evaluation(args.checkpoint_path, args.tasks, args.batch_size, args.output_dir)
    
if __name__ == "__main__":
    main()
