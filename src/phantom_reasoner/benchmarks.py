"""
Model Evaluation Script for Reasoning Benchmarks

This script evaluates a model checkpoint on standard reasoning benchmarks like ARC, 
CommonSenseQA, etc. using LightEval.

Usage:
    TODO: Write example usage here
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

# Enable accelerate if possible
if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None

def load_model(checkpoint_path: str):
    """Load a model and tokenizer from a Hugging Face-style checkpoint."""
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer

def run_evaluation(checkpoint_path: str, tasks: list):
    evaluation_tracker = EvaluationTracker(
        output_dir="./results",
        save_details=True,
        push_to_hub=False,
        # hub_results_org="your user name", # Replace with username if pushing to hub
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=EnvConfig(cache_dir="tmp/"),
    )

    # Refere to this link: https://huggingface.co/docs/lighteval/package_reference/models
    # We can use LightevalModel (not sure what it does), TransformersModel (Huggingface),
    # AdapterModel, DeltaModel, Endpoints-based Models, Nanotron Models, and VLLMModel (
    # example use in the provided link). For now, using TransformersModels since we want 
    # to load from a huggingface checkpoint.
    model_config = TransformersModelConfig(
        pretrained="/share/nikola/phantom-reasoning/runs/grpo/ak2426/qwen3b__method=cot__cur=random/runs/grpo/ak2426/qwen3b__method=cot__curr=random/checkpoint-900", # checkpoint_path,
        # tokenizer=checkpoint_path, # TODO: Should be tokenizer ID, not sure how to retrieve that -> Optional argument, so maybe if not provided will get correct one
        accelerator=accelerator,
        # batch_size = 8, # TODO: Figure out if I need to parse any other inputs https://github.com/huggingface/lighteval/blob/main/src/lighteval/models/transformers/transformers_model.py
    )

    task = "|".join(tasks)

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
        custom_task_directory=None,
    )

    pipeline.evaluate() # This simply prints "Killed" and the process exits, not sure why
    pipeline.save_and_push_results()
    pipeline.show_results()

def main():
    parser = argparse.ArgumentParser(description="Evaluate an LLM with lighteval")
    parser.add_argument("--checkpoint", "-c", type=str, help="Path to the model checkpoint")
    args = parser.parse_args()
    
    # Define reasoning benchmarks to evaluate and run evaluation
    tasks = ["arc_easy"]# For now-> after add ["arc_challenge", "commonsense_qa", "hellaswag"]
    run_evaluation(args.checkpoint, tasks)
    
if __name__ == "__main__":
    main()
