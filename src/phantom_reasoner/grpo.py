"""
Training script for the GRPO model using Zeroshot or CoT prompt from PhantomEval.

Usage:
```bash
./scripts/train_grpo.sh \
    recipes/accelerate_configs/zero1.yaml \
    recipes/Qwen/Qwen2.5-1.5B-Instruct/grpo/config_base.yaml \
    --prompt_method cot \
```
"""

import logging
import os
import shutil
import subprocess
from datetime import datetime

import torch
from datasets import Dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from phantom_eval.agents.cot import CoTAgent
from phantom_eval.agents.nshot import NshotAgent
from phantom_eval.constants import answer_sep
from phantom_eval.score import exact_match, f1, precision, recall
from phantom_eval.utils import setup_logging
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, TrlParser, get_peft_config, get_quantization_config

from phantom_reasoner.configs import GRPOConfig, GRPOScriptArguments
from phantom_reasoner.datasets import GSMInfiniteDataset, PhantomWikiDataset
from phantom_reasoner.trainers.custom_grpo_trainer import CustomGRPOTrainer
from phantom_reasoner.utils import callbacks, exp_utils

logger = logging.getLogger(__name__)


############################################
# REWARD FUNCTIONS
############################################
# TODO refactor reward functions to a separate python module
def format_pred(pred: str, prompt_method: str) -> str:
    # TODO: use script_arguments.ignore_think_tags_in_outputs
    # HACK: check if pred contains <think> tag
    match prompt_method:
        case "zeroshot":
            if "<think>" in pred:
                # Zeroshot prompt does not use thinking tags, so we remove them
                return NshotAgent.parse_thinking_answer(pred)
            else:
                return pred
        case "cot":
            try:
                return CoTAgent.parse_answer(pred)
            except ValueError:
                return ""
        case _:
            raise ValueError(f"Invalid {prompt_method=}")


def reward_exact_match(
    completions: list[list[dict[str, str]]], answer: list[list[str]], prompt_method: list[str], **kwargs
) -> list[float]:
    """
    Args:
        completions (shape (batch, len of convo)): Batch of completions,
            where each is a conversation (i.e. a list of dicts).
        answer: (shape (batch, # answers)): The true answers for the prompts.
        prompt_method: (shape (batch,)): The prompt method used for each sample.
    """
    # Format the model output text based on the prompting format
    preds = [
        format_pred(completion[0]["content"], method)
        for completion, method in zip(completions, prompt_method)
    ]
    return [float(exact_match(pred, answer_sep.join(a))) for pred, a in zip(preds, answer)]


def reward_precision(
    completions: list[list[dict[str, str]]], answer: list[list[str]], prompt_method: list[str], **kwargs
) -> list[float]:
    """
    Args:
        completions (shape (batch, len of convo)): Batch of completions,
            where each is a conversation (i.e. a list of dicts).
        answer: (shape (batch, # answers)): The true answers for the prompts.
        prompt_method: (shape (batch,)): The prompt method used for each sample.
    """
    # Format the model output text based on the prompting format
    preds = [
        format_pred(completion[0]["content"], method)
        for completion, method in zip(completions, prompt_method)
    ]
    return [float(precision(pred, answer_sep.join(a))) for pred, a in zip(preds, answer)]


def reward_recall(
    completions: list[list[dict[str, str]]], answer: list[list[str]], prompt_method: list[str], **kwargs
) -> list[float]:
    """
    Args:
        completions (shape (batch, len of convo)): Batch of completions,
            where each is a conversation (i.e. a list of dicts).
        answer: (shape (batch, # answers)): The true answers for the prompts.
        prompt_method: (shape (batch,)): The prompt method used for each sample.
    """
    # Format the model output text based on the prompting format
    preds = [
        format_pred(completion[0]["content"], method)
        for completion, method in zip(completions, prompt_method)
    ]
    return [float(recall(pred, answer_sep.join(a))) for pred, a in zip(preds, answer)]


def reward_f1(
    completions: list[list[dict[str, str]]], answer: list[list[str]], prompt_method: list[str], **kwargs
) -> list[float]:
    """
    Args:
        completions (shape (batch, len of convo)): Batch of completions,
            where each is a conversation (i.e. a list of dicts).
        answer: (shape (batch, # answers)): The true answers for the prompts.
        prompt_method: (shape (batch,)): The prompt method used for each sample.
    """
    # Format the model output text based on the prompting format
    preds = [
        format_pred(completion[0]["content"], method)
        for completion, method in zip(completions, prompt_method)
    ]
    return [float(f1(pred, answer_sep.join(a))) for pred, a in zip(preds, answer)]


def get_reward_func(reward_type_name: str) -> callable:
    match reward_type_name:
        case "exact_match":
            return reward_exact_match
        case "precision":
            return reward_precision
        case "recall":
            return reward_recall
        case "f1":
            return reward_f1
        case _:
            raise ValueError(f"Invalid {reward_type_name=}")


def arrange_dataset(dataset: Dataset, data_curriculum: str, seed: int) -> Dataset:
    match data_curriculum:
        case "random":
            return dataset.shuffle(seed=seed)
        case "difficulty_asc":
            return dataset.sort("difficulty")
        case "difficulty_desc":
            return dataset.sort("difficulty", reverse=True)
        case _:
            raise ValueError(f"Invalid {data_curriculum=}")


def train_grpo(script_args: GRPOScriptArguments, training_args: GRPOConfig, model_args: ModelConfig) -> None:
    """Training script for the GRPO model using Zeroshot prompt from PhantomEval.

    Args:
        script_args: Script arguments.
        training_args: Training arguments.
        model_args: Model arguments.
    """
    # Ensure GRPO does not shuffle dataset by itself
    training_args.shuffle_dataset = False

    # Get train and eval datasets and use a curriculum on the train dataset
    match script_args.training_mode:
        case "pw":
            dataset_for_grpo = PhantomWikiDataset(script_args)
        case "gsminfinite":
            dataset_for_grpo = GSMInfiniteDataset(script_args)
        case _:
            raise ValueError(f"Invalid {script_args.training_mode=}")

    train_dataset: Dataset = dataset_for_grpo.get_dataset(is_eval=False)
    train_dataset = arrange_dataset(train_dataset, script_args.data_curriculum, training_args.seed)
    logger.info(f"*** Arranged in curriculum={script_args.data_curriculum}.")

    eval_dataset: Dataset = dataset_for_grpo.get_dataset(is_eval=True)

    # Count number of tokens in train dataset
    # NOTE: depth_20_size_50_seed_1 prompts have num_tokens ~ 4k
    # logger.info(
    #     train_dataset.map(lambda x: {"num_tokens": len(x["prompt"][0]["content"].split()) })\
    #     .to_pandas().sort_values(by="num_tokens", ascending=False)
    # )

    # Load tokenizer
    # Set padding side to left for GRPO. If we don't create tokenizer here, the GRPOTrainer will create it
    # and set padding_side="left". So we do it here with other kwargs.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_funcs: list[callable] = [
        get_reward_func(reward_func_name) for reward_func_name in script_args.reward_func_names
    ]
    logger.info(f"*** Selected reward functions: {script_args.reward_func_names}")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    # Get glibc version from ldd --version. If less than 2.32, set attn_implementation to None
    # This is because flash-attn==2.8.2 requires GLIBC 2.32, and Anvil has GLIBC 2.82
    try:
        glibc_version = (
            subprocess.check_output(["ldd", "--version"]).decode("utf-8").split("\n")[0].split(" ")[-1]
        )
        logger.info(f"*** Available GLIBC version: {glibc_version}, required: 2.32 ***")
        # glibc_version is like "2.28"
        if float(glibc_version) < 2.32:
            logger.info("*** Setting attn_implementation to None***")
            model_args.attn_implementation = None
    except Exception as e:
        logger.warning(f"*** Error getting glibc version: {e} ***")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        quantization_config=get_quantization_config(model_args) if model_args.use_peft else None,
    )
    logger.info(f"*** Model kwargs: {model_kwargs} ***")
    # NOTE: CustomGRPOTrainer does not prepare model for kbit training,
    # so we do it outside of the trainer manually
    # Reference: https://huggingface.co/docs/peft/en/developer_guides/quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    if model_kwargs["quantization_config"] is not None:
        logger.info("*** Preparing model for kbit training ***")
        model = prepare_model_for_kbit_training(model)
    if model_args.use_peft:
        logger.info("*** Initializing PEFT model ***")
        lora_config = get_peft_config(model_args)
        model = get_peft_model(model, lora_config)

    # Instantiate the trainer
    trainer = CustomGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        callbacks=callbacks.get_callbacks(training_args, model_args),
    )
    logger.info(f"*** Instantiated GRPO trainer for model {model_args.model_name_or_path}")

    # Training loop
    # Check for last checkpoint
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Training loop
    logger.info(
        f"*** Starting training {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"for {training_args.num_train_epochs} epochs"
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"*** Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"*** Tokenizer saved to {training_args.output_dir}")

    # Delete the last checkpoint to save space
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None:
        logger.info(f"Removing checkpoint {last_checkpoint}")
        shutil.rmtree(last_checkpoint, ignore_errors=True)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    setup_logging(training_args.log_level.upper())
    set_seed(training_args.seed)

    run_flags_str = f"curr={script_args.data_curriculum}__prompt={script_args.prompt_method}"
    run_name: str = exp_utils.get_run_name(
        training_algo_name="grpo",
        script_args=script_args,
        model_args=model_args,
        run_flags_str=run_flags_str,
    )
    training_args.run_name = run_name
    # output_dir = RUN_BASE_DIR environment variable + run_name
    training_args.output_dir = os.path.join(os.environ.get("RUN_BASE_DIR", "."), run_name)
    os.makedirs(training_args.output_dir, exist_ok=True)

    train_grpo(script_args, training_args, model_args)
