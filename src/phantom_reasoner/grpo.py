"""
Training script for the GRPO model using Zeroshot or CoT prompt from PhantomEval.

Usage:
```bash
bash scripts/create_train_grpo__vllm_colocate.sh <cluster_name>
bash scripts/train_grpo__vllm_colocate.sub \
    recipes/accelerate_configs/zero1.yaml \
    recipes/Qwen/Qwen3-0.6B-Instruct/grpo/config_pw_4gpu.yaml
```
"""

import logging
import os
import shutil
import typing
from datetime import datetime
from functools import partial

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

from phantom_reasoner._types import CONVO_T
from phantom_reasoner.configs import GRPOConfig, GRPOScriptArguments
from phantom_reasoner.datasets_for_grpo import (
    GSMInfiniteDataset,
    HotpotQADataset,
    MuSiQueDataset,
    PhantomWikiDataset,
    TwoWikiDataset,
)
from phantom_reasoner.trainers.custom_grpo_trainer import CustomGRPOTrainer
from phantom_reasoner.utils import exp_utils
from phantom_reasoner.utils.callbacks import DeleteAllButLastOptimizerCheckpointCallback

# import HP metrics
from phantom_reasoner.utils.hp.hotpot_evaluate_v1 import exact_match_score, f1_score

# import msq metrics
from phantom_reasoner.utils.msq.evaluate_utils import score_pred as score_pred_msq

# import 2wiki metrics
from phantom_reasoner.utils.twowiki.evaluate_2wiki import score_pred as score_pred_2wiki

logger = logging.getLogger(__name__)


############################################
# REWARD FUNCTIONS
############################################
# TODO refactor reward functions to a separate python module
def format_pred(pred: str, prompt_method: str) -> str:
    # TODO partial reward for correct parsing but wrong values?
    match prompt_method:
        case "zeroshot":
            # parse_answer takes care of any <think> tags
            return NshotAgent.parse_answer(pred)
        case "cot":
            try:
                return CoTAgent.parse_answer(pred)
            except ValueError:
                return ""
        case _:
            raise ValueError(f"Invalid {prompt_method=}")


def reward_with_metric(
    metric: typing.Callable[[str, str], float],
    completions: list[CONVO_T],
    answer: list[list[str]],
    prompt_method: list[str],
    **kwargs,
) -> list[float]:
    """
    Args:
        metric: A function that takes a predicted string and a target string and returns a float score.
            E.g. `exact_match`, `precision`, `recall`, `f1`.
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
    return [float(metric(pred, answer_sep.join(a))) for pred, a in zip(preds, answer)]


def reward_with_metric_single_string(
    metric: typing.Callable[[str, str], float],
    completions: list[CONVO_T],
    answer: list[str],
    prompt_method: list[str],
    **kwargs,
) -> list[float]:
    """
    Args:
        metric: A function that takes a predicted string and a target string and returns a float score.
            E.g. `exact_match`, `precision`, `recall`, `f1`.
        completions (shape (batch, len of convo)): Batch of completions,
            where each is a conversation (i.e. a list of dicts).
        answer: (shape (batch,)): The true answers for the prompts.
        prompt_method: (shape (batch,)): The prompt method used for each sample.
    """
    # Format the model output text based on the prompting format
    preds = [
        format_pred(completion[0]["content"], method)
        for completion, method in zip(completions, prompt_method)
    ]
    # import pdb; pdb.set_trace()
    return [float(metric(pred, a)) for pred, a in zip(preds, answer)]


def get_reward_func(training_mode: str, reward_type_name: str) -> typing.Callable:
    match training_mode:
        case "pw":
            match reward_type_name:
                case "exact_match":
                    f = partial(reward_with_metric, exact_match)
                case "precision":
                    f = partial(reward_with_metric, precision)
                case "recall":
                    f = partial(reward_with_metric, recall)
                case "f1":
                    f = partial(reward_with_metric, f1)
                case _:
                    raise ValueError(f"Invalid {reward_type_name=}")
        case "hp":
            match reward_type_name:
                case "exact_match":
                    f = partial(reward_with_metric_single_string, exact_match_score)
                case "precision":
                    # NOTE: f1_score returns (f1, precision, recall)
                    f = partial(reward_with_metric_single_string, lambda x, y: f1_score(x, y)[1])
                case "recall":
                    # NOTE: f1_score returns (f1, precision, recall)
                    f = partial(reward_with_metric_single_string, lambda x, y: f1_score(x, y)[2])
                case "f1":
                    # NOTE: f1_score returns (f1, precision, recall)
                    f = partial(reward_with_metric_single_string, lambda x, y: f1_score(x, y)[0])
                case _:
                    raise ValueError(f"Invalid {reward_type_name=}")
        case "2wiki":
            match reward_type_name:
                case "exact_match":
                    f = partial(
                        reward_with_metric_single_string,
                        lambda x, y: score_pred_2wiki({"pred": x, "answer": y})["em"],
                    )
                case "precision":
                    f = partial(
                        reward_with_metric_single_string,
                        lambda x, y: score_pred_2wiki({"pred": x, "answer": y})["prec"],
                    )
                case "recall":
                    f = partial(
                        reward_with_metric_single_string,
                        lambda x, y: score_pred_2wiki({"pred": x, "answer": y})["recall"],
                    )
                case "f1":
                    f = partial(
                        reward_with_metric_single_string,
                        lambda x, y: score_pred_2wiki({"pred": x, "answer": y})["f1"],
                    )
                case _:
                    raise ValueError(f"Invalid {reward_type_name=}")
        case "msq":
            match reward_type_name:
                case "exact_match":
                    f = partial(
                        reward_with_metric_single_string,
                        lambda x, y: score_pred_msq({"pred": x, "answer": y})["em"],
                    )
                case "f1":
                    f = partial(
                        reward_with_metric_single_string,
                        lambda x, y: score_pred_msq({"pred": x, "answer": y})["f1"],
                    )
                case _:
                    raise ValueError(f"Invalid {reward_type_name=}")
        case _:
            raise ValueError(f"Invalid {training_mode=}")
    # Add a __name__ attribute because CustomGRPOTrainer uses the attribute
    f.__name__ = f"reward_{reward_type_name}"
    return f


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
        case "hp":
            dataset_for_grpo = HotpotQADataset(script_args)
        case "2wiki":
            dataset_for_grpo = TwoWikiDataset(script_args)
        case "msq":
            dataset_for_grpo = MuSiQueDataset(script_args)
        case _:
            raise ValueError(f"Invalid {script_args.training_mode=}")

    train_dataset: Dataset = dataset_for_grpo.get_dataset(is_eval=False)
    # TODO: remove the slicing
    train_dataset = train_dataset.select(range(10000))
    train_dataset = arrange_dataset(train_dataset, script_args.data_curriculum, training_args.seed)
    logger.info(f"*** Arranged in curriculum={script_args.data_curriculum}.")

    eval_dataset: Dataset = dataset_for_grpo.get_dataset(is_eval=True)

    # Count number of tokens in train dataset
    # NOTE: depth_20_size_50_seed_1 prompts have num_tokens ~ 4k
    logger.info(
        train_dataset.map(lambda x: {"num_tokens": len(x["prompt"][0]["content"].split())})
        .to_pandas()
        .sort_values(by="num_tokens", ascending=False)
    )

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
        get_reward_func(script_args.training_mode, reward_func_name)
        for reward_func_name in script_args.reward_func_names
    ]
    logger.info(f"*** Selected reward functions: {script_args.reward_func_names}")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    exp_utils.disable_flash_attn_if_unsupported_glibc(model_args)

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
    callbacks = [DeleteAllButLastOptimizerCheckpointCallback()]
    trainer = CustomGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        callbacks=callbacks,
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
