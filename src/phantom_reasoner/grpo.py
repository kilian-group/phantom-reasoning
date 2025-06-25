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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import torch
from datasets import Dataset, concatenate_datasets
from peft import get_peft_model, prepare_model_for_kbit_training
from phantom_eval.agents.common import get_all_evidence
from phantom_eval.agents.cot import CoTAgent
from phantom_eval.agents.nshot import NshotAgent
from phantom_eval.constants import answer_sep
from phantom_eval.prompts import COT_EXAMPLES, CoTLLMPrompt, ZeroshotLLMPrompt
from phantom_eval.score import exact_match, f1, precision, recall
from phantom_eval.utils import load_data, setup_logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    set_seed,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint
from trl import (
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
    get_quantization_config,
)

from phantom_reasoner.configs import GRPOConfig
from phantom_reasoner.utils import exp_utils

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
                # TODO partial reward for correct parsing but wrong values?
                if "<think>" in pred:
                    return CoTAgent.parse_thinking_answer(pred)
                else:
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


############################################
# SCRIPT ARGUMENTS
############################################
@dataclass
class GRPOScriptArguments(ScriptArguments):
    # Train dataset arguments
    dataset_name: str = "kilian-group/phantom-wiki-v1"
    split_list: list[str] = field(
        default_factory=lambda: ["depth_20_size_50_seed_1", "depth_20_size_50_seed_2"]
    )
    from_local: bool = False
    # Eval dataset arguments
    eval_dataset_name: str = "kilian-group/phantom-wiki-v1"
    eval_split_list: str = field(default_factory=lambda: ["depth_20_size_50_seed_3"])
    eval_from_local: bool = False
    # Script arguments
    run_dir: str = "runs"
    reward_func_names: list[str] = field(default_factory=lambda: ["f1"])
    data_curriculum: Literal[
        "random",
        "difficulty_asc",
        "difficulty_desc",
    ] = "random"
    prompt_method: Literal["zeroshot", "cot"] = "cot"
    ignore_think_tags_in_outputs: bool = False
    exclude_aggregation_questions: bool = True


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


def get_prompt_for_sample(sample: dict[str, Any], evidence: str, prompt_method: str) -> list[dict[str, str]]:
    """
    Get the prompt for a sample, depending on the prompt method.

    Args:
        sample (dict[str, Any]): A sample from the dataset, with key "question".
        evidence (str): The evidence text to include in the prompt.
        prompt_method (str): Either "zeroshot" or "cot".

    Returns:
        list[dict[str, str]]: A list of messages for the conversational-style prompt.
            Each message is a dict with keys "role" and "content".
    """
    match prompt_method:
        case "zeroshot":
            llm_prompt = ZeroshotLLMPrompt()
            prompt = [
                {
                    "role": "user",
                    "content": llm_prompt.get_prompt().format(evidence=evidence, question=sample["question"]),
                },
            ]
            return prompt

        case "cot":
            llm_prompt = CoTLLMPrompt()
            return [
                {
                    "role": "user",
                    "content": llm_prompt.get_prompt().format(
                        evidence=evidence, examples=COT_EXAMPLES, question=sample["question"]
                    ),
                },
            ]
        case _:
            raise ValueError(f"Invalid {prompt_method=}")


def get_pw_dataset(script_args: GRPOScriptArguments, is_eval: bool) -> Dataset:
    if is_eval:
        dataset_name = script_args.eval_dataset_name
        split_list = script_args.eval_split_list
        from_local = script_args.eval_from_local
    else:
        dataset_name = script_args.dataset_name
        split_list = script_args.split_list
        from_local = script_args.from_local

    all_datasets: list[Dataset] = []
    for split_name in split_list:
        dataset: dict[str, Dataset] = load_data(
            dataset_name,
            split=split_name,
            from_local=from_local,
            exclude_aggregation_questions=script_args.exclude_aggregation_questions,
        )
        text_corpus: Dataset = dataset["text"]
        qa_pairs: Dataset = dataset["qa_pairs"]
        evidence: str = get_all_evidence(text_corpus)

        dataset: Dataset = qa_pairs.map(
            lambda sample: {
                "prompt": get_prompt_for_sample(sample, evidence, script_args.prompt_method),
                "answer": sample["answer"],  # x['answer'] is a list of strings
                "prompt_method": script_args.prompt_method,
            }
        )
        all_datasets.append(dataset)

    dataset = concatenate_datasets(all_datasets)
    logger.info(
        f"*** Loaded {is_eval=} dataset {script_args.dataset_name}::{script_args.split_list} "
        f"with {len(dataset)} samples."
    )
    return dataset


class PhantomEvalCallback(TrainerCallback):
    """Callback to run phantom_eval on saving a checkpoint."""

    def __init__(self, script_args: GRPOScriptArguments):
        """
        Args:
            script_args (GRPOScriptArguments): Script arguments containing evaluation parameters.
        """
        self.script_args = script_args

    def on_save(self, args: GRPOConfig, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        eval_out_dir = os.path.join(args.output_dir, "out")

        # Run phantom_eval on the saved checkpoint ONLY on the last GPU (n-1).
        # Get the number of available GPUs
        # num_gpus = torch.cuda.device_count()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        # Run phantom_eval as a subprocess
        cmd = [
            "python",
            "-m",
            "phantom_eval",
            "--method",
            self.script_args.prompt_method,
            "--server",
            "vllm",
            "--inf_vllm_offline",
            "--model_name",
            checkpoint_folder,
            "--dataset",
            self.script_args.eval_dataset_name,
            "--split_list",
            *self.script_args.eval_split_list,
            "--inf_vllm_tensor_parallel_size",
            "1",
            "-od",
            eval_out_dir,
        ]
        if self.script_args.eval_from_local:
            cmd.append("--from_local")

        if self.script_args.ignore_think_tags_in_outputs:
            # NOTE: in phantom_eval, this flag is used to ignore <think> tags in outputs
            cmd.append("--inf_is_deepseek_r1_model")

        if args.should_save:
            # In multi-GPU training, we only run phantom_eval from the main process
            # that was trying to save the checkpoint.
            _ = subprocess.run(cmd, env=env, check=True)
        # try:
        #     result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        #     logger.info(f"PhantomEval output:\n{result.stdout}")
        # except subprocess.CalledProcessError as e:
        #     logger.error(f"PhantomEval failed with error:\n{e.stderr}")
        #     raise e

        return control


def train_grpo(script_args: GRPOScriptArguments, training_args: GRPOConfig, model_args: ModelConfig) -> None:
    """Training script for the GRPO model using Zeroshot prompt from PhantomEval.

    Args:
        script_args: Script arguments.
        training_args: Training arguments.
        model_args: Model arguments.
    """
    # Ensure GRPO does not shuffle dataset by itself
    training_args.shuffle_dataset = False

    # Get train dataset and use a curriculum
    train_dataset = get_pw_dataset(script_args, is_eval=False)
    train_dataset = arrange_dataset(train_dataset, script_args.data_curriculum, training_args.seed)
    logger.info(f"*** Arranged in curriculum={script_args.data_curriculum}.")

    # NOTE: getting rid of stage now, only sorting
    # if script_args.data_curriculum in ["difficulty_asc_stage_on", "difficulty_desc_stage_on"]:
    #     # Repeat each dataset entry num_train_epochs times and reduce num_train_epochs to 1
    #     # This creates a curriculum where the easy questions are processed num_train_epochs times
    #     # before the harder questions are processed
    #     train_dataset = train_dataset.select(
    #         [i for i in range(len(train_dataset)) for _ in range(training_args.num_train_epochs)]
    #     )
    #     training_args.num_train_epochs = 1

    # Get eval dataset
    eval_dataset = get_pw_dataset(script_args, is_eval=True)
    # Count number of tokens in train dataset
    # NOTE: depth_20_size_50_seed_1 prompts have num_tokens ~ 4k
    # logger.info(
    #     train_dataset.map(lambda x: {"num_tokens": len(x["prompt"][0]["content"].split()) })\
    #     .to_pandas().sort_values(by="num_tokens", ascending=False)
    # )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
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
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        quantization_config=get_quantization_config(model_args) if model_args.use_peft else None,
    )
    logger.info(f"*** Model kwargs: {model_kwargs} ***")
    # NOTE: GRPOTrainer does not prepare model for kbit training,
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
    callbacks: list[TrainerCallback] = [PhantomEvalCallback(script_args)]
    trainer = GRPOTrainer(
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
