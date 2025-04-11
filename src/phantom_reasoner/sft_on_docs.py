"""
Supervised fine-tuning on PhantomWiki documents.

Usage:
```bash
./scripts/train_sft_on_docs.sh \
    recipes/accelerate_configs/zero1.yaml \
    recipes/qwen2.5-1.5b-instruct/sft_on_docs/config_base.yaml \
    --output_dir runs/sft_on_docs/username/qwen1.5b__MMDD__flags
```
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import torch
from datasets import Dataset, concatenate_datasets
from peft import get_peft_model, prepare_model_for_kbit_training
from phantom_eval.agents.common import get_all_evidence
from phantom_eval.prompts import ZeroshotLLMPrompt
from phantom_eval.utils import load_data, setup_logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import (
    ModelConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
    get_quantization_config,
)

from phantom_reasoner.configs import SFTConfig
from phantom_reasoner.utils.callbacks import get_callbacks
from phantom_reasoner.utils.score import answer_sep

logger = logging.getLogger(__name__)


############################################
# SCRIPT ARGUMENTS
############################################
@dataclass
class ScriptArguments:
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
    data_curriculum: Literal[
        "random",
        "difficulty_asc_stage_off",
        "difficulty_desc_stage_off",
        "difficulty_asc_stage_on",
        "difficulty_asc_stage_off",
    ] = "random"


def get_checkpoint(training_args: SFTConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def get_pw_dataset(dataset_name: str, split_list: list[str], from_local: bool) -> Dataset:
    llm_prompt = ZeroshotLLMPrompt()
    all_datasets: list[Dataset] = []
    for split_name in split_list:
        dataset: dict[str, Dataset] = load_data(dataset_name, split=split_name, from_local=from_local)
        text_corpus: Dataset = dataset["text"]
        qa_pairs: Dataset = dataset["qa_pairs"]
        evidence: str = get_all_evidence(text_corpus)

        dataset: Dataset = qa_pairs.map(
            lambda sample: {
                "prompt": llm_prompt.get_prompt().format(
                    evidence=evidence,
                    question=sample["question"],
                ),
                "completion": answer_sep.join(sample["answer"]),
            }
        )
        all_datasets.append(dataset)
    return concatenate_datasets(all_datasets)


def arrange_dataset(dataset: Dataset, data_curriculum: str, seed: int) -> Dataset:
    match data_curriculum:
        case "random":
            return dataset.shuffle(seed=seed)
        case "difficulty_asc_stage_off" | "difficulty_asc_stage_on":
            return dataset.sort("difficulty")
        case "difficulty_desc_stage_off" | "difficulty_desc_stage_on":
            return dataset.sort("difficulty", reverse=True)
        case _:
            raise ValueError(f"Invalid {data_curriculum=}")


def train_sft_on_docs(script_args: ScriptArguments, training_args: SFTConfig, model_args: ModelConfig):
    # Get train dataset and use a curriculum
    train_dataset = get_pw_dataset(script_args.dataset_name, script_args.split_list, script_args.from_local)
    train_dataset = arrange_dataset(train_dataset, script_args.data_curriculum, training_args.seed)

    if script_args.data_curriculum in ["difficulty_asc_stage_on", "difficulty_desc_stage_on"]:
        # Repeat each dataset entry num_train_epochs times and reduce num_train_epochs to 1
        # This creates a curriculum where the easy questions are processed num_train_epochs times
        # before the harder questions are processed
        train_dataset = train_dataset.select(
            [i for i in range(len(train_dataset)) for _ in range(training_args.num_train_epochs)]
        )
        training_args.num_train_epochs = 1

    logger.info(
        f"*** Loaded train dataset {script_args.dataset_name}::{script_args.split_list} "
        f"with {len(train_dataset)} samples, and arranged in curriculum={script_args.data_curriculum}."
    )

    # Get eval dataset
    eval_dataset = get_pw_dataset(
        script_args.eval_dataset_name, script_args.eval_split_list, script_args.eval_from_local
    )
    logger.info(
        f"*** Loaded eval dataset {script_args.eval_dataset_name}::{script_args.eval_split_list} "
        f"with {len(eval_dataset)} samples."
    )

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###################
    # Model init kwargs
    ###################
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

    ############################
    # Initialize the SFT Trainer
    ############################
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

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
    # Training loop
    logger.info(
        f"*** Starting training {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"for {training_args.num_train_epochs} epochs"
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    metrics = train_result.metrics
    # Log and save metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"*** Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["phantom-reasoner"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        # trainer.model.config.use_cache = True
        # trainer.model.config.save_pretrained(training_args.output_dir)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    setup_logging(training_args.log_level.upper())

    train_sft_on_docs(script_args, training_args, model_args)
