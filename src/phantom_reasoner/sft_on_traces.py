"""Supervised fine-tuning script for decoder language models.

Adapted from the Open-R1 SFT script:
    Source: https://github.com/huggingface/open-r1/blob/main/src/open_r1/sft.py
    License: Apache 2.0

Usage:
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes NUM_GPUS \
    --config_file recipes/accelerate_configs/zero3.yaml \
    src/phantom_reasoner/sft.py \
    --config recipes/qwen2.5-1.5b-instruct/sft/config_demo.yaml
```
Here NUM_GPUS is the number of GPUs you want to use.
NOTE: when changing the number of GPUs, the total batch size will be scaled by NUM_GPUS.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass

import datasets
import torch
import transformers
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import (
    ModelConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from phantom_reasoner.configs import SFTConfig
from phantom_reasoner.utils.callbacks import get_callbacks
from phantom_reasoner.utils.wandb_logging import init_wandb_training

logger = logging.getLogger(__name__)


############################################
# SCRIPT ARGUMENTS
############################################
# TODO: add functionality to load from multiple folders
@dataclass
class ScriptArguments:
    dataset_name: str  # name to use in the model card
    data_dir: str  # output directory where the predictions are stored
    method: str  # method to filter by
    split: str  # split to filter by
    model_name: str  # model_name to filter by
    dataset_test_split: str = "test"


def get_data_by_split_model(data_dir: str, method: str, split: str, model_name: str):
    """
    Filter the predictions by split and model
    Args:
        data_dir: the directory where the predictions are stored as .json files
            each json file is a dictionary in the format of
            {"question_id": {
                "interaction": {"messages": list[{"role": str, "content": str}]},
                ...
                }
            }
        split: the split of PhantomWiki to filter by
        model_name: the model that made the predictions
    Returns:
        a dictionary containing the filtered predictions
    """
    dir = os.path.join(data_dir, "preds", method)
    filtered_predictions = {}
    for file in os.listdir(dir):
        if file.endswith(".json"):
            if file.startswith(f"split={split}") and f"model_name={model_name}" in file:
                with open(os.path.join(dir, file)) as f:
                    # load the predictions from the file and add them to the dictionary
                    data = json.load(f)
                    filtered_predictions.update(data)
    return filtered_predictions


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device},"
        + f" n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)},"
        + f" 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load datasets
    ################
    logger.info("*** Loading datasets ***")
    # filter the predictions data by split and model
    predictions_data = get_data_by_split_model(
        data_dir=script_args.data_dir,
        method=script_args.method,
        split=script_args.split,
        # NOTE: the preds are saved with model name using -- in the filename
        model_name=script_args.model_name.replace("/", "--"),
    )
    # NOTE: the raw prediction data follows the Conversation schema from phantom_eval:
    # https://github.com/kilian-group/phantom-wiki/blob/main/src/phantom_eval/_types.py#L21
    # For example:
    # {
    #     "messages": [
    #         {"role": "user", "content": [{"text": "What's the capital of France?", "type": "text"}]},
    #         {"role": "assistant", "content": [{"text": "...", "type": "text"}]}
    #     ]
    # }

    # NOTE: we convert the raw data to the conversational input format as described in
    # https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
    # For example:
    # {
    #     "messages": [
    #         {"role": "system", "content": "You are helpful"},
    #         {"role": "user", "content": "What's the capital of France?"},
    #         {"role": "assistant", "content": "..."}
    #     ]
    # }
    dataset = datasets.Dataset.from_list(list(predictions_data.values()))
    logger.info("*** Converting data to conversational format ***")
    dataset = dataset.map(
        lambda x: {
            "messages": [
                {"role": msg["role"], "content": msg["content"][0]["text"]}
                for msg in x["interaction"]["messages"]
            ]
        },
        remove_columns=dataset.column_names,  # This removes all existing columns
    )
    # split the dataset into train and eval
    dataset = dataset.train_test_split(test_size=0.1, seed=training_args.seed)

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
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
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        # HACK: currently we use the train set from train_test_split() as the train split
        train_dataset=dataset["train"],
        # HACK: currently we use the test set from train_test_split() as the eval set
        # eval_dataset=dataset["test"],
        # train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None
        ),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    # HACK: currently we use the train set from train_test_split() as the train split
    # metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    metrics["train_samples"] = len(dataset["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["phantom-reasoner"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        # metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        # HACK: currently we use the test set from train_test_split() as the eval set
        metrics["eval_samples"] = len(dataset["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
