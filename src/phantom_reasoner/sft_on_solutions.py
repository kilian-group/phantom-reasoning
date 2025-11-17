"""
Supervised fine-tuning on GSMInfinite solutions.

Usage:
```bash
bash scripts/train_sft_on_solutions.sub \
    recipes/accelerate_configs/zero1.yaml \
    recipes/Qwen/Qwen3-0.6B/sft_on_solutions/config_gsminfinite_1gpu.yaml
```
"""

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
from peft import get_peft_model, prepare_model_for_kbit_training
from phantom_eval.utils import setup_logging
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import (
    ModelConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
    get_quantization_config,
)

from phantom_reasoner.configs import GRPOScriptArguments, SFTConfig
from phantom_reasoner.datasets_for_grpo import GSMInfiniteDataset
from phantom_reasoner.grpo import arrange_dataset
from phantom_reasoner.utils import exp_utils
from phantom_reasoner.utils.callbacks import get_callbacks

logger = logging.getLogger(__name__)


def train_sft_on_solutions(
    script_args: GRPOScriptArguments, training_args: SFTConfig, model_args: ModelConfig
):
    # Get train dataset and use a curriculum
    dataset_for_sft = GSMInfiniteDataset(script_args)
    train_dataset = dataset_for_sft.get_dataset(is_eval=False, get_solutions=True)

    # Map to convert from GRPO format (prompt/answer) to SFT format (prompt/completion)
    def convert_to_sft_format(sample):
        # Prompt is in CONVO format when loaded from GSMInfiniteDataset,
        # take the first message's content
        return {
            "prompt": sample["prompt"][0]["content"],
            "completion": sample["solution"],
        }

    train_dataset = train_dataset.map(
        convert_to_sft_format,
        desc="Converting to SFT format",
    )

    train_dataset = arrange_dataset(train_dataset, script_args.data_curriculum, training_args.seed)
    num_train_samples = min(script_args.max_num_train_samples, len(train_dataset))
    train_dataset = train_dataset.select(range(num_train_samples))
    train_dataset = arrange_dataset(train_dataset, script_args.data_curriculum, training_args.seed)
    logger.info(f"*** Arranged in curriculum={script_args.data_curriculum}.")

    # Get eval dataset
    eval_dataset = dataset_for_sft.get_dataset(is_eval=True, get_solutions=True)
    eval_dataset = eval_dataset.map(
        convert_to_sft_format,
        desc="Converting eval to SFT format",
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
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
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

    # Delete the last checkpoint's optimizer state to save space
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None:
        glob_optimizer_states = [str(x) for x in Path(last_checkpoint).glob("global_step*")]
        for optimizer_state in glob_optimizer_states:
            logger.info(f"Deleting optimizer state {optimizer_state}")
            shutil.rmtree(optimizer_state, ignore_errors=True)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    setup_logging(training_args.log_level.upper())
    set_seed(training_args.seed)

    assert script_args.training_mode == "gsminfinite", "Only supported for GSMInfinite"

    run_flags_str = f"curr={script_args.data_curriculum}__training_seed={training_args.seed}"
    run_name: str = exp_utils.get_run_name(
        training_algo_name="sft_on_solutions",
        script_args=script_args,
        model_args=model_args,
        run_flags_str=run_flags_str,
    )
    training_args.run_name = run_name
    training_args.output_dir = os.path.join(os.environ.get("RUN_BASE_DIR", "."), run_name)
    os.makedirs(training_args.output_dir, exist_ok=True)

    train_sft_on_solutions(script_args, training_args, model_args)
