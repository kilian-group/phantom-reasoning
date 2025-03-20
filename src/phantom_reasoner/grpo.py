import logging
import os
from dataclasses import dataclass, field
from datetime import datetime

from datasets import Dataset
from phantom_eval.agents.common import get_all_evidence
from phantom_eval.prompts import ZeroshotLLMPrompt
from phantom_eval.utils import load_data, setup_logging
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser

from phantom_reasoner.configs import GRPOConfig
from phantom_reasoner.utils.score import answer_sep, exact_match, f1, precision, recall

logger = logging.getLogger(__name__)


############################################
# REWARD FUNCTIONS
############################################
# TODO refactor reward functions to a separate python module
def reward_exact_match(
    completions: list[list[dict[str, str]]], answer: list[list[str]], **kwargs
) -> list[float]:
    """
    Args:
        completions (shape (batch, len of convo)): Batch of completions,
            where each is a conversation (i.e. a list of dicts).
        answer: (shape (batch, # answers)): The true answers for the prompts.
    """
    return [
        float(exact_match(completion[0]["content"], answer_sep.join(a)))
        for completion, a in zip(completions, answer)
    ]


def reward_precision(
    completions: list[list[dict[str, str]]], answer: list[list[str]], **kwargs
) -> list[float]:
    """
    Args:
        completions (shape (batch, len of convo)): Batch of completions,
            where each is a conversation (i.e. a list of dicts).
        answer: (shape (batch, # answers)): The true answers for the prompts.
    """
    return [
        float(precision(completion[0]["content"], answer_sep.join(a)))
        for completion, a in zip(completions, answer)
    ]


def reward_recall(completions: list[list[dict[str, str]]], answer: list[list[str]], **kwargs) -> list[float]:
    """
    Args:
        completions (shape (batch, len of convo)): Batch of completions,
            where each is a conversation (i.e. a list of dicts).
        answer: (shape (batch, # answers)): The true answers for the prompts.
    """
    return [
        float(recall(completion[0]["content"], answer_sep.join(a)))
        for completion, a in zip(completions, answer)
    ]


def reward_f1(completions: list[list[dict[str, str]]], answer: list[list[str]], **kwargs) -> list[float]:
    """
    Args:
        completions (shape (batch, len of convo)): Batch of completions,
            where each is a conversation (i.e. a list of dicts).
        answer: (shape (batch, # answers)): The true answers for the prompts.
    """
    return [
        float(f1(completion[0]["content"], answer_sep.join(a))) for completion, a in zip(completions, answer)
    ]


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
    # TODO convert to list of dataset_names, split_names, from_locals to support multiple datasets
    # TODO do this for both train and eval datasets
    dataset_name: str
    split_name: str = "depth_20_size_50_seed_1"
    # TODO add support for eval dataset
    from_local: bool = False
    reward_func_names: list[str] = field(default_factory=lambda: ["f1"])
    # TODO add flag --curriculum: str = "none"  # "none" or "difficulty_asc" or "difficulty_desc" options


# TODO move to utils.py or something
def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def get_pw_train_dataset(dataset_name: str, split_name: str, from_local: bool) -> Dataset:
    dataset: dict[str, Dataset] = load_data(dataset_name, split=split_name, from_local=from_local)
    text_corpus: Dataset = dataset["text"]
    question_answer: Dataset = dataset["qa_pairs"]

    evidence: str = get_all_evidence(text_corpus)

    # TODO support multiple prompting methods, use get_llm_prompt somehow
    llm_prompt = ZeroshotLLMPrompt()

    train_dataset: Dataset = question_answer.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": llm_prompt.get_prompt().format(evidence=evidence, question=x["question"]),
                },
            ],
            "answer": x["answer"],  # x['answer'] is a list of strings
        }
    )
    return train_dataset


def train_grpo(script_args: GRPOScriptArguments, training_args: GRPOConfig, model_args: ModelConfig) -> None:
    """Training script for the GRPO model using Zeroshot prompt from PhantomEval.

    Args:
        script_args: Script arguments.
        training_args: Training arguments.
        model_args: Model arguments.
    """
    # Get the dataset
    train_dataset = get_pw_train_dataset(
        script_args.dataset_name, script_args.split_name, script_args.from_local
    )
    logger.info(
        f"*** Loaded dataset {script_args.dataset_name}::{script_args.split_name} "
        f"with {len(train_dataset)} samples."
    )
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

    # GRPOConfig options from @willccbb's script
    # TODO Could need Lora for larger models
    # peft_config = LoraConfig(
    #     r=16,
    #     lora_alpha=64,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    #     task_type="CAUSAL_LM",
    #     lora_dropout=0.05,
    # )
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map=None
    # ).to("cuda")

    reward_funcs: list[callable] = [
        get_reward_func(reward_func_name) for reward_func_name in script_args.reward_func_names
    ]
    logger.info(f"*** Selected reward functions: {script_args.reward_func_names}")

    # TODO setup run_name

    # Instantiate the trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # TODO: add eval dataset
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        # peft_config=peft_config  # "Use at your own risk, didn't work for multi-GPU setup"
        # TODO additional options from @willccbb's script
    )
    logger.info(f"*** Instantiated GRPO trainer for model {model_args.model_name_or_path}")

    # Training loop
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

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl", "grpo", "phantom-wiki"]})
    # push to hub if needed
    if training_args.push_to_hub:
        logger.info("*** Pushing to hub...")
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    setup_logging(training_args.log_level.upper())

    train_grpo(script_args, training_args, model_args)
