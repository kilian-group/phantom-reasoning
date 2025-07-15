import logging
import os
import random
import re  # Import regex for answer extraction
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

PRINT_SAMPLE_PROB = 0.01  # Probability to print sample prompts for debugging

import torch
from datasets import (  # For loading and combining JSONL datasets
    Dataset,
    concatenate_datasets,
    load_dataset,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

# Import GRPOTrainer and related classes from TRL (TRL library for RL fine-tuning)
from trl import (
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
    get_quantization_config,
)

# Optionally import Phantom-Reasoner metrics if available (for reward calculation)
from phantom_reasoner.configs import GRPOConfig
from phantom_reasoner.utils.score import answer_sep, exact_match, f1, precision

# Removed phantom_eval imports, since we will handle prompts and parsing manually for GSM-Infinite
# from phantom_eval.agents.common import get_all_evidence
# from phantom_eval.agents.cot import CoTAgent
# from phantom_eval.prompts import COT_EXAMPLES, CoTLLMPrompt, ZeroshotLLMPrompt


logger = logging.getLogger(__name__)

############################################################
# Reward Functions and Utility for Parsing Model Outputs
############################################################


def extract_final_answer(solution: str) -> str:
    """
    Extract the final answer from a solution string.
    For example, if solution contains "Answer: 3.", it returns "3".
    If no "Answer:" pattern is found, returns the original solution stripped.
    """
    # Use regex to find the text after "Answer:" up to a newline or period.
    match = re.search(r"Answer:\s*([^\n\.]*)", solution)
    if match:
        return match.group(1).strip()
    # If "Answer:" is not found, return the whole solution (stripped of whitespace).
    return solution.strip()


def parse_answer(pred: str) -> str:
    pattern = r"[tT]he answer is (.*?)\."
    matches = list(re.finditer(pattern, pred))
    if matches:
        # 只取最后一个
        return matches[-1].group(1).strip()
    else:
        raise ValueError(f"Answer '{pred}' cannot be parsed.")


def format_pred(pred: str, prompt_method: str) -> str:
    """
    Format the model's prediction (completion) based on the prompting method.
    - For zeroshot prompting, return the prediction as-is.
    - For chain-of-thought (cot) prompting, attempt to extract the final answer from the prediction.
      If extraction fails (no clear answer found), return an empty string.
    """
    if prompt_method == "zeroshot":
        # Zeroshot: assume the prediction is directly the answer (no chain-of-thought to parse)
        return pred
    elif prompt_method == "cot":
        try:
            # Try to parse out the final answer from the chain-of-thought output
            return parse_answer(pred)
        except Exception:
            # If parsing fails, return empty string (indicates incorrect format or answer missing)
            return ""
    else:
        raise ValueError(f"Invalid prompt_method: {prompt_method}")


def reward_exact_match(
    completions: list[list[dict[str, str]]], answers: list[list[str]], prompt_methods: list[str], **kwargs
) -> list[float]:
    """
    Reward function: exact match between the model's output and the true answer.
    Returns 1.0 if the model's final answer matches exactly, otherwise 0.0.
    """
    # `completions` is a batch of conversations (each a list of message dicts).
    #  We use the first (and only) assistant message.
    preds = [
        format_pred(completion[0]["content"], method)
        for completion, method in zip(completions, prompt_methods)
    ]
    # Compare each predicted answer with the correct answer (answers list may contain
    # one or multiple acceptable answers per sample).
    return [float(exact_match(pred, answer_sep.join(ans_list))) for pred, ans_list in zip(preds, answers)]


def reward_precision(
    completions: list[list[dict[str, str]]], answers: list[list[str]], prompt_methods: list[str], **kwargs
) -> list[float]:
    """
    Reward function: precision of the model's output compared to the true answer
    (treated as tokens or text segments).
    Useful if answers contain multiple components.
    """
    preds = [
        format_pred(completion[0]["content"], method)
        for completion, method in zip(completions, prompt_methods)
    ]
    return [float(precision(pred, answer_sep.join(ans_list))) for pred, ans_list in zip(preds, answers)]


def reward_f1(completions: list[list[dict[str, str]]], answer: list[list[str]], **kwargs) -> list[float]:
    """
    Compute F1 reward for a batch of completions.

    Args:
        completions: Batch of completions (each is a list of messages).
        answer: The true answers for the prompts (list of lists of strings).
    Returns:
        list[float]: F1 score for each sample in the batch.
    """

    # 全部写死为 cot
    prompt_method = "cot"

    preds = []
    for i, completion in enumerate(completions):
        content = completion[0]["content"]

        # 抽样打印 model 原始生成
        if random.random() < PRINT_SAMPLE_PROB:
            print(f"[DEBUG COMPLETION {i}] method={prompt_method}\n{content}\n{'-'*50}")

        pred = format_pred(content, prompt_method)

        # 抽样打印 format 之后的 pred
        if random.random() < PRINT_SAMPLE_PROB:
            print(f"[DEBUG FORMATTED PRED {i}] {pred}\n{'='*50}")

        preds.append(pred)

    rewards = []
    for i, (pred, a) in enumerate(zip(preds, answer)):
        joined = answer_sep.join(a)
        score = f1(pred, joined)

        # 抽样打印 reward
        if random.random() < PRINT_SAMPLE_PROB:
            print(f"[DEBUG REWARD {i}] pred='{pred}' | answer='{joined}' | f1={score}")

        rewards.append(float(score))

    return rewards


# def reward_f1(
#     completions: list[list[dict[str, str]]],
#     answers: list[list[str]],
#     **kwargs
# ) -> list[float]:
#     prompt_method = "cot"   # 或者 zeroshot
#     preds = [
#         format_pred(completion[0]["content"], prompt_method)
#         for completion in completions
#     ]
#     return [float(f1(pred, answer_sep.join(ans_list))) for pred, ans_list in zip(preds, answers)]


def get_reward_func(name: str) -> callable:
    """
    Retrieve the reward function by name.
    Supported: "exact_match", "precision", "recall", "f1".
    """
    if name == "exact_match":
        return reward_exact_match
    elif name == "precision":
        return reward_precision
    # elif name == "recall":
    #     return reward_recall
    elif name == "f1":
        return reward_f1
    else:
        raise ValueError(f"Invalid reward function name: {name}")


############################################################
# Dataset Preparation for GSM-Infinite
############################################################


def get_prompt_for_sample(sample: dict[str, Any], prompt_method: str) -> list[dict[str, str]]:
    """
    Construct a prompt (as a conversation list) for a given sample based on the prompt method.
    For GSM-Infinite, each sample has a 'problem' description and a 'question'.
    - Zeroshot: provide the problem and question directly.
    - CoT (chain-of-thought): provide problem and question, and prompt the model to think step by step.

    Returns:
        A list of message dicts (role + content) to represent the conversation prompt.
    """
    problem = sample["problem"]
    question = sample["question"]
    if prompt_method == "zeroshot":
        # Zeroshot prompt: just the problem and question, expecting a direct answer.
        prompt_text = f"{problem}\nQuestion: {question}"
        return [{"role": "user", "content": prompt_text}]
    elif prompt_method == "cot":
        # CoT prompt: problem and question, plus an instruction to encourage step-by-step reasoning.
        prompt_text = (
            f"{problem}\n"
            f"Question: {question}\n"
            f'Let\'s think step by step. Please conclude your answer in the form: "The answer is ...".'
        )
        if random.random() < PRINT_SAMPLE_PROB:
            print(f"[DEBUG PROMPT] {prompt_text}\n{'='*50}")
        return [{"role": "user", "content": prompt_text}]
    else:
        raise ValueError(f"Invalid prompt_method: {prompt_method}")


def get_gsm_dataset(base_path: str, difficulty_list: list[str] = None, prompt_method: str = "cot") -> Dataset:
    """
    Load and combine the GSM-Infinite dataset from the local filesystem.

    Args:
        base_path (str): Root directory of GSM-Infinite dataset (contains subdirs like 'hard/', 'medium/').
        difficulty_list (list of str): Which difficulty subdirectories to include (e.g., ["hard", "medium"]).
                                       If None, include all subdirectories in base_path.
        prompt_method (str): "zeroshot" or "cot", determines how to format the prompt.

    Returns:
        Dataset: a HuggingFace Dataset object with the combined data. Each sample has:
            - "prompt": a conversation (list of role-content dicts) ready for the model.
            - "answer": a list of true answer strings (for reward computation).
            - "prompt_method": the prompt method used (same for all samples in this dataset).
            - "op": the 'op' value from the data (operation count or difficulty measure in GSM).
            - "template": the template category from the data.
            - "id": the sample ID.
    """
    if difficulty_list is None:
        # If no specific difficulties provided, use all directories in base_path
        difficulty_list = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    all_datasets = []
    for diff in difficulty_list:
        diff_dir = os.path.join(base_path, diff)
        if not os.path.isdir(diff_dir):
            continue  # skip if the difficulty directory doesn't exist
        # Iterate through each subdirectory (which represent different "op" values)
        for op_subdir in os.listdir(diff_dir):
            sub_dir_path = os.path.join(diff_dir, op_subdir)
            if not os.path.isdir(sub_dir_path):
                continue
            # Iterate through all JSONL files in this subdirectory
            for filename in os.listdir(sub_dir_path):
                if not filename.endswith(".jsonl"):
                    continue  # skip non-JSONL files
                jsonl_path = os.path.join(sub_dir_path, filename)
                # Load the JSON lines file as a HuggingFace dataset
                ds = load_dataset("json", data_files=jsonl_path, split="train")
                # Transform each entry of this dataset file to the desired format
                ds = ds.map(
                    lambda x: {
                        "prompt": get_prompt_for_sample(
                            x, prompt_method
                        ),  # conversation prompt for the model
                        "answer": extract_final_answer(x["solution"]),
                        "answers": [extract_final_answer(x["solution"])],  # true answer(s) as a list
                        "prompt_method": prompt_method,  # store the prompt method for reference
                        "op": x.get("op", None),  # operation count or difficulty level (if present)
                        "template": x.get("template", None),  # template type of the problem (if present)
                        "id": x.get("id", None),  # problem ID
                    }
                )
                if random.random() < 0.1:
                    print(f"[DEBUG DATASET SAMPLE] {ds}\n{'='*80}")
                all_datasets.append(ds)
    # Combine all small datasets into one
    if len(all_datasets) == 0:
        raise RuntimeError(
            "No data loaded from GSM-Infinite. Please check the base_path and difficulty_list."
        )
    combined_dataset = concatenate_datasets(all_datasets)
    return combined_dataset


############################################################
# Script Argument Definitions
############################################################


@dataclass
class GRPOScriptArguments(ScriptArguments):
    # Training dataset parameters
    train_dataset_path: str = ""  # Path to the root of GSM-Infinite dataset
    difficulty_list: list[str] = field(default_factory=lambda: ["hard", "medium"])
    # Evaluation dataset parameters (optional)
    eval_dataset_path: str = (
        ""  # Path for evaluation dataset (could be same as train path or a different subset)
    )
    eval_difficulty_list: list[str] = field(default_factory=lambda: [])
    # Reward and prompting settings
    reward_func_names: list[str] = field(default_factory=lambda: ["f1"])
    data_curriculum: Literal[
        "random",
        "difficulty_asc_stage_off",
        "difficulty_desc_stage_off",
        "difficulty_asc_stage_on",
        "difficulty_desc_stage_on",
    ] = "random"
    prompt_method: Literal["zeroshot", "cot"] = "zeroshot"


# Note: We keep the curriculum options in the Literal for compatibility,
# but only "random" is supported for GSM-Infinite.


def get_checkpoint(training_args: GRPOConfig):
    """
    Check if there is an existing checkpoint in the output directory to resume from.
    Returns the path to the last checkpoint if available, otherwise None.
    """
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def arrange_dataset(dataset: Dataset, data_curriculum: str, seed: int) -> Dataset:
    """
    Arrange (order or shuffle) the dataset based on the specified curriculum strategy.
    For GSM-Infinite, we primarily use random shuffle. Other strategies (difficulty-based)
    are not recommended.
    """
    if data_curriculum == "random":
        # Random shuffle of the dataset
        return dataset.shuffle(seed=seed)
    elif data_curriculum in ["difficulty_asc_stage_off", "difficulty_desc_stage_off"]:
        # Sort by difficulty if the field exists (not typically present in GSM-Infinite data)
        return dataset.sort("difficulty", reverse=(data_curriculum.startswith("difficulty_desc")))
    elif data_curriculum in ["difficulty_asc_stage_on", "difficulty_desc_stage_on"]:
        # Stage_on: sort by difficulty and replicate questions for curriculum staging
        sorted_dataset = dataset.sort("difficulty", reverse=(data_curriculum.startswith("difficulty_desc")))
        # If using staged curriculum, repeat each entry for num_train_epochs times
        # (staging easy -> hard per epoch)
        # (In GSM-Infinite usage, we generally avoid this complexity to not degrade
        # performance on easy questions.)
        repeated_indices = [
            i for i in range(len(sorted_dataset)) for _ in range(seed)
        ]  # Using seed as a proxy for num_train_epochs if needed
        return sorted_dataset.select(repeated_indices)
    else:
        # If an invalid curriculum is passed, default to random shuffle
        logger.warning(f"Unsupported curriculum '{data_curriculum}'. Using random shuffle instead.")
        return dataset.shuffle(seed=seed)


############################################################
# Main Training Function
############################################################


def train_grpo(script_args: GRPOScriptArguments, training_args: GRPOConfig, model_args: ModelConfig) -> None:
    """Main training routine for GRPO model using prompts from GSM-Infinite dataset."""
    # Load and prepare the training dataset
    train_dataset = get_gsm_dataset(
        script_args.train_dataset_path, script_args.difficulty_list, script_args.prompt_method
    )
    # Arrange the dataset (shuffle or sort) according to curriculum setting. We recommend random
    # for GSM-Infinite.
    if script_args.data_curriculum != "random":
        logger.warning(f"Curriculum '{script_args.data_curriculum}' is not fully supported for GSM-Infinite.")
    train_dataset = train_dataset.shuffle(seed=training_args.seed)

    logger.info(
        f"*** Loaded train dataset from {script_args.train_dataset_path} \
            (difficulties: {script_args.difficulty_list}) "
        f"with {len(train_dataset)} samples. Curriculum used: {script_args.data_curriculum}."
    )

    # Load and prepare the evaluation dataset, if provided
    eval_dataset = None
    if script_args.eval_dataset_path:
        # If eval difficulties not specified, use the same list as training
        eval_diffs = script_args.eval_difficulty_list or script_args.difficulty_list
        eval_dataset = get_gsm_dataset(script_args.eval_dataset_path, eval_diffs, script_args.prompt_method)
        logger.info(
            f"*** Loaded eval dataset from {script_args.eval_dataset_path} (difficulties: {eval_diffs}) "
            f"with {len(eval_dataset)} samples."
        )
    else:
        logger.info("*** No eval dataset specified, proceeding without separate evaluation data.")

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    # Ensure the tokenizer has a pad token (if not, assign the EOS token as pad)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare the list of reward function callables based on names provided
    reward_funcs = [get_reward_func(name) for name in script_args.reward_func_names]
    logger.info(f"*** Selected reward functions: {script_args.reward_func_names}")

    # Set up model loading arguments
    logger.info("*** Initializing model with specified configurations ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": torch_dtype,
        "use_cache": False if training_args.gradient_checkpointing else True,
        "quantization_config": get_quantization_config(model_args) if model_args.use_peft else None,
    }
    logger.info(f"*** Model loading kwargs: {model_kwargs} ***")

    # Load the base model (for causal LM) from HuggingFace
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    # If quantization is enabled (for 4-bit or 8-bit training), prepare the model for low-bit training
    if model_kwargs.get("quantization_config", None) is not None:
        logger.info("*** Preparing model for low-bit (k-bit) training ***")
        model = prepare_model_for_kbit_training(model)
    # If using PEFT (Parameter-Efficient Fine-Tuning, e.g., LoRA), wrap the model with PEFT adapters
    if model_args.use_peft:
        logger.info("*** Applying PEFT (e.g., LoRA) configuration ***")
        lora_config = get_peft_config(model_args)
        model = get_peft_model(model, lora_config)

    # Instantiate the GRPO Trainer with model, datasets, tokenizer, and reward functions
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
    )

    logger.info(f"*** Initialized GRPOTrainer for model {model_args.model_name_or_path} ***")

    # Resume from checkpoint if available
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"*** Resuming training from checkpoint {last_checkpoint} ***")

    # Begin training
    logger.info(
        f"*** Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"for {training_args.num_train_epochs} epoch(s) ***"
    )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    print(f"[DEBUG] train_result: {train_result}")

    # After training, log and save final metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("*** Training complete ***")

    ########################################################
    # Save the fine-tuned model and tokenizer, and optionally push to hub
    ########################################################
    logger.info("*** Saving model and tokenizer ***")
    trainer.model.config.use_cache = True  # re-enable cache if it was disabled for training
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"*** Model and tokenizer saved to {training_args.output_dir} ***")

    # Only the main process should create a model card or push to HuggingFace Hub
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl", "grpo", "gsm-infinite"]})
    if training_args.push_to_hub:
        logger.info("*** Pushing model to the Hugging Face Hub ***")
        trainer.push_to_hub()


# Entry point: parse arguments and start training
if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # Setup logging level
    logging_level = training_args.log_level.upper() if hasattr(training_args, "log_level") else "INFO"
    logging.basicConfig(level=logging_level)
    # Begin training
    train_grpo(script_args, training_args, model_args)
