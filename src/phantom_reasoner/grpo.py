from dataclasses import dataclass

from datasets import Dataset
from phantom_eval.agents.common import get_all_evidence
from phantom_eval.prompts import ZeroshotLLMPrompt
from phantom_eval.utils import load_data
from transformers import AutoTokenizer
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser

from phantom_reasoner.configs import GRPOConfig
from phantom_reasoner.utils.score import answer_sep, exact_match, f1, precision, recall


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
        float(exact_match(completion[0]["content"], a.join(answer_sep)))
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
        float(precision(completion[0]["content"], a.join(answer_sep)))
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
        float(recall(completion[0]["content"], a.join(answer_sep)))
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
        float(f1(completion[0]["content"], a.join(answer_sep))) for completion, a in zip(completions, answer)
    ]


REWARD_FUNC_TYPE2FUNC = {
    "exact_match": reward_exact_match,
    "precision": reward_precision,
    "recall": reward_recall,
    "f1": reward_f1,
}


@dataclass
class GRPOScriptArguments(ScriptArguments):
    dataset_name: str
    split_name: str = "depth_20_size_50_seed_1"
    from_local: bool = False
    reward_func_types: list[str] = ["exact_match"]


def get_pw_train_dataset(dataset_name: str, split_name: str, from_local: bool) -> Dataset:
    dataset: dict[str, Dataset] = load_data(
        script_args.dataset_name, split=script_args.split_name, from_local=script_args.from_local
    )
    text_corpus: Dataset = dataset["text"]
    question_answer: Dataset = dataset["qa_pairs"]

    evidence: str = get_all_evidence(text_corpus)

    llm_prompt = ZeroshotLLMPrompt()

    train_dataset: Dataset = question_answer.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": llm_prompt.get_prompt(evidence=evidence, question=x["question"])},
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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    # GRPOConfig options from @willccbb's script
    # TODO is any of this needed?
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
        REWARD_FUNC_TYPE2FUNC[reward_func_type] for reward_func_type in script_args.reward_func_types
    ]

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # TODO: add eval dataset
        processing_class=tokenizer,
        # peft_config=peft_config  # "Use at your own risk, didn't work for multi-GPU setup"
        # TODO make reward_funcs an argument to the training script
        reward_funcs=reward_funcs,
        # TODO additional options from @willccbb's script
    )
    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    train_grpo(script_args, training_args, model_args)
