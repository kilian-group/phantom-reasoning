# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

from phantom_reasoner.utils.score import exact_match

from ._types import LLMChatResponse


def reward_exact_match(completions: LLMChatResponse, **kwargs) -> list[float]:
    """Reward exact answer match."""
    # TODO I'm assuming output is of LLMChatResponse type--which is probably wrong?
    return [1.0 if exact_match(completion.pred, completion.true) else 0.0 for completion in completions]


def train_grpo(script_args, training_args, model_args):
    # TODO types of these arguments and convenions
    #   e.g. are these dictionaries?
    """Training script for the GRPO model.

    script_args:
        dataset_name: HuggingFace dataset on which the training occurs


    training_args:
        logging_steps
        output_dir: path to the training outputs (e.g. checkpoints and logs)

    model_args:
        model_name_or_path

    """
    dataset = load_dataset(script_args.dataset_name, split="train")

    # GRPOConfig options from @willccbb's script
    # TODO clean up---what is the smallest sufficient config?
    #   (HF tutorial only suggests output_dir and logging_steps)
    training_args_ = GRPOConfig(
        output_dir=training_args.output_dir,
        run_name=training_args.run_name,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,  # HF tutorial suggests 10
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=16,
        max_prompt_length=256,
        max_completion_length=786,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        # report_to="wandb",
        log_on_each_node=False,
    )
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

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        # TODO make reward_funcs an argument to the training script
        reward_funcs=reward_exact_match,
        train_dataset=dataset,
        # TODO additional options from @willccbb's script
        args=training_args_,
        # processing_class=tokenizer,
        # args=training_args,
        # train_dataset=dataset,
        # peft_config=peft_config  # "Use at your own risk, didn't work for multi-GPU setup"
    )
    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a GRPO model")
    
    # Script arguments
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="HuggingFace dataset name on which the training occurs")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the training outputs (e.g. checkpoints and logs)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name of the run for logging purposes")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Number of steps between logging updates")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path or name of the model to fine-tune")
    
    # Additional GRPO specific arguments
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate for training")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per GPU/TPU core/CPU for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--num_train_epochs", type=float, default=1.0,
                        help="Total number of training epochs to perform")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every X updates steps")
    
    return parser.parse_args()


def main():
    """Main function to parse arguments and train the GRPO model."""
    args = parse_args()
    
    # Create the dataclass objects from parsed arguments
    script_args = ScriptArgs(
        dataset_name=args.dataset_name,
    )
    
    training_args = TrainingArgs(
        output_dir=args.output_dir,
        run_name=args.run_name,
        logging_steps=args.logging_steps,
    )
    
    model_args = ModelArgs(
        model_name_or_path=args.model_name_or_path,
    )
    
    # Call the training function
    train_grpo(script_args, training_args, model_args)


if __name__ == "__main__":
    main()
