"""Script for generating final predictions on HotpotQA dataset.

Example usage:
```bash
python pred.py -od OUTPUT_DIR --dataset hp --split minidev --method cot --server vllm -m qwen/qwen3-1.7b
```
"""

import asyncio
import json
import logging
import math
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from data_utils import get_parser, load_data
from few_shot_examples import COT_EXAMPLES_2WIKI, COT_EXAMPLES_HP, COT_EXAMPLES_MSQ
from phantom_eval.agents import Agent, get_agent
from phantom_eval.llm import Conversation, InferenceGenerationConfig, LLMChat, get_llm

# Import CoT baseline prompts
from phantom_eval.prompts import LLMPrompt, get_llm_prompt
from phantom_eval.utils import setup_logging
from tqdm.asyncio import tqdm as tqdm_async

logger = logging.getLogger(__name__)


def get_agent_kwargs(args: ArgumentParser, text_corpus: pd.DataFrame, llm_prompt: LLMPrompt) -> dict:
    """Get agent initialization kwargs based on method type."""
    match args.method:
        case "zeroshot":
            raise NotImplementedError("Zeroshot is not implemented")
        case "fewshot":
            raise NotImplementedError("Fewshot is not implemented")
        case "cot":
            match args.dataset:
                case "hp":
                    cot_examples = COT_EXAMPLES_HP
                case "2wiki":
                    cot_examples = COT_EXAMPLES_2WIKI
                case "msq":
                    cot_examples = COT_EXAMPLES_MSQ
            agent_kwargs = dict(text_corpus=text_corpus, llm_prompt=llm_prompt, cot_examples=cot_examples)
        case _:
            agent_kwargs = dict()
    return agent_kwargs


def get_model_kwargs(args: ArgumentParser) -> dict:
    match args.server:
        case "vllm":
            model_kwargs = dict(
                max_model_len=args.inf_vllm_max_model_len,
                tensor_parallel_size=args.inf_vllm_tensor_parallel_size,
                # NOTE: for simplicity, we will always use the vLLM server API for inference
                # This reduces the prompt throughput somewhat over the offline batch inference
                # But simplifies the code by avoiding the need to handle vLLM differently
                # from other models.
                use_api=True,
                port=args.inf_vllm_port,
                is_deepseek_r1_model=args.inf_is_deepseek_r1_model,
            )
        case _:
            model_kwargs = dict(
                usage_tier=args.inf_usage_tier,
                enforce_rate_limits=not args.inf_relax_rate_limits,
                llms_rpm_tpm_config_fpath=args.inf_llms_rpm_tpm_config_fpath,
            )
    return model_kwargs


async def main(args: ArgumentParser) -> None:
    logger.info(f"Loading LLM='{args.model_name}'")
    model_kwargs: dict = get_model_kwargs(args)
    logger.info(f"Model kwargs: {model_kwargs}")

    logger.info(f"Loading dataset='{args.dataset}'")
    dataset: dict[str, list[dict]] = load_data(args.data_dir, args.dataset, args.split)
    df_qa_pairs: pd.DataFrame = pd.DataFrame(dataset["qa_pairs"])
    df_text: pd.DataFrame = pd.DataFrame(dataset["text"])
    # Merge qa_pairs with text to get articles for each question
    df_qa_pairs = df_qa_pairs.merge(
        df_text[["id", "article", "title"]],
        on="id",
        how="left",
        suffixes=("", "_text"),
    )

    logger.info(f"Loading prompt for method='{args.method}'")
    llm_prompt: LLMPrompt = get_llm_prompt(args.method, args.model_name)
    logger.info(f"LLM prompt: {llm_prompt}")

    logger.info("Loading inference generation config")
    inf_gen_config: InferenceGenerationConfig = InferenceGenerationConfig(
        max_tokens=args.inf_max_tokens,
        temperature=args.inf_temperature,
        top_k=args.inf_top_k,
        top_p=args.inf_top_p,
        repetition_penalty=args.inf_repetition_penalty,
        max_retries=args.inf_max_retries,
        wait_seconds=args.inf_wait_seconds,
    )
    logger.info(f"Inference generation config: {inf_gen_config}")

    logger.info("Loading agent kwargs")
    agent_kwargs_list: list[dict] = []
    for i, row in df_qa_pairs.iterrows():
        titles = df_qa_pairs.iloc[i]["title"]
        articles = df_qa_pairs.iloc[i]["article"]
        text_corpus = pd.DataFrame({"title": titles, "article": articles})
        agent_kwargs: dict = get_agent_kwargs(
            args=args,
            text_corpus=text_corpus,
            llm_prompt=llm_prompt,
        )
        agent_kwargs_list.append(agent_kwargs)

    logger.info("Running agent loop")
    assert len(agent_kwargs_list) == len(
        df_qa_pairs
    ), "agent_kwargs_list must be a list of kwargs for each question"

    # Get the LLM chat
    llm_chat: LLMChat = get_llm(args.server, args.model_name, model_kwargs=model_kwargs)

    # Get the split, seed, and batch size
    split = args.split
    seed = args.seed
    num_df_qa_pairs = len(df_qa_pairs)
    batch_size = args.batch_size

    logger.info(f"Running inference for method='{args.method}' with {seed=}")
    for batch_number in range(1, math.ceil(num_df_qa_pairs / batch_size) + 1):
        run_name = (
            f"dataset={args.dataset}"
            + f"__split={split}"
            + f"__model_name={args.model_name.replace('/', '--')}"
            + f"__bs={batch_size}"
            + f"__bn={batch_number:03d}"
            + f"__seed={seed}"
        )
        pred_path = Path(args.output_dir) / "preds" / args.method / f"{run_name}.json"

        # Skip if the batch number is not the one specified
        if (args.batch_number is not None) and (batch_number != args.batch_number):
            continue
        # Skip if the output file already exists and --force is not set
        if pred_path.exists() and not args.force:
            logger.info(f"Skipping {pred_path} as it already exists. Use --force to overwrite.")
            continue

        # Get batch
        batch_start_idx = (batch_number - 1) * batch_size
        batch_end_idx = batch_start_idx + batch_size
        logger.info(
            f"Getting predictions for questions [{batch_start_idx}, {batch_end_idx})"
            f" out of {num_df_qa_pairs}"
        )
        batch_df_qa_pairs = df_qa_pairs.iloc[batch_start_idx:batch_end_idx]

        # Construct agent for each question
        agents: list[Agent] = [
            get_agent(
                args.method,
                text_corpus=agent_kwargs_list[i]["text_corpus"],
                llm_prompt=agent_kwargs_list[i]["llm_prompt"],
                agent_kwargs={
                    k: v for k, v in agent_kwargs_list[i].items() if k not in ["text_corpus", "llm_prompt"]
                },
            )
            for i in range(batch_start_idx, min(batch_end_idx, num_df_qa_pairs))
        ]

        # Run predictions in parallel
        match args.method:
            case "zeroshot" | "fewshot" | "cot":
                responses = await tqdm_async.gather(
                    *[
                        agent.run(
                            llm_chat=llm_chat,
                            question=instance.question,
                            inf_gen_config=inf_gen_config,
                        )
                        for agent, (_, instance) in zip(agents, batch_df_qa_pairs.iterrows())
                    ]
                )
            case _:
                raise ValueError(f"Invalid method: {args.method}")
        # Get the agent interactions
        agent_interactions: list[Conversation | None] = [agent.agent_interactions for agent in agents]

        # Collect predictions for this batch
        preds = {}
        for i, (_, instance) in enumerate(batch_df_qa_pairs.iterrows()):
            uid = instance.id
            if agent_interactions:
                if isinstance(agent_interactions[i], Conversation):
                    interaction = agent_interactions[i].model_dump()
                elif isinstance(agent_interactions[i], list):
                    # interaction = {
                    #     "planning": agent_interactions[i][0].model_dump(),
                    #     "executing": agent_interactions[i][1].model_dump(),
                    # }
                    interaction = [x.model_dump() for x in agent_interactions[i]]
                elif agent_interactions[i] is None:
                    interaction = []
                else:
                    raise ValueError(f"Invalid agent interaction: {agent_interactions[i]}")
            else:
                interaction = []
            preds[uid] = {
                "true": instance.answer,
                "pred": responses[i].pred,
                "error": responses[i].error,
                "interaction": interaction,
                "metadata": {
                    "model": args.model_name,
                    "dataset": args.dataset,
                    "split": split,
                    "batch_size": batch_size,
                    "batch_number": batch_number,
                    "type": instance.type,
                },
                "inference_params": inf_gen_config.model_dump(),
                "model_kwargs": model_kwargs,
                # HACK: remove entries from agent_kwargs that are not JSON serializable
                "agent_kwargs": {
                    k: v
                    for k, v in agent_kwargs_list[i].items()
                    if k
                    not in [
                        "llm_prompts",
                        "planning_llm_prompt",
                        "executing_llm_prompt",
                        "text_corpus",
                        "llm_prompt",
                        "db",
                        "agent_constructor",
                    ]
                },
                "usage": responses[i].usage,
            }

        # Save all predictions after processing completes
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving all predictions to {pred_path}")
        with open(pred_path, "w") as f:
            json.dump(preds, f, indent=4)
            f.flush()

    logger.info("Agent loop complete")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    asyncio.run(main(args))
