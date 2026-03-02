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
from cot_examples import COT_EXAMPLES_2WIKI, COT_EXAMPLES_HP, COT_EXAMPLES_MSQ
from phantom_eval.agents import Agent
from phantom_eval.llm import Conversation, InferenceGenerationConfig, LLMChat, get_llm

# Import CoT baseline prompts
from phantom_eval.prompts import LLMPrompt, get_llm_prompt
from phantom_eval.utils import setup_logging
from utils.agent_utils import CoTWikiAgent
from utils.data_utils import get_parser, load_data

logger = logging.getLogger(__name__)


def get_agent_kwargs(args: ArgumentParser, llm_prompt: LLMPrompt) -> dict:
    """Get agent initialization kwargs based on method type."""
    match args.method:
        case "zeroshot":
            raise NotImplementedError("Zeroshot is not implemented")
        case "fewshot":
            raise NotImplementedError("Fewshot is not implemented")
        case "cot":
            match args.dataset:
                case "hp" | "hp500":
                    cot_examples = COT_EXAMPLES_HP
                case "2wiki" | "2wiki500":
                    cot_examples = COT_EXAMPLES_2WIKI
                case "msq" | "msq500":
                    cot_examples = COT_EXAMPLES_MSQ
                case "cofca500":
                    logger.info(
                        "Using HotpotQA CoT examples for CofCA, because CofCA does not have training data."
                    )
                    cot_examples = COT_EXAMPLES_HP
                case "synthrm500":
                    logger.info(
                        "Using MuSiQue CoT examples for SynthWorlds-RM, "
                        "because SynthWorlds-RM does not have training data."
                    )
                    cot_examples = COT_EXAMPLES_MSQ
            agent_kwargs = dict(llm_prompt=llm_prompt, cot_examples=cot_examples)
        case _:
            agent_kwargs = dict()
    return agent_kwargs


def get_model_kwargs(args: ArgumentParser) -> dict:
    match args.server:
        case "vllm":
            model_kwargs = dict(
                max_model_len=args.inf_vllm_max_model_len,
                tensor_parallel_size=args.inf_vllm_tensor_parallel_size,
                use_api=False,  # NOTE: we use offline inference to maximize throughput
            )
        case _:
            raise ValueError(f"Invalid server: {args.server}")
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

    logger.info("Loading agent kwargs")
    if args.no_evidence:
        logger.info("*** No evidence: using empty text_corpus ***")

    corpora: list[pd.DataFrame] = []
    for i, row in df_qa_pairs.iterrows():
        titles = df_qa_pairs.iloc[i]["title"]
        articles = df_qa_pairs.iloc[i]["article"]
        if args.no_evidence:
            # Empty text_corpus
            text_corpus = pd.DataFrame(columns=["title", "article"])
        else:
            text_corpus = pd.DataFrame({"title": titles, "article": articles})
        corpora.append(text_corpus)

    logger.info("Running agent loop")
    assert len(corpora) == len(df_qa_pairs), "corpora must be a list of the same length as df_qa_pairs"

    # Get the LLM chat
    llm_chat: LLMChat = get_llm(args.server, args.model_name, model_kwargs=model_kwargs)

    # Get the split, seed, and batch size
    split = args.split
    # Use one batch for offline inference
    batch_size = num_df_qa_pairs = len(df_qa_pairs)

    for seed in args.inf_seed_list:
        logger.info("Loading inference generation config")
        inf_gen_config: InferenceGenerationConfig = InferenceGenerationConfig(
            max_tokens=args.inf_max_tokens,
            temperature=args.inf_temperature,
            top_k=args.inf_top_k,
            top_p=args.inf_top_p,
            repetition_penalty=args.inf_repetition_penalty,
            max_retries=args.inf_max_retries,
            wait_seconds=args.inf_wait_seconds,
            seed=seed,
        )
        logger.info(f"Inference generation config: {inf_gen_config}")

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
            batch_corpora = corpora[batch_start_idx:batch_end_idx]

            # Construct agent
            agent_kwargs: dict = get_agent_kwargs(args=args, llm_prompt=llm_prompt)
            agent: Agent = CoTWikiAgent(**agent_kwargs)

            # Run predictions in parallel
            match args.method:
                case "cot":
                    responses = await agent.batch_run(
                        llm_chat=llm_chat,
                        questions=batch_df_qa_pairs["question"].tolist(),
                        inf_gen_config=inf_gen_config,
                        corpora=batch_corpora,
                    )
                    agent_interactions: list[Conversation] = agent.agent_interactions
                case "zeroshot" | "fewshot":
                    raise NotImplementedError("Zeroshot and fewshot are not implemented")
                case _:
                    raise ValueError(f"Invalid method: {args.method}")

            # Collect predictions for this batch
            preds = {}
            for i, (_, instance) in enumerate(batch_df_qa_pairs.iterrows()):
                uid = instance.id
                interaction = agent_interactions[i].model_dump()
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
                        for k, v in agent_kwargs.items()
                        if k
                        not in [
                            "llm_prompt",
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
    parser.add_argument(
        "--no_evidence",
        action="store_true",
        help="run without any evidence (empty text_corpus)",
    )
    args = parser.parse_args()
    args.server = "vllm"  # NOTE: we use vllm with offline inference to maximize throughput
    setup_logging(args.log_level)
    asyncio.run(main(args))
