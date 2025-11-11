import argparse
import glob
import json
import os

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from langchain.prompts import PromptTemplate
from model_handler import ModelHandler
from no_rag_pipeline import NoRAGPipeline

from phantom_reasoner.datasets_for_grpo import GSMInfiniteDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample with command line arguments.")
    parser.add_argument("--output-dir", "-od", type=str, help="Output directory", default="out")
    parser.add_argument("--save-name", type=str, help="Save model name", default="base")
    parser.add_argument("--save-dataset", type=str, help="Save dataset name", default="base")
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="The name of the dataset for organizing the folders",
    )
    # Required arguments
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to use in api call.",
    )
    parser.add_argument(
        "--backend-type",
        type=str,
        default="openai",
        help="backend type in ['openai', 'anthropic', 'gemini']",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate per example.",
    )

    # Optional arguments with default values
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: None).",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3072,
        help="Maximum number of tokens (default: 3072).",
    )

    parser.add_argument("--batch-size", type=int, default=200, help="Batch size (default: 200).")

    parser.add_argument("--length", type=str, default="0", help="noise context length")

    parser.add_argument("--limit", type=int, default=100, help="max number of examples per op")

    parser.add_argument(
        "--filter-config",
        type=json.loads,
        help="Filter configuration as a JSON string.",
    )

    parser.add_argument(
        "--op-range",
        type=str,
        help="Operating range, can be an integer, or a list of integers separated by commas.",
    )
    args = parser.parse_args()

    if args.op_range:
        try:
            # Attempt to parse as a single integer
            args.op_range = [int(args.op_range)]
        except ValueError:
            # If not a single integer, split by comma and convert to integers
            try:
                args.op_range = [int(x.strip()) for x in args.op_range.split(",")]
            except ValueError:
                raise ValueError(
                    "Invalid input for --op-range. Please provide an integer \
                        or a comma-separated list of integers."
                )

    subsets = [f"ops_{x}" for x in args.op_range]
    use_full_query = True

    model_handler = ModelHandler(model_name=args.model_name, backend_type=args.backend_type)
    pipeline = NoRAGPipeline(
        model_handler=model_handler,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    use_full_query = True

    # for length in [0, 8000, 16000, 32000, 64000, 128000]:
    length = args.length
    try:
        if os.path.isdir(args.dataset_name):
            base = os.path.join(args.dataset_name, "zero_context", "realistic", "medium")
            requested = {
                f"ops_{k}": sorted(glob.glob(os.path.join(base, str(k), "*.jsonl"))) for k in args.op_range
            }

            present = {k: v for k, v in requested.items() if v}
            missing = [k for k, v in requested.items() if not v]
            if missing:
                print(f"[warn] missing splits skipped: {missing}")
            if not present:
                raise FileNotFoundError(f"No JSONL found under {base}/<op>/*.jsonl")

            full_dataset = load_dataset("json", data_files=present)

            subsets = list(full_dataset.keys())
        else:
            full_dataset = load_dataset(f"{args.dataset_name}_{length}")

        # === Normalize local/remote dataset to ensure 'messages' exists ===
        def _ensure_messages(ex):
            prompt_template = PromptTemplate(
                input_variables=["problem", "examples", "question"],
                template=GSMInfiniteDataset.COT_INSTRUCTION,
            )
            ex["messages"] = [
                {
                    "role": "user",
                    "content": prompt_template.format(
                        problem=ex["problem"],
                        examples=GSMInfiniteDataset.COT_EXAMPLES,
                        question=ex["question"],
                    ),
                },
            ]
            return ex

        # map normalization over all splits
        if isinstance(full_dataset, DatasetDict):
            for sname in list(full_dataset.keys()):
                full_dataset[sname] = full_dataset[sname].map(
                    _ensure_messages, desc=f"ensure_messages::{sname}"
                )
        else:
            full_dataset = full_dataset.map(_ensure_messages, desc="ensure_messages::single")

        filter_config = args.filter_config
        if filter_config:
            filtered_datasets = []
            for split in subsets:
                dataset_split = full_dataset[split]
                total_samples = min(args.limit, len(dataset_split))
                filtered_data = []
                for config in filter_config:
                    num_to_add = int(total_samples * config["percentage"])
                    current_filter = {
                        key: value for key, value in config.items() if key not in ["percentage"]
                    }
                    filtered_subset = dataset_split.filter(
                        lambda example: all(
                            example.get(key) == value for key, value in current_filter.items()
                        )
                    )
                    filtered_data.extend(filtered_subset.select(range(min(num_to_add, len(filtered_subset)))))
                filtered_datasets.append(Dataset.from_list(filtered_data))
            unprocessed_dataset = concatenate_datasets(filtered_datasets)
        else:
            unprocessed_dataset = concatenate_datasets(
                [
                    full_dataset[split].select(range(min(args.limit, len(full_dataset[split]))))
                    for split in subsets
                ]
            )

        # TODO change the queries to elicit <answer> and </answer>
        len_dataset = len(unprocessed_dataset)
        contexts = []
        queries = []
        for i in range(0, len_dataset):
            for _ in range(args.num_samples):
                queries.append(unprocessed_dataset[i]["messages"])
        print(json.dumps(unprocessed_dataset[0]["messages"], ensure_ascii=False, indent=2))
        replies = pipeline.process_batch(queries=queries, max_workers=args.batch_size)
        processed_examples = []

        for i in range(0, len_dataset):
            newline = unprocessed_dataset[i]
            newline["replies"] = replies[i * args.num_samples : (i + 1) * args.num_samples]
            newline.pop("problem", "")
            newline.pop("question", "")
            newline.pop("messages", "")
            processed_examples.append(newline)

        os.makedirs(args.output_dir, exist_ok=True)  # Create directory if it doesn't exist

        save_file_path = os.path.join(args.output_dir, f"{args.save_dataset}-{args.save_name}_{length}.json")
        with open(save_file_path, "w") as f:
            json.dump(processed_examples, f, indent=4)
        print(f"Successfully saved generations to {save_file_path}")

    except Exception as e:
        print(e)
        raise
