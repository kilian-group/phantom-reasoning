"""
Script for evaluating the accuracy of the generated answers on the GSM-Infinite dataset.

Example usage:
```bash
python eval_realistic.py --output-dir OUTPUT_DIR --model-name MODEL_NAME --save-dataset SAVE_DATASET
```
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from phantom_eval.agents.cot import CoTAgent
from tabulate import tabulate


def criteriaoutput(generatedtext: list[str], inputexample: dict) -> tuple[int, list[int | None]]:
    """
    Returns a tuple of (number of correct answers, list of generated answers).
    Each generated text is parsed and matched with the answer text from the solution.
    """
    num_correct_answers = 0
    all_generated_answers: list[int | None] = []
    for i in range(len(generatedtext)):
        # Get the answer text from solution
        idx_answer_start = inputexample["solution"].find("Answer: ")
        idx_answer_end = inputexample["solution"].find(".", idx_answer_start)
        answer_text = inputexample["solution"][idx_answer_start + len("Answer: ") : idx_answer_end]
        answer_text = int(answer_text.lower())

        try:
            # Parse the answer from the generated text
            answer_generated_text = int(CoTAgent.parse_answer(generatedtext[i]))
            num_correct_answers += int(answer_generated_text == answer_text)
        except Exception:
            answer_generated_text = None
            # print(f"[PARSE_ERR] {type(e).__name__}: {e}")
        finally:
            all_generated_answers.append(answer_generated_text)

    return num_correct_answers, all_generated_answers


def postprocess_line(input_example: dict, generatedtext: list[str]) -> dict:
    """
    Postprocess the input example dictionary with the generated text, adding two keys:
    - correct_num: number of correct answers
    - reply_answers: list of generated answers
    """
    num_correct_answers, all_generated_answers = criteriaoutput(generatedtext, input_example)
    input_example["correct_num"] = num_correct_answers
    input_example["reply_answers"] = all_generated_answers
    return input_example


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval with command line arguments.")
    parser.add_argument("--output-dir", "-od", type=str, help="Output directory", default="out")
    parser.add_argument("--save-dataset", type=str, help="Save dataset name", default="medium")
    parser.add_argument("--length", type=str, default="0", help="noise context length")
    parser.add_argument(
        "--model-name",
        type=str,
        help="The name or path of the model to evaluate",
        default="Qwen/Qwen3-0.6B",
    )

    args = parser.parse_args()

    length = args.length
    try:
        file_path = os.path.join(
            args.output_dir, f"{args.save_dataset}-{args.model_name.replace('/', '--')}_{str(length)}.json"
        )
        with open(file_path) as f:
            unprocessed_dataset = json.load(f)

        def process_dataset(unprocessed_dataset):
            results = []
            count_dict = {}
            correct_dict = {}

            submission_list = []
            num_samples = len(unprocessed_dataset[0]["replies"])

            len_dataset = len(unprocessed_dataset)

            for i in range(len_dataset):
                submission_list.extend(unprocessed_dataset[i]["replies"])
                results.append(
                    postprocess_line(
                        unprocessed_dataset[i],
                        [submission_list[j] for j in range(i * num_samples, (i + 1) * num_samples)],
                    )
                )

            for processed_example in results:
                op = processed_example["op"]
                count_dict.setdefault(op, 0)
                correct_dict.setdefault(op, [])

                count_dict[op] += 1
                correct_dict[op].append(
                    processed_example["correct_num"] / len(processed_example["reply_answers"])
                )

            sorted_keys = sorted(count_dict.keys())

            save_file_path = os.path.join(
                args.output_dir, "scores", f"{args.save_dataset}_{args.model_name.replace('/', '--')}.csv"
            )
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
            # Save the scores per op to a csv file
            records = []
            for op in sorted_keys:
                records.append(
                    {
                        "model_name": args.model_name,
                        "difficulty": op,
                        "accuracy": np.mean(correct_dict[op]),
                        "num_correct": np.sum(correct_dict[op]),
                        "count": len(correct_dict[op]),
                    }
                )
            df = pd.DataFrame(records)
            df.to_csv(save_file_path, index=False)
            print(f"Saved to {save_file_path}")

            # Print the overall accuracy by model name
            acc = df.groupby("model_name")["num_correct"].agg("sum") / df.groupby("model_name")["count"].agg(
                "sum"
            )
            print(tabulate(acc.reset_index(), headers="keys", tablefmt="github"))

        process_dataset(unprocessed_dataset)

    except Exception as e:
        print(e)
        raise
