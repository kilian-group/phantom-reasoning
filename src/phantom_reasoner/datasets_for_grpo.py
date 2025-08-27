"""
Implements dataset generators/loaders for GRPO training.
"""

import abc
import glob
import logging
import os
import re
from typing import Any

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset
from langchain.prompts import PromptTemplate
from phantom_eval.agents.common import get_all_evidence
from phantom_eval.prompts import COT_EXAMPLES, CoTLLMPrompt, ZeroshotLLMPrompt
from phantom_eval.utils import load_data as load_pw_data

from phantom_reasoner._types import CONVO_T
from phantom_reasoner.configs import GRPOScriptArguments
from phantom_reasoner.utils.data_utils import load_hp_data

logger = logging.getLogger(__name__)


class DatasetForGRPO(abc.ABC):
    def __init__(self, script_args: GRPOScriptArguments) -> None:
        self.script_args = script_args

    @abc.abstractmethod
    def get_dataset(self, is_eval: bool) -> Dataset:
        """
        Get the dataset for training or evaluation.
        """

    @staticmethod
    @abc.abstractmethod
    def get_prompt_for_sample(sample: dict[str, Any], prompt_method: str, **kwargs) -> CONVO_T:
        """
        Get the prompt for a sample, depending on the prompt method.

        Args:
            sample (dict[str, Any]): A sample from the dataset, with key "question".
            prompt_method (str): Either "zeroshot" or "cot".

        Returns:
            CONVO_T: A list of messages for the conversational-style prompt.
                Each message is a dict with keys "role" and "content".
        """


class PhantomWikiDataset(DatasetForGRPO):
    def __init__(self, script_args: GRPOScriptArguments):
        super().__init__(script_args)

    def get_dataset(self, is_eval: bool) -> Dataset:
        if is_eval:
            dataset_name = self.script_args.eval_dataset_name
            split_list = self.script_args.eval_split_list
            from_local = self.script_args.eval_from_local
        else:
            dataset_name = self.script_args.dataset_name
            split_list = self.script_args.split_list
            from_local = self.script_args.from_local

        all_datasets: list[Dataset] = []
        for split_name in split_list:
            dataset: dict[str, Dataset] = load_pw_data(
                dataset_name,
                split=split_name,
                from_local=from_local,
                exclude_aggregation_questions=self.script_args.exclude_aggregation_questions,
            )
            text_corpus: Dataset = dataset["text"]
            qa_pairs: Dataset = dataset["qa_pairs"]
            evidence: str = get_all_evidence(text_corpus)

            dataset: Dataset = qa_pairs.map(
                lambda sample: {
                    "prompt": PhantomWikiDataset.get_prompt_for_sample(
                        sample, self.script_args.prompt_method, evidence=evidence
                    ),
                    "answer": sample["answer"],  # x['answer'] is a list of strings
                    "prompt_method": self.script_args.prompt_method,
                }
            )
            all_datasets.append(dataset)

        dataset = concatenate_datasets(all_datasets)
        logger.info(
            f"*** Loaded {is_eval=} dataset {self.script_args.dataset_name}::{self.script_args.split_list} "
            f"with {len(dataset)} samples."
        )
        return dataset

    @staticmethod
    def get_prompt_for_sample(sample: dict[str, Any], prompt_method: str, evidence: str = "") -> CONVO_T:
        """
        Get the prompt for a PhantomWiki sample, depending on the prompt method.

        Args:
            sample (dict[str, Any]): A sample from the dataset, with key "question".
            prompt_method (str): Either "zeroshot" or "cot".
            evidence (str): The evidence text to include in the prompt. Default is "".

        Returns:
            CONVO_T: A list of messages for the conversational-style prompt.
                Each message is a dict with keys "role" and "content".
        """
        match prompt_method:
            case "zeroshot":
                llm_prompt = ZeroshotLLMPrompt()
                prompt = [
                    {
                        "role": "user",
                        "content": llm_prompt.get_prompt().format(
                            evidence=evidence, question=sample["question"]
                        ),
                    },
                ]
                return prompt

            case "cot":
                llm_prompt = CoTLLMPrompt()
                return [
                    {
                        "role": "user",
                        "content": llm_prompt.get_prompt().format(
                            evidence=evidence, examples=COT_EXAMPLES, question=sample["question"]
                        ),
                    },
                ]
            case _:
                raise ValueError(f"Invalid {prompt_method=}")


class GSMInfiniteDataset(DatasetForGRPO):
    def __init__(self, script_args: GRPOScriptArguments):
        super().__init__(script_args)

    COT_INSTRUCTION = f"""
    You are given the following problem:
    (BEGIN PROBLEM)
    {{problem}}
    (END PROBLEM)

    You will be provided a question on the above problem. Your response must end with the final answer enclosed in tags: <answer>FINAL_ANSWER</answer>

    Here, FINAL_ANSWER must be a number.

    Here are some examples:
    (START OF EXAMPLES)
    {{examples}}
    (END OF EXAMPLES)

    Question: {{question}}
    Answer: """  # noqa: F541, E501

    # From /share/nikola/phantom-reasoning/data/gsm-infinite-train.zip
    # taken from
    # - igsm_op2_ip20_force_True_15.jsonl
    # - igsm_op7_ip20_force_True_3.jsonl
    # - igsm_op16_ip20_force_True_0.jsonl
    COT_EXAMPLES = """
    Example 1:
    Question: What is the total number of adult animals in Maple Creek?
    Answer: Define adult wolf in Maple Creek as r; so r = 2. Define total number of adult animals in Maple Creek as p; so p = r = 2. <answer>2</answer>.

    Example 2:
    Question: What is the total number of schools in Clearwater Bay?
    Answer: Define elementary school in Riverton City as b; so b = 3. Define private middle school in Clearwater Bay as i; so i = b = 3. Define public highschool in Clearwater Bay as M; so M = i = 3. Define elementary school in Clearwater Bay as G; so G = 2. Define total number of schools in Clearwater Bay as W; V = G + i = 2 + 3 = 5; so W = V + M = 5 + 3 = 8. <answer>8</answer>.

    Example 3:
    Question: What is the total number of movies in Festival de Clairmont?
    Answer: Define upbeat metropolis comedy in Festival de Saint-Rivage as m; so m = 4. Define total number of movies in Festival de Saint-Rivage as k; so k = m = 4. Define intense detective thriller in Festival Lumi\u00e8re de Valmont as C; l = k - m = 4 - 4 = 0; so C = 3 + l = 3 + 0 = 3. Define total number of movies in Festival Lumi\u00e8re de Valmont as Q; so Q = C = 3. Define solemn period drama in R\u00eaves de Belleville as N; t = Q + C = 3 + 3 = 6; T = t + k = 6 + 4 = 10; so N = 4 + T = 4 + 10 = 14. Define total number of movies in R\u00eaves de Belleville as y; so y = N = 14. Define futuristic sci-fi movie in Festival de Clairmont as A; z = y + N = 14 + 14 = 28; q = z + C = 28 + 3 = 31; so A = 3 * q = 3 * 31 = 93. Define total number of movies in Festival de Clairmont as p; so p = A = 93. <answer>93</answer>.
    """  # noqa: F541, E501

    def get_prompt(self) -> PromptTemplate:
        """Get the Chain-of-Thought prompt template.

        Returns:
            A PromptTemplate object containing the Chain-of-Thought prompt template.
        """
        return PromptTemplate(
            input_variables=["problem", "examples", "question"],
            template=self.COT_INSTRUCTION,
        )

    def get_dataset(self, is_eval: bool) -> Dataset:
        if is_eval:
            base_path = self.script_args.eval_dataset_name
            difficulty_list = self.script_args.eval_split_list
        else:
            base_path = self.script_args.dataset_name
            difficulty_list = self.script_args.split_list
        prompt_method = self.script_args.prompt_method

        if difficulty_list is None:
            difficulty_list = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

        all_datasets = []
        for diff in difficulty_list:
            diff_dir = os.path.join(base_path, diff)
            if not os.path.isdir(diff_dir):
                continue

            for op_subdir in os.listdir(diff_dir):
                sub_dir_path = os.path.join(diff_dir, op_subdir)
                if not os.path.isdir(sub_dir_path):
                    continue

                for filename in glob.glob(os.path.join(sub_dir_path, "*.jsonl")):
                    ds: Dataset = load_dataset("json", data_files=filename, split="train")
                    ds = ds.map(
                        lambda x: {
                            "prompt": GSMInfiniteDataset.get_prompt_for_sample(x, prompt_method),
                            "answer": GSMInfiniteDataset.extract_gsm_final_answer_from_ground_truth(
                                x["solution"]
                            ),
                            "prompt_method": prompt_method,
                            "difficulty": x.get("op", None),
                            "id": x.get("id", None),
                        }
                    )
                    all_datasets.append(ds)

        if len(all_datasets) == 0:
            raise RuntimeError(
                "No data loaded from GSM-Infinite. Please check the base_path and difficulty_list."
            )

        combined_dataset = concatenate_datasets(all_datasets)
        logger.info(
            f"*** Loaded {is_eval=} dataset {self.script_args.dataset_name}::{self.script_args.split_list} "
            f"with {len(combined_dataset)} samples."
        )
        return combined_dataset

    @staticmethod
    def extract_gsm_final_answer_from_ground_truth(solution: str) -> str:
        match = re.search(r"Answer:\s*([^\n\.]*)", solution)
        if match:
            return match.group(1).strip()
        raise ValueError(f"No final answer found in solution: {solution}")

    @staticmethod
    def get_prompt_for_sample(sample: dict[str, Any], prompt_method: str) -> CONVO_T:
        """
        Get the prompt for a GSM-Infinite sample, depending on the prompt method.

        Args:
            sample (dict[str, Any]): A sample from the dataset, with key "problem" and "question".
            prompt_method (str): Either "zeroshot" or "cot".

        Returns:
            CONVO_T: A list of messages for the conversational-style prompt.
                Each message is a dict with keys "role" and "content".
        """
        problem = sample["problem"]
        question = sample["question"]
        match prompt_method:
            case "zeroshot":
                prompt = [
                    {
                        "role": "user",
                        "content": f"{problem}\nQuestion: {question}\nAnswer: ",
                    },
                ]
                return prompt

            case "cot":
                prompt_template = PromptTemplate(
                    input_variables=["problem", "examples", "question"],
                    template=GSMInfiniteDataset.COT_INSTRUCTION,
                )
                return [
                    {
                        "role": "user",
                        "content": prompt_template.format(
                            problem=problem, examples=GSMInfiniteDataset.COT_EXAMPLES, question=question
                        ),
                    },
                ]
            case _:
                raise ValueError(f"Invalid {prompt_method=}")


class HotpotQADataset(DatasetForGRPO):
    COT_EXAMPLES = f"""\
    Question: Which magazine was started first Arthur's Magazine or First for Women?
    Answer: First I need to find the year Arthur's Magazine was started. Based on the evidence, Arthur's Magazine was started in 1844. Next, I need to find the year First for Women was started. Based on the evidence, First for Women was started in 1989. Since 1844 is before 1989, Arthur's Magazine was started first. <answer>Arthur's Magazine</answer>.

    Question: The Oberoi family is part of a hotel company that has a head office in what city?
    Answer: First I need to find what hotel company the Oberoi family is part of. Based on the evidence, the Oberoi family is part of the The Oberoi Group. Next, I need to find the head office of The Oberoi Group. Based on the evidence, the head office of The Oberoi Group is in Delhi. <answer>Delhi</answer>.

    Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
    Answer: First I need to find out who Milhouse was named after. Based on the evidence, Milhouse was named after Richard Nixon. <answer>Richard Nixon</answer>.

    Question: What nationality was James Henry Miller's wife?
    Answer: First I need to find out who James Henry Miller's wife is. Based on the evidence, James Henry Miller's wife is named Peggy Seeger. Next, I need to find out the nationality of Peggy Seeger. Based on the evidence, Peggy Seeger is an American folksinger. <answer>American</answer>.

    Question: Cadmium Chloride is slightly soluble in this chemical, it is also called what?
    Answer: First I need to find out what chemical Cadmium Chloride is slightly soluble in. Based on the evidence, Cadmium Chloride is slightly soluble in alcohol. <answer>alcohol</answer>.

    Question: Which tennis player won more Grand Slam titles, Henri Leconte or Jonathan Stark?
    Answer: First I need to find out how many Grand Slam titles Henri Leconte has won. Based on the evidence, Henri Leconte has won 0 Grand Slam titles. Next, I need to find out how many Grand Slam titles Jonathan Stark has won. Based on the evidence, Jonathan Stark has won 2 Grand Slam titles. <answer>Jonathan Stark</answer>.

    Question: Which genus of moth in the world's seventh-largest country contains only one species?
    Answer: First I need to find out the world's seventh-largest country. Based on the evidence, the world's seventh-largest country is India. Next, I need to find out the genus of moth in India. Based on the evidence, Nepita is a genus of moth in India, Indogrammodes is a genus of moth in India. Next, I need to figure out the number of species in each genus. Based on the evidence, Nepita has 1 species and Indogrammodes has 1 species. <answer>Nepita,Indogrammodes</answer>.

    Question: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring?
    Answer: First I need to find out who is the best kick boxer in the world. Based on the evidence, the best kick boxer in the world is Badr Hari. Next, I need to find out whether Badr Hari has been involved in controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring. Based on the evidence, Badr Hari has been involved in controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring. <answer>Badr Hari</answer>.

    Question: The Dutch-Belgian television series that "House of Anubis" was based on first aired in what year?
    Answer: First I need to find the Dutch-Belgian television series that "House of Anubis" was based on. Based on the evidence, "House of Anubis" was based on "Het Huis Anubis". Next, I need to find out when "Het Huis Anubis" was first aired. Based on the evidence, "Het Huis Anubis" was first aired in September 2006. <answer>2006</answer>.

    Question: What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?
    Answer: First I need to find out what track the 2013 Liqui Moly Bathurst 12 Hour was staged on. Based on the evidence, the 2013 Liqui Moly Bathurst 12 Hour was staged on the Mount Panorama Circuit. Next, I need to find out the length of the track. Based on the evidence, the length of the track is 6.213 km. <answer>6.213 km</answer>.
    """  # noqa: E501, F541

    def __init__(self, script_args: GRPOScriptArguments):
        super().__init__(script_args)

    def get_dataset(self, is_eval: bool) -> Dataset:
        if is_eval:
            dataset_name = self.script_args.eval_dataset_name
            split_list = self.script_args.eval_split_list
        else:
            dataset_name = self.script_args.dataset_name
            split_list = self.script_args.split_list

        all_datasets: list[Dataset] = []
        for split_name in split_list:
            # dataset["qa_pairs"] is records of ("id", "question", "answer", "type")
            # dataset["text"] is records of ("title", "article", "id")
            # id can be used to merge the two dataframes
            dataset: dict[str, list] = load_hp_data(dataset_name, split=split_name, setting="distractor")
            df_qa_pairs: pd.DataFrame = pd.DataFrame(dataset["qa_pairs"])
            df_text: pd.DataFrame = pd.DataFrame(dataset["text"])

            # Merge qa_pairs with text to get articles for each question
            df_qa_pairs: pd.DataFrame = df_qa_pairs.merge(
                df_text[["id", "article", "title"]],
                on="id",
                how="left",
                suffixes=("", "_text"),
            )

            dataset: Dataset = Dataset.from_pandas(df_qa_pairs)
            dataset = dataset.map(
                lambda sample: {
                    "prompt": HotpotQADataset.get_prompt_for_sample(
                        sample,
                        self.script_args.prompt_method,
                    ),
                    "answer": sample["answer"],
                    "prompt_method": self.script_args.prompt_method,
                }
            )

            all_datasets.append(dataset)

        dataset = concatenate_datasets(all_datasets)
        logger.info(
            f"*** Loaded {is_eval=} dataset {self.script_args.dataset_name}::{self.script_args.split_list} "
            f"with {len(dataset)} samples."
        )
        return dataset

    @staticmethod
    def get_prompt_for_sample(sample: dict[str, Any], prompt_method: str) -> CONVO_T:
        """
        Get the prompt for a PhantomWiki sample, depending on the prompt method.

        Args:
            sample (dict[str, Any]): A sample from the dataset, with keys "question", "title", "article".
            prompt_method (str): Either "zeroshot" or "cot".

        Returns:
            CONVO_T: A list of messages for the conversational-style prompt.
                Each message is a dict with keys "role" and "content".
        """
        text_corpus = pd.DataFrame({"title": sample["title"], "article": sample["article"]})
        evidence = get_all_evidence(text_corpus)
        match prompt_method:
            case "zeroshot":
                llm_prompt = ZeroshotLLMPrompt()
                prompt = [
                    {
                        "role": "user",
                        "content": llm_prompt.get_prompt().format(
                            evidence=evidence, question=sample["question"]
                        ),
                    },
                ]
                return prompt

            case "cot":
                llm_prompt = CoTLLMPrompt()
                return [
                    {
                        "role": "user",
                        "content": llm_prompt.get_prompt().format(
                            evidence=evidence,
                            examples=HotpotQADataset.COT_EXAMPLES,
                            question=sample["question"],
                        ),
                    },
                ]
            case _:
                raise ValueError(f"Invalid {prompt_method=}")
