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
from phantom_eval.prompts import COT_EXAMPLES, CoTLLMPrompt, ZeroeshotLLMPrompt
from phantom_eval.utils import load_data as load_pw_data

from phantom_reasoner._types import CONVO_T
from phantom_reasoner.configs import GRPOScriptArguments
from phantom_reasoner.utils.hp.data_utils import load_hp_data
from phantom_reasoner.utils.msq.data_utils import load_msq_data
from phantom_reasoner.utils.twowiki.data_utils import load_2wiki_data

logger = logging.getLogger(__name__)


class DatasetForGRPO(abc.ABC):
    def __init__(self, script_args: GRPOScriptArguments) -> None:
        self.script_args = script_args

    @abc.abstractmethod
    def get_dataset(self, is_eval: bool) -> Dataset:
        """
        Get the dataset for training or evaluation.
        """

    @abc.abstractmethod
    def get_prompt_for_sample(self, sample: dict[str, Any], prompt_method: str, **kwargs) -> CONVO_T:
        """
        Get the prompt for a sample, depending on the prompt method.

        Args:
            sample (dict[str, Any]): A sample from the dataset, with key "question".
            prompt_method (str): Either "zeroeshot" or "cot".

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

            # Define a named function for better caching
            def add_prompt_formatting_pw(sample):
                return {
                    "prompt": self.get_prompt_for_sample(
                        sample, self.script_args.prompt_method, evidence=evidence
                    ),
                    "answer": sample["answer"],  # x['answer'] is a list of strings
                    "prompt_method": self.script_args.prompt_method,
                }

            dataset: Dataset = qa_pairs.map(
                add_prompt_formatting_pw,
                desc="Formatting PhantomWiki prompts",
            )
            all_datasets.append(dataset)

        dataset = concatenate_datasets(all_datasets)
        logger.info(
            f"*** Loaded {is_eval=} dataset {self.script_args.dataset_name}::{self.script_args.split_list} "
            f"with {len(dataset)} samples."
        )
        return dataset

    def get_prompt_for_sample(
        self, sample: dict[str, Any], prompt_method: str, evidence: str = ""
    ) -> CONVO_T:
        """
        Get the prompt for a PhantomWiki sample, depending on the prompt method.

        Args:
            sample (dict[str, Any]): A sample from the dataset, with key "question".
            prompt_method (str): Either "zeroeshot" or "cot".
            evidence (str): The evidence text to include in the prompt. Default is "".

        Returns:
            CONVO_T: A list of messages for the conversational-style prompt.
                Each message is a dict with keys "role" and "content".
        """
        match prompt_method:
            case "zeroeshot":
                llm_prompt = ZeroeshotLLMPrompt()
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
    # Example 1: igsm_op2_ip20_force_True_15.jsonl
    # Example 2: igsm_op7_ip20_force_True_3.jsonl
    # Example 3: igsm_op16_ip20_force_True_0.jsonl
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

                    # Define a named function for better caching
                    def add_prompt_formatting_gsm(x):
                        return {
                            "prompt": self.get_prompt_for_sample(x, prompt_method),
                            "answer": GSMInfiniteDataset.extract_gsm_final_answer_from_ground_truth(
                                x["solution"]
                            ),
                            "prompt_method": prompt_method,
                            "difficulty": x.get("op", None),
                            "id": x.get("id", None),
                        }

                    ds = ds.map(
                        add_prompt_formatting_gsm,
                        desc="Formatting GSM prompts",
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

    def get_prompt_for_sample(self, sample: dict[str, Any], prompt_method: str) -> CONVO_T:
        """
        Get the prompt for a GSM-Infinite sample, depending on the prompt method.

        Args:
            sample (dict[str, Any]): A sample from the dataset, with key "problem" and "question".
            prompt_method (str): Either "zeroeshot" or "cot".

        Returns:
            CONVO_T: A list of messages for the conversational-style prompt.
                Each message is a dict with keys "role" and "content".
        """
        problem = sample["problem"]
        question = sample["question"]
        match prompt_method:
            case "zeroeshot":
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


class ReasoningGymDataset(DatasetForGRPO):
    """
    Loader for Reasoning Gym style data.

    Directory layout example:
        train: <base_path>/train.jsonl
        eval: <base_path>/eval.jsonl

    Each JSONL record:
      {
        "question": "<text>",
        "answer": "<number as string or int>",
        "metadata": {
            "source_dataset": "<dataset name like 'family_relationships' or 'knights_knaves'>",
            "source_index": <int>,
            ...
        }
      }
    """

    # Base instruction. Examples will be injected dynamically by split list.
    COT_INSTRUCTION = f"""
    You will be provided with a question. Your response must end with the final answer enclosed in tags: <answer>FINAL_ANSWER</answer>

    Here are some examples:
    (START OF EXAMPLES)
    {{examples}}
    (END OF EXAMPLES)

    You are given the following problem:
    Question: {{question}}
    Answer: """  # noqa: F541, E501

    RG_TASK2DIFFICULTY_KEY: dict[str, str | None] = {
        "rg-family_relationships": "family_size",
        "rg-knights_knaves": "n_people",
    }

    # Registry of CoT examples by RG task name
    RG_TASK2COT_EXAMPLES: dict[str, str] = {
        # From /share/nikola/phantom-reasoning/data/rg-family_relationships/train.jsonl
        # Example 1: source_index=3091, family_size=3
        # Example 2: source_index=4624, family_size=4
        # Example 3: source_index=11671, family_size=5
        # Example 4: source_index=7319, family_size=6
        # Example 5: source_index=8442, family_size=7
        # Example 6: source_index=3027, family_size=8
        # Example 7: source_index=4706, family_size=9
        # Example 8: source_index=7680, family_size=10
        # Example 9: source_index=3729, family_size=11
        # Example 10: source_index=1153, family_size=12
        # Example 11: source_index=1972, family_size=13
        "rg-family_relationships": (
            """
            Example 1:
            Question: Harry is married to Emily. They have a child called Daniel.
            What is Emily to Daniel? Respond only with the word that describes their relationship.
            Answer: Harry and Emily are the parents of Daniel, so Emily is the mother of Daniel. <answer>mother</answer>

            Example 2:
            Question: James is married to Aria. They have a child called George. George is married to Olivia.
            What is George to Olivia? Respond only with the word that describes their relationship.
            Answer: Since George is married to Olivia, George is the husband of Olivia. <answer>husband</answer>

            Example 3:
            Question: John is married to Olivia. They have a child called Liam. Liam is married to Willow. They have a child called Logan.
            What relation is John to Olivia? Answer with a single word.
            Answer: John is married to Olivia, so John is the husband of Olivia. <answer>husband</answer>

            Example 4:
            Question: Ryder is married to Lily. They have a child called John. John is married to Aria. They have children called Noah and Daniel.
            How is John related to Daniel? Provide the relationship in one word.
            Answer: Noah and Daniel are children of John, so John is the father of Daniel. <answer>father</answer>

            Example 5:
            Question: Alexander is married to Lisa. They have a child called Joseph. Joseph is married to Luna. They have a child called Eleanor. William is married to Amelia. They have a child called Luna.
            How is Eleanor related to Luna? Provide the relationship in one word.
            Answer: Joseph is married to Luna, and together they have a child called Eleanor, so Eleanor is the daughter of Luna. <answer>daughter</answer>

            Example 6:
            Question: Charles is married to Eleanor. They have a child called Mason. Mason is married to Susan. They have children called Lucy and Ryder. Christopher is married to Patricia. They have a child called Susan.
            What is Lucy to Mason? Respond only with the word that describes their relationship.
            Answer: Lucy is the daughter of Mason, so Lucy is the daughter of Mason. <answer>daughter</answer>

            Example 7:
            Question: John is married to Barbara. They have children called Kai and Atlas. Atlas is married to River. Kai is married to Luna. They have a child called Joseph. Michael is married to Zoe. They have a child called Luna.
            What is Luna to Kai? Respond only with the word that describes their relationship.
            Answer: Luna is married to Kai, so Luna is the wife of Kai. <answer>wife</answer>

            Example 8:
            Question: Noah is married to Barbara. They have children called Aiden and Charles. Charles is married to Lisa. They have a child called River. Aiden is married to Lucy. They have a child called Atlas. Matthew is married to Sarah. They have a child called Lucy.
            What relation is Noah to Atlas? Answer with a single word.
            Answer: Noah has a child called Aiden. Moreover, Aiden is married to Lucy, who together have a child called Atlas. So Noah is the father of Aiden, and Aiden is the father of Atlas. Therefore, Noah is the grandfather of Atlas. <answer>grandfather</answer>

            Example 9:
            Question: Phoenix is married to Amelia. They have children called Lucas, Aiden and Sophia. Daniel is married to Willow. They have a child called Nova. Aiden is married to Grace. Lucas is married to Nova. They have a child called Sebastian. Sophia is married to Noah.
            What relation is Sebastian to Lucas? Answer with a single word.
            Answer: Lucas is married to Nova, and together they have a child called Sebastian. So Sebastian is the son of Lucas. <answer>son</answer>

            Example 10:
            Question: Sebastian is married to Ava. They have children called David, Aiden and Hannah. Thomas is married to Luna. They have a child called Karen. Aiden is married to Sky. David is married to Karen. They have a child called Matthew. Hannah is married to James. They have a child called Daniel.
            What relation is Matthew to Karen? Answer with a single word.
            Answer: David is married to Karen, and together they have a child called Matthew. So Matthew is the son of Karen. <answer>son</answer>

            Example 11:
            Question: Zion is married to Grace. They have children called Andrew, Logan and Patricia. Matthew is married to Emily. They have a child called Sophie. Logan is married to Nova. They have a child called Margaret. Andrew is married to Sophie. They have a child called Henry. Patricia is married to Lucas. They have a child called Karen.
            How is Andrew related to Karen? Provide the relationship in one word.
            Answer: Patricia has a child called Karen. Moreover, Andrew and Patricia are siblings, so Andrew is the uncle of Karen. <answer>uncle</answer>
            """  # noqa: F541, E501
        ),
        "rg-knights_knaves": (
            """
            Example 1:
            Question: A very special island is inhabited only by heroes and villains. Heroes always tell the truth, and villains always lie. You meet 2 inhabitants: Benjamin, and Scarlett. Benjamin was heard saying, "if Benjamin is a hero then Scarlett is a hero". Scarlett stated, "Scarlett is a hero or Benjamin is a hero". So who is a hero and who is a villain? (Format your answer like: "Benjamin is a hero/villain, and Scarlett is a hero/villain")
            Answer: Assume Benjamin is a hero (tells truth). His statement "if Benjamin is hero then Scarlett is hero" means Scarlett must be a hero. If Scarlett is a hero, her statement "Scarlett is hero OR Benjamin is hero" is true (both parts are true). This is consistent. <answer>Benjamin is a hero, and Scarlett is a hero.</answer>

            Example 2:
            Question: A very special island is inhabited only by altruists and egoists. Altruists always tell the truth, and egoists always lie. You meet 2 inhabitants: Luke, and Riley. In a statement by Luke: "if Riley is an egoist then Luke is an altruist". Riley remarked, "Luke is an egoist if and only if Riley is an altruist". So who is an altruist and who is an egoist? (Format your answer like: "Luke is a altruist/egoist, and Riley is a altruist/egoist")
            Answer: Assume Luke is an egoist (lies). His statement "if Riley is egoist then Luke is altruist" is false. For a conditional to be false, the premise must be true and conclusion false. So Riley is an egoist and Luke is not an altruist (Luke is egoist), which is consistent. Now check Riley's statement "Luke is egoist ↔ Riley is altruist". Since both Luke and Riley are egoists, left side is true and right side is false, making the biconditional false, which is consistent with Riley being an egoist who lies. <answer>Luke is an egoist, and Riley is an egoist.</answer>

            Example 3:
            Question: A very special island is inhabited only by angels and devils. Angels always tell the truth, and devils always lie. You meet 3 inhabitants: Logan, Aurora, and Riley. Logan asserted: "Aurora is an angel". As Aurora put it, "if Logan is an angel then Riley is a devil". Riley stated, "Logan is a devil". So who is an angel and who is a devil? (Format your answer like: "Logan is a angel/devil, Aurora is a angel/devil, and Riley is a angel/devil")
            Answer: Assume Logan is an angel (tells truth). Then Aurora is an angel (as Logan stated). If Aurora is an angel, her statement "if Logan is angel then Riley is devil" means Riley is a devil. If Riley is a devil (lies), he says "Logan is a devil", which is false, consistent with lying. <answer>Logan is an angel, Aurora is an angel, and Riley is a devil.</answer>

            Example 4:
            Question: A very special island is inhabited only by heroes and villains. Heroes always tell the truth, and villains always lie. You meet 3 inhabitants: Luke, Henry, and Zoey. "Luke is a hero or Henry is a hero," Luke claimed. Henry said that if Zoey is a villain then Henry is a hero. In a statement by Zoey: "if Henry is a hero then Luke is a villain". So who is a hero and who is a villain? (Format your answer like: "Luke is a hero/villain, Henry is a hero/villain, and Zoey is a hero/villain")
            Answer: Assume Luke is a hero (tells truth). His statement "Luke is hero OR Henry is hero" is true. Let's assume Henry is also a hero. Henry's statement "if Zoey is villain then Henry is hero" is true since Henry is a hero. Now for Zoey's statement "if Henry is hero then Luke is villain" - if Zoey is a villain (lies), her statement should be false. Since Henry is a hero and Luke is not a villain, the conditional is false, which is consistent with Zoey lying. <answer>Luke is a hero, Henry is a hero, and Zoey is a villain.</answer>

            Example 5:
            Question: A very special island is inhabited only by sages and fools. Sages always tell the truth, and fools always lie. You meet 4 inhabitants: Alexander, Elizabeth, Amelia, and Penelope. In a statement by Alexander: "if Amelia is a fool then Amelia is a sage". Elizabeth said that Penelope is a sage if and only if Amelia is a fool. "Alexander is a sage and Penelope is a sage" - Amelia. Penelope stated, "Amelia is a sage or Elizabeth is a fool". So who is a sage and who is a fool? (Format your answer like: "Alexander is a sage/fool, Elizabeth is a sage/fool, Amelia is a sage/fool, and Penelope is a sage/fool")
            Answer: Alexander's statement "if Amelia is fool then Amelia is sage" is a contradiction if Amelia is a fool, so for Alexander to be a sage (truth-teller), Amelia must be a sage. If Amelia is a sage, her statement "Alexander is sage AND Penelope is sage" means both are sages. Penelope's statement "Amelia is sage OR Elizabeth is fool" is true if Penelope is a sage - since Amelia is sage, the OR is true. Elizabeth's statement "Penelope is sage ↔ Amelia is fool" - if Penelope is sage and Amelia is sage (not fool), the biconditional is false, so Elizabeth is a fool. <answer>Alexander is a sage, Elizabeth is a fool, Amelia is a sage, and Penelope is a sage.</answer>

            Example 6:
            Question: A very special island is inhabited only by heroes and villains. Heroes always tell the truth, and villains always lie. You meet 4 inhabitants: Sophia, Alexander, Grace, and Liam. Sophia stated, "if Sophia is a hero then Alexander is a villain". "Grace is a villain if and only if Liam is a hero," Alexander mentioned. "Sophia is a villain if and only if Sophia is a hero," Grace declared. As Liam put it, "Grace is a villain and Liam is a hero". So who is a hero and who is a villain? (Format your answer like: "Sophia is a hero/villain, Alexander is a hero/villain, Grace is a hero/villain, and Liam is a hero/villain")
            Answer: Grace's statement "Sophia is villain ↔ Sophia is hero" is a contradiction, so it's false, meaning Grace is a villain. Assume Sophia is a hero. Her statement "if Sophia is hero then Alexander is villain" means Alexander is a villain. Alexander's statement "Grace is villain ↔ Liam is hero" - if Alexander is a villain (lies), this should be false. Since Grace is a villain, for the biconditional to be false, Liam must not be a hero (Liam is villain). Liam's statement "Grace is villain AND Liam is hero" - since Liam is a villain, this false statement is consistent. <answer>Sophia is a hero, Alexander is a villain, Grace is a villain, and Liam is a villain.</answer>

            Example 7:
            Question: A very special island is inhabited only by angels and devils. Angels always tell the truth, and devils always lie. You meet 5 inhabitants: Ava, Amelia, Daniel, Mia, and Jack. "Mia is an angel if and only if Jack is a devil," Ava mentioned. In Amelia's words: "Daniel is a devil or Mia is an angel". Daniel was heard saying, "Ava is a devil". Mia noted, "Jack is a devil". Jack was heard saying, "Mia is a devil if and only if Mia is an angel". So who is an angel and who is a devil? (Format your answer like: "Ava is a angel/devil, Amelia is a angel/devil, Daniel is a angel/devil, Mia is a angel/devil, and Jack is a angel/devil")
            Answer: Jack's statement "Mia is devil ↔ Mia is angel" is a contradiction, so it's false, meaning Jack is a devil. If Mia is an angel, her statement "Jack is devil" is true. Ava's statement "Mia is angel ↔ Jack is devil" - both parts are true, so the biconditional is true, meaning Ava is an angel. Amelia's statement "Daniel is devil OR Mia is angel" - since Mia is angel, the OR is true, so Amelia is an angel. Daniel's statement "Ava is devil" is false since Ava is angel, so Daniel is a devil. <answer>Ava is an angel, Amelia is an angel, Daniel is a devil, Mia is an angel, and Jack is a devil.</answer>

            Example 8:
            Question: A very special island is inhabited only by saints and sinners. Saints always tell the truth, and sinners always lie. You meet 5 inhabitants: Penelope, Lily, Riley, Mia, and Aria. "Mia is a saint and Lily is a sinner," Penelope claimed. Lily noted, "Penelope is a sinner". Riley asserted: "if Lily is a sinner then Mia is a sinner". Mia noted, "Penelope is a sinner if and only if Penelope is a saint". Aria expressed that Lily is a saint. So who is a saint and who is a sinner? (Format your answer like: "Penelope is a saint/sinner, Lily is a saint/sinner, Riley is a saint/sinner, Mia is a saint/sinner, and Aria is a saint/sinner")
            Answer: Mia's statement "Penelope is sinner ↔ Penelope is saint" is a contradiction, so it's false, meaning Mia is a sinner. Assume Lily is a saint. Lily's statement "Penelope is sinner" means Penelope is a sinner. Penelope's statement "Mia is saint AND Lily is sinner" - since Mia is sinner and Lily is saint, this is false, consistent with Penelope being a sinner. Riley's statement "if Lily is sinner then Mia is sinner" has a false premise (Lily is saint), making it vacuously true, so Riley is a saint. Aria's statement "Lily is saint" is true, so Aria is a saint. <answer>Penelope is a sinner, Lily is a saint, Riley is a saint, Mia is a sinner, and Aria is a saint.</answer>

            Example 9:
            Question: A very special island is inhabited only by angels and devils. Angels always tell the truth, and devils always lie. You meet 6 inhabitants: Liam, Zoey, Ava, Logan, James, and Avery. Liam commented, "James is a devil if and only if Ava is a devil". Zoey asserted: "Zoey is an angel and Logan is a devil". In Ava's words: "Liam is a devil". "Avery is a devil," Logan mentioned. James said that if Avery is an angel then Zoey is an angel. Avery asserted: "Avery is an angel and Liam is a devil". So who is an angel and who is a devil? (Format your answer like: "Liam is a angel/devil, Zoey is a angel/devil, Ava is a angel/devil, Logan is a angel/devil, James is a angel/devil, and Avery is a angel/devil")
            Answer: Assume Ava is an angel. Her statement "Liam is devil" means Liam is a devil. Liam's statement "James is devil ↔ Ava is devil" - if Liam is a devil (lies), since Ava is angel (not devil), for the biconditional to be false, James must also not be a devil, so James is a devil (making both sides false, thus true). Wait, let me reconsider. If Liam lies, the biconditional is false. Ava is angel (not devil), so for false biconditional, James must be devil. Assume Avery is an angel. Avery's statement "Avery is angel AND Liam is devil" is true. Zoey's statement "Zoey is angel AND Logan is devil" - if this is false, Zoey is a devil. Logan's statement "Avery is devil" is false, so Logan is a devil. James's statement "if Avery is angel then Zoey is angel" - since Avery is angel and Zoey is devil, this is false, so James is a devil. <answer>Liam is a devil, Zoey is a devil, Ava is an angel, Logan is a devil, James is a devil, and Avery is an angel.</answer>

            Example 10:
            Question: A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 6 inhabitants: Aria, Ava, Amelia, Grace, Charlotte, and Jack. "Jack is a knight," Aria claimed. In a statement by Ava: "Jack is a knight". Amelia asserted: "Jack is a knave and Grace is a knight". Grace commented, "Aria is a knight if and only if Charlotte is a knave". As Charlotte put it, "Aria is a knight". Jack noted, "Ava is a knave if and only if Charlotte is a knave". So who is a knight and who is a knave? (Format your answer like: "Aria is a knight/knave, Ava is a knight/knave, Amelia is a knight/knave, Grace is a knight/knave, Charlotte is a knight/knave, and Jack is a knight/knave")
            Answer: Assume Aria is a knight. Her statement "Jack is knight" means Jack is a knight. Ava's statement "Jack is knight" is true, so Ava is a knight. Charlotte's statement "Aria is knight" is true, so Charlotte is a knight. Jack's statement "Ava is knave ↔ Charlotte is knave" - both Ava and Charlotte are knights (not knaves), so both sides are false, making the biconditional true, consistent with Jack being a knight. Grace's statement "Aria is knight ↔ Charlotte is knave" - Aria is knight and Charlotte is knight (not knave), so the biconditional is false, meaning Grace is a knave. Amelia's statement "Jack is knave AND Grace is knight" - Jack is knight and Grace is knave, so this is false, meaning Amelia is a knave. <answer>Aria is a knight, Ava is a knight, Amelia is a knave, Grace is a knave, Charlotte is a knight, and Jack is a knight.</answer>
            """  # noqa: F541, E501
        ),
    }

    def __init__(self, script_args: GRPOScriptArguments):
        super().__init__(script_args)

    def get_dataset(self, is_eval: bool) -> Dataset:
        if is_eval:
            base_path = self.script_args.eval_dataset_name
            jsonl_file_path = os.path.join(base_path, "eval.jsonl")
        else:
            base_path = self.script_args.dataset_name
            jsonl_file_path = os.path.join(base_path, "train.jsonl")
        prompt_method = self.script_args.prompt_method

        # Load JSONL file
        # For knights_knaves, we need to manually load due to complex nested structures
        # that cause PyArrow parsing issues
        if self.script_args.training_mode == "rg-knights_knaves":
            import json

            data_list = []
            with open(jsonl_file_path, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    # Convert problematic nested list fields to strings to avoid PyArrow issues
                    # We'll keep the original nested structure since we don't need these fields for training
                    if "statements" in data.get("metadata", {}):
                        data["metadata"]["statements"] = json.dumps(data["metadata"]["statements"])
                    if "solution" in data.get("metadata", {}):
                        data["metadata"]["solution"] = json.dumps(data["metadata"]["solution"])
                    data_list.append(data)
            dataset: Dataset = Dataset.from_list(data_list)
        else:
            dataset: Dataset = load_dataset("json", data_files=jsonl_file_path, split="train")

        # Define a named function for better caching
        difficulty_key = ReasoningGymDataset.RG_TASK2DIFFICULTY_KEY[self.script_args.training_mode]

        def add_prompt_formatting_rg(x):
            # Get difficulty value based on the task's difficulty key
            if difficulty_key is None:
                difficulty_value = None
            elif difficulty_key in x["metadata"]:
                # Direct key in metadata (e.g., "family_size")
                difficulty_value = x["metadata"][difficulty_key]
            else:
                # Nested in metadata["difficulty"] (e.g., "n_people")
                difficulty_value = x["metadata"]["difficulty"][difficulty_key]

            return {
                "prompt": self.get_prompt_for_sample(x, prompt_method),
                "answer": str(x["answer"]),
                "prompt_method": prompt_method,
                "difficulty": difficulty_value,
                "id": x["metadata"]["source_index"],
            }

        dataset = dataset.map(
            add_prompt_formatting_rg,
            desc="Formatting ReasoningGym prompts",
        )

        logger.info(f"*** Loaded {is_eval=} dataset from {jsonl_file_path} with {len(dataset)} samples.")
        return dataset

    def get_prompt_for_sample(self, sample: dict[str, Any], prompt_method: str) -> CONVO_T:
        """
        Get the prompt for a Reasoning-Gym sample, depending on the prompt method.

        Args:
            sample (dict[str, Any]): A sample from the dataset, with keys "question".
            prompt_method (str): Either "zeroeshot" or "cot".

        Returns:
            CONVO_T: A list of messages for the conversational-style prompt.
                Each message is a dict with keys "role" and "content".
        """
        question = sample["question"]

        match prompt_method:
            case "zeroeshot":
                prompt = [
                    {
                        "role": "user",
                        "content": f"Question: {question}\nAnswer: ",
                    },
                ]
                return prompt
            case "cot":
                prompt_template = PromptTemplate(
                    input_variables=["question"],
                    template=ReasoningGymDataset.COT_INSTRUCTION,
                )
                examples = ReasoningGymDataset.RG_TASK2COT_EXAMPLES[self.script_args.training_mode]
                return [
                    {
                        "role": "user",
                        "content": prompt_template.format(question=question, examples=examples),
                    },
                ]
            case _:
                raise ValueError(f"Invalid {prompt_method=}")


class WikiDataset(DatasetForGRPO):
    """Parent class for wiki datasets.

    HotpotQA, 2WikiMultiHopQA, and MuSiQue inherit from this class.
    """

    COT_EXAMPLES = None

    @staticmethod
    def load_data_func(data_path: str, split: str) -> dict:
        raise NotImplementedError("Subclasses must implement this method")

    def get_all_evidence(self, text_corpus: pd.DataFrame) -> str:
        """
        Format the text corpus into a string of evidence.

        NOTE: the wiki passages do not already contain the title, so we need to add it.
        """
        formatted_articles = text_corpus.apply(lambda x: f"{x['title']}\n{x['article']}", axis=1)
        evidence = "\n================\n\n".join(formatted_articles)
        return evidence

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
            # import pdb; pdb.set_trace()
            dataset: dict[str, list] = self.load_data_func(dataset_name, split=split_name)
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

            # Define a named function for better caching
            def add_prompt_formatting(sample):
                return {
                    "prompt": self.get_prompt_for_sample(
                        sample,
                        self.script_args.prompt_method,
                    ),
                    "answer": sample["answer"],
                    "prompt_method": self.script_args.prompt_method,
                }

            dataset = dataset.map(
                add_prompt_formatting,
                desc="Formatting prompts",
            )

            all_datasets.append(dataset)

        dataset = concatenate_datasets(all_datasets)
        logger.info(
            f"*** Loaded {is_eval=} dataset {self.script_args.dataset_name}::{self.script_args.split_list} "
            f"with {len(dataset)} samples."
        )
        return dataset

    def get_prompt_for_sample(self, sample: dict[str, Any], prompt_method: str) -> CONVO_T:
        """
        Get the prompt for a question-context pair, depending on the prompt method.

        NOTE: the context comes from the distractor setting for each wiki dataset.

        Args:
            sample (dict[str, Any]): A sample from the dataset, with keys "question", "title", "article".
            prompt_method (str): Either "zeroeshot" or "cot".

        Returns:
            CONVO_T: A list of messages for the conversational-style prompt.
                Each message is a dict with keys "role" and "content".
        """
        text_corpus = pd.DataFrame({"title": sample["title"], "article": sample["article"]})
        evidence = self.get_all_evidence(text_corpus)
        match prompt_method:
            case "zeroeshot":
                llm_prompt = ZeroeshotLLMPrompt()
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
                            examples=self.COT_EXAMPLES,
                            question=sample["question"],
                        ),
                    },
                ]
            case _:
                raise ValueError(f"Invalid {prompt_method=}")


class HotpotQADataset(WikiDataset):
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

    @staticmethod
    def load_data_func(data_path: str, split: str) -> dict:
        return load_hp_data(data_path, split, setting="distractor")


class TwoWikiDataset(WikiDataset):
    COT_EXAMPLES = f"""\
Example 1:
Question: Are director of film Move (1970 Film) and director of film M\u00e9diterran\u00e9e (1963 Film) from the same country?
Answer: Based on the evidence, the director of the film Move (1970 Film) is Stuart Rosenberg. Also, the director of the film M\u00e9diterran\u00e9e (1963 Film) is Jean-Daniel Pollet. The country of citizenship of Stuart Rosenberg is American. Also, the country of citizenship of Jean-Daniel Pollet is French. Because American and French aren't the same, the answer is no. <answer>no</answer>.

Example 2:
Question: What nationality is the director of film Borunbabur Bondhu?
Answer: Based on the evidence, the director of the film Borunbabur Bondhu is Anik Dutta. The country of citizenship of Anik Dutta is India. <answer>India</answer>.

Example 3:
Question: Where was the place of burial of the performer of song There Is So Much World To See?
Answer: Based on the evidence, the former of the song There Is So Much World To See is Elvis. The place of burial of Elvis is Graceland. <answer>Graceland</answer>.

Example 4:
Question: Why did the director of film The Light Of Western Stars (1930 Film) die?
Answer: Based on the evidence, the director of the film The Light Of Western Stars (1930 Film) is Otto Brower. The cause of death of Otto Brower is heart attack. <answer>heart attack</answer>.

Example 5:
Question: Which film came out first, The Love Route or Engal Aasan?
Answer: Based on the evidence, the release date of the film The Love Route is 1915. Also, the release date of the film Engal Aasan is 2009. Because 1915 is before 2009, the answer is The Love Route. <answer>The Love Route</answer>.

Example 6:
Question: Where was the director of film The Fascist born?
Answer: Based on the evidence, the director of the film The Fascist is Luciano Salce. The birthplace of Luciano Salce is Rome. <answer>Rome</answer>.

Example 7:
Question: Are Matraville Sports High School and Wabash High School both located in the same country?
Answer: Based on the evidence, the country that Matraville Sports High School is in is United States. Also, the country that Wabash High School is in is Australia. Because United States and Australia aren't the same, the answer is no. <answer>no</answer>.

Example 8:
Question: Which country the performer of song Soldier (Neil Young Song) is from?
Answer: Based on the evidence, the performer of the song Soldier is Neil Young. The country of citizenship of Neil Young is Canadian. <answer>Canadian</answer>.

Example 9:
Question: Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?
Answer: Based on the evidence, the director of the film A Flame In My Heart is Alain Tanner. Also, the director of the film Butcher, Baker, Nightmare Maker is William Asher. The date of birth of Alain Tanner is 6 December 1929. Also, the date of birth of William Asher is August 8, 1921. Because 6 December 1929 is later than August 8, 1921, the answer is A Flame In My Heart. <answer>A Flame In My Heart</answer>.

Example 10:
Question: Which film has the director who died later, Aaranya Kandam or One Hundred Nails?
Answer: Based on the evidence, the director of the film Aaranya Kandam is J. Sasikumar. Also, the director of the film One Hundred Nails is Ermanno Olmi. The date of death of J. Sasikumar is 17 July 2014. The date of death of Ermanno Olmi is 7 May 2018. Because 7 May 2018 is later than 17 July 2014, the answer is One Hundred Nails. <answer>One Hundred Nails</answer>.
"""  # noqa: E501, F541

    @staticmethod
    def load_data_func(data_path: str, split: str) -> dict:
        return load_2wiki_data(data_path, split)


class MuSiQueDataset(WikiDataset):
    COT_EXAMPLES = """\
Example 1:
Question: Who was ordered to force a Tibetan assault into the region conquered by Yellow Tiger in the mid-17th century?
Answer: Based on the evidence, Yellow Tiger conquered the region of Sichuan in the mid-17th century. Based on the evidence, Ming general Qu Neng was ordered to force a Tibetan assault into Sichuan. <answer>Qu Neng</answer>.

Example 2:
Question: When did the publisher of Tetrisphere unveil their new systems?
Answer: Based on the evidence, Nintendo published Tetrisphere. Based on the evidence, Nintendo unveiled their new systems on October 18, 1985. <answer>October 18, 1985</answer>.

Example 3:
Question: Who is the composer of Rhapsody No. 1, named after and inspired by the county where Alfred Seaman was born?
Answer: Based on the evidence, Alfred Seaman was born in Norfolk. Based on the evidence, the Norfolk Rhapsodies were composed by Ralph Vaughan Williams. <answer>Ralph Vaughan Williams</answer>.

Example 4:
Question: What region is Qaleh Now-e Khaleseh in Mahdi Tajik's birth city located?
Answer: Based on the evidence, the birth city of Mahdi Tajik is Tehran. Based on the evidence, Qaleh Now-e Khaleseh is located in the Qaleh Now Rural District. <answer>Qaleh Now Rural District</answer>.

Example 5:
Question: What is Nasir Zaidi's birthplace the capital of?
Answer: Based on the evidence, Nasir Zaidi was born in the city of Karachi. Based on the evidence, Karachi is the capital of West Pakistan. <answer>West Pakistan</answer>.

Example 6:
Question: Who founded the publisher of Journal of Bisexuality?
Answer: Based on the evidence, the publisher of Journal of Bisexuality is Routledge. Based on the evidence, Routledge was founded by George Routledge. <answer>George Routledge</answer>.

Example 7:
Question: Where is the headquarters of located of the 48th Highlanders of the country which released the performer of As Seen Through Windows?
Answer: Based on the evidence, As Seen Through Windows was performed by the Bell Orchestre. Based on the evidence, Canada released the Bell Orchestre. Based on the evidence, the 48th Highlanders of Canada is headquartered in the Moss Park Armoury. <answer>Moss Park Armoury</answer>.

Example 8:
Question: Who is the spouse of the director of The Yellow Ticket?
Answer: Based on the evidence, the director of The Yellow Ticket is Sidney Lumet. Based on the evidence, the spouse of Sidney Lumet is Miriam Cooper. <answer>Miriam Cooper</answer>.

Example 9:
Question: When did the torch arrive in the country where Drunken Master was filmed?
Answer: Based on the evidence, Drunken Master was filmed in Hong Kong. Based on the evidence, the torch arrived in Hong Kong on May 2. <answer>May 2</answer>.

Example 10:
Question: Who is the mother of the singer of Catch You?
Answer: Based on the evidence, the singer of Catch You is Sophie Ellis-Bextor. Based on the evidence, the mother of Sophie Ellis-Bextor is Janet Ellis. <answer>Janet Ellis</answer>.
"""  # noqa: E501, F541

    @staticmethod
    def load_data_func(data_path: str, split: str) -> dict:
        return load_msq_data(data_path, split, answerable_only=True)
