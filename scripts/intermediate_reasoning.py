#!/usr/bin/env python3
"""
Script to analyze intermediate reasoning step accuracy for both training configurations.
This script processes models from both:
- out__train=pw__eval=wiki (trained on PW, evaluated on wiki)
- out__train=wiki__eval=wiki (trained on wiki, evaluated on wiki)
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class IntermediateReasoningAnalyzer:
    """
    Analyzer for intermediate reasoning steps across both training configurations.
    """

    def __init__(self, eval_dir: str = "eval"):
        self.eval_dir = Path(eval_dir)
        self.datasets_dir = self.eval_dir / "datasets"

        # All three prediction directories
        self.pw_preds_dir = self.eval_dir / "out__train=pw__eval=wiki" / "preds" / "cot"
        self.wiki_preds_dir = self.eval_dir / "out__train=wiki__eval=wiki" / "preds" / "cot"
        self.notrain_preds_dir = self.eval_dir / "out__notrain__0626-other-datasets" / "preds" / "cot"

    def load_ground_truth_data(self, dataset: str) -> dict[str, Any]:
        """Load ground truth data for a dataset."""
        logger.info(f"Loading ground truth data for {dataset}...")

        if dataset == "2wiki":
            gt_file = self.datasets_dir / "2wiki" / "minidev.json"
            with open(gt_file) as f:
                data = json.load(f)
            return {"data": data}  # Wrap in dict for consistency

        elif dataset == "hp":
            gt_file = self.datasets_dir / "hp" / "hotpot_minidev_distractor_v1.json"
            with open(gt_file) as f:
                data = json.load(f)
            return {"data": data}  # This already has the full structure

        elif dataset == "msq":
            gt_file = self.datasets_dir / "msq" / "musique_ans_v1.0_minidev.jsonl"
            data = []
            with open(gt_file) as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return {"data": data}

        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def load_model_predictions(self, dataset: str) -> dict[str, dict[str, Any]]:
        """Load model predictions for a dataset from ALL THREE directories."""
        logger.info(f"Loading model predictions for {dataset}...")

        pattern = f"dataset={dataset}500__split=minidev*.json"
        all_predictions = {}

        # Load from PW training directory
        pw_pred_files = list(self.pw_preds_dir.glob(pattern))
        for pred_file in pw_pred_files:
            model_name = pred_file.stem.split("__model_name=")[1].split("__bs=")[0]
            model_name = model_name.replace("--", "/")
            # Mark as PW training
            model_key = f"PW_TRAIN::{model_name}"

            logger.info(f"  Loading PW predictions from {model_name}...")
            with open(pred_file) as f:
                preds = json.load(f)
            all_predictions[model_key] = preds

        # Load from Wiki training directory
        wiki_pred_files = list(self.wiki_preds_dir.glob(pattern))
        for pred_file in wiki_pred_files:
            model_name = pred_file.stem.split("__model_name=")[1].split("__bs=")[0]
            model_name = model_name.replace("--", "/")
            # Mark as Wiki training
            model_key = f"WIKI_TRAIN::{model_name}"

            logger.info(f"  Loading Wiki predictions from {model_name}...")
            with open(pred_file) as f:
                preds = json.load(f)
            all_predictions[model_key] = preds

        # Load from NoTrain directory
        notrain_pred_files = list(self.notrain_preds_dir.glob(pattern))
        for pred_file in notrain_pred_files:
            model_name = pred_file.stem.split("__model_name=")[1].split("__bs=")[0]
            model_name = model_name.replace("--", "/")
            # Mark as NoTrain
            model_key = f"NOTRAIN::{model_name}"

            logger.info(f"  Loading NoTrain predictions from {model_name}...")
            with open(pred_file) as f:
                preds = json.load(f)
            all_predictions[model_key] = preds

        return all_predictions

    def extract_reasoning_text(self, prediction: dict[str, Any]) -> str:
        """Extract the reasoning text from a model prediction."""
        try:
            interaction = prediction.get("interaction", {})
            messages = interaction.get("messages", [])

            # Get the assistant's response (last message)
            if messages and len(messages) > 1:
                assistant_msg = messages[-1]
                content = assistant_msg.get("content", [])
                if content and isinstance(content, list):
                    text_content = content[0].get("text", "")
                    return text_content

            return ""
        except Exception as e:
            logger.warning(f"Error extracting reasoning text: {e}")
            return ""

    def _entity_mentioned_in_text(self, entity: str, text: str) -> bool:
        """Check if an entity is mentioned in the text using exact string matching."""
        if not entity or not text:
            return False

        # Convert to lowercase for case-insensitive matching
        entity_lower = entity.lower().strip()
        text_lower = text.lower()

        # Exact substring match only
        return entity_lower in text_lower

    def _extract_supporting_sentence(self, question: dict, title: str, sent_id: int) -> str:
        """Extract the actual supporting sentence from HotpotQA context."""
        context = question.get("context", [])

        # Find the paragraph with matching title
        for paragraph_title, sentences in context:
            if paragraph_title == title:
                # Check if sent_id is valid for this paragraph
                if 0 <= sent_id < len(sentences):
                    return sentences[sent_id].strip()
                break

        return ""  # Return empty if not found

    def _sentence_mentioned_in_text(self, sentence: str, text: str) -> bool:
        """Check if key terms from a sentence are mentioned in the reasoning text.

        Uses proximity constraint: found terms must appear within sentence-level proximity
        (roughly within 100 characters of each other) to ensure coherent mention.
        """
        if not sentence or not text:
            return False

        # Convert to lowercase for case-insensitive matching
        sentence_lower = sentence.lower().strip()
        text_lower = text.lower()

        # For short sentences, use direct substring matching
        if len(sentence_lower) <= 20:
            return sentence_lower in text_lower

        # Extract key terms from the sentence
        key_terms = self._extract_key_terms(sentence_lower)

        if not key_terms:
            return False

        # Find positions of key terms in the reasoning text
        term_positions = []
        for term in key_terms:
            pos = text_lower.find(term)
            if pos != -1:
                term_positions.append((term, pos))

        # Check if we have enough terms found
        match_ratio = len(term_positions) / len(key_terms)
        if match_ratio < 0.5:  # Less than 50% of terms found
            return False

        # Proximity constraint: check if >match_ratio of terms appear in some 2-sentence window
        # This handles cases where the same word can appear multiple times
        if len(term_positions) >= 2:
            # Split text into sentences using periods as delimiters
            sentences = [s.strip() for s in text_lower.split(".") if s.strip()]

            if len(sentences) < 2:
                # If less than 2 sentences, all terms are effectively in the same "window"
                return True

            # Check each possible 2-sentence window
            for start_sent in range(len(sentences) - 1):
                end_sent = start_sent + 1  # 2-sentence window

                # Combine the sentences in this window
                window_text = (sentences[start_sent] + " " + sentences[end_sent]).lower()

                # Count how many key terms appear in this window
                terms_in_window = 0
                for term in key_terms:
                    if term in window_text:
                        terms_in_window += 1

                # Check if this window contains enough terms (≥50% threshold)
                window_ratio = terms_in_window / len(key_terms)
                if window_ratio >= 0.5:  # Use consistent 50% threshold for proximity
                    return True

            # No 2-sentence window contains enough terms
            return False

        return True

    def _extract_key_terms(self, sentence: str) -> list[str]:
        """Extract key terms from a sentence, filtering out common stop words.

        Stopwords are categorized as:
        - Articles: the, a, an
        - Conjunctions: and, or, but
        - Prepositions: in, on, at, to, for, of, with, by, as
        - Common verbs: is, was, are, were, be, been, being, have, has, had, do, does, did
        - Modal verbs: will, would, could, should, may, might, can
        - Pronouns: it, its, he, she, his, her, they, them, their,
          this, that, these, those, i, you, we, us, me, him
        """
        import re

        # Categorized stopwords (common words that don't carry semantic meaning)
        stopwords = {
            # Articles
            "the",
            "a",
            "an",
            # Conjunctions
            "and",
            "or",
            "but",
            # Prepositions
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "as",
            # Common verbs (copula and auxiliaries)
            "is",
            "was",
            "are",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            # Modal verbs
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            # Pronouns
            "it",
            "its",
            "he",
            "she",
            "his",
            "her",
            "they",
            "them",
            "their",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "we",
            "us",
            "me",
            "him",
        }

        # Extract words, keeping numbers, names, and meaningful terms
        # Remove punctuation but keep apostrophes in contractions
        words = re.findall(r"\b\w+(?:'\w+)?\b", sentence.lower())

        key_terms = []
        for word in words:
            # Keep the word if it's:
            # 1. Not a stopword
            # 2. Longer than 2 characters (unless it's a number)
            # 3. Contains digits (years, dates, etc.)
            if (word not in stopwords and len(word) > 2) or any(char.isdigit() for char in word):
                key_terms.append(word)

        return key_terms

    def _answers_match(self, predicted: str, correct: str) -> bool:
        """Check if predicted answer matches the correct answer (case-insensitive)."""
        if not predicted or not correct:
            return False

        pred_clean = predicted.lower().strip()
        correct_clean = correct.lower().strip()

        return pred_clean == correct_clean

    def analyze_2wiki_evidences(
        self, ground_truth: dict[str, Any], predictions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze how well models identify evidence entities in 2Wiki dataset."""
        logger.info("Analyzing 2Wiki evidences...")

        results = {}
        gt_data = ground_truth["data"]

        for model_name, model_preds in predictions.items():
            logger.info(f"  Processing model: {model_name}")

            total_questions = 0
            partial_evidence_matches = 0
            full_evidence_matches = 0
            bridging_entity_matches = 0
            final_answer_in_reasoning = 0
            final_answer_in_pred = 0
            sequential_reasoning = 0

            # Track partial breakdowns
            evidence_breakdown = defaultdict(int)  # evidence_length -> count
            bridging_breakdown = defaultdict(int)  # bridging_length -> count

            for question in gt_data:
                qid = question.get("_id") or question.get("id") or question.get("question_id")
                if qid not in model_preds:
                    continue

                total_questions += 1
                evidences = question.get("evidences", [])
                if not evidences:
                    continue

                bridging_entities = [ev[2] for ev in evidences]  # object entities

                pred = model_preds[qid]
                reasoning_text = self.extract_reasoning_text(pred)

                # Track evidence complexity
                evidence_length = len(evidences)
                evidence_breakdown[f"{evidence_length}_evidence_questions"] += 1

                # Check evidence mentions
                evidence_found = []
                for i, (subj, rel, obj) in enumerate(evidences):
                    # Only check the object entity (third element of triplet)
                    if self._entity_mentioned_in_text(obj, reasoning_text):
                        evidence_found.append((reasoning_text.lower().find(obj.lower()), i))

                # Check bridging entity mentions
                bridging_found = []
                for i, bridging_entity in enumerate(bridging_entities):
                    if self._entity_mentioned_in_text(bridging_entity, reasoning_text):
                        bridging_found.append((reasoning_text.lower().find(bridging_entity.lower()), i))

                # Count matches
                if len(evidence_found) > 0:
                    partial_evidence_matches += 1
                if len(evidence_found) == len(evidences):
                    full_evidence_matches += 1
                if len(bridging_found) > 0:
                    bridging_entity_matches += 1

                # Check final answer presence
                correct_answer = question.get("answer", "")
                if correct_answer and self._entity_mentioned_in_text(correct_answer, reasoning_text):
                    final_answer_in_reasoning += 1

                pred_answer = pred.get("pred", "").strip()
                if self._answers_match(pred_answer, correct_answer):
                    final_answer_in_pred += 1

                # Check sequential reasoning for evidence
                if len(evidence_found) >= 2:
                    evidence_found.sort(key=lambda x: x[0])  # Sort by text position
                    is_sequential = all(
                        evidence_found[i][1] < evidence_found[i + 1][1]  # Check step index order
                        for i in range(len(evidence_found) - 1)
                    )
                    if is_sequential:
                        sequential_reasoning += 1

                # Update partial breakdown stats
                evidence_found_count = len(evidence_found)
                for i in range(1, evidence_length + 1):
                    if evidence_found_count >= i:
                        evidence_breakdown[f"{evidence_length}_evidence_at_least_{i}_found"] += 1

                bridging_found_count = len(bridging_found)
                bridging_length = len(bridging_entities)
                bridging_breakdown[f"{bridging_length}_bridging_questions"] += 1
                for i in range(1, bridging_length + 1):
                    if bridging_found_count >= i:
                        bridging_breakdown[f"{bridging_length}_bridging_at_least_{i}_found"] += 1

            # Convert counts to cumulative statistics
            partial_breakdown_stats = dict(evidence_breakdown)
            partial_breakdown_stats.update(dict(bridging_breakdown))

            # Fix cumulative calculation for "at least N found" statistics
            for evidence_len in range(2, 6):  # Support up to 5 evidence questions
                total_questions_key = f"{evidence_len}_evidence_questions"
                if total_questions_key in partial_breakdown_stats:
                    total_q = partial_breakdown_stats[total_questions_key]

                    # Calculate cumulative "at least N found" from right to left
                    cumulative_sum = 0
                    for found_count in range(evidence_len, 0, -1):  # From max to 1
                        at_least_key = f"{evidence_len}_evidence_at_least_{found_count}_found"
                        if at_least_key in partial_breakdown_stats:
                            cumulative_sum = partial_breakdown_stats[at_least_key]
                            partial_breakdown_stats[at_least_key] = f"{cumulative_sum}/{total_q}"

            for bridging_len in range(2, 6):  # Support up to 5 bridging questions
                total_questions_key = f"{bridging_len}_bridging_questions"
                if total_questions_key in partial_breakdown_stats:
                    total_q = partial_breakdown_stats[total_questions_key]

                    # Calculate cumulative "at least N found" from right to left
                    cumulative_sum = 0
                    for found_count in range(bridging_len, 0, -1):  # From max to 1
                        at_least_key = f"{bridging_len}_bridging_at_least_{found_count}_found"
                        if at_least_key in partial_breakdown_stats:
                            cumulative_sum = partial_breakdown_stats[at_least_key]
                            partial_breakdown_stats[at_least_key] = f"{cumulative_sum}/{total_q}"

            results[model_name] = {
                "total_questions": total_questions,
                "partial_evidence_matches": partial_evidence_matches,
                "full_evidence_matches": full_evidence_matches,
                "bridging_entity_matches": bridging_entity_matches,
                "final_answer_in_reasoning": final_answer_in_reasoning,
                "final_answer_in_pred": final_answer_in_pred,
                "sequential_reasoning": sequential_reasoning,
                "partial_breakdown": partial_breakdown_stats,
                "partial_evidence_accuracy": partial_evidence_matches / total_questions
                if total_questions > 0
                else 0,
                "full_evidence_accuracy": full_evidence_matches / total_questions
                if total_questions > 0
                else 0,
                "bridging_entity_accuracy": bridging_entity_matches / total_questions
                if total_questions > 0
                else 0,
                "final_answer_in_reasoning_accuracy": final_answer_in_reasoning / total_questions
                if total_questions > 0
                else 0,
                "final_answer_in_pred_accuracy": final_answer_in_pred / total_questions
                if total_questions > 0
                else 0,
                "sequential_accuracy": sequential_reasoning / total_questions if total_questions > 0 else 0,
            }

        return results

    def analyze_hp_supporting_facts(
        self, ground_truth: dict[str, Any], predictions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze how well models identify supporting facts in HotpotQA dataset."""
        logger.info("Analyzing HotpotQA supporting facts...")

        results = {}
        gt_data = ground_truth["data"]

        for model_name, model_preds in predictions.items():
            logger.info(f"  Processing model: {model_name}")

            total_questions = 0
            partial_supporting_fact_matches = 0
            full_supporting_fact_matches = 0
            final_answer_in_reasoning = 0
            final_answer_in_pred = 0
            sequential_reasoning = 0

            # Track partial breakdowns
            fact_breakdown = defaultdict(int)

            for question in gt_data:
                qid = question.get("_id") or question.get("id") or question.get("question_id")
                if qid not in model_preds:
                    continue

                total_questions += 1
                supporting_facts = question.get("supporting_facts", [])
                if not supporting_facts:
                    continue

                pred = model_preds[qid]
                reasoning_text = self.extract_reasoning_text(pred)

                # Track fact complexity
                fact_length = len(supporting_facts)
                fact_breakdown[f"{fact_length}_fact_questions"] += 1

                # Check supporting fact mentions - look for the actual supporting sentences
                fact_found = []
                for i, fact in enumerate(supporting_facts):
                    title = fact[0]  # Title is the first element
                    sent_id = fact[1]  # Sentence ID is the second element

                    # Find the actual supporting sentence from context
                    supporting_sentence = self._extract_supporting_sentence(question, title, sent_id)
                    if supporting_sentence and self._sentence_mentioned_in_text(
                        supporting_sentence, reasoning_text
                    ):
                        fact_found.append((reasoning_text.lower().find(supporting_sentence.lower()[:50]), i))

                # Count matches
                if len(fact_found) > 0:
                    partial_supporting_fact_matches += 1
                if len(fact_found) == len(supporting_facts):
                    full_supporting_fact_matches += 1

                # Check final answer presence
                correct_answer = question.get("answer", "")
                if correct_answer and self._entity_mentioned_in_text(correct_answer, reasoning_text):
                    final_answer_in_reasoning += 1

                pred_answer = pred.get("pred", "").strip()
                if self._answers_match(pred_answer, correct_answer):
                    final_answer_in_pred += 1

                # Check sequential reasoning
                if len(fact_found) >= 2:
                    fact_found.sort(key=lambda x: x[0])  # Sort by text position
                    is_sequential = all(
                        fact_found[i][1] < fact_found[i + 1][1]  # Check step index order
                        for i in range(len(fact_found) - 1)
                    )
                    if is_sequential:
                        sequential_reasoning += 1

                # Update partial breakdown stats
                fact_found_count = len(fact_found)
                for i in range(1, fact_length + 1):
                    if fact_found_count >= i:
                        fact_breakdown[f"{fact_length}_fact_at_least_{i}_found"] += 1

            # Convert counts to statistics with proper cumulative calculation
            partial_breakdown_stats = dict(fact_breakdown)

            # Fix cumulative calculation for "at least N found" statistics
            for fact_len in range(2, 6):  # Support up to 5 fact questions
                total_questions_key = f"{fact_len}_fact_questions"
                if total_questions_key in partial_breakdown_stats:
                    total_q = partial_breakdown_stats[total_questions_key]

                    # Calculate cumulative "at least N found" from right to left
                    cumulative_sum = 0
                    for found_count in range(fact_len, 0, -1):  # From max to 1
                        at_least_key = f"{fact_len}_fact_at_least_{found_count}_found"
                        if at_least_key in partial_breakdown_stats:
                            cumulative_sum = partial_breakdown_stats[at_least_key]
                            partial_breakdown_stats[at_least_key] = f"{cumulative_sum}/{total_q}"

            results[model_name] = {
                "total_questions": total_questions,
                "partial_supporting_fact_matches": partial_supporting_fact_matches,
                "full_supporting_fact_matches": full_supporting_fact_matches,
                "final_answer_in_reasoning": final_answer_in_reasoning,
                "final_answer_in_pred": final_answer_in_pred,
                "sequential_reasoning": sequential_reasoning,
                "partial_breakdown": partial_breakdown_stats,
                "partial_supporting_fact_accuracy": partial_supporting_fact_matches / total_questions
                if total_questions > 0
                else 0,
                "full_supporting_fact_accuracy": full_supporting_fact_matches / total_questions
                if total_questions > 0
                else 0,
                "final_answer_in_reasoning_accuracy": final_answer_in_reasoning / total_questions
                if total_questions > 0
                else 0,
                "final_answer_in_pred_accuracy": final_answer_in_pred / total_questions
                if total_questions > 0
                else 0,
                "sequential_accuracy": sequential_reasoning / total_questions if total_questions > 0 else 0,
            }

        return results

    def analyze_msq_decomposition(
        self, ground_truth: dict[str, Any], predictions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze how well models identify decomposition steps in MuSiQue dataset."""
        logger.info("Analyzing MuSiQue decomposition...")

        results = {}
        gt_data = ground_truth["data"]

        for model_name, model_preds in predictions.items():
            logger.info(f"  Processing model: {model_name}")

            total_questions = 0
            partial_matches = 0
            full_decomposition_matches = 0
            reasoning_text_matches = 0
            final_answer_in_reasoning = 0
            final_answer_in_pred = 0
            sequential_reasoning = 0

            # Track partial breakdowns
            step_breakdown = defaultdict(int)

            for question in gt_data:
                qid = question.get("id") or question.get("_id") or question.get("question_id")
                if qid not in model_preds:
                    continue

                total_questions += 1
                decomposition = question.get("question_decomposition", [])
                if not decomposition:
                    continue

                pred = model_preds[qid]
                reasoning_text = self.extract_reasoning_text(pred)

                # Track step complexity
                step_length = len(decomposition)
                step_breakdown[f"{step_length}_step_questions"] += 1

                # Check decomposition step answers
                answer_positions = []
                for i, step in enumerate(decomposition):
                    step_answer = step.get("answer", "")
                    if step_answer and self._entity_mentioned_in_text(step_answer, reasoning_text):
                        answer_positions.append((reasoning_text.lower().find(step_answer.lower()), i))

                # Count matches
                if len(answer_positions) > 0:
                    partial_matches += 1
                if len(answer_positions) == len(decomposition):
                    full_decomposition_matches += 1

                # Check if any decomposition answer is in reasoning text (new metric)
                reasoning_text_found = False
                for step in decomposition:
                    step_answer = step.get("answer", "")
                    if step_answer and self._entity_mentioned_in_text(step_answer, reasoning_text):
                        reasoning_text_found = True
                        break
                if reasoning_text_found:
                    reasoning_text_matches += 1

                # Check final answer presence
                correct_answer = question.get("answer", "")
                if correct_answer and self._entity_mentioned_in_text(correct_answer, reasoning_text):
                    final_answer_in_reasoning += 1

                pred_answer = pred.get("pred", "").strip()
                if self._answers_match(pred_answer, correct_answer):
                    final_answer_in_pred += 1

                # Check sequential reasoning: first occurrence of each found answer appears in correct order
                if len(answer_positions) >= 2:
                    answer_positions.sort(key=lambda x: x[0])  # Sort by text position
                    is_sequential = all(
                        answer_positions[i][1] < answer_positions[i + 1][1]  # Check step index order
                        for i in range(len(answer_positions) - 1)
                    )
                    if is_sequential:
                        sequential_reasoning += 1

                # Update partial breakdown stats
                step_found_count = len(answer_positions)
                for i in range(1, step_length + 1):
                    if step_found_count >= i:
                        step_breakdown[f"{step_length}_step_at_least_{i}_found"] += 1

            # Convert counts to statistics with proper cumulative calculation
            partial_breakdown_stats = dict(step_breakdown)

            # Fix cumulative calculation for "at least N found" statistics
            for step_len in range(2, 6):  # Support up to 5 step questions
                total_questions_key = f"{step_len}_step_questions"
                if total_questions_key in partial_breakdown_stats:
                    total_q = partial_breakdown_stats[total_questions_key]

                    # Calculate cumulative "at least N found" from right to left
                    cumulative_sum = 0
                    for found_count in range(step_len, 0, -1):  # From max to 1
                        at_least_key = f"{step_len}_step_at_least_{found_count}_found"
                        if at_least_key in partial_breakdown_stats:
                            cumulative_sum = partial_breakdown_stats[at_least_key]
                            partial_breakdown_stats[at_least_key] = f"{cumulative_sum}/{total_q}"

            results[model_name] = {
                "total_questions": total_questions,
                "partial_matches": partial_matches,
                "full_decomposition_matches": full_decomposition_matches,
                "reasoning_text_matches": reasoning_text_matches,
                "final_answer_in_reasoning": final_answer_in_reasoning,
                "final_answer_in_pred": final_answer_in_pred,
                "sequential_reasoning": sequential_reasoning,
                "partial_breakdown": partial_breakdown_stats,
                "partial_accuracy": partial_matches / total_questions if total_questions > 0 else 0,
                "full_decomposition_accuracy": full_decomposition_matches / total_questions
                if total_questions > 0
                else 0,
                "reasoning_text_accuracy": reasoning_text_matches / total_questions
                if total_questions > 0
                else 0,
                "final_answer_in_reasoning_accuracy": final_answer_in_reasoning / total_questions
                if total_questions > 0
                else 0,
                "final_answer_in_pred_accuracy": final_answer_in_pred / total_questions
                if total_questions > 0
                else 0,
                "sequential_accuracy": sequential_reasoning / total_questions if total_questions > 0 else 0,
            }

        return results

    def save_detailed_results(self, all_results: dict[str, dict[str, Any]], output_path: str):
        """Save detailed results to JSON file."""
        logger.info(f"Saving detailed results to {output_path}...")

        # Structure results for JSON output
        output_data = {
            "analysis_type": "intermediate_reasoning",
            "detailed_results": all_results,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

    def run_full_analysis(self) -> dict[str, Any]:
        """Run the complete intermediate reasoning analysis for all three folders."""
        logger.info("Starting intermediate reasoning analysis for all three folders...")

        datasets = ["2wiki", "hp", "msq"]  # Include all datasets
        all_results = {}

        for dataset in datasets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing dataset: {dataset}")
            logger.info(f"{'='*60}")

            try:
                # Load data
                ground_truth = self.load_ground_truth_data(dataset)
                predictions = self.load_model_predictions(dataset)

                if not predictions:
                    logger.warning(f"No predictions found for {dataset}")
                    continue

                # Run dataset-specific analysis
                if dataset == "2wiki":
                    results = self.analyze_2wiki_evidences(ground_truth, predictions)
                elif dataset == "hp":
                    results = self.analyze_hp_supporting_facts(ground_truth, predictions)
                elif dataset == "msq":
                    results = self.analyze_msq_decomposition(ground_truth, predictions)

                all_results[dataset] = results

            except Exception as e:
                logger.error(f"Error analyzing {dataset}: {e}")
                continue

        return all_results


def main():
    """Main function to run the intermediate reasoning analysis."""
    analyzer = IntermediateReasoningAnalyzer()

    # Run full analysis
    results = analyzer.run_full_analysis()

    # Save detailed results
    output_path = "scripts/intermediate_reasoning_analysis_results.json"
    analyzer.save_detailed_results(results, output_path)

    logger.info("Analysis complete! Results saved to JSON file.")


if __name__ == "__main__":
    main()
