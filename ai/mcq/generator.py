"""Topic-based MCQ generator using local LLM."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import json
import random
import re
import threading
import time
from typing import Dict, List, Optional

from web_context import build_topic_web_context


class MCQGenerator:
    """
    Generates Multiple Choice Questions using a language model.
    This class is responsible for prompt engineering, model inference, and parsing the output.
    """
    def __init__(self) -> None:
        self.llm = None
        self._llm_lock = threading.Lock()
        self._result_cache: OrderedDict[tuple, List[Dict]] = OrderedDict()
        self._cache_limit = 128
        self.last_generation_meta: Dict = {}

    def ensure_model_loaded(self):
        """Initializes the language model if it hasn't been already."""
        from models.llm import get_llm_model
        if self.llm is not None:
            return self.llm
        with self._llm_lock:
            if self.llm is None:
                self.llm = get_llm_model()
        return self.llm

    def is_model_loaded(self) -> bool:
        """Checks if the model has been loaded into memory."""
        return self.llm is not None

    def _cache_key(
        self,
        *,
        topic: str,
        num_questions: int,
        difficulty: str,
        subject: str,
        grade: str,
        seed: Optional[int],
        test_title: str,
        test_description: str,
    ) -> tuple:
        """Creates a unique tuple key for caching generation requests."""
        return (
            str(topic or "").strip().lower(),
            int(num_questions),
            str(difficulty or "medium").strip().lower(),
            str(subject or "science").strip().lower(),
            str(grade or "high").strip().lower(),
            int(seed) if seed is not None else None,
            str(test_title or "").strip().lower()[:160],
            str(test_description or "").strip().lower()[:320],
        )

    def generate_from_topic(
        self,
        topic: str,
        num_questions: int = 5,
        difficulty: str = "medium",
        subject: str = "science",
        grade: str = "high",
        seed: Optional[int] = None,
        test_title: Optional[str] = None,
        test_description: Optional[str] = None,
    ) -> List[Dict]:
        self.ensure_model_loaded()

        safe_topic = str(topic or "").strip()
        safe_test_title = str(test_title or "").strip()
        safe_test_description = str(test_description or "").strip()
        if not safe_topic:
            safe_topic = " ".join(
                part for part in [safe_test_title, safe_test_description] if part
            ).strip() or "general science"

        requested_count = int(max(1, min(num_questions, 50)))
        cache_key = self._cache_key(
            topic=safe_topic, num_questions=requested_count, difficulty=difficulty,
            subject=subject, grade=grade, seed=seed,
            test_title=safe_test_title, test_description=safe_test_description,
        )

        cached_rows = self._result_cache.get(cache_key)
        if cached_rows is not None:
            self._result_cache.move_to_end(cache_key)
            return deepcopy(cached_rows[:requested_count])

        # Fetch web context with a hard 3-second timeout so it never blocks generation
        facts: List[str] = []
        try:
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError("web context timed out")
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(3)
            try:
                context = build_topic_web_context(safe_topic)
                facts = self._extract_facts(context, safe_topic)
            finally:
                signal.alarm(0)
        except Exception:
            # Web context is optional — fall back to topic name as a fact
            facts = [f"{safe_topic} is an important concept in {subject}."]

        final_rows = self._generate_llm_mcqs(
            topic=safe_topic, subject=subject, grade=grade, difficulty=difficulty,
            requested_count=requested_count, facts=facts,
            test_title=safe_test_title, test_description=safe_test_description,
            seed=seed,
        )

        self.last_generation_meta = {
            "requested": requested_count,
            "produced": len(final_rows),
            "llm_count": len(final_rows),
            "template_count": 0,
            "facts_count": len(facts),
            "cache_hit": False,
        }

        if final_rows:
            self._result_cache[cache_key] = deepcopy(final_rows)
            if len(self._result_cache) > self._cache_limit:
                self._result_cache.popitem(last=False)

        return final_rows

    def _grade_guidance(self, grade: str) -> str:
        """Provides specific instructions based on the grade level."""
        grade_key = str(grade or "").strip().lower()
        if grade_key == "elementary":
            return "Use simple, concrete vocabulary suitable for young learners. Focus on foundational concepts."
        if grade_key == "middle":
            return "Use standard academic vocabulary. Questions can involve 1-2 steps of reasoning."
        if grade_key == "high":
            return "Use rigorous, exam-style language. Questions should test for deep conceptual understanding and application."
        if grade_key == "college":
            return "Use concise, technical language. Assume foundational knowledge and test deeper or more abstract concepts."
        return "Match question wording to a standard curriculum for the grade level."

    def _difficulty_guidance(self, difficulty: str) -> str:
        """Provides specific instructions based on the difficulty level."""
        difficulty_key = str(difficulty or "").strip().lower()
        if difficulty_key == "easy":
            return "Focus on direct fact recall and single-step problems. Distractors can be clearly incorrect."
        if difficulty_key == "medium":
            return "Require application of concepts or 2-3 steps of reasoning. Distractors should be plausible."
        if difficulty_key == "hard":
            return "Require synthesis of multiple concepts or nuanced, multi-step reasoning. Distractors should target common misconceptions."
        return "Align the question's challenge level to the requested difficulty."

    def _create_llm_prompt(
        self,
        *,
        topic: str,
        subject: str,
        grade: str,
        num_questions: int,
        difficulty: str,
        facts: List[str],
        test_title: str = "",
        test_description: str = "",
        existing_questions: Optional[List[str]] = None,
    ) -> str:
        # Provide up to 12 facts for richer, factual context
        fact_lines = "\n".join(f"- {fact}" for fact in facts[:12])
        
        # Build strict context constraints dynamically
        context_block = ""
        if test_title or test_description:
            context_block += "TEST CONTEXT (Must be deeply integrated into the questions):\n"
            if test_title:
                context_block += f"Test Title: {test_title}\n"
            if test_description:
                context_block += f"Test Description: {test_description}\n"
            context_block += "\n"

        return (
            f"You are an expert educational assessment designer. Your task is to generate exactly {num_questions} original, highly sensible, and factual multiple-choice questions.\n\n"
            f"TARGET AUDIENCE & PARAMETERS:\n"
            f"- STEM Subject: {subject.capitalize()}\n"
            f"- Sub-Topic: {topic}\n"
            f"- Grade Level: {grade.capitalize()}\n"
            f"- Difficulty: {difficulty.capitalize()}\n\n"
            f"{context_block}"
            f"STRICT INSTRUCTIONS:\n"
            f"1. Alignment: The questions MUST align perfectly with the Test Title, Description, Subject, and Sub-Topic.\n"
            f"2. Complexity: Calibrate the depth of the question, the vocabulary, and the reasoning required strictly to the '{grade.capitalize()}' grade level and '{difficulty.capitalize()}' difficulty.\n"
            f"3. Practicality: Make questions sensible, practical, and highly useful for actual student practice. Use multi-line scenarios or real-world examples if they fit the difficulty.\n"
            f"4. Options: Ensure all 4 options are plausible and challenging (especially for medium/hard difficulty), but only ONE is definitively correct.\n"
            f"5. Accuracy: Use the following reference facts to ensure strict factual accuracy:\n"
            f"{fact_lines}\n\n"
            "DO NOT OUTPUT JSON. Output EXACTLY this plain-text format for each question, separated by a blank line:\n\n"
            "Question: <the question text>\n"
            "A) <first option>\n"
            "B) <second option>\n"
            "C) <third option>\n"
            "D) <fourth option>\n"
            "Answer: <A, B, C, or D>\n"
            "Explanation: <clear, educational explanation of why the answer is correct>\n"
        )

    def _generate_llm_mcqs(
        self,
        *,
        topic: str,
        subject: str,
        grade: str,
        difficulty: str,
        requested_count: int,
        facts: List[str],
        test_title: str = "",
        test_description: str = "",
        seed: Optional[int] = None,
    ) -> List[Dict]:
        if requested_count <= 0:
            return []

        llm = self.ensure_model_loaded()
        collected: List[Dict] = []
        attempts = 0
        
        # Max 2 attempts. If the first fails, the second will rescue it.
        while len(collected) < requested_count and attempts < 2:
            remaining = requested_count - len(collected)
            prompt = self._create_llm_prompt(
                topic=topic,
                subject=subject,
                grade=grade,
                num_questions=remaining,
                difficulty=difficulty,
                facts=facts,
                test_title=test_title,
                test_description=test_description,
            )

            # Generate text
            response = llm.generate(
                prompt=prompt,
                max_new_tokens=800,
                temperature=0.1,
                top_p=0.95,
                max_time=45.0, # Timeout the LLM at 45s so the backend doesn't crash
            )
            
            # Use your original parser to cleanly chop the text into dicts
            parsed = self._parse_llm_output(response, difficulty, topic)
            
            if parsed:
                collected.extend(parsed)
                collected = self._dedupe_questions(collected)

            attempts += 1

        return collected[:requested_count]

    def _parse_llm_output(self, response: str, difficulty: str, topic: str) -> List[Dict]:
        """
        Parses plain text output from the LLM into structured dictionaries.
        Highly robust to minor AI formatting mistakes (e.g. A. instead of A) ).
        """
        candidates: List[Dict] = []
        text = str(response or "").strip()

        # Split the entire response into blocks, where each block starts with "Question:"
        # (?i) makes it case-insensitive just in case the AI writes "question:"
        question_blocks = re.split(r"(?i)(?=^Question:)", text, flags=re.MULTILINE)

        for block in question_blocks:
            block = block.strip()
            if not block:
                continue

            try:
                # Extract question
                question_match = re.search(r"(?i)^Question:\s*(.+)", block, re.MULTILINE)
                if not question_match:
                    continue
                question = question_match.group(1).strip()

                # Extract options (Supports "A)" or "A.")
                options_matches = re.findall(r"(?i)^[A-D][\)\.]\s*(.+)", block, re.MULTILINE)
                if len(options_matches) < 4:
                    continue
                options = [opt.strip() for opt in options_matches[:4]]

                # Extract answer
                answer_match = re.search(r"(?i)^Answer:\s*([A-D])", block, re.MULTILINE)
                if not answer_match:
                    continue
                answer = answer_match.group(1).upper().strip()

                # Extract explanation (If missing, gracefully fallback)
                explanation_match = re.search(r"(?i)^Explanation:\s*(.+)", block, re.MULTILINE)
                explanation = explanation_match.group(1).strip() if explanation_match else f"The correct answer is {answer}."
                
                normalized = self._normalize_candidate(
                    item={
                        "question": question,
                        "options": options,
                        "answer": answer,
                        "explanation": explanation,
                    },
                    difficulty=difficulty,
                    topic=topic,
                    source="llm_text"
                )
                if normalized:
                    candidates.append(normalized)

            except Exception as e:
                # If a single question is hopelessly garbled, ignore it and parse the rest
                print(f"[Parser Warning] Failed to parse a block: {e}")
                continue
                
        return self._dedupe_questions(candidates)


    def _parse_json_output(self, response: str, difficulty: str, topic: str) -> List[Dict]:
        """Parse JSON array output from the LLM. Robust to common model quirks."""
    
        text = str(response or "").strip()

        # Strip markdown fences if model added them anyway
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # Extract the first [...] block
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            print(f"[parser] No JSON array found in output. First 200 chars: {text[:200]}")
            return []

        json_str = text[start:end + 1]

        # Attempt 1: direct parse
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Attempt 2: fix common model errors — trailing commas, single quotes
            json_str_fixed = re.sub(r",\s*([\]}])", r"\1", json_str)  # trailing commas
            json_str_fixed = json_str_fixed.replace("'", '"')           # single → double quotes
            # Fix unescaped newlines inside strings
            json_str_fixed = re.sub(r'(?<!\\)\n(?=[^"]*")', ' ', json_str_fixed)
            try:
                data = json.loads(json_str_fixed)
            except json.JSONDecodeError as e:
                print(f"[parser] JSON decode failed after fix attempt: {e}. Snippet: {json_str[:300]}")
                return []

        if not isinstance(data, list):
            # Model may have returned {"questions": [...]}
            if isinstance(data, dict):
                for key in ("questions", "mcqs", "items", "data"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
            if not isinstance(data, list):
                return []

        candidates: List[Dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            normalized = self._normalize_candidate(
                item=item, difficulty=difficulty, topic=topic, source="llm_json"
            )
            if normalized:
                candidates.append(normalized)

        return self._dedupe_questions(candidates)

    def _normalize_candidate(self, item: Dict, difficulty: str, topic: str, source: str) -> Optional[Dict]:
        """Validates and standardizes a single parsed question dictionary."""
        question = str(item.get("question") or "").strip()
        if not question.endswith("?"):
            question += "?"

        options = item.get("options")
        if not isinstance(options, list) or len(options) != 4:
            return None
        
        answer = str(item.get("answer") or item.get("correct_answer") or "").strip().upper()
        if answer not in ["A", "B", "C", "D"]:
            return None
            
        explanation = str(item.get("explanation") or "").strip()
        if not all([question, explanation]):
            return None

        correct_index = ["A", "B", "C", "D"].index(answer)

        return {
            "question": question,
            "options": options,
            "correct_answer": answer,
            "correct_index": correct_index,
            "explanation": explanation,
            "hint": f"Review the facts about {topic} related to this question.",
            "difficulty": difficulty,
            "topic": topic,
            "source": source,
        }

    def _dedupe_questions(self, rows: List[Dict]) -> List[Dict]:
        """Removes duplicate questions based on text and options."""
        unique: List[Dict] = []
        seen = set()
        for row in rows:
            question_key = " ".join(str(row.get("question", "")).lower().split())
            options_key = "|".join(sorted(" ".join(str(opt).lower().split()) for opt in row.get("options", [])))
            key = f"{question_key}|{options_key}"
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(row)
        return unique
    
    def _extract_facts(self, context: str, topic: str) -> List[str]:
        """A simple utility to extract sentences from web context."""
        text = str(context or "").replace("\r", " ").strip()
        # Basic sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        facts = [s.strip() for s in sentences if len(s.strip()) > 20 and len(s.strip()) < 300]
        
        if not facts:
            return [f"{topic} is a key area of study in this subject."]
        return facts


_mcq_generator: MCQGenerator | None = None
_mcq_generator_lock = threading.Lock()


def get_mcq_generator() -> MCQGenerator:
    """Singleton factory for the MCQGenerator."""
    global _mcq_generator
    if _mcq_generator is None:
        with _mcq_generator_lock:
            if _mcq_generator is None:
                _mcq_generator = MCQGenerator()
    return _mcq_generator
