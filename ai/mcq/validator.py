"""Validation and scoring helpers for generated MCQs."""

from __future__ import annotations

from typing import Dict, List


class MCQValidator:
    @staticmethod
    def validate_mcq(row: Dict) -> bool:
        if not isinstance(row, dict):
            return False

        question = str(row.get("question") or "").strip()
        options = row.get("options") if isinstance(row.get("options"), list) else []
        answer = str(row.get("correct_answer") or row.get("answer") or "").strip().upper()

        if not question or len(question) < 6:
            return False
        if len(options) != 4:
            return False
        if any(not str(opt).strip() for opt in options):
            return False
        if answer not in {"A", "B", "C", "D"}:
            return False

        return True

    @staticmethod
    def score_answers(mcqs: List[Dict], user_answers: Dict[int, str]) -> dict:
        total = len(mcqs or [])
        correct = 0
        breakdown = []

        for idx, row in enumerate(mcqs or []):
            expected = str(row.get("correct_answer") or row.get("answer") or "").strip().upper()
            given = str(user_answers.get(idx, "")).strip().upper() if isinstance(user_answers, dict) else ""
            is_correct = bool(expected and given and expected == given)
            if is_correct:
                correct += 1
            breakdown.append(
                {
                    "index": idx,
                    "expected": expected,
                    "given": given,
                    "correct": is_correct,
                }
            )

        percent = round((correct / total) * 100, 2) if total else 0.0
        return {
            "total_questions": total,
            "correct_answers": correct,
            "score_percent": percent,
            "breakdown": breakdown,
        }
