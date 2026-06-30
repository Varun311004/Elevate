"""Build a simple SFT JSONL dataset for TinyLlama LoRA fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def to_record(item: dict) -> dict | None:
    question = str(item.get("question") or item.get("text") or "").strip()
    options = item.get("options") if isinstance(item.get("options"), list) else []
    answer = str(item.get("answer") or item.get("correct_answer") or "").strip().upper()
    explanation = str(item.get("explanation") or "").strip()

    if not question or len(options) < 4:
        return None
    if answer not in {"A", "B", "C", "D"}:
        return None

    payload = {
        "question": question,
        "options": {
            "A": str(options[0]),
            "B": str(options[1]),
            "C": str(options[2]),
            "D": str(options[3]),
        },
        "answer": answer,
        "explanation": explanation or "Based on reference curriculum facts.",
    }

    prompt = (
        "Generate one factual STEM MCQ and return JSON with keys "
        "question, options(A-D), answer, explanation."
    )
    return {
        "prompt": prompt,
        "completion": json.dumps(payload, ensure_ascii=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file (list of question rows)")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Input file must contain a JSON array")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            rec = to_record(row if isinstance(row, dict) else {})
            if not rec:
                continue
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            written += 1

    print(f"Wrote {written} records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
