import random

def generate_fallback_mcqs(topic: str, count: int, difficulty: str, subject: str, grade: str) -> list:
    """
    Generate readable fallback MCQs when external AI generation is unavailable.
    These are deterministic STEM-safe templates with clear, domain-relevant wording.
    """
    safe_topic = (topic or "general concept").strip().replace("_", " ").title()
    safe_subject = (subject or "stem").strip().lower()
    safe_grade = (grade or "high").strip().lower()
    safe_difficulty = (difficulty or "medium").strip().lower()

    def _subject_noun():
        mapping = {
            "mathematics": "model",
            "science": "investigation",
            "physics": "system",
            "chemistry": "reaction setup",
            "biology": "biological process",
            "technology": "software workflow",
            "engineering": "design process",
        }
        return mapping.get(safe_subject, "problem-solving workflow")

    def _difficulty_hint():
        if safe_difficulty == "easy":
            return "Start from the core definition and identify the most direct concept."
        if safe_difficulty == "hard":
            return "Check assumptions, constraints, and which option is technically defensible."
        return "Focus on method selection and evidence-based reasoning."

    templates = [
        {
            "text": f"In a {safe_grade} {safe_subject.capitalize()} class, what is the best first step when solving a {safe_topic} problem?",
            "correct": f"Define the goal, list known information, and choose an approach that fits the { _subject_noun() }.",
            "distractors": [
                "Memorize one previous answer pattern and apply it without checking context.",
                "Skip the problem statement and begin with random option elimination.",
                "Ignore constraints because they usually do not affect the final answer.",
            ],
            "explanation": f"Strong solutions in {safe_topic} begin with clear goals, knowns, and method fit."
        },
        {
            "text": f"Which statement is most accurate about using {safe_topic} in practical {safe_subject.capitalize()} tasks?",
            "correct": "Reliable results depend on correct assumptions, valid steps, and verification of outcomes.",
            "distractors": [
                "Any method is acceptable if it produces an answer quickly.",
                "Context never matters once you know one formula.",
                "Verification is optional when the topic appears familiar.",
            ],
            "explanation": "Practical application requires valid assumptions, method correctness, and checks."
        },
        {
            "text": f"While reviewing answers in {safe_topic}, what is the strongest quality check?",
            "correct": "Test whether the result is logically consistent with the problem conditions and units/constraints.",
            "distractors": [
                "Prefer the longest explanation, because longer answers are usually correct.",
                "Accept the first answer that matches a keyword from class notes.",
                "Ignore edge cases because typical examples are sufficient for all situations.",
            ],
            "explanation": "Quality checks should validate consistency, conditions, and constraints."
        },
    ]

    mcqs = []
    for idx in range(max(1, int(count or 1))):
        template = templates[idx % len(templates)]
        correct = template["correct"]
        options = [correct] + list(template["distractors"])
        random.shuffle(options)
        mcqs.append({
            "text": template["text"],
            "options": options,
            "correct_index": options.index(correct),
            "explanation": template["explanation"],
            "hint": _difficulty_hint(),
            "difficulty": safe_difficulty,
            "topic": safe_topic,
            "source": "robust_fallback_v2",
        })
    return mcqs