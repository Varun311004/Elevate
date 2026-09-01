"""
Elevate Practice Question Bank Generator
=========================================

Single-file demo + production generator for the PostgreSQL/Supabase
practice_questions table.

IMPORTANT ARCHITECTURE
----------------------
Generation remains STRICTLY SEQUENTIAL:

    Model 1 -> Model 2 -> Model 3 -> ... -> Model 8

The dedicated validator is independent and runs concurrently.

The validator:
- owns its own queue;
- takes up to a configured maximum number of questions at a time;
- does not wait for more questions once it has a batch;
- validates the current batch independently;
- immediately checks the queue again;
- continues until generation is finished AND the queue is empty;
- supports different batch sizes for demo and production.

Demo:
    python seed_questions.py --mode demo

Production:
    python seed_questions.py --mode production
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field, replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from sqlalchemy import (bindparam, create_engine, text)
from sqlalchemy.engine import Engine


# ============================================================================
# PATHS / ENVIRONMENT
# ============================================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"

STATE_FILE = DATA_DIR / "question_generation_state.json"

# Optional file where you can paste in the CURRENT rate limits shown on
# your own Google AI Studio / Groq console, without touching code.
# Provider free-tier numbers change over time and vary by account, so the
# constants baked into MODEL_HIERARCHY below are defaults, not guarantees.
# See apply_rate_limit_overrides() for the file format.
RATE_LIMIT_OVERRIDE_FILE = DATA_DIR / "model_limits_override.json"

VALIDATION_PENDING_DIR = (
    DATA_DIR / "pending_validation"
)

load_dotenv(
    PROJECT_ROOT / ".env",
    override=False,
)

load_dotenv(
    BACKEND_DIR / ".env",
    override=False,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_DEMO_QUESTIONS_PER_MODEL = 5

DEFAULT_PRODUCTION_BATCH_GEMINI = 5
DEFAULT_PRODUCTION_BATCH_GROQ = 4

# Validator batch sizes.
#
# Demo:
#   up to 5 questions per validation pass
#
# Production:
#   up to 10 questions per validation pass
#
VALIDATOR_BATCH_DEMO = 5
VALIDATOR_BATCH_PRODUCTION = 10

# Small grace period after receiving the first question.
#
# This gives the producer thread a moment to place other questions
# into the same validator queue before the worker decides its batch size.
VALIDATOR_BATCH_GATHER_DELAY = 0.10

DEFAULT_HTTP_TIMEOUT = 120

WARNING_LEVELS = (
    0.75,
    0.90,
    0.95,
)

VALIDATOR_MODELS = (
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
)
VALIDATOR_API_KEY_ENV = "GEMINI_VALIDATOR_API_KEY"

VALIDATION_QUEUE_SIZE = 64
VALIDATOR_WORKER_COUNT = len(
    VALIDATOR_MODELS
)

VALIDATOR_HTTP_TIMEOUT = 120

VALIDATOR_MIN_SCORE = 85

# A rejected question gets at most ONE repair attempt.
# Repair is used only when the validator believes the correction
# is obvious and can be made confidently.
VALIDATOR_REPAIR_MIN_CONFIDENCE = 95

GENERATION_PARSE_RETRY_LIMIT = 4
VALIDATOR_RETRY_LIMIT = 5

# ============================================================================
# SMART VALIDATOR SCHEDULER
# ============================================================================

# Pause generation when the total validator backlog reaches this fraction
# of the total queue capacity.
VALIDATOR_QUEUE_HIGH_WATERMARK = 0.80

# Resume generation only after the backlog falls to this fraction.
VALIDATOR_QUEUE_RESUME_WATERMARK = 0.50

# When validators are near their daily quota, begin slowing generation once
# the backlog reaches this fraction of total capacity.
VALIDATOR_QUEUE_SOFT_WATERMARK = 0.35

# A validator is considered "expiring" when its daily quota usage reaches this.
VALIDATOR_QUOTA_PRESSURE_RATIO = 0.90

# Polling/smoothing delays for the scheduler.
VALIDATOR_PRESSURE_POLL_SECONDS = 0.50
VALIDATOR_PRESSURE_SLOWDOWN_SECONDS = 2.0

# ============================================================================
# PROVIDER-AUTHORITATIVE QUOTA MODE
# ============================================================================

# Gemini does not expose a supported API endpoint that returns the active
# AI Studio RPM/TPM/RPD limits before a request. Therefore provider responses
# are authoritative; local MODEL_HIERARCHY values are only fallback hints.
PROVIDER_AUTHORITATIVE_LIMITS = True

# Gemini RPD resets at midnight Pacific time.
GEMINI_RESET_ZONE = "America/Los_Angeles"

# Never deliberately hammer a provider while discovering a live limit.
# A provider 429 is handled by retry/backoff instead.
PROVIDER_MIN_BACKOFF_SECONDS = 2.0
PROVIDER_MAX_BACKOFF_SECONDS = 300.0

SUPPORTED_GRADES = (
    "elementary",
    "middle",
    "high",
    "college",
)

SUPPORTED_SUBJECTS = (
    "science",
    "technology",
    "engineering",
    "mathematics",
)

SUPPORTED_DIFFICULTIES = (
    "easy",
    "medium",
    "hard",
)


# ============================================================================
# GRADE-AWARE STEM CURRICULUM
# ============================================================================
#
# IMPORTANT:
# These are curriculum boundaries, not merely topic suggestions.
#
# Difficulty changes the cognitive difficulty INSIDE the grade-appropriate
# curriculum. It does NOT allow a learner to jump into an inappropriate
# academic level.
#
# Example:
#   middle + mathematics + hard
# can contain harder algebra/geometry/statistics reasoning,
# but must NOT suddenly contain differential equations or calculus.
#
# The structure is:
#
# GRADE -> SUBJECT -> DIFFICULTY -> SUB-TOPICS
#
# This replaces the previous subject-only syllabus mapping.
# ============================================================================

STEM_CURRICULUM = {

    # ========================================================================
    # ELEMENTARY
    # ========================================================================

    "elementary": {

        "science": {

            "easy": [
                "Living and Non-Living Things",
                "Plants and Their Parts",
                "Animals and Their Habitats",
                "Weather and Seasons",
                "Light and Shadows",
                "Sound and Vibrations",
                "Pushes and Pulls",
                "States of Matter",
            ],

            "medium": [
                "Plant Growth and Needs",
                "Animal Adaptations",
                "Food Chains and Simple Ecosystems",
                "Properties of Materials",
                "Changes in Matter",
                "Energy in Everyday Life",
                "Water Cycle",
                "Earth, Sun, and Moon Patterns",
            ],

            "hard": [
                "Forces and Motion in Everyday Situations",
                "Energy Transfer in Simple Systems",
                "Matter and Its Observable Properties",
                "Ecosystem Relationships",
                "Adaptations and Survival",
                "Weather Patterns and Data",
                "Earth and Its Natural Processes",
                "Simple Scientific Investigation and Evidence",
            ],
        },

        "technology": {

            "easy": [
                "Computer Parts and Devices",
                "Using a Keyboard and Mouse",
                "Basic Digital Files and Folders",
                "Digital Safety",
                "Simple Algorithms",
                "Patterns and Sequences",
                "Introduction to Coding",
                "Digital Communication Basics",
            ],

            "medium": [
                "Block-Based Programming",
                "Loops and Repetition",
                "Conditions and Decisions",
                "Variables as Data Containers",
                "Debugging Simple Programs",
                "Information and Data Representation",
                "Internet and Web Basics",
                "Responsible Technology Use",
            ],

            "hard": [
                "Decomposing Problems into Steps",
                "Nested Patterns and Loops",
                "Simple Event-Driven Programs",
                "Debugging Multi-Step Programs",
                "Simple Data Collection and Analysis",
                "Computational Thinking",
                "Digital Privacy and Security Basics",
                "How Search and Recommendation Systems Work",
            ],
        },

        "engineering": {

            "easy": [
                "What Engineers Do",
                "Simple Machines",
                "Structures and Stability",
                "Materials and Their Uses",
                "Forces in Everyday Objects",
                "Simple Tools and Mechanisms",
                "Building and Testing Simple Designs",
                "Engineering Safety",
            ],

            "medium": [
                "Engineering Design Process",
                "Criteria and Constraints",
                "Strong and Stable Structures",
                "Simple Mechanical Systems",
                "Simple Electrical Systems",
                "Material Selection",
                "Testing and Improving Designs",
                "Trade-Offs in Simple Designs",
            ],

            "hard": [
                "Designing for Multiple Constraints",
                "Structural Strength and Stability",
                "Energy Transfer in Designed Systems",
                "Mechanical Advantage",
                "Simple Circuit Design",
                "Failure Analysis of Simple Designs",
                "Optimization of Simple Products",
                "Environmental Impact of Designs",
            ],
        },

        "mathematics": {

            "easy": [
                "Counting and Place Value",
                "Addition and Subtraction",
                "Basic Multiplication",
                "Basic Division",
                "Fractions as Parts of Wholes",
                "Basic Measurement",
                "Basic Shapes and Geometry",
                "Simple Data and Graphs",
            ],

            "medium": [
                "Multi-Digit Arithmetic",
                "Multiplication and Division Word Problems",
                "Equivalent Fractions",
                "Decimals",
                "Perimeter and Area",
                "Time and Money Problems",
                "Bar Graphs and Line Plots",
                "Patterns and Number Relationships",
            ],

            "hard": [
                "Multi-Step Word Problems",
                "Fraction Operations",
                "Decimal Operations",
                "Ratios in Simple Contexts",
                "Area and Perimeter Reasoning",
                "Volume of Simple Shapes",
                "Interpreting Tables and Graphs",
                "Multi-Step Numerical Patterns",
            ],
        },
    },


    # ========================================================================
    # MIDDLE
    # ========================================================================

    "middle": {

        "science": {

            "easy": [
                "Matter and Its Properties",
                "Chemical and Physical Changes",
                "Forces and Motion",
                "Energy Transfer",
                "Cells and Cell Structure",
                "Ecosystems",
                "Earth Systems",
                "Weather and Climate",
            ],

            "medium": [
                "Particle Model of Matter",
                "Chemical Reactions",
                "Newtonian Motion Concepts",
                "Energy Transfer in Systems",
                "Cell Structure and Function",
                "Matter and Energy in Ecosystems",
                "Natural Selection and Adaptation",
                "Earth-Sun-Moon Systems",
            ],

            "hard": [
                "Interactions of Forces",
                "Energy Conservation and Transfer",
                "Chemical Reaction Evidence",
                "Structure and Function in Living Systems",
                "Population Changes and Natural Selection",
                "Ecosystem Stability",
                "Climate Patterns and Human Impacts",
                "Earth's Geological Processes",
            ],
        },

        "technology": {

            "easy": [
                "Algorithms and Flowcharts",
                "Variables and Data",
                "Basic Programming",
                "Conditional Statements",
                "Loops",
                "Digital Information",
                "Computer Hardware and Software",
                "Internet Fundamentals",
            ],

            "medium": [
                "Functions and Modular Programming",
                "Lists and Collections",
                "Algorithm Efficiency Concepts",
                "Data Representation",
                "Basic Database Concepts",
                "Networks and Internet Protocols",
                "Cybersecurity Fundamentals",
                "Computing Impacts and Ethics",
            ],

            "hard": [
                "Searching and Sorting Algorithms",
                "Data Structures Fundamentals",
                "Algorithm Complexity Concepts",
                "Binary and Other Data Representations",
                "Database Design Fundamentals",
                "Network Architecture",
                "Cybersecurity and Authentication",
                "Algorithmic Bias and Digital Ethics",
            ],
        },

        "engineering": {

            "easy": [
                "Engineering Design Process",
                "Forces and Simple Machines",
                "Structural Stability",
                "Basic Electrical Circuits",
                "Mechanical Systems",
                "Properties of Engineering Materials",
                "Energy in Designed Systems",
                "Engineering Safety",
            ],

            "medium": [
                "Engineering Constraints",
                "Mechanical Advantage",
                "Series and Parallel Circuits",
                "Structural Loads",
                "Fluid Properties",
                "Energy Efficiency",
                "Material Selection",
                "Prototype Testing",
            ],

            "hard": [
                "Force Analysis in Simple Systems",
                "Circuit Analysis Fundamentals",
                "Fluid Pressure and Flow",
                "Structural Failure and Safety",
                "Energy Conversion Systems",
                "Engineering Optimization",
                "Design Trade-Offs",
                "Environmental Engineering Concepts",
            ],
        },

        "mathematics": {

            "easy": [
                "Ratios and Proportions",
                "Integers",
                "Fractions and Decimals",
                "Percentages",
                "Expressions and Equations",
                "Area and Volume",
                "Basic Statistics",
                "Basic Probability",
            ],

            "medium": [
                "Linear Equations",
                "Ratios and Proportional Relationships",
                "Percent Applications",
                "Inequalities",
                "Coordinate Geometry",
                "Surface Area and Volume",
                "Statistics and Data Distributions",
                "Probability Models",
            ],

            "hard": [
                "Systems of Linear Relationships",
                "Multi-Step Linear Equations",
                "Inequalities and Absolute Value",
                "Pythagorean Theorem Applications",
                "Transformations and Similarity",
                "Probability and Compound Events",
                "Statistical Sampling and Variability",
                "Nonlinear Patterns and Functions",
            ],
        },
    },


    # ========================================================================
    # HIGH
    # ========================================================================

    "high": {

        "science": {

            "easy": [
                "Atomic Structure",
                "Chemical Bonding",
                "Newton's Laws",
                "Energy and Work",
                "Cell Biology",
                "Genetics",
                "Ecosystems",
                "Earth and Space Systems",
            ],

            "medium": [
                "Chemical Reactions and Stoichiometry",
                "Molecular Structure",
                "Momentum and Conservation Laws",
                "Electricity and Magnetism",
                "Cellular Respiration and Photosynthesis",
                "Inheritance and Genetic Variation",
                "Population Dynamics",
                "Climate and Earth Systems",
            ],

            "hard": [
                "Reaction Rates and Equilibrium",
                "Thermodynamics",
                "Electromagnetic Interactions",
                "Wave Phenomena",
                "Gene Expression and Regulation",
                "Natural Selection and Evolution",
                "Biogeochemical Cycles",
                "Advanced Earth-System Interactions",
            ],
        },

        "technology": {

            "easy": [
                "Programming Fundamentals",
                "Data Structures Fundamentals",
                "Algorithms Fundamentals",
                "Computer Architecture",
                "Databases",
                "Networking Basics",
                "Cybersecurity Basics",
                "Web and Software Development",
            ],

            "medium": [
                "Object-Oriented Programming",
                "Stacks, Queues, and Trees",
                "Searching and Sorting",
                "Database Design",
                "Computer Networks",
                "Operating System Concepts",
                "Cybersecurity Concepts",
                "Introduction to Artificial Intelligence",
            ],

            "hard": [
                "Algorithm Complexity",
                "Graphs and Graph Algorithms",
                "Advanced Data Structures",
                "Operating Systems and Concurrency",
                "Computer Networking and Protocols",
                "Database Transactions",
                "Machine Learning Fundamentals",
                "Cybersecurity and Cryptography Fundamentals",
            ],
        },

        "engineering": {

            "easy": [
                "Engineering Measurements",
                "Statics Fundamentals",
                "Basic Circuit Analysis",
                "Material Properties",
                "Thermal Systems",
                "Fluid Mechanics Fundamentals",
                "Engineering Design",
                "Engineering Safety",
            ],

            "medium": [
                "Free-Body Diagrams",
                "Stress and Strain",
                "Circuit Laws",
                "Thermodynamic Processes",
                "Fluid Flow Fundamentals",
                "Control System Basics",
                "Engineering Materials",
                "Engineering Economics",
            ],

            "hard": [
                "Static Equilibrium",
                "Stress-Strain Analysis",
                "AC Circuit Analysis",
                "Thermodynamic Cycles",
                "Fluid Dynamics",
                "Control Systems",
                "Engineering Optimization",
                "Engineering Ethics and Risk",
            ],
        },

        "mathematics": {

            "easy": [
                "Algebraic Expressions",
                "Linear Equations",
                "Quadratic Equations",
                "Functions",
                "Coordinate Geometry",
                "Basic Probability",
                "Statistics",
                "Trigonometric Ratios",
            ],

            "medium": [
                "Polynomial Functions",
                "Exponential and Logarithmic Functions",
                "Sequences and Series",
                "Trigonometric Functions",
                "Analytic Geometry",
                "Probability Distributions",
                "Statistical Inference Basics",
                "Limits and Introductory Calculus",
            ],

            "hard": [
                "Advanced Functions",
                "Complex Numbers",
                "Trigonometric Identities",
                "Limits and Continuity",
                "Differentiation",
                "Integration Fundamentals",
                "Advanced Probability",
                "Vectors and Three-Dimensional Geometry",
            ],
        },
    },


    # ========================================================================
    # COLLEGE
    # ========================================================================

    "college": {

        "science": {

            "easy": [
                "Classical Mechanics",
                "Thermodynamics Fundamentals",
                "Electricity and Magnetism Fundamentals",
                "General Chemistry",
                "Cell Biology",
                "Genetics Fundamentals",
                "Ecology Fundamentals",
                "Earth and Planetary Science",
            ],

            "medium": [
                "Advanced Mechanics",
                "Electromagnetic Fields",
                "Thermodynamics and Statistical Concepts",
                "Organic Chemistry Fundamentals",
                "Molecular Biology",
                "Evolutionary Biology",
                "Environmental Science",
                "Geological Processes",
            ],

            "hard": [
                "Quantum Mechanics Fundamentals",
                "Advanced Electromagnetism",
                "Statistical Mechanics",
                "Chemical Thermodynamics and Equilibrium",
                "Molecular Genetics and Gene Regulation",
                "Advanced Evolutionary Biology",
                "Climate System Modeling",
                "Astrophysics Fundamentals",
            ],
        },

        "technology": {

            "easy": [
                "Programming Fundamentals",
                "Data Structures",
                "Algorithms",
                "Computer Architecture",
                "Database Fundamentals",
                "Operating Systems Fundamentals",
                "Computer Networking",
                "Software Engineering Fundamentals",
            ],

            "medium": [
                "Advanced Data Structures",
                "Algorithm Design and Analysis",
                "Operating Systems",
                "Database Systems",
                "Computer Networks",
                "Distributed Systems Fundamentals",
                "Machine Learning Fundamentals",
                "Cybersecurity Fundamentals",
            ],

            "hard": [
                "Advanced Algorithms",
                "Graph Algorithms",
                "Distributed Systems",
                "Concurrency and Parallelism",
                "Advanced Database Systems",
                "Machine Learning Algorithms",
                "Cryptography and Network Security",
                "Artificial Intelligence and Reasoning",
            ],
        },

        "engineering": {

            "easy": [
                "Engineering Mechanics",
                "Electrical Circuit Fundamentals",
                "Materials Science Fundamentals",
                "Thermodynamics Fundamentals",
                "Fluid Mechanics Fundamentals",
                "Engineering Measurements",
                "Engineering Design",
                "Engineering Safety",
            ],

            "medium": [
                "Statics and Dynamics",
                "Stress and Strain Analysis",
                "Circuit Analysis",
                "Thermodynamic Cycles",
                "Fluid Flow",
                "Heat Transfer Fundamentals",
                "Control Systems Fundamentals",
                "Engineering Economics",
            ],

            "hard": [
                "Advanced Mechanics",
                "Finite Element Concepts",
                "Advanced Circuit Analysis",
                "Heat Transfer",
                "Fluid Dynamics",
                "Control System Design",
                "Engineering Optimization",
                "Reliability and Risk Engineering",
            ],
        },

        "mathematics": {

            "easy": [
                "Calculus I",
                "Linear Algebra",
                "Differential Equations Fundamentals",
                "Probability Fundamentals",
                "Statistics",
                "Discrete Mathematics",
                "Multivariable Calculus Fundamentals",
                "Numerical Methods Fundamentals",
            ],

            "medium": [
                "Multivariable Calculus",
                "Ordinary Differential Equations",
                "Linear Algebra",
                "Discrete Mathematics",
                "Probability Theory",
                "Statistical Inference",
                "Numerical Analysis",
                "Complex Variables Fundamentals",
            ],

            "hard": [
                "Advanced Differential Equations",
                "Partial Differential Equations",
                "Real Analysis Fundamentals",
                "Abstract Algebra Fundamentals",
                "Advanced Probability",
                "Stochastic Processes",
                "Numerical Analysis",
                "Complex Analysis",
            ],
        },
    },
}


# ============================================================================
# CURRICULUM LOADER
# ============================================================================

def load_syllabus() -> Dict[
    str,
    Dict[
        str,
        Dict[
            str,
            List[str],
        ],
    ],
]:

    """
    Return the built-in grade-aware curriculum.

    We intentionally keep this function name because the rest of the
    generator already calls load_syllabus().
    """

    return STEM_CURRICULUM


# ============================================================================
# CURRICULUM COMBINATIONS
# ============================================================================

def build_all_combinations(
    syllabus: Dict[
        str,
        Dict[
            str,
            Dict[
                str,
                List[str],
            ],
        ],
    ],
) -> List[
    Dict[str, str]
]:

    combinations: List[
        Dict[str, str]
    ] = []

    for grade in SUPPORTED_GRADES:

        for subject in SUPPORTED_SUBJECTS:

            for difficulty in (
                SUPPORTED_DIFFICULTIES
            ):

                topics = (
                    syllabus
                    [grade]
                    [subject]
                    [difficulty]
                )

                for topic in topics:

                    combinations.append(
                        {
                            "grade": grade,
                            "subject": subject,
                            "syllabus_topic": topic,
                            "difficulty": difficulty,
                        }
                    )

    return combinations


def build_demo_combinations(
    syllabus: Dict[
        str,
        Dict[
            str,
            Dict[
                str,
                List[str],
            ],
        ],
    ],
    count: int,
) -> List[
    Dict[str, str]
]:

    """
    Deterministic demo targets.

    This keeps the existing one-target-per-model demo behavior,
    but now selects from the correct grade + subject + difficulty
    curriculum pool.
    """

    combinations: List[
        Dict[str, str]
    ] = []

    index = 0

    while (
        len(combinations)
        < count
    ):

        grade = (
            SUPPORTED_GRADES[
                index
                % len(
                    SUPPORTED_GRADES
                )
            ]
        )

        subject = (
            SUPPORTED_SUBJECTS[
                index
                % len(
                    SUPPORTED_SUBJECTS
                )
            ]
        )

        difficulty = (
            SUPPORTED_DIFFICULTIES[
                index
                % len(
                    SUPPORTED_DIFFICULTIES
                )
            ]
        )

        topics = (
            syllabus
            [grade]
            [subject]
            [difficulty]
        )

        topic = topics[
            (
                index
                // len(
                    SUPPORTED_DIFFICULTIES
                )
            )
            % len(topics)
        ]

        combinations.append(
            {
                "grade": grade,
                "subject": subject,
                "syllabus_topic": topic,
                "difficulty": difficulty,
            }
        )

        index += 1

    return combinations


# ============================================================================
# MODEL REGISTRY
# ============================================================================

@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    rank: int
    rpm: Optional[int]
    rpd: Optional[int]
    tpm: Optional[int]
    tpd: Optional[int]
    batch_size_production: int
    quality_tier: str
    reset_zone: str

    @property
    def key(self) -> str:
        return (
            f"{self.provider}:{self.model}"
        )


# IMPORTANT:
# This hierarchy is intentionally STRICTLY SEQUENTIAL.
MODEL_HIERARCHY: List[ModelSpec] = [
    # ------------------------------ Gemini ------------------------------

    ModelSpec(
        "gemini",
        "gemini-3.6-flash",
        1,
        5,
        20,
        250_000,
        None,
        DEFAULT_PRODUCTION_BATCH_GEMINI,
        "frontier",
        "America/Los_Angeles",
    ),

    ModelSpec(
        "gemini",
        "gemini-3.5-flash-lite",
        2,
        15,
        500,
        250_000,
        None,
        DEFAULT_PRODUCTION_BATCH_GEMINI,
        "high-throughput",
        "America/Los_Angeles",
    ),

    ModelSpec(
        "gemini",
        "gemini-3.5-flash",
        3,
        5,
        20,
        250_000,
        None,
        DEFAULT_PRODUCTION_BATCH_GEMINI,
        "frontier",
        "America/Los_Angeles",
    ),

    ModelSpec(
        "gemini",
        "gemini-3-flash-preview",
        4,
        5,
        20,
        250_000,
        None,
        DEFAULT_PRODUCTION_BATCH_GEMINI,
        "high",
        "America/Los_Angeles",
    ),

    ModelSpec(
        "gemini",
        "gemini-3.1-flash-lite",
        5,
        15,
        500,
        250_000,
        None,
        DEFAULT_PRODUCTION_BATCH_GEMINI,
        "frontier",
        "America/Los_Angeles",
    ),

    # ------------------------------- Groq --------------------------------

    ModelSpec(
        "groq",
        "openai/gpt-oss-120b",
        6,
        30,
        1_000,
        8_000,
        200_000,
        DEFAULT_PRODUCTION_BATCH_GROQ,
        "frontier",
        "UTC",
    ),

    ModelSpec(
        "groq",
        "qwen/qwen3.6-27b",
        7,
        30,
        1_000,
        8_000,
        200_000,
        DEFAULT_PRODUCTION_BATCH_GROQ,
        "high",
        "UTC",
    ),

    ModelSpec(
        "groq",
        "openai/gpt-oss-20b",
        8,
        30,
        1_000,
        8_000,
        200_000,
        DEFAULT_PRODUCTION_BATCH_GROQ,
        "high",
        "UTC",
    ),
]


def load_rate_limit_overrides() -> Dict[str, Dict[str, Any]]:
    """
    Read an optional JSON file of per-model rate-limit overrides.

    Provider free-tier RPM/RPD/TPM numbers are account- and tier-specific
    and change without notice, and Google/Groq only show the numbers that
    actually apply to YOUR project in their own dashboards. Rather than
    guessing at numbers here, this generator lets you paste your current
    dashboard values into a small JSON file and picks them up automatically.

    Expected shape of the file (any subset of models/fields is fine):

        {
          "gemini:gemini-3.6-flash": {"rpm": 5, "rpd": 20},
          "groq:openai/gpt-oss-120b": {"rpm": 30, "rpd": 1000, "tpm": 8000}
        }

    Keys are "<provider>:<model>" (matches ModelSpec.key). Any of
    rpm / rpd / tpm / tpd / batch_size_production may be set; omitted
    fields keep the built-in default.
    """

    if not RATE_LIMIT_OVERRIDE_FILE.exists():
        return {}

    try:

        with RATE_LIMIT_OVERRIDE_FILE.open(
            "r",
            encoding="utf-8",
        ) as handle:

            data = json.load(
                handle
            )

        return (
            data
            if isinstance(data, dict)
            else {}
        )

    except Exception as exc:

        print(
            (
                "[WARN] Could not read "
                f"{RATE_LIMIT_OVERRIDE_FILE.name}; "
                f"using built-in default rate limits: {exc}"
            ),
            flush=True,
        )

        return {}


def apply_rate_limit_overrides() -> None:
    """Apply load_rate_limit_overrides() on top of MODEL_HIERARCHY, in place."""

    overrides = load_rate_limit_overrides()

    if not overrides:
        return

    overridable_fields = (
        "rpm",
        "rpd",
        "tpm",
        "tpd",
        "batch_size_production",
    )

    updated: List[ModelSpec] = []

    for spec in MODEL_HIERARCHY:

        override = (
            overrides.get(spec.key)
            or overrides.get(spec.model)
        )

        if override:

            changes = {
                name: override[name]
                for name in overridable_fields
                if name in override
                and override[name] is not None
            }

            if changes:

                spec = dataclass_replace(
                    spec,
                    **changes,
                )

                print(
                    f"[CONFIG] {spec.model}: "
                    f"applied rate-limit override {changes}",
                    flush=True,
                )

        updated.append(spec)

    MODEL_HIERARCHY[:] = updated


# ============================================================================
# EXCEPTIONS
# ============================================================================

class ProviderError(RuntimeError):

    def __init__(
        self,
        message: str,
        *,
        temporary: bool = True,
        daily_exhausted: bool = False,
        retry_after: float = 0.0,
    ) -> None:

        super().__init__(message)

        self.temporary = temporary
        self.daily_exhausted = daily_exhausted

        self.retry_after = max(
            0.0,
            float(
                retry_after or 0.0
            ),
        )


class GenerationStopped(RuntimeError):
    """Raised internally when a Ctrl+C / stop request interrupts a wait,
    so the current generation attempt is abandoned cleanly instead of
    being started anyway."""


def trim_one_line(
    text: str,
    limit: int = 180,
) -> str:
    """Collapse a (possibly multi-line, possibly huge) error string into
    one short line so raw HTTP/JSON error bodies never get dumped to the
    console verbatim."""

    single_line = " ".join(
        str(
            text
            or ""
        ).split()
    )

    if len(single_line) > limit:
        return single_line[: limit - 3] + "..."

    return single_line


def friendly_provider_message(
    exc: "ProviderError",
    model: str,
) -> str:
    """
    Turn a ProviderError into a single short line instead of a raw
    HTTP error body (which can be a multi-hundred-character JSON blob
    from the provider). The only "error-shaped" message that should
    routinely show up during a healthy run is the high-usage/rate-limit
    one; everything else is trimmed to one line so nothing resembling a
    stack trace or raw JSON reaches the console.
    """

    raw = str(exc)
    lowered = raw.lower()

    if exc.daily_exhausted:

        return (
            f"{model}: today's free quota is used up. "
            "Moving on to the next model."
        )

    is_rate_limited = (
        "429" in raw
        or "rate limit" in lowered
        or "resource_exhausted" in lowered
        or "quota" in lowered
    )

    if exc.temporary and is_rate_limited:

        return (
            f"{model}: currently cannot generate "
            "because of high usage."
        )

    # Non-rate-limit errors (bad key, model unavailable, network blip,
    # 5xx, unexpected payload shape, etc.) still get surfaced, but as a
    # single trimmed line rather than a raw response dump.
    return f"{model}: {trim_one_line(raw)}"


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def now_iso() -> str:
    return (
        datetime.now(
            timezone.utc
        ).isoformat()
    )


def normalize_text(
    value: Any,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(
            value or ""
        ).strip(),
    ).lower()


def provider_day_label(
    tz_name: str,
) -> str:

    try:

        from zoneinfo import ZoneInfo

        return datetime.now(
            ZoneInfo(tz_name)
        ).strftime(
            "%Y-%m-%d"
        )

    except Exception:

        return datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%d"
        )


def parse_retry_after(
    headers: Dict[str, str],
) -> float:

    value = (
        headers.get(
            "retry-after"
        )
        or headers.get(
            "Retry-After"
        )
    )

    if not value:
        return 0.0

    try:

        return max(
            0.0,
            float(
                str(value).strip()
            ),
        )

    except ValueError:

        return 0.0


def response_text_safe(
    response: requests.Response,
) -> str:

    try:
        return response.text[:6000]
    except Exception:
        return ""


def looks_daily_exhausted(
    message: str,
    headers: Dict[str, str],
    spec: ModelSpec,
) -> bool:
    lowered = (
        message or ""
    ).lower()

    # --------------------------------------------------------------
    # Explicit Gemini machine-readable daily quota errors.
    # --------------------------------------------------------------
    daily_markers = (
        "quota_exceeded",
        "requests per day",
        "requests per day per project per model",
        "daily quota",
        "quota exhausted",
        "quota exceeded",
        "daily limit",
        "generaterequestsperdayperprojectpermodelfreetier",
    )

    if any(
        marker in lowered
        for marker in daily_markers
    ):
        return True

    # --------------------------------------------------------------
    # Inspect Gemini's structured QuotaFailure details.
    # Example:
    #
    # quotaMetric:
    #   generate_content_free_tier_requests
    #
    # quotaId:
    #   GenerateRequestsPerDayPerProjectPerModel-FreeTier
    # --------------------------------------------------------------
    try:
        payload = json.loads(
            message
        )

        error = (
            payload.get(
                "error"
            )
            if isinstance(
                payload,
                dict,
            )
            else None
        )

        if isinstance(
            error,
            dict,
        ):
            error_status = str(
                error.get(
                    "status"
                )
                or ""
            ).lower()

            error_code = str(
                error.get(
                    "code"
                )
                or ""
            )

            if (
                "quota_exceeded"
                in error_status
                or error_code == "429"
            ):
                details = (
                    error.get(
                        "details"
                    )
                    or []
                )

                for detail in details:
                    if not isinstance(
                        detail,
                        dict,
                    ):
                        continue

                    quota_id = str(
                        detail.get(
                            "quotaId"
                        )
                        or ""
                    ).lower()

                    quota_metric = str(
                        detail.get(
                            "quotaMetric"
                        )
                        or ""
                    ).lower()

                    if (
                        "day"
                        in quota_id
                        or "day"
                        in quota_metric
                    ):
                        return True

    except Exception:
        pass

    return False


def clean_json_text(
    raw: str,
) -> str:

    text_value = (
        str(
            raw or ""
        ).strip()
    )

    if text_value.startswith(
        "```"
    ):

        text_value = re.sub(
            r"^```(?:json)?\s*",
            "",
            text_value,
            flags=re.IGNORECASE,
        )

        text_value = re.sub(
            r"\s*```$",
            "",
            text_value,
        )

    return text_value.strip()

# ============================================================================
# MATHEMATICAL / LATEX NORMALIZATION
# ============================================================================

_LATEX_COMMANDS_WITHOUT_BACKSLASH = (
    "begin",
    "end",
    "mathbb",
    "mathcal",
    "mathbf",
    "mathrm",
    "text",
    "frac",
    "dfrac",
    "tfrac",
    "sqrt",
    "sum",
    "prod",
    "int",
    "lim",
    "lambda",
    "leq",
    "geq",
    "rightarrow",
    "times",
    "neq",
    "pmatrix",
    "bmatrix",
    "vmatrix",
    "cases",
)


def normalize_latex_text(
    value: Any,
) -> str:
    """
    Repair common model-output LaTeX damage while preserving
    normal English prose.

    The generator prompt requires delimiters, but models can still
    occasionally emit:
        mathbb{R}
        pmatrix ... endpmatrix

    instead of:
        \\mathbb{R}
        \\begin{pmatrix} ... \\end{pmatrix}

    This function repairs only recognizable LaTeX command damage.
    """

    text_value = str(
        value or ""
    )

    if not text_value.strip():
        return text_value

    # ------------------------------------------------------------------
    # 1. Repair common command names that lost their leading slash.
    # ------------------------------------------------------------------

    for command in _LATEX_COMMANDS_WITHOUT_BACKSLASH:
        text_value = re.sub(
            rf"(?<![\\A-Za-z]){command}(?=\{{|\s)",
            rf"\\{command}",
            text_value,
        )

    # ------------------------------------------------------------------
    # 2. Repair matrix environments.
    #
    # Example:
    #   \pmatrix ... \endpmatrix
    #
    # becomes:
    #   \begin{pmatrix} ... \end{pmatrix}
    # ------------------------------------------------------------------

    text_value = re.sub(
        r"\\pmatrix\b",
        r"\\begin{pmatrix}",
        text_value,
    )

    text_value = re.sub(
        r"\\bmatrix\b",
        r"\\begin{bmatrix}",
        text_value,
    )

    text_value = re.sub(
        r"\\vmatrix\b",
        r"\\begin{vmatrix}",
        text_value,
    )

    text_value = re.sub(
        r"\\cases\b",
        r"\\begin{cases}",
        text_value,
    )

    text_value = re.sub(
        r"\\endpmatrix\b",
        r"\\end{pmatrix}",
        text_value,
    )

    text_value = re.sub(
        r"\\endbmatrix\b",
        r"\\end{bmatrix}",
        text_value,
    )

    text_value = re.sub(
        r"\\endvmatrix\b",
        r"\\end{vmatrix}",
        text_value,
    )

    text_value = re.sub(
        r"\\endcases\b",
        r"\\end{cases}",
        text_value,
    )

    # ------------------------------------------------------------------
    # 3. Repair an already-damaged matrix where "begin" was lost.
    #
    # Example:
    #   \beginpmatrix ... \end{pmatrix}
    # ------------------------------------------------------------------

    text_value = re.sub(
        r"\\beginpmatrix\b",
        r"\\begin{pmatrix}",
        text_value,
    )

    text_value = re.sub(
        r"\\beginbmatrix\b",
        r"\\begin{bmatrix}",
        text_value,
    )

    text_value = re.sub(
        r"\\beginvmatrix\b",
        r"\\begin{vmatrix}",
        text_value,
    )

    text_value = re.sub(
        r"\\begincases\b",
        r"\\begin{cases}",
        text_value,
    )

    # ------------------------------------------------------------------
    # 4. If a recognizable matrix environment exists without any
    #    surrounding math delimiters, add inline delimiters.
    # ------------------------------------------------------------------

    has_matrix_environment = bool(
        re.search(
            r"\\begin\{(?:pmatrix|bmatrix|vmatrix|cases)\}",
            text_value,
        )
    )

    has_math_delimiter = bool(
        "\\(" in text_value
        or "\\[" in text_value
        or "$$" in text_value
        or re.search(
            r"(^|[^\\])\$[^$]+\$",
            text_value,
        )
    )

    if (
        has_matrix_environment
        and not has_math_delimiter
    ):
        # Do NOT wrap the entire text. A question may contain
        # ordinary English plus a matrix.
        #
        # Replace only the matrix fragment with an inline
        # KaTeX expression.
        text_value = re.sub(
            r"(\\begin\{(?:pmatrix|bmatrix|vmatrix|cases)\}"
            r"[\s\S]*?"
            r"\\end\{(?:pmatrix|bmatrix|vmatrix|cases)\})",
            lambda match: (
                "\\("
                + match.group(1)
                + "\\)"
            ),
            text_value,
            count=1,
        )

    return text_value

# ============================================================================
# STATE
# ============================================================================

@dataclass
class ModelRuntime:

    spec: ModelSpec

    requests_started: List[
        float
    ] = field(
        default_factory=list
    )

    token_events: List[
        Tuple[float, int]
    ] = field(
        default_factory=list
    )

    daily_requests: int = 0
    daily_tokens: int = 0

    status: str = "ACTIVE"

    request_count: int = 0
    question_count: int = 0
    rejected_count: int = 0
    duplicate_count: int = 0

    last_warning_level: float = 0.0

    exhausted_reason: Optional[
        str
    ] = None

    def trim_windows(
        self,
    ) -> None:

        now = time.time()
        minute_ago = (
            now - 60.0
        )

        self.requests_started = [
            timestamp
            for timestamp
            in self.requests_started
            if timestamp
            > minute_ago
        ]

        self.token_events = [
            (
                timestamp,
                tokens,
            )
            for (
                timestamp,
                tokens,
            )
            in self.token_events
            if timestamp
            > minute_ago
        ]

    @property
    def rpm_used(self) -> int:
        self.trim_windows()

        return len(
            self.requests_started
        )

    @property
    def tpm_used(self) -> int:
        self.trim_windows()

        return sum(
            tokens
            for (
                _,
                tokens,
            )
            in self.token_events
        )

    @property
    def rpd_ratio(self) -> float:

        return (
            self.daily_requests
            / self.spec.rpd
            if self.spec.rpd
            else 0.0
        )

    @property
    def tpd_ratio(self) -> float:

        if not self.spec.tpd:
            return 0.0

        return (
            self.daily_tokens
            / self.spec.tpd
        )


class GenerationState:

    def __init__(
        self,
        path: Path,
    ) -> None:

        self.path = path

        self.data: Dict[
            str,
            Any,
        ] = {
            "version": 1,
            "day": {},
            "models": {},
            "totals": {
                "requests": 0,
                "generated": 0,
                "inserted": 0,
                "rejected": 0,
                "duplicates": 0,
            },
        }

        self.lock = threading.Lock()

        self._load()

    def _load(
        self,
    ) -> None:

        if not self.path.exists():
            return

        try:

            with self.path.open(
                "r",
                encoding="utf-8",
            ) as handle:

                loaded = json.load(
                    handle
                )

            if isinstance(
                loaded,
                dict,
            ):

                self.data.update(
                    loaded
                )

        except Exception as exc:

            print(
                "[WARN] Could not read "
                f"state file; starting fresh: {exc}"
            )

    def save(
        self,
    ) -> None:

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.lock:

            temp = (
                self.path.with_suffix(
                    ".tmp"
                )
            )

            with temp.open(
                "w",
                encoding="utf-8",
            ) as handle:

                json.dump(
                    self.data,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )

            temp.replace(
                self.path
            )

    def model_bucket(
        self,
        spec: ModelSpec,
    ) -> Dict[str, Any]:

        models = self.data.setdefault(
            "models",
            {},
        )

        return models.setdefault(
            spec.key,
            {
                "status": "ACTIVE",
                "daily_requests": 0,
                "daily_tokens": 0,
                "question_count": 0,
                "rejected_count": 0,
                "duplicate_count": 0,
                "exhausted_reason": None,
            },
        )

    def reset_daily_if_needed(
        self,
    ) -> None:
        """
        Reset persisted daily usage whenever the provider's current quota
        day has changed.

        Gemini daily quotas reset at midnight Pacific time, so all Gemini
        generation + validation state uses the Gemini reset boundary.
        """
        for spec in MODEL_HIERARCHY:

            reset_zone = (
                GEMINI_RESET_ZONE
                if spec.provider == "gemini"
                else spec.reset_zone
            )

            bucket = (
                self.model_bucket(
                    spec
                )
            )

            today = provider_day_label(
                reset_zone
            )

            previous = (
                bucket.get(
                    "day"
                )
            )

            if previous != today:
                bucket.update(
                    {
                        "day": today,
                        "status": "ACTIVE",
                        "daily_requests": 0,
                        "daily_tokens": 0,
                        "exhausted_reason": None,
                    }
                )

        validators = (
            self.data.setdefault(
                "validator_models",
                {},
            )
        )

        for validator_model in VALIDATOR_MODELS:
            bucket = (
                validators.setdefault(
                    validator_model,
                    {},
                )
            )

            today = provider_day_label(
                GEMINI_RESET_ZONE
            )

            if (
                bucket.get("day")
                != today
            ):
                bucket.update(
                    {
                        "day": today,
                        "status": "ACTIVE",
                        "daily_requests": 0,
                        "daily_tokens": 0,
                        "exhausted_reason": None,
                    }
                )

        self.save()

    def mark_exhausted(
        self,
        spec: ModelSpec,
        reason: str,
    ) -> None:

        bucket = (
            self.model_bucket(
                spec
            )
        )

        bucket[
            "status"
        ] = "DAILY_EXHAUSTED"

        bucket[
            "exhausted_reason"
        ] = reason

        self.save()

    def increment_usage(
        self,
        spec: ModelSpec,
        request_count: int,
        token_count: int,
    ) -> None:

        bucket = (
            self.model_bucket(
                spec
            )
        )

        bucket[
            "daily_requests"
        ] = (
            int(
                bucket.get(
                    "daily_requests",
                    0,
                )
            )
            + request_count
        )

        bucket[
            "daily_tokens"
        ] = (
            int(
                bucket.get(
                    "daily_tokens",
                    0,
                )
            )
            + token_count
        )

        self.data[
            "totals"
        ][
            "requests"
        ] = (
            int(
                self.data[
                    "totals"
                ].get(
                    "requests",
                    0,
                )
            )
            + request_count
        )

        self.save()


# ============================================================================
# DATABASE
# ============================================================================

def resolve_database_url() -> str:

    candidates = (
        os.environ.get(
            "DATABASE_URL"
        ),
        os.environ.get(
            "SUPABASE_DIRECT_CONNECTION_STRING"
        ),
        os.environ.get(
            "SUPABASE_DB_URL"
        ),
        os.environ.get(
            "SUPABASE_DATABASE_URL"
        ),
        os.environ.get(
            "SUPABASE_POOLER_CONNECTION_STRING"
        ),
    )

    for candidate in candidates:

        value = str(
            candidate or ""
        ).strip()

        if not value:
            continue

        if value.startswith(
            "postgres://"
        ):

            value = value.replace(
                "postgres://",
                "postgresql://",
                1,
            )

        return value

    raise RuntimeError(
        "No PostgreSQL database URL found. "
        "Set DATABASE_URL or "
        "SUPABASE_DIRECT_CONNECTION_STRING in .env."
    )


def build_db_engine() -> Engine:

    url = resolve_database_url()

    if not url.startswith(
        "postgresql://"
    ):

        raise RuntimeError(
            "Safety stop: the question-bank "
            "generator only permits PostgreSQL/Supabase. "
            f"Resolved URL scheme: "
            f"{url.split(':', 1)[0]!r}"
        )

    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=2,
        max_overflow=1,
        connect_args={
            "connect_timeout": 10,
            "options": (
                "-c statement_timeout=30000"
            ),
        },
    )

    with engine.connect() as conn:

        conn.execute(
            text("SELECT 1")
        )

        exists = conn.execute(
            text(
                """
                SELECT to_regclass(
                    'public.practice_questions'
                )
                """
            )
        ).scalar_one_or_none()

        if not exists:

            raise RuntimeError(
                "Supabase table "
                "public.practice_questions "
                "does not exist. "
                "Run the Alembic migration first."
            )

    return engine


def load_existing_fingerprints_for_batch(
    engine: Engine,
    fingerprints: Iterable[str],
) -> set[str]:

    values = [
        str(
            fingerprint
        ).strip()
        for fingerprint
        in fingerprints
        if str(
            fingerprint
        ).strip()
    ]

    if not values:
        return set()

    statement = text(
        """
        SELECT question_fingerprint
        FROM practice_questions
        WHERE question_fingerprint IN :fingerprints
        """
    ).bindparams(
        bindparam(
            "fingerprints",
            expanding=True,
        )
    )

    with engine.connect() as conn:

        rows = conn.execute(
            statement,
            {
                "fingerprints": values
            },
        )

        return {
            str(
                row[0]
            ).strip()
            for row in rows
            if row[0]
        }

def load_combination_counts(
    engine: Engine,
) -> Dict[
    Tuple[str, str, str, str],
    int,
]:

    counts: Dict[
        Tuple[str, str, str, str],
        int,
    ] = {}

    with engine.connect() as conn:

        rows = conn.execute(
            text(
                """
                SELECT
                    grade,
                    subject,
                    syllabus_topic,
                    difficulty,
                    COUNT(*)
                FROM practice_questions
                GROUP BY
                    grade,
                    subject,
                    syllabus_topic,
                    difficulty
                """
            )
        )

        for (
            grade,
            subject,
            topic,
            difficulty,
            count,
        ) in rows:

            counts[
                (
                    str(grade),
                    str(subject),
                    str(topic),
                    str(difficulty),
                )
            ] = int(
                count
            )

    return counts

def load_database_inventory(
    engine: Engine,
    syllabus: Dict[
        str,
        Dict[
            str,
            Dict[
                str,
                List[str],
            ],
        ],
    ],
) -> Dict[str, Any]:

    db_counts = (
        load_combination_counts(
            engine
        )
    )

    combinations = []

    for target in build_all_combinations(
        syllabus
    ):

        key = (
            target["grade"],
            target["subject"],
            target["syllabus_topic"],
            target["difficulty"],
        )

        combinations.append(
            {
                "grade":
                    target["grade"],

                "subject":
                    target["subject"],

                "syllabus_topic":
                    target[
                        "syllabus_topic"
                    ],

                "difficulty":
                    target[
                        "difficulty"
                    ],

                "question_count":
                    int(
                        db_counts.get(
                            key,
                            0,
                        )
                    ),
            }
        )

    database_total = (
        load_database_totals(
            engine
        )
    )

    represented_total = sum(
        int(
            row[
                "question_count"
            ]
        )
        for row in combinations
    )

    return {
        "refreshed_at":
            now_iso(),

        "total_questions":
            database_total,

        "represented_total":
            represented_total,

        "unclassified_questions":
            max(
                0,
                database_total
                - represented_total,
            ),

        "combinations":
            combinations,
    }

# ============================================================================
# NEW:
# Final database-backed insertion counts.
#
# This is deliberately queried at the END rather than relying only on
# runtime counters.
# ============================================================================

def load_inserted_counts_by_model(
    engine: Engine,
) -> Dict[str, int]:

    counts: Dict[
        str,
        int,
    ] = {}

    with engine.connect() as conn:

        rows = conn.execute(
            text(
                """
                SELECT
                    generation_model,
                    COUNT(*)
                FROM practice_questions
                WHERE generation_model IS NOT NULL
                GROUP BY generation_model
                """
            )
        )

        for model, count in rows:

            model_name = str(
                model or ""
            ).strip()

            if model_name:

                counts[
                    model_name
                ] = int(
                    count
                )

    return counts


def load_database_totals(
    engine: Engine,
) -> int:

    with engine.connect() as conn:

        result = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM practice_questions
                """
            )
        ).scalar_one()

    return int(
        result or 0
    )


def insert_questions(
    engine: Engine,
    rows: List[
        Dict[str, Any]
    ],
) -> int:

    if not rows:
        return 0

    sql = text(
        """
        INSERT INTO practice_questions (
            grade,
            subject,
            syllabus_topic,
            difficulty,
            question_text,
            options,
            correct_index,
            explanation,
            question_fingerprint,
            generation_model,
            generation_batch_id,
            generation_meta,
            created_at
        ) VALUES (
            :grade,
            :subject,
            :syllabus_topic,
            :difficulty,
            :question_text,
            CAST(:options AS jsonb),
            :correct_index,
            :explanation,
            :question_fingerprint,
            :generation_model,
            :generation_batch_id,
            CAST(:generation_meta AS jsonb),
            CURRENT_TIMESTAMP
        )
        ON CONFLICT (question_fingerprint)
        DO NOTHING
        """
    )

    payload = []

    for row in rows:

        payload.append(
            {
                **row,
                "options": json.dumps(
                    row["options"],
                    ensure_ascii=False,
                ),
                "generation_meta": json.dumps(
                    row.get(
                        "generation_meta"
                    )
                    or {},
                    ensure_ascii=False,
                ),
            }
        )

    with engine.begin() as conn:

        result = conn.execute(
            sql,
            payload,
        )

        return int(
            result.rowcount or 0
        )


# ============================================================================
# QUESTION GENERATION SCHEMA
# ============================================================================

# IMPORTANT:
#
# No question-length limits.
# No option-length limits.
# No explanation-length limits.
# No minItems/maxItems/minLength/maxLength/minimum/maximum.
#
# The application itself validates the structural requirements that are
# actually required for the database:
# - question exists
# - exactly 4 options
# - correct option is A-D
#
# This prevents provider-side Groq schema restrictions from causing failures.

def build_json_schema(
    count: int,
) -> Dict[str, Any]:

    return {
        "type": "object",

        "properties": {
            "questions": {
                "type": "array",

                "items": {
                    "type": "object",

                    "properties": {
                        "question_text": {
                            "type": "string",
                        },

                        "options": {
                            "type": "array",

                            "items": {
                                "type": "string",
                            },
                        },

                        "correct_option": {
                            "type": "string",

                            "enum": [
                                "A",
                                "B",
                                "C",
                                "D",
                            ],
                        },

                        "explanation": {
                            "type": "string",
                        },
                    },

                    "required": [
                        "question_text",
                        "options",
                        "correct_option",
                        "explanation",
                    ],

                    "additionalProperties": False,
                },
            }
        },

        "required": [
            "questions"
        ],

        "additionalProperties": False,
    }


# ============================================================================
# PROMPT
# ============================================================================

def build_prompt(
    target: Dict[str, str],
    count: int,
) -> str:

    grade_guidance = {
        "elementary": (
            "Use concrete vocabulary, foundational concepts, "
            "short reasoning chains, and age-appropriate examples."
        ),

        "middle": (
            "Use standard academic vocabulary and require one "
            "or two steps of reasoning when appropriate."
        ),

        "high": (
            "Use rigorous curriculum-aligned language and test "
            "conceptual understanding, application, and common misconceptions."
        ),

        "college": (
            "Use technically accurate terminology and deeper "
            "conceptual or multi-step reasoning appropriate for undergraduate study."
        ),
    }[
        target["grade"]
    ]

    difficulty_guidance = {
        "easy": (
            "Direct understanding or one-step application. "
            "Do not make distractors absurdly obvious."
        ),

        "medium": (
            "Application, interpretation, or multi-step reasoning. "
            "Distractors should reflect realistic mistakes."
        ),

        "hard": (
            "Deeper synthesis, nuanced reasoning, multi-step analysis, "
            "or difficult application. Distractors should target genuine misconceptions."
        ),
    }[
        target["difficulty"]
    ]

    return rf"""
You are a senior STEM curriculum and assessment designer creating a production-grade practice question bank.

Generate exactly {count} ORIGINAL multiple-choice questions.

Quality is more important than completing the batch quickly. Never fill a missing question with a broken, uncertain, approximate, self-contradictory, or unverified question.

TARGET:
- Grade level: {target['grade']}
- STEM subject: {target['subject']}
- Sub-topic: {target['syllabus_topic']}
- Difficulty: {target['difficulty']}

GRADE CALIBRATION:
{grade_guidance}

DIFFICULTY CALIBRATION:
{difficulty_guidance}

NON-NEGOTIABLE QUALITY RULES:

1. Every question must genuinely test the specified sub-topic.
2. Every question must be useful for real student practice.
3. Do not reuse the same question pattern repeatedly within this batch.
4. Make all four options plausible and mutually distinct.
5. Exactly one option must be correct.
6. The correct answer must actually follow from the question.
7. Never invent facts.
8. Explanations must teach the reasoning and justify the correct answer.
9. Do not create trick questions based on ambiguous wording.
10. For mathematics, physics, chemistry, statistics, and engineering:
    - perform every numerical calculation before finalizing;
    - independently recompute the result;
    - verify units and conversions;
    - verify that the selected correct option matches the computed result;
    - never choose an answer merely because it is the closest option;
    - if the numbers and options cannot produce one unambiguous answer, regenerate it.
11. For computing/technology, prefer practical and technically correct concepts over vague buzzwords.
12. Avoid duplicated questions and near-identical wording.
13. Do not mention AI, prompts, quotas, APIs, models, or this generation process.
14. Output only the requested JSON object.
15. Do not use markdown.
16. Do not use code fences.
17. Set correct_option to exactly one of A, B, C, D.
18. Never output correct_index.
19. MATHEMATICAL / TECHNICAL FORMATTING:
    - Whenever mathematics or mathematical notation appears in the
      question, any option, or the explanation, represent it using
      KaTeX-compatible LaTeX.
    - Inline mathematics MUST use \( ... \).
    - Display mathematics MUST use \[ ... \].
    - Fractions MUST use \frac{{numerator}}{{denominator}}.
    - Matrices MUST use KaTeX-compatible matrix environments such as
      \begin{{pmatrix}} ... \end{{pmatrix}}.
    - Use LaTeX commands for mathematical symbols such as
      \mathbb{{R}}, \mathbb{{N}}, \sqrt{{}}, \sum, \int, \lambda, \leq,
      \geq, \rightarrow, \times, \neq, etc.
    - Do not output raw LaTeX commands outside a math delimiter.
    - Do not use Markdown $$ ... $$ math delimiters.
    - Do not use plain ASCII substitutes such as [[1,2],[3,4]]
      when proper matrix notation is appropriate.
    - Keep ordinary explanatory English as ordinary text.
    - Use LaTeX only for mathematical or technical notation that
      benefits from mathematical typesetting.
    - Ensure every LaTeX expression is syntactically valid for KaTeX.
    - When returning JSON, escape every LaTeX backslash correctly for JSON.
    - For example, JSON must contain "\\mathbb{{R}}" to represent
      \mathbb{{R}}, and "\\begin{{pmatrix}} ... \\end{{pmatrix}}"
      to represent a KaTeX matrix.
    - Never write a single unescaped backslash before a LaTeX command
      inside a JSON string.
    - The parsed question text, every parsed option, and the parsed
      explanation must preserve the actual "\" character used by LaTeX.
    - Never emit an unescaped double quote inside a JSON string.
    - Never emit literal control characters (newlines/tabs) inside a JSON
      string; encode them as JSON escapes when needed.

Before finalizing each question, silently verify:
- exactly four options;
- exactly one correct option;
- correct_option is A/B/C/D;
- explanation agrees with the answer;
- correct grade;
- correct subject;
- correct topic;
- correct difficulty;
- no ambiguity;
- useful educational value;
- calculations are independently checked;
- units are checked;
- no contradiction exists.
- if mathematical notation is used, it is enclosed in KaTeX-compatible
  delimiters and uses valid KaTeX-compatible LaTeX;
- no raw LaTeX command is left outside a math delimiter;
- fractions, matrices, superscripts, subscripts, roots and symbols use
  proper LaTeX notation.

ADDITIONAL FINAL-ANSWER INTEGRITY RULES:

Before outputting EVERY question, independently solve the question yourself.

For every generated question:

1. Determine the correct answer independently.
2. Compare your independently calculated/reasoned answer against all four options.
3. Make sure the selected correct_option exactly matches that answer.
4. Make sure the explanation proves the selected answer.
5. The explanation must NEVER say that the question is wrong, broken, ambiguous, inconsistent, or needs rewriting.
6. The explanation must NEVER contain a calculation that contradicts the selected option.
7. If your first draft is wrong, silently FIX the question, options, answer, and explanation before outputting it.
8. NEVER output a question that you know is incorrect.
9. NEVER output an explanation that admits the generated question has a problem.
10. For mathematics and engineering, recompute the final result independently before emitting the JSON.
11. For multiple-choice questions involving formulas, derive the answer first and only then construct the four options around the verified result.
12. The final JSON must contain only finished, internally consistent questions.

IMPORTANT:
A question is not finished until:
QUESTION -> OPTIONS -> CORRECT ANSWER -> EXPLANATION
all agree with each other.


OUTPUT:

{{
  "questions": [
    {{
      "question_text": "...",
      "options": [
        "...",
        "...",
        "...",
        "..."
      ],
      "correct_option": "A",
      "explanation": "..."
    }}
  ]
}}
""".strip()


def parse_questions(
    raw: str,
) -> List[
    Dict[str, Any]
]:

    cleaned = clean_json_text(
        raw
    )

    data = json.loads(
        cleaned
    )

    if isinstance(
        data,
        dict,
    ):

        questions = data.get(
            "questions"
        )

    else:

        questions = data

    if not isinstance(
        questions,
        list,
    ):

        raise ValueError(
            "AI response does not contain "
            "a questions array"
        )

    normalized_questions = []

    for item in questions:
        if not isinstance(
            item,
            dict,
        ):
            continue

        normalized_item = dict(
            item
        )

        normalized_item["question_text"] = (
            normalize_latex_text(
                normalized_item.get(
                    "question_text",
                    "",
                )
            )
        )

        raw_options = normalized_item.get(
            "options",
            [],
        )

        if isinstance(
            raw_options,
            list,
        ):
            normalized_item["options"] = [
                normalize_latex_text(
                    option
                )
                for option in raw_options
            ]

        normalized_item["explanation"] = (
            normalize_latex_text(
                normalized_item.get(
                    "explanation",
                    "",
                )
            )
        )

        normalized_questions.append(
            normalized_item
        )

    return normalized_questions


# ============================================================================
# LOCAL QUESTION VALIDATION
# ============================================================================

def question_fingerprint(
    grade: str,
    subject: str,
    topic: str,
    difficulty: str,
    question: str,
    options: List[str],
) -> str:

    normalized = "||".join(
        [
            normalize_text(
                grade
            ),
            normalize_text(
                subject
            ),
            normalize_text(
                topic
            ),
            normalize_text(
                difficulty
            ),
            normalize_text(
                question
            ),
            "|".join(
                sorted(
                    normalize_text(
                        option
                    )
                    for option
                    in options
                )
            ),
        ]
    )

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


def basic_quality_check(
    question: str,
    options: List[str],
    correct_index: int,
    explanation: str,
) -> Optional[str]:

    # --------------------------------------------------------------
    # IMPORTANT:
    # NO question length check.
    # NO explanation length check.
    # NO option length check.
    #
    # Only required structural checks remain.
    # --------------------------------------------------------------

    if not question.strip():

        return (
            "question is empty"
        )

    if len(options) != 4:

        return (
            "options must contain exactly "
            "4 entries"
        )

    if correct_index not in (
        0,
        1,
        2,
        3,
    ):

        return (
            "correct answer outside "
            "A-D / 0-3"
        )

    if any(
        not option.strip()
        for option
        in options
    ):

        return (
            "empty option"
        )

    normalized_options = [
        normalize_text(
            option
        )
        for option
        in options
    ]

    if len(
        set(
            normalized_options
        )
    ) != 4:

        return (
            "duplicate options"
        )

    if not explanation.strip():

        return (
            "explanation is empty"
        )

    forbidden = {
        "none of the above",
        "all of the above",
        "not sure",
    }

    if any(
        normalize_text(
            option
        ) in forbidden
        for option in options
    ):

        return (
            "generic/ambiguous option detected"
        )

    # --------------------------------------------------------------
    # Mathematical formatting integrity
    # --------------------------------------------------------------

    math_texts = [
        question,
        *options,
        explanation,
    ]

    for text_value in math_texts:
        text_value = str(
            text_value or ""
        )

        # A recognizable LaTeX command without a leading backslash
        # usually means the model produced malformed math.
        suspicious_raw_command = re.search(
            r"(?<![\\A-Za-z])"
            r"(?:"
            r"mathbb|"
            r"mathcal|"
            r"mathbf|"
            r"mathrm|"
            r"frac|"
            r"dfrac|"
            r"tfrac|"
            r"sqrt|"
            r"pmatrix|"
            r"bmatrix|"
            r"vmatrix|"
            r"begin|"
            r"end"
            r")"
            r"\s*(?:\{|[\(\[]|\b)",
            text_value,
        )

        if suspicious_raw_command:
            return (
                "possible malformed LaTeX command: "
                f"{suspicious_raw_command.group(0)!r}"
            )

        # Any control character other than normal whitespace is a
        # strong signal that JSON escaping damaged the text.
        for character in text_value:
            if (
                ord(character) < 32
                and character not in "\n\r\t"
            ):
                return (
                    "control character detected in "
                    "question content; likely malformed "
                    "JSON/LaTeX escaping"
                )

    return None


def validate_question(
    item: Dict[str, Any],
    target: Dict[str, str],
) -> Tuple[
    Optional[
        Dict[str, Any]
    ],
    Optional[str],
]:

    question = str(
        item.get(
            "question_text"
        )
        or ""
    ).strip()

    options = item.get(
        "options"
    )

    explanation = str(
        item.get(
            "explanation"
        )
        or ""
    ).strip()

    raw_correct_option = item.get(
        "correct_option"
    )

    raw_correct_index = item.get(
        "correct_index"
    )

    correct_index: Optional[
        int
    ] = None

    if (
        raw_correct_option
        is not None
    ):

        value = str(
            raw_correct_option
        ).strip().upper()

        if value in {
            "A",
            "B",
            "C",
            "D",
        }:

            correct_index = {
                "A": 0,
                "B": 1,
                "C": 2,
                "D": 3,
            }[
                value
            ]

        else:

            return (
                None,
                (
                    "invalid correct_option: "
                    f"{raw_correct_option!r}"
                ),
            )

    elif (
        raw_correct_index
        is not None
    ):

        try:

            if isinstance(
                raw_correct_index,
                bool,
            ):

                return (
                    None,
                    "correct_index cannot be boolean",
                )

            if isinstance(
                raw_correct_index,
                str,
            ):

                value = (
                    raw_correct_index
                    .strip()
                    .upper()
                )

                if value in {
                    "A",
                    "B",
                    "C",
                    "D",
                }:

                    correct_index = {
                        "A": 0,
                        "B": 1,
                        "C": 2,
                        "D": 3,
                    }[
                        value
                    ]

                elif re.fullmatch(
                    r"[0-3]",
                    value,
                ):

                    correct_index = int(
                        value
                    )

                else:

                    return (
                        None,
                        (
                            "invalid correct_index "
                            f"value: {raw_correct_index!r}"
                        ),
                    )

            elif isinstance(
                raw_correct_index,
                int,
            ):

                correct_index = (
                    raw_correct_index
                )

            elif (
                isinstance(
                    raw_correct_index,
                    float,
                )
                and raw_correct_index.is_integer()
            ):

                correct_index = int(
                    raw_correct_index
                )

            else:

                return (
                    None,
                    "correct_index is not usable",
                )

        except Exception:

            return (
                None,
                "correct_index normalization failed",
            )

    else:

        return (
            None,
            "missing correct_option",
        )

    if not isinstance(
        options,
        list,
    ):

        return (
            None,
            "options must be a list",
        )

    clean_options = [
        str(
            option or ""
        ).strip()
        for option
        in options
    ]

    quality_error = (
        basic_quality_check(
            question,
            clean_options,
            correct_index,
            explanation,
        )
    )

    if quality_error:

        return (
            None,
            quality_error,
        )

    fingerprint = (
        question_fingerprint(
            target["grade"],
            target["subject"],
            target["syllabus_topic"],
            target["difficulty"],
            question,
            clean_options,
        )
    )

    return (
        {
            "grade": target["grade"],
            "subject": target["subject"],
            "syllabus_topic": target["syllabus_topic"],
            "difficulty": target["difficulty"],
            "question_text": question,
            "options": clean_options,
            "correct_index": correct_index,
            "explanation": explanation,
            "question_fingerprint": fingerprint,
        },
        None,
    )


# ============================================================================
# PROVIDER DISCOVERY
# ============================================================================

def fetch_available_model_ids() -> Dict[
    str,
    set[str],
]:

    available = {
        "gemini": set(),
        "groq": set(),
    }

    gemini_key = str(
        os.environ.get(
            "GEMINI_API_KEY"
        )
        or ""
    ).strip()

    if gemini_key:

        try:

            response = requests.get(
                (
                    "https://generativelanguage.googleapis.com/"
                    "v1beta/models"
                ),
                params={
                    "key": gemini_key,
                    "pageSize": 1000,
                },
                timeout=30,
            )

            if response.ok:

                data = response.json()

                for model in data.get(
                    "models",
                    [],
                ):

                    name = str(
                        model.get(
                            "name"
                        )
                        or ""
                    )

                    if name.startswith(
                        "models/"
                    ):

                        name = name.split(
                            "/",
                            1,
                        )[1]

                    methods = (
                        model.get(
                            "supportedGenerationMethods"
                        )
                        or []
                    )

                    if (
                        name
                        and "generateContent"
                        in methods
                    ):

                        available[
                            "gemini"
                        ].add(
                            name
                        )

            else:

                print(
                    "[WARN] Gemini model discovery "
                    f"failed: HTTP {response.status_code}"
                )

        except requests.RequestException as exc:

            print(
                "[WARN] Gemini model discovery "
                f"failed: {exc}"
            )

    groq_key = str(
        os.environ.get(
            "GROQ_API_KEY"
        )
        or ""
    ).strip()

    if groq_key:

        try:

            response = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={
                    "Authorization": (
                        f"Bearer {groq_key}"
                    )
                },
                timeout=30,
            )

            if response.ok:

                data = response.json()

                for model in data.get(
                    "data",
                    [],
                ):

                    model_id = str(
                        model.get(
                            "id"
                        )
                        or ""
                    ).strip()

                    if (
                        model_id
                        and model.get(
                            "active",
                            True,
                        )
                    ):

                        available[
                            "groq"
                        ].add(
                            model_id
                        )

            else:

                print(
                    "[WARN] Groq model discovery "
                    f"failed: HTTP {response.status_code}"
                )

        except requests.RequestException as exc:

            print(
                "[WARN] Groq model discovery "
                f"failed: {exc}"
            )

    return available


def mark_unavailable_models(
    runtime: Dict[
        str,
        ModelRuntime,
    ],
    available: Dict[
        str,
        set[str],
    ],
) -> None:

    for spec in MODEL_HIERARCHY:

        provider_models = (
            available.get(
                spec.provider
            )
        )

        if (
            provider_models is None
            or not provider_models
        ):

            continue

        if (
            spec.model
            not in provider_models
        ):

            runtime[
                spec.key
            ].status = (
                "UNAVAILABLE"
            )

            runtime[
                spec.key
            ].exhausted_reason = (
                "model not exposed by provider model catalog"
            )

            print(
                f"[SKIP] "
                f"{spec.provider}:{spec.model} "
                "is not currently exposed "
                "by the provider API."
            )


# ============================================================================
# GEMINI GENERATION
# ============================================================================

def gemini_thinking_level(model: str) -> str:
    """
    Keep Gemini 3.x reasoning bounded so hidden reasoning does not consume
    most of the output/token budget before the JSON batch is emitted.
    """
    if model == "gemini-3.5-flash-lite":
        return "minimal"
    return "low"


def call_gemini(
    spec: ModelSpec,
    prompt: str,
    count: int,
) -> Tuple[
    str,
    int,
]:

    api_key = str(
        os.environ.get(
            "GEMINI_API_KEY"
        )
        or ""
    ).strip()

    if not api_key:

        raise ProviderError(
            "GEMINI_API_KEY is not configured",
            temporary=False,
        )

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{spec.model}:generateContent"
        f"?key={api_key}"
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": prompt
                    }
                ],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": build_json_schema(
                count
            ),
            "maxOutputTokens": 16_000,
            "thinkingConfig": {
                "thinkingLevel": gemini_thinking_level(
                    spec.model
                ),
            },
        },
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=DEFAULT_HTTP_TIMEOUT,
        )

    except requests.RequestException as exc:

        raise ProviderError(
            f"Gemini network error: {exc}",
            temporary=True,
        )

    headers = dict(
        response.headers
    )

    body = response_text_safe(
        response
    )

    if response.status_code == 429:

        raise ProviderError(
            f"Gemini 429: {body}",
            temporary=True,
            daily_exhausted=(
                looks_daily_exhausted(
                    body,
                    headers,
                    spec,
                )
            ),
            retry_after=(
                parse_retry_after(
                    headers
                )
            ),
        )

    if response.status_code in (
        401,
        403,
        404,
    ):

        raise ProviderError(
            (
                "Gemini configuration/model "
                f"access error "
                f"({response.status_code}): "
                f"{body}"
            ),
            temporary=False,
        )

    if response.status_code >= 500:

        raise ProviderError(
            (
                "Gemini server error "
                f"({response.status_code}): "
                f"{body}"
            ),
            temporary=True,
            retry_after=(
                parse_retry_after(
                    headers
                )
            ),
        )

    if response.status_code >= 400:

        raise ProviderError(
            (
                "Gemini API error "
                f"({response.status_code}): "
                f"{body}"
            ),
            temporary=False,
        )

    try:

        data = response.json()

        candidates = data.get(
            "candidates",
            [],
        )

        if not candidates:

            raise ValueError(
                "Gemini returned no candidates"
            )

        parts = (
            candidates[0]
            .get(
                "content",
                {},
            )
            .get(
                "parts",
                [],
            )
        )

        raw = "".join(
            str(
                part.get(
                    "text"
                )
                or ""
            )
            for part
            in parts
        )

        usage = (
            data.get(
                "usageMetadata"
            )
            or {}
        )

        total_tokens = int(
            usage.get(
                "totalTokenCount"
            )
            or 0
        )

    except Exception as exc:

        raise ProviderError(
            f"Gemini response parsing failed: {exc}",
            temporary=True,
        )

    if not raw.strip():

        raise ProviderError(
            "Gemini returned empty output",
            temporary=True,
        )

    return (
        raw,
        total_tokens,
    )


# ============================================================================
# GROQ GENERATION
# ============================================================================

def call_groq(
    spec: ModelSpec,
    prompt: str,
    count: int,
) -> Tuple[
    str,
    int,
]:

    api_key = str(
        os.environ.get(
            "GROQ_API_KEY"
        )
        or ""
    ).strip()

    if not api_key:

        raise ProviderError(
            "GROQ_API_KEY is not configured",
            temporary=False,
        )

    url = (
        "https://api.groq.com/openai/v1/chat/completions"
    )

    headers = {
        "Authorization": (
            f"Bearer {api_key}"
        ),
        "Content-Type": (
            "application/json"
        ),
    }

    schema = build_json_schema(
        count
    )

    if spec.model in {
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    }:

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "mcq_batch",
                "schema": schema,
                "strict": True,
            },
        }

        extra_payload = {
            # Groq GPT-OSS exposes reasoning separately. Do not send
            # reasoning_format to GPT-OSS; use include_reasoning=False.
            # Keep 20B at low reasoning effort so a 5-question batch stays
            # comfortably inside the 8K TPM free-tier window.
            "reasoning_effort": (
                "low"
                if spec.model == "openai/gpt-oss-20b"
                else "medium"
            ),
            "include_reasoning": False,
        }

        temperature = 0.2

    elif spec.model == (
        "qwen/qwen3.6-27b"
    ):

        response_format = {
            "type": "json_object"
        }

        extra_payload = {
            "reasoning_effort": "none",
            "reasoning_format": "hidden",
        }

        temperature = 0.7

    else:

        raise ProviderError(
            (
                "Unsupported Groq model: "
                f"{spec.model}"
            ),
            temporary=False,
        )

    base_payload = {
        "model": spec.model,

        "messages": [
            {
                "role": "user",
                "content": (
                    "Return ONLY the requested JSON object. "
                    "Do not include reasoning, markdown, "
                    "code fences, or text outside the JSON.\n\n"
                    + prompt
                ),
            }
        ],

        "temperature": temperature,

        "response_format": response_format,

        "max_completion_tokens": (
            3_072
            if spec.model == "openai/gpt-oss-20b"
            else 4_096
        ),

        **extra_payload,
    }

    try:

        response = requests.post(
            url,
            json=base_payload,
            headers=headers,
            timeout=DEFAULT_HTTP_TIMEOUT,
        )

    except requests.RequestException as exc:

        raise ProviderError(
            f"Groq network error: {exc}",
            temporary=True,
        )

    response_headers = dict(
        response.headers
    )

    body = response_text_safe(
        response
    )

    # Strict JSON schema is supported by GPT-OSS 20B/120B.
    # Do not retry the same batch with unstructured JSON after a schema error:
    # the second request can consume the entire 8K TPM window and trigger
    # the "high usage" error seen in the demo.

    if response.status_code == 429:

        raise ProviderError(
            f"Groq 429: {body}",
            temporary=True,
            daily_exhausted=(
                looks_daily_exhausted(
                    body,
                    response_headers,
                    spec,
                )
            ),
            retry_after=(
                parse_retry_after(
                    response_headers
                )
            ),
        )

    if response.status_code in (
        401,
        403,
        404,
    ):

        raise ProviderError(
            (
                "Groq configuration/model "
                f"access error "
                f"({response.status_code}): "
                f"{body}"
            ),
            temporary=False,
        )

    if response.status_code >= 500:

        raise ProviderError(
            (
                "Groq server error "
                f"({response.status_code}): "
                f"{body}"
            ),
            temporary=True,
            retry_after=(
                parse_retry_after(
                    response_headers
                )
            ),
        )

    if response.status_code >= 400:

        lowered = body.lower()

        temporary = (
            "rate_limit"
            in lowered
            or "temporarily"
            in lowered
        )

        raise ProviderError(
            (
                "Groq API error "
                f"({response.status_code}): "
                f"{body}"
            ),
            temporary=temporary,
            retry_after=(
                parse_retry_after(
                    response_headers
                )
            ),
        )

    try:

        data = response.json()

        choices = data.get(
            "choices",
            [],
        )

        if not choices:

            raise ValueError(
                "Groq returned no choices"
            )

        message = (
            choices[0]
            .get(
                "message",
                {},
            )
        )

        raw = str(
            message.get(
                "content"
            )
            or ""
        )

        usage = (
            data.get(
                "usage"
            )
            or {}
        )

        total_tokens = int(
            usage.get(
                "total_tokens"
            )
            or 0
        )

    except Exception as exc:

        raise ProviderError(
            f"Groq response parsing failed: {exc}",
            temporary=True,
        )

    if not raw.strip():

        raise ProviderError(
            "Groq returned empty output",
            temporary=True,
        )

    return (
        raw,
        total_tokens,
    )


# ============================================================================
# VALIDATOR QUEUE DATA
# ============================================================================

@dataclass
class ValidationItem:
    source_model: str
    target: Dict[str, str]
    row: Dict[str, Any]


@dataclass
class ValidationBatch:
    batch_id: str
    validator_model: str
    items: List[
        ValidationItem
    ]


@dataclass
class ValidationResult:
    batch_id: str
    validator_model: str
    items: List[
        ValidationItem
    ]
    approved: List[
        Dict[str, Any]
    ]
    rejected: int
    repaired_count: int = 0
    error: Optional[str] = None


# ============================================================================
# VALIDATOR SCHEMA
# ============================================================================

def build_validator_schema() -> Dict[
    str,
    Any,
]:

    return {
        "type": "object",

        "properties": {
            "reviews": {
                "type": "array",

                "items": {
                    "type": "object",

                    "properties": {
                        "question_number": {
                            "type": "integer",
                        },

                        "approved": {
                            "type": "boolean",
                        },

                        "score": {
                            "type": "integer",
                        },

                        "reason": {
                            "type": "string",
                        },
                    },

                    "required": [
                        "question_number",
                        "approved",
                        "score",
                        "reason",
                    ],

                    "additionalProperties": False,
                },
            }
        },

        "required": [
            "reviews"
        ],

        "additionalProperties": False,
    }


def build_validator_repair_schema() -> Dict[
    str,
    Any,
]:

    return {
        "type": "object",

        "properties": {
            "repairs": {
                "type": "array",

                "items": {
                    "type": "object",

                    "properties": {
                        "question_number": {
                            "type": "integer",
                        },

                        "repairable": {
                            "type": "boolean",
                        },

                        "confidence": {
                            "type": "integer",
                        },

                        "question_text": {
                            "type": "string",
                        },

                        "options": {
                            "type": "array",

                            "items": {
                                "type": "string",
                            },
                        },

                        "correct_option": {
                            "type": "string",

                            "enum": [
                                "A",
                                "B",
                                "C",
                                "D",
                            ],
                        },

                        "explanation": {
                            "type": "string",
                        },

                        "reason": {
                            "type": "string",
                        },
                    },

                    "required": [
                        "question_number",
                        "repairable",
                        "confidence",
                        "question_text",
                        "options",
                        "correct_option",
                        "explanation",
                        "reason",
                    ],

                    "additionalProperties": False,
                },
            }
        },

        "required": [
            "repairs"
        ],

        "additionalProperties": False,
    }
    

def build_validator_prompt(
    items: List[
        ValidationItem
    ],
) -> str:

    blocks: List[
        str
    ] = []

    for index, item in enumerate(
        items,
        start=1,
    ):

        row = item.row

        options = "\n".join(
            f"{label}) {option}"
            for label, option
            in zip(
                [
                    "A",
                    "B",
                    "C",
                    "D",
                ],
                row[
                    "options"
                ],
            )
        )

        correct = [
            "A",
            "B",
            "C",
            "D",
        ][
            row[
                "correct_index"
            ]
        ]

        target = item.target

        blocks.append(
            f"""
QUESTION {index}

TARGET:
- Grade: {target["grade"]}
- Subject: {target["subject"]}
- Sub-topic: {target["syllabus_topic"]}
- Difficulty: {target["difficulty"]}

Question:
{row["question_text"]}

Options:
{options}

Model-selected correct option:
{correct}

Explanation:
{row["explanation"]}
""".strip()
        )

    joined = "\n\n".join(
        blocks
    )

    return f"""
You are the final quality-control reviewer for a production STEM question bank.

Review EVERY question independently.

A question is APPROVED only if ALL of these are true:

1. The question is factually correct.
2. Exactly one option is correct.
3. The selected answer is actually correct.
4. The explanation agrees with the selected answer.
5. The question genuinely belongs to its specified sub-topic.
6. The grade is appropriate.
7. The requested difficulty is appropriate.
8. The wording is clear and unambiguous.
9. The question is useful for actual student practice.
10. Mathematical calculations are correct.
11. Scientific reasoning is correct.
12. Engineering reasoning is correct.
13. Technical/computing claims are correct.
14. Units and conversions are correct.
15. The question is not broken.
16. The question is not internally contradictory.
17. The options are plausible and distinct.

IMPORTANT:

Do NOT approve a question merely because it looks well written.

For numerical questions:
- independently recompute the result;
- check units;
- compare the result directly against all options.

For factual questions:
- check the actual claim;
- reject factual mistakes.

For conceptual questions:
- reason through the principle yourself;
- reject subtle conceptual mistakes.

For every question, return:
- question_number
- approved
- score
- reason

question_number corresponds to the question's number
in THIS validation batch.

Output ONLY JSON.

Example shape:

{{
  "reviews": [
    {{
      "question_number": 1,
      "approved": true,
      "score": 96,
      "reason": "The answer is factually and logically correct."
    }},
    {{
      "question_number": 2,
      "approved": false,
      "score": 52,
      "reason": "The selected option is not mathematically correct."
    }}
  ]
}}

Review all questions.

{joined}
""".strip()


def build_validator_repair_prompt(
    rejected_items: List[
        Tuple[
            int,
            ValidationItem,
            int,
            str,
        ]
    ],
) -> str:

    blocks: List[str] = []

    for (
        number,
        item,
        score,
        reason,
    ) in rejected_items:

        row = item.row
        target = item.target

        options = "\n".join(
            f"{label}) {option}"
            for label, option
            in zip(
                [
                    "A",
                    "B",
                    "C",
                    "D",
                ],
                row["options"],
            )
        )

        current_correct = [
            "A",
            "B",
            "C",
            "D",
        ][
            row["correct_index"]
        ]

        blocks.append(
            f"""
QUESTION {number}

TARGET:
- Grade: {target["grade"]}
- Subject: {target["subject"]}
- Sub-topic: {target["syllabus_topic"]}
- Difficulty: {target["difficulty"]}

CURRENT QUESTION:
{row["question_text"]}

OPTIONS:
{options}

CURRENT MODEL-SELECTED CORRECT OPTION:
{current_correct}

CURRENT EXPLANATION:
{row["explanation"]}

INITIAL VALIDATOR SCORE:
{score}

INITIAL VALIDATOR REASON:
{reason}
""".strip()
        )

    joined = "\n\n".join(
        blocks
    )

    return f"""
You are performing ONE AND ONLY ONE repair attempt on rejected STEM multiple-choice questions.

Your job is NOT to rewrite every rejected question.

Only repair a question when the problem is clearly and safely fixable with ONE obvious correction.

Typical repairable cases:
- the question is correct but the selected correct option letter is wrong;
- the explanation and selected answer disagree but the correct answer is unambiguous;
- a simple arithmetic/result mismatch can be corrected;
- a minor answer-label inconsistency can be corrected.

Do NOT repair when:
- the question is fundamentally broken;
- the wording is ambiguous;
- multiple options could reasonably be correct;
- the underlying fact is uncertain;
- the question needs substantial rewriting;
- the question is outside the target topic;
- the difficulty/grade is fundamentally wrong;
- fixing it would require inventing information.

For every question:

1. Independently solve the question yourself.
2. Determine the actually correct answer.
3. Compare it against all four options.
4. Inspect the original explanation.
5. Decide whether there is ONE obvious safe correction.
6. Set repairable=true ONLY when confidence is at least {VALIDATOR_REPAIR_MIN_CONFIDENCE}.
7. If repairable=true, return the COMPLETE corrected question, options, correct_option, and explanation.
8. If repairable=false, return the original question unchanged and explain why it must remain rejected.
9. Never invent facts.
10. Never silently perform a major rewrite.
11. The corrected question must have exactly four distinct options.
12. The corrected explanation must agree with the corrected answer.
13. For numerical questions, independently recompute the result.
14. If the original correct option is wrong but another existing option is clearly correct, change ONLY the answer/explanation as needed.
15. A successful repair must produce a question that would pass the normal validator.

Output ONLY JSON.

Return exactly one repair result for every input question.

Shape:

{{
  "repairs": [
    {{
      "question_number": 1,
      "repairable": true,
      "confidence": 98,
      "question_text": "...",
      "options": [
        "...",
        "...",
        "...",
        "..."
      ],
      "correct_option": "A",
      "explanation": "...",
      "reason": "The original selected option was incorrect; option A is the uniquely correct answer."
    }}
  ]
}}

{joined}
""".strip()


# ============================================================================
# SAVE VALIDATION BATCH
# ============================================================================

def save_pending_validation(
    batch: ValidationBatch,
) -> Path:

    VALIDATION_PENDING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        VALIDATION_PENDING_DIR
        / f"{batch.batch_id}.json"
    )

    payload = {
        "batch_id": batch.batch_id,

        "validator_model": batch.validator_model,

        "items": [
            {
                "source_model": item.source_model,
                "target": item.target,
                "row": item.row,
            }
            for item in batch.items
        ],
    }

    temp = path.with_suffix(
        ".tmp"
    )

    with temp.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    temp.replace(
        path
    )

    return path


def delete_pending_validation(
    batch_id: str,
) -> None:

    path = (
        VALIDATION_PENDING_DIR
        / f"{batch_id}.json"
    )

    try:

        path.unlink()

    except FileNotFoundError:

        pass


# ============================================================================
# VALIDATOR API
# ============================================================================

def call_validator(
    items: List[
        ValidationItem
    ],
    validator_model: str,
    spec: ModelSpec,
) -> Tuple[
    List[
        Dict[str, Any]
    ],
    int,
]:

    api_key = str(
        os.environ.get(
            VALIDATOR_API_KEY_ENV
        )
        or ""
    ).strip()

    if not api_key:

        raise ProviderError(
            "Validator API key missing. "
            "Set GEMINI_VALIDATOR_API_KEY.",
            temporary=False,
        )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": build_validator_prompt(
                            items
                        )
                    }
                ],
            }
        ],

        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": (
                build_validator_schema()
            ),
            "maxOutputTokens": 8_000,
        },
    }

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{validator_model}:generateContent"
        f"?key={api_key}"
    )

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=VALIDATOR_HTTP_TIMEOUT,
        )

    except requests.RequestException as exc:

        raise ProviderError(
            f"Validator network error: {exc}",
            temporary=True,
        )

    body = response_text_safe(
        response
    )

    headers = dict(
        response.headers
    )

    if response.status_code == 429:
        raise ProviderError(
            (
                "Validator rate limit reached: "
                f"{body}"
            ),
            temporary=True,
            daily_exhausted=(
                looks_daily_exhausted(
                    body,
                    headers,
                    spec,
                )
            ),
            retry_after=(
                parse_retry_after(
                    headers
                )
            ),
        )

    if response.status_code in (
        401,
        403,
        404,
    ):

        raise ProviderError(
            (
                "Validator configuration/model "
                f"error ({response.status_code}): "
                f"{body}"
            ),
            temporary=False,
        )

    if response.status_code >= 500:

        raise ProviderError(
            (
                "Validator server error "
                f"({response.status_code}): "
                f"{body}"
            ),
            temporary=True,
            retry_after=(
                parse_retry_after(
                    headers
                )
            ),
        )

    if response.status_code >= 400:

        raise ProviderError(
            (
                "Validator API error "
                f"({response.status_code}): "
                f"{body}"
            ),
            temporary=False,
        )

    try:

        data = response.json()

        usage_metadata = (
            data.get(
                "usageMetadata"
            )
            or {}
        )

        total_tokens = int(
            usage_metadata.get(
                "totalTokenCount",
                0,
            )
            or 0
        )

        candidates = data.get(
            "candidates",
            [],
        )

        if not candidates:

            raise ValueError(
                "Validator returned no candidates"
            )

        parts = (
            candidates[0]
            .get(
                "content",
                {},
            )
            .get(
                "parts",
                [],
            )
        )

        raw = "".join(
            str(
                part.get(
                    "text"
                )
                or ""
            )
            for part
            in parts
        ).strip()

        parsed = json.loads(
            clean_json_text(
                raw
            )
        )

        reviews = (
            parsed.get(
                "reviews"
            )
            if isinstance(
                parsed,
                dict,
            )
            else None
        )

    except Exception as exc:

        raise ProviderError(
            (
                "Validator response "
                f"parsing failed: {exc}"
            ),
            temporary=True,
        )

    if not isinstance(
        reviews,
        list,
    ):

        raise ProviderError(
            (
                "Validator response does not "
                "contain a reviews array."
            ),
            temporary=True,
        )

    if len(reviews) != len(
        items
    ):

        raise ProviderError(
            (
                f"Validator returned "
                f"{len(reviews)} reviews for "
                f"{len(items)} questions."
            ),
            temporary=True,
        )

    return (
        reviews,
        total_tokens,
    )


def call_validator_repair(
    rejected_items: List[
        Tuple[
            int,
            ValidationItem,
            int,
            str,
        ]
    ],
    validator_model: str,
    spec: ModelSpec,
) -> Tuple[
    List[
        Dict[str, Any]
    ],
    int,
]:

    api_key = str(
        os.environ.get(
            VALIDATOR_API_KEY_ENV
        )
        or ""
    ).strip()

    if not api_key:

        raise ProviderError(
            "Validator API key missing for repair.",
            temporary=False,
        )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": build_validator_repair_prompt(
                            rejected_items
                        )
                    }
                ],
            }
        ],

        "generationConfig": {
            "responseMimeType": "application/json",

            "responseJsonSchema": (
                build_validator_repair_schema()
            ),

            "maxOutputTokens": 10_000,
        },
    }

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{validator_model}:generateContent"
        f"?key={api_key}"
    )

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=VALIDATOR_HTTP_TIMEOUT,
        )

    except requests.RequestException as exc:

        raise ProviderError(
            f"Validator repair network error: {exc}",
            temporary=True,
        )

    body = response_text_safe(
        response
    )

    headers = dict(
        response.headers
    )

    if response.status_code == 429:
        raise ProviderError(
            (
                "Validator repair rate limit reached: "
                f"{body}"
            ),
            temporary=True,
            daily_exhausted=(
                looks_daily_exhausted(
                    body,
                    headers,
                    spec,
                )
            ),
            retry_after=(
                parse_retry_after(
                    headers
                )
            ),
        )

    if response.status_code in (
        401,
        403,
        404,
    ):

        raise ProviderError(
            (
                "Validator repair configuration/model "
                f"error ({response.status_code}): "
                f"{body}"
            ),
            temporary=False,
        )

    if response.status_code >= 500:

        raise ProviderError(
            (
                "Validator repair server error "
                f"({response.status_code}): "
                f"{body}"
            ),
            temporary=True,
            retry_after=(
                parse_retry_after(
                    headers
                )
            ),
        )

    if response.status_code >= 400:

        raise ProviderError(
            (
                "Validator repair API error "
                f"({response.status_code}): "
                f"{body}"
            ),
            temporary=False,
        )

    try:

        data = response.json()

        usage_metadata = (
            data.get(
                "usageMetadata"
            )
            or {}
        )

        total_tokens = int(
            usage_metadata.get(
                "totalTokenCount",
                0,
            )
            or 0
        )

        candidates = data.get(
            "candidates",
            [],
        )

        if not candidates:

            raise ValueError(
                "Validator repair returned no candidates"
            )

        parts = (
            candidates[0]
            .get(
                "content",
                {},
            )
            .get(
                "parts",
                [],
            )
        )

        raw = "".join(
            str(
                part.get(
                    "text"
                )
                or ""
            )
            for part in parts
        ).strip()

        parsed = json.loads(
            clean_json_text(
                raw
            )
        )

        repairs = (
            parsed.get(
                "repairs"
            )
            if isinstance(
                parsed,
                dict,
            )
            else None
        )

    except Exception as exc:

        raise ProviderError(
            (
                "Validator repair response "
                f"parsing failed: {exc}"
            ),
            temporary=True,
        )

    if not isinstance(
        repairs,
        list,
    ):

        raise ProviderError(
            (
                "Validator repair response "
                "does not contain a repairs array."
            ),
            temporary=True,
        )

    if len(repairs) != len(
        rejected_items
    ):

        raise ProviderError(
            (
                f"Validator repair returned "
                f"{len(repairs)} repairs for "
                f"{len(rejected_items)} rejected questions."
            ),
            temporary=True,
        )

    return (
        repairs,
        total_tokens,
    )


# ============================================================================
# VALIDATOR WORKER
# ============================================================================

def validator_worker(
    generator: "QuestionGenerator",
    validator_model: str,
) -> None:

    while True:

        # --------------------------------------------------------------
        # Wait for the FIRST question.
        #
        # This means the validator is independent:
        # it does not wait for a full batch to be generated.
        # --------------------------------------------------------------

        validator_queue = (
            generator.validator_queues[
                validator_model
            ]
        )

        try:
            first_item = (
                validator_queue.get(
                    timeout=0.5
                )
            )
            
            if not generator._validator_is_active(
                validator_model
            ):
                generator._reroute_validator_queue(
                    validator_model,
                    first_item,
                )
                return

        except queue.Empty:
        
            if (
                generator.generation_finished
                and validator_queue.empty()
            ):
                return

            continue
        
        
        # --------------------------------------------------------------
        # SMART BATCH GATHERING
        # --------------------------------------------------------------
        #
        # Give the producer a tiny amount of time to enqueue additional
        # questions before deciding the batch size.
        # --------------------------------------------------------------

        if (
            VALIDATOR_BATCH_GATHER_DELAY > 0
        ):
            time.sleep(
                VALIDATOR_BATCH_GATHER_DELAY
            )


        # We already consumed the first question.
        #
        # If 4 more are waiting:
        #     available_now = 4
        #     desired_batch_size = 5
        #
        # If 14 more are waiting and production max = 10:
        #     desired_batch_size = 10
        desired_batch_size = (
            generator.get_validator_batch_size(
                validator_model
            )
        )

        items = [
            first_item
        ]

        for _ in range(
            desired_batch_size - 1
        ):

            try:
                items.append(
                    validator_queue.get_nowait()
                )

            except queue.Empty:
                break
        
        items = (
            generator._fit_validator_batch_to_tpm(
                validator_model,
                items,
            )
        )
            
            
        batch = ValidationBatch(
            batch_id=str(
                uuid.uuid4()
            ),
            validator_model=validator_model,
            items=items,
        )

        try:

            pending_path = (
                save_pending_validation(
                    batch
                )
            )

            print(
                (
                    f"\n[QA BATCH] "
                    f"validator={validator_model} | "
                    f"Taking {len(batch.items)} "
                    "questions from queue | "
                    f"remaining={validator_queue.qsize()} | "
                    f"max={generator.validator_batch_size} | "
                    f"pending={pending_path.name}"
                ),
                flush=True,
            )

            reviews: Optional[
                List[
                    Dict[str, Any]
                ]
            ] = None

            last_error = ""

            validator_runtime = (
                generator.validator_runtime[
                    validator_model
                ]
            )

            validator_spec = (
                validator_runtime.spec
            )

            # ----------------------------------------------------------
            # Validator execution loop.
            #
            # Temporary provider failures do NOT discard this batch.
            # The same batch remains pending until:
            #   1. validation succeeds,
            #   2. this validator's daily quota is exhausted,
            #   3. this validator becomes permanently unavailable.
            # ----------------------------------------------------------
            retry_attempt = 0

            while (
                reviews is None
                and not generator.stop_requested
            ):
                try:
                    estimated_tokens = (
                        generator._validator_prompt_tokens(
                            batch.items
                        )
                    )

                    generator._wait_for_validator_limits(
                        validator_model,
                        estimated_tokens,
                    )

                    generator._record_validator_request(
                        validator_model
                    )

                    (
                        reviews,
                        actual_tokens,
                    ) = call_validator(
                        batch.items,
                        validator_model,
                        validator_spec,
                    )

                    generator._record_validator_tokens(
                        validator_model,
                        max(
                            actual_tokens,
                            estimated_tokens,
                        ),
                    )

                    retry_attempt = 0

                    break

                except ProviderError as exc:
                    last_error = str(
                        exc
                    )

                    if exc.daily_exhausted:
                        generator._mark_validator_exhausted(
                            validator_model,
                            "provider daily quota exhausted",
                        )

                        # Reroute this batch to another validator
                        # if one still has usable capacity.
                        delete_pending_validation(
                            batch.batch_id
                        )

                        generator._requeue_validator_items(
                            batch.items,
                            exclude_model=validator_model,
                        )

                        reviews = None

                        break

                    if not exc.temporary:
                        generator._mark_validator_unavailable(
                            validator_model,
                            last_error,
                        )

                        delete_pending_validation(
                            batch.batch_id
                        )

                        generator._requeue_validator_items(
                            batch.items,
                            exclude_model=validator_model,
                        )

                        reviews = None

                        break

                    retry_attempt += 1

                    delay = (
                        exc.retry_after
                        or min(
                            30.0,
                            2.0
                            * retry_attempt,
                        )
                    )

                    if (
                        retry_attempt
                        >= VALIDATOR_RETRY_LIMIT
                    ):
                        delay = max(
                            delay,
                            60.0,
                        )
                        retry_attempt = 0

                    print(
                        (
                            f"[QA RETRY] "
                            f"validator={validator_model} "
                            f"batch={batch.batch_id} "
                            f"retrying in "
                            f"{delay:.1f}s"
                        ),
                        flush=True,
                    )

                    generator._interruptible_sleep(
                        delay
                    )

                except Exception as exc:
                    last_error = str(
                        exc
                    )

                    retry_attempt += 1

                    delay = min(
                        30.0,
                        2.0
                        * retry_attempt,
                    )

                    if (
                        retry_attempt
                        >= VALIDATOR_RETRY_LIMIT
                    ):
                        delay = 60.0
                        retry_attempt = 0

                    print(
                        (
                            f"[QA RETRY] "
                            f"validator={validator_model} "
                            f"batch={batch.batch_id} "
                            f"unexpected error; "
                            f"retrying in {delay:.1f}s"
                        ),
                        flush=True,
                    )

                    generator._interruptible_sleep(
                        delay
                    )

            # Stop requested while waiting for this batch.
            # Keep the pending JSON on disk.
            if (
                reviews is None
                and generator.stop_requested
            ):
                continue

            # If the current validator was exhausted/unavailable,
            # this worker must not discard the batch.
            if reviews is None:
                continue

            approved_rows: List[
                Dict[str, Any]
            ] = []

            rejected = 0

            repaired_count = 0

            seen_numbers: set[
                int
            ] = set()

            # Questions rejected by the first validation pass.
            # Each receives at most ONE repair attempt.
            rejected_for_repair: List[
                Tuple[
                    int,
                    ValidationItem,
                    int,
                    str,
                ]
            ] = []

            for review in reviews:

                if not isinstance(
                    review,
                    dict,
                ):

                    rejected += 1
                    continue

                raw_number = review.get(
                    "question_number"
                )

                try:

                    if isinstance(
                        raw_number,
                        bool,
                    ):

                        raise ValueError

                    if isinstance(
                        raw_number,
                        str,
                    ):

                        raw_number = (
                            raw_number.strip()
                        )

                    number = int(
                        raw_number
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    rejected += 1

                    continue

                if (
                    number < 1
                    or number > len(
                        batch.items
                    )
                ):

                    rejected += 1
                    continue

                if number in seen_numbers:

                    rejected += 1
                    continue

                seen_numbers.add(
                    number
                )

                review_approved = (
                    review.get(
                        "approved"
                    )
                )

                if isinstance(
                    review_approved,
                    bool,
                ):

                    approved = (
                        review_approved
                    )

                elif isinstance(
                    review_approved,
                    str,
                ):

                    approved = (
                        review_approved
                        .strip()
                        .lower()
                        == "true"
                    )

                else:

                    approved = False

                raw_score = review.get(
                    "score"
                )

                try:

                    score = int(
                        str(
                            raw_score
                            or 0
                        ).strip()
                    )

                except (
                    ValueError,
                    TypeError,
                ):

                    score = 0

                reason = str(
                    review.get(
                        "reason"
                    )
                    or ""
                ).strip()

                item = batch.items[
                    number - 1
                ]

                if (
                    approved
                    and score
                    >= VALIDATOR_MIN_SCORE
                ):

                    item.row[
                        "generation_meta"
                    ] = {
                        **(
                            item.row.get(
                                "generation_meta"
                            )
                            or {}
                        ),

                        "validator_model": (
                            validator_model
                        ),

                        "validator_score": (
                            score
                        ),

                        "validator_status": (
                            "approved"
                        ),

                        "validator_reason": (
                            reason
                        ),

                        "validated_at": (
                            now_iso()
                        ),
                    }

                    approved_rows.append(
                        item.row
                    )

                else:

                    # Do NOT reject immediately.
                    # Give the validator exactly ONE opportunity to repair
                    # a simple/high-confidence inconsistency.
                    rejected_for_repair.append(
                        (
                            number,
                            item,
                            score,
                            reason,
                        )
                    )

            rejected += max(
                0,
                len(
                    batch.items
                )
                - len(
                    seen_numbers
                ),
            )
            
            # ----------------------------------------------------------------------
            # ONE-TIME REPAIR PASS
            # ----------------------------------------------------------------------
            #
            # Only questions rejected by the first validator pass reach this point.
            # Each rejected question gets exactly ONE repair opportunity.
            #
            # The repair itself is performed in ONE validator API call for the
            # rejected questions in this batch.
            # ----------------------------------------------------------------------

            if rejected_for_repair:
            
                print(
                    (
                        f"[QA REPAIR] "
                        f"Attempting one repair pass for "
                        f"{len(rejected_for_repair)} rejected questions."
                    ),
                    flush=True,
                )

                repairs: Optional[
                    List[
                        Dict[str, Any]
                    ]
                ] = None

                repair_error = ""

                for repair_attempt in range(
                    1,
                    3,
                ):

                    try:
                    
                        estimated_repair_tokens = (
                            generator._validator_repair_prompt_tokens(
                                rejected_for_repair
                            )
                        )

                        generator._wait_for_validator_limits(
                            validator_model,
                            estimated_repair_tokens,
                        )

                        generator._record_validator_request(
                            validator_model
                        )

                        (
                            repairs,
                            actual_repair_tokens,
                        ) = call_validator_repair(
                            rejected_for_repair,
                            validator_model,
                            validator_spec,
                        )

                        generator._record_validator_tokens(
                            validator_model,
                            max(
                                actual_repair_tokens,
                                estimated_repair_tokens,
                            ),
                        )

                        break
                    
                    except ProviderError as exc:
                    
                        repair_error = str(
                            exc
                        )

                        if not exc.temporary:
                            break
                        
                        delay = (
                            exc.retry_after
                            or min(
                                20.0,
                                2.0 * repair_attempt,
                            )
                        )

                        print(
                            (
                                f"[QA REPAIR RETRY] "
                                f"attempt={repair_attempt}/2 "
                                f"retrying in "
                                f"{delay:.1f}s"
                            ),
                            flush=True,
                        )

                        generator._interruptible_sleep(
                            delay
                        )

                    except Exception as exc:
                    
                        repair_error = str(
                            exc
                        )

                        break
                    
                if repairs is not None:

                    repaired_numbers: set[
                        int
                    ] = set()
                
                    for (
                        repair,
                        rejected_entry,
                    ) in zip(
                        repairs,
                        rejected_for_repair,
                    ):

                        (
                            number,
                            item,
                            original_score,
                            original_reason,
                        ) = rejected_entry

                        if not isinstance(
                            repair,
                            dict,
                        ):
                            continue
                        
                        try:
                        
                            repair_number = int(
                                repair.get(
                                    "question_number"
                                )
                            )

                        except (
                            TypeError,
                            ValueError,
                        ):

                            continue
                        
                        if (
                            repair_number
                            != number
                        ):
                            continue
                        
                        if (
                            repair_number
                            in repaired_numbers
                        ):
                            continue
                        
                        repaired_numbers.add(
                            repair_number
                        )

                        repairable = (
                            repair.get(
                                "repairable"
                            )
                            is True
                        )

                        try:
                        
                            confidence = int(
                                repair.get(
                                    "confidence",
                                    0,
                                )
                            )

                        except (
                            TypeError,
                            ValueError,
                        ):

                            confidence = 0

                        if not (
                            repairable
                            and confidence
                            >= VALIDATOR_REPAIR_MIN_CONFIDENCE
                        ):

                            print(
                                (
                                    f"[QA REPAIR REJECT] "
                                    f"{item.source_model} "
                                    f"Q{number}: "
                                    "repair not sufficiently "
                                    f"confident ({confidence}%)."
                                ),
                                flush=True,
                            )

                            continue
                        
                        repaired_payload = {
                            "question_text": normalize_latex_text(
                                repair.get(
                                    "question_text"
                                )
                            ),
                        
                            "options": [
                                normalize_latex_text(
                                    option
                                )
                                for option in (
                                    repair.get(
                                        "options"
                                    )
                                    or []
                                )
                            ],
                        
                            "correct_option": (
                                repair.get(
                                    "correct_option"
                                )
                            ),
                        
                            "explanation": normalize_latex_text(
                                repair.get(
                                    "explanation"
                                )
                            ),
                        }

                        # ----------------------------------------------------------
                        # Run the repaired question through our existing local
                        # structural validator.
                        #
                        # This does NOT count as another AI repair attempt.
                        # ----------------------------------------------------------

                        repaired_row, local_error = (
                            validate_question(
                                repaired_payload,
                                item.target,
                            )
                        )

                        if (
                            repaired_row is None
                            or local_error
                        ):

                            print(
                                (
                                    f"[QA REPAIR REJECT] "
                                    f"{item.source_model} "
                                    f"Q{number}: "
                                    f"repaired question failed "
                                    f"local validation: "
                                    f"{local_error}"
                                ),
                                flush=True,
                            )

                            continue
                        
                        # ----------------------------------------------------------
                        # Preserve the original generation metadata/identity.
                        # Only replace the corrected question content.
                        # ----------------------------------------------------------

                        original_meta = dict(
                            item.row.get(
                                "generation_meta"
                            )
                            or {}
                        )

                        original_generation_model = (
                            item.row.get(
                                "generation_model"
                            )
                        )

                        original_batch_id = (
                            item.row.get(
                                "generation_batch_id"
                            )
                        )

                        item.row.update(
                            {
                                "question_text": (
                                    repaired_row[
                                        "question_text"
                                    ]
                                ),

                                "options": (
                                    repaired_row[
                                        "options"
                                    ]
                                ),

                                "correct_index": (
                                    repaired_row[
                                        "correct_index"
                                    ]
                                ),

                                "explanation": (
                                    repaired_row[
                                        "explanation"
                                    ]
                                ),

                                "question_fingerprint": (
                                    repaired_row[
                                        "question_fingerprint"
                                    ]
                                ),

                                "generation_model": (
                                    original_generation_model
                                ),

                                "generation_batch_id": (
                                    original_batch_id
                                ),

                                "generation_meta": {
                                    **original_meta,

                                    "validator_repair": {
                                        "attempted": True,
                                        "confidence": confidence,
                                        "original_score": original_score,
                                        "original_reason": original_reason,
                                        "repair_reason": str(
                                            repair.get(
                                                "reason"
                                            )
                                            or ""
                                        ).strip(),
                                        "repaired_at": now_iso(),
                                    },

                                    "validator_model": (
                                        validator_model
                                    ),

                                    "validator_score": (
                                        confidence
                                    ),

                                    "validator_status": (
                                        "approved_after_repair"
                                    ),

                                    "validator_reason": (
                                        str(
                                            repair.get(
                                                "reason"
                                            )
                                            or ""
                                        ).strip()
                                    ),

                                    "validated_at": now_iso(),
                                },
                            }
                        )

                        approved_rows.append(
                            item.row
                        )

                        repaired_count += 1

                        print(
                            (
                                f"[QA REPAIRED] "
                                f"{item.source_model} "
                                f"Q{number}: "
                                f"repair accepted "
                                f"({confidence}% confidence)."
                            ),
                            flush=True,
                        )

                    rejected += (
                        len(
                            rejected_for_repair
                        )
                        - repaired_count
                    )

                else:
                
                    rejected += len(
                        rejected_for_repair
                    )

                    print(
                        (
                            f"[QA REPAIR FAILED] "
                            f"{repair_error}"
                        ),
                        flush=True,
                    )

            else:
            
                print(
                    "[QA REPAIR] No rejected questions; "
                    "no repair call required.",
                    flush=True,
                )

            generator.validation_results.put(
                ValidationResult(
                    batch_id=batch.batch_id,
                    validator_model=validator_model,
                    items=batch.items,
                    approved=approved_rows,
                    rejected=rejected,
                    repaired_count=repaired_count,
                )
            )

            print(
                (
                    f"[QA COMPLETE] "
                    f"reviewed={len(batch.items)} "
                    f"approved={len(approved_rows)} "
                    f"rejected={rejected}"
                ),
                flush=True,
            )

        except Exception as exc:

            print(
                (
                    f"[QA ERROR] "
                    f"validator={validator_model} "
                    f"batch={batch.batch_id}: "
                    f"{trim_one_line(str(exc))}"
                ),
                flush=True,
            )

            generator.validation_results.put(
                ValidationResult(
                    batch_id=batch.batch_id,
                    validator_model=validator_model,
                    items=batch.items,
                    approved=[],
                    rejected=len(
                        batch.items
                    ),
                    repaired_count=0,
                    error=str(exc),
                )
            )

        finally:

            # Every ValidationItem removed from the queue must have
            # exactly one matching task_done().
            for _ in batch.items:
                validator_queue.task_done()


# ============================================================================
# QUESTION GENERATOR
# ============================================================================

class QuestionGenerator:

    def __init__(
        self,
        engine: Engine,
        state: GenerationState,
        syllabus: Dict[str, List[str]],
    ) -> None:

        self.engine = engine
        self.state = state
        self.syllabus = syllabus


        self.combination_counts = (
            load_combination_counts(
                engine
            )
        )

        self.stop_requested = False

        self.generation_finished = False

        self.mode = "demo"

        self.validator_batch_size = (
            VALIDATOR_BATCH_DEMO
        )

        self.runtime = {
            spec.key: ModelRuntime(
                spec
            )
            for spec
            in MODEL_HIERARCHY
        }
        
        # Separate runtime ledger for validator API usage.
        # Validator and generation usage must never share counters.
        validator_specs = {
            spec.model: spec
            for spec in MODEL_HIERARCHY
        }

        self.validator_runtime: Dict[
            str,
            ModelRuntime,
        ] = {
            validator_model: ModelRuntime(
                validator_specs[validator_model]
            )
            for validator_model in VALIDATOR_MODELS
            if validator_model in validator_specs
        }

        self.validator_state_lock = threading.Lock()

        self.validator_queues: Dict[
            str,
            queue.Queue
        ] = {
            model: queue.Queue(
                maxsize=VALIDATION_QUEUE_SIZE
            )
            for model in VALIDATOR_MODELS
        }
        
        self.validation_results: queue.Queue[
            ValidationResult
        ] = queue.Queue()
        
        self.validator_threads: List[
            threading.Thread
        ] = []
        
        self.validator_stats: Dict[
            str,
            Dict[str, Any]
        ] = {
            model: {
                "batches": 0,
                "validated_questions": 0,
                "approved": 0,
                "repaired": 0,
                "rejected": 0,
                "errors": 0,
            }
            for model in VALIDATOR_MODELS
        }

        VALIDATION_PENDING_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "[QA START] Parallel validators active:",
            flush=True,
        )

        for validator_model in VALIDATOR_MODELS:
            print(
                f"  - {validator_model}",
                flush=True,
            )

        self._install_signal_handlers()

        self.state.reset_daily_if_needed()

        self._hydrate_validator_runtime_from_state()

        self._hydrate_runtime_from_state()

        available_models = (
            fetch_available_model_ids()
        )

        mark_unavailable_models(
            self.runtime,
            available_models,
        )
        
        for validator_model in VALIDATOR_MODELS:
            thread = threading.Thread(
                target=validator_worker,
                args=(
                    self,
                    validator_model,
                ),
                name=(
                    "question-validator-"
                    + validator_model
                ),
                daemon=True,
            )
        
            thread.start()
        
            self.validator_threads.append(
                thread
            )
        
    def _validator_bucket(
        self,
        validator_model: str,
    ) -> Dict[str, Any]:
        validators = (
            self.state.data.setdefault(
                "validator_models",
                {},
            )
        )

        runtime = self.validator_runtime[
            validator_model
        ]

        return validators.setdefault(
            validator_model,
            {
                "day": provider_day_label(
                    runtime.spec.reset_zone
                ),
                "status": "ACTIVE",
                "daily_requests": 0,
                "daily_tokens": 0,
                "exhausted_reason": None,
            },
        )

    def _hydrate_validator_runtime_from_state(
        self,
    ) -> None:
        for validator_model, runtime in (
            self.validator_runtime.items()
        ):
            spec = runtime.spec
            bucket = self._validator_bucket(
                validator_model
            )

            today = provider_day_label(
                spec.reset_zone
            )

            if bucket.get("day") != today:
                bucket.update(
                    {
                        "day": today,
                        "status": "ACTIVE",
                        "daily_requests": 0,
                        "daily_tokens": 0,
                        "exhausted_reason": None,
                    }
                )

            runtime.daily_requests = int(
                bucket.get(
                    "daily_requests",
                    0,
                )
            )

            runtime.daily_tokens = int(
                bucket.get(
                    "daily_tokens",
                    0,
                )
            )

            runtime.status = str(
                bucket.get(
                    "status"
                )
                or "ACTIVE"
            )

            runtime.exhausted_reason = (
                bucket.get(
                    "exhausted_reason"
                )
            )

            if (
                runtime.daily_requests
                >= spec.rpd
            ):
                runtime.status = (
                    "DAILY_EXHAUSTED"
                )

            if (
                spec.tpd
                and runtime.daily_tokens
                >= spec.tpd
            ):
                runtime.status = (
                    "DAILY_EXHAUSTED"
                )

        self.state.save()

    def _validator_is_active(
        self,
        validator_model: str,
    ) -> bool:
        runtime = self.validator_runtime[
            validator_model
        ]

        if runtime.status != "ACTIVE":
            return False

        spec = runtime.spec

        if (
            runtime.daily_requests
            >= spec.rpd
        ):
            runtime.status = (
                "DAILY_EXHAUSTED"
            )
            runtime.exhausted_reason = "RPD"
            self._mark_validator_exhausted(
                validator_model,
                "RPD",
            )
            return False

        if (
            spec.tpd
            and runtime.daily_tokens
            >= spec.tpd
        ):
            runtime.status = (
                "DAILY_EXHAUSTED"
            )
            runtime.exhausted_reason = "TPD"
            self._mark_validator_exhausted(
                validator_model,
                "TPD",
            )
            return False

        return True

    def _active_validator_models(
        self,
    ) -> List[str]:
        return [
            validator_model
            for validator_model in VALIDATOR_MODELS
            if self._validator_is_active(
                validator_model
            )
        ]

    def _mark_validator_exhausted(
        self,
        validator_model: str,
        reason: str,
    ) -> None:
        runtime = self.validator_runtime[
            validator_model
        ]

        runtime.status = (
            "DAILY_EXHAUSTED"
        )
        runtime.exhausted_reason = reason

        with self.validator_state_lock:
            bucket = self._validator_bucket(
                validator_model
            )
            bucket["status"] = (
                "DAILY_EXHAUSTED"
            )
            bucket["exhausted_reason"] = (
                reason
            )
            bucket["daily_requests"] = (
                runtime.daily_requests
            )
            bucket["daily_tokens"] = (
                runtime.daily_tokens
            )
            self.state.save()

        print(
            (
                f"[QA EXHAUSTED] "
                f"{validator_model}: "
                f"{reason}; removing validator "
                "from future scheduling."
            ),
            flush=True,
        )

    def _mark_validator_unavailable(
        self,
        validator_model: str,
        reason: str,
    ) -> None:
        runtime = self.validator_runtime[
            validator_model
        ]

        runtime.status = "UNAVAILABLE"
        runtime.exhausted_reason = reason

        with self.validator_state_lock:
            bucket = self._validator_bucket(
                validator_model
            )
            bucket["status"] = (
                "UNAVAILABLE"
            )
            bucket["exhausted_reason"] = (
                reason
            )
            self.state.save()

    def _validator_prompt_tokens(
        self,
        items: List[ValidationItem],
    ) -> int:
        prompt = build_validator_prompt(
            items
        )

        return max(
            600,
            (
                len(prompt)
                // 4
            )
            + (
                len(items)
                * 450
            ),
        )

    def _validator_repair_prompt_tokens(
        self,
        rejected_items: List[
            Tuple[
                int,
                ValidationItem,
                int,
                str,
            ]
        ],
    ) -> int:
        prompt = (
            build_validator_repair_prompt(
                rejected_items
            )
        )

        return max(
            600,
            (
                len(prompt)
                // 4
            )
            + (
                len(rejected_items)
                * 600
            ),
        )

    def _wait_for_validator_limits(
        self,
        validator_model: str,
        estimated_tokens: int,
    ) -> None:
        runtime = self.validator_runtime[
            validator_model
        ]
        spec = runtime.spec

        while (
            not self.stop_requested
        ):
            runtime.trim_windows()

            # ----------------------------------------------------------
            # DAILY REQUEST LIMIT
            #
            # In provider-authoritative mode Gemini itself decides when
            # the daily quota is exhausted.
            # ----------------------------------------------------------
            if (
                not PROVIDER_AUTHORITATIVE_LIMITS
                and spec.rpd is not None
                and runtime.daily_requests
                >= spec.rpd
            ):
                self._mark_validator_exhausted(
                    validator_model,
                    "RPD",
                )

                raise ProviderError(
                    (
                        f"{validator_model}: "
                        "daily request quota exhausted."
                    ),
                    temporary=False,
                    daily_exhausted=True,
                )

            # ----------------------------------------------------------
            # DAILY TOKEN LIMIT
            # ----------------------------------------------------------
            if (
                not PROVIDER_AUTHORITATIVE_LIMITS
                and spec.tpd
                and (
                    runtime.daily_tokens
                    + estimated_tokens
                    > spec.tpd
                )
            ):
                self._mark_validator_exhausted(
                    validator_model,
                    "TPD",
                )
                raise ProviderError(
                    (
                        f"{validator_model}: "
                        "daily token quota would be exceeded."
                    ),
                    temporary=False,
                    daily_exhausted=True,
                )

            now = time.time()

            # ----------------------------------------------------------
            # RPM
            # ----------------------------------------------------------
            if (
                runtime.rpm_used
                >= spec.rpm
            ):
                wait_for = max(
                    0.5,
                    60.0
                    - (
                        now
                        - runtime.requests_started[
                            0
                        ]
                    )
                    + 0.5,
                )

                print(
                    (
                        f"[QA WAIT] "
                        f"{validator_model}: "
                        f"RPM full; waiting "
                        f"{wait_for:.1f}s."
                    ),
                    flush=True,
                )

                self._interruptible_sleep(
                    wait_for
                )
                continue

            # ----------------------------------------------------------
            # TPM
            # ----------------------------------------------------------
            if (
                spec.tpm is not None
                and (
                    runtime.tpm_used
                    + estimated_tokens
                    > spec.tpm
                )
            ):
                if runtime.token_events:
                    wait_for = max(
                        0.5,
                        60.0
                        - (
                            now
                            - runtime.token_events[
                                0
                            ][0]
                        )
                        + 0.5,
                    )

                    print(
                        (
                            f"[QA WAIT] "
                            f"{validator_model}: "
                            f"TPM window full; "
                            f"waiting {wait_for:.1f}s."
                        ),
                        flush=True,
                    )

                    self._interruptible_sleep(
                        wait_for
                    )
                    continue

                raise ProviderError(
                    (
                        f"{validator_model}: "
                        "single validation batch is "
                        "larger than the configured TPM limit."
                    ),
                    temporary=False,
                )

            return

        raise GenerationStopped()

    def _record_validator_request(
        self,
        validator_model: str,
    ) -> None:
        runtime = self.validator_runtime[
            validator_model
        ]

        timestamp = time.time()

        runtime.requests_started.append(
            timestamp
        )

        runtime.daily_requests += 1

        with self.validator_state_lock:
            bucket = self._validator_bucket(
                validator_model
            )

            bucket[
                "daily_requests"
            ] = runtime.daily_requests

            bucket[
                "daily_tokens"
            ] = runtime.daily_tokens

            self.state.save()

    def _record_validator_tokens(
        self,
        validator_model: str,
        token_count: int,
    ) -> None:
        runtime = self.validator_runtime[
            validator_model
        ]

        token_count = max(
            0,
            int(
                token_count
                or 0
            ),
        )

        runtime.token_events.append(
            (
                time.time(),
                token_count,
            )
        )

        runtime.daily_tokens += (
            token_count
        )

        with self.validator_state_lock:
            bucket = self._validator_bucket(
                validator_model
            )

            bucket[
                "daily_requests"
            ] = runtime.daily_requests

            bucket[
                "daily_tokens"
            ] = runtime.daily_tokens

            self.state.save()

    def _fit_validator_batch_to_tpm(
        self,
        validator_model: str,
        items: List[ValidationItem],
    ) -> List[ValidationItem]:
        runtime = self.validator_runtime[
            validator_model
        ]

        spec = runtime.spec

        if spec.tpm is None:
            return items

        fitted = list(items)

        while (
            len(fitted) > 1
            and (
                self._validator_prompt_tokens(
                    fitted
                )
                > spec.tpm
            )
        ):
            item = fitted.pop()

            self.validator_queues[
                validator_model
            ].put(
                item
            )

        return fitted

    def _requeue_validator_items(
        self,
        items: List[ValidationItem],
        exclude_model: Optional[str] = None,
    ) -> None:
        if not items:
            return

        active_models = [
            model
            for model in self._active_validator_models()
            if model != exclude_model
        ]

        if not active_models:
            # No validator has remaining usable capacity.
            # Leave the work pending rather than assigning it to
            # an exhausted/unavailable validator.
            print(
                (
                    "[QA HOLD] No validator has remaining "
                    "capacity; validation work remains pending."
                ),
                flush=True,
            )
            return

        loads = {
            model: self.validator_queues[
                model
            ].unfinished_tasks
            for model in active_models
        }

        for item in items:
            validator_model = min(
                active_models,
                key=lambda model: (
                    loads[model],
                    self.validator_runtime[
                        model
                    ].rpd_ratio,
                    self.validator_runtime[
                        model
                    ].tpd_ratio,
                ),
            )

            self.validator_queues[
                validator_model
            ].put(
                item
            )

            loads[
                validator_model
            ] += 1

    def _reroute_validator_queue(
        self,
        validator_model: str,
        first_item: ValidationItem,
    ) -> None:
        validator_queue = (
            self.validator_queues[
                validator_model
            ]
        )

        items = [
            first_item
        ]

        while True:
            try:
                items.append(
                    validator_queue.get_nowait()
                )
            except queue.Empty:
                break

        # Every item removed from this queue is being transferred
        # elsewhere, so every removed item must be completed here.
        for _ in range(
            len(items)
        ):
            validator_queue.task_done()

        active_models = [
            model
            for model in self._active_validator_models()
            if model != validator_model
        ]

        if active_models:
            self._requeue_validator_items(
                items,
                exclude_model=validator_model,
            )
            return

        # No validator has capacity left.
        # Put the first item back and leave the remaining queue untouched.
        validator_queue.put(
            first_item
        )

    def get_validator_batch_size(
        self,
        validator_model: str,
    ) -> int:
        validator_queue = (
            self.validator_queues[
                validator_model
            ]
        )

        pending = (
            validator_queue.unfinished_tasks
        )

        max_batch = (
            self.validator_batch_size
        )

        if pending <= 0:
            return 1

        return min(
            max_batch,
            pending,
        )

    def wait_for_validator_pressure(
        self,
    ) -> bool:
        total_capacity = (
            VALIDATION_QUEUE_SIZE
            * VALIDATOR_WORKER_COUNT
        )

        high_watermark = int(
            total_capacity
            * VALIDATOR_QUEUE_HIGH_WATERMARK
        )

        resume_watermark = int(
            total_capacity
            * VALIDATOR_QUEUE_RESUME_WATERMARK
        )

        soft_watermark = int(
            total_capacity
            * VALIDATOR_QUEUE_SOFT_WATERMARK
        )

        while not self.stop_requested:
            self.drain_validation_results()

            pending = (
                self.total_validator_queue_size()
            )

            active_models = (
                self._active_validator_models()
            )

            if (
                pending > 0
                and not active_models
            ):
                print(
                    (
                        "\n[QA STOP] "
                        "All validator models have "
                        "exhausted/unavailable capacity. "
                        f"{pending} validation tasks remain pending."
                    ),
                    flush=True,
                )
                return False

            # ----------------------------------------------------------
            # HARD BACKPRESSURE
            # ----------------------------------------------------------
            if pending >= high_watermark:
                print(
                    (
                        f"[QA BACKPRESSURE] "
                        f"queue={pending}/"
                        f"{total_capacity}; "
                        "pausing generation until queue "
                        f"falls to {resume_watermark}."
                    ),
                    flush=True,
                )

                while (
                    not self.stop_requested
                ):
                    self.drain_validation_results()

                    pending = (
                        self.total_validator_queue_size()
                    )

                    active_models = (
                        self._active_validator_models()
                    )

                    if (
                        pending
                        <= resume_watermark
                    ):
                        break

                    if not active_models:
                        return False

                    self._interruptible_sleep(
                        VALIDATOR_PRESSURE_POLL_SECONDS
                    )

                if self.stop_requested:
                    return False

                continue

            # ----------------------------------------------------------
            # DAILY QUOTA PRESSURE
            # ----------------------------------------------------------
            expiring_models = []

            for validator_model in active_models:
                runtime = self.validator_runtime[
                    validator_model
                ]

                ratio = max(
                    runtime.rpd_ratio,
                    runtime.tpd_ratio,
                )

                if (
                    ratio
                    >= VALIDATOR_QUOTA_PRESSURE_RATIO
                ):
                    expiring_models.append(
                        validator_model
                    )

            healthy_models = [
                model
                for model in active_models
                if model
                not in expiring_models
            ]

            if (
                pending >= soft_watermark
                and expiring_models
                and not healthy_models
            ):
                print(
                    (
                        "[QA PRESSURE] Validator daily "
                        "capacity is near exhaustion; "
                        "slowing generation while backlog drains."
                    ),
                    flush=True,
                )

                self._interruptible_sleep(
                    VALIDATOR_PRESSURE_SLOWDOWN_SECONDS
                )
                continue

            if (
                pending > 0
                and expiring_models
            ):
                self._interruptible_sleep(
                    VALIDATOR_PRESSURE_SLOWDOWN_SECONDS
                )

            return (
                not self.stop_requested
            )

        return False

    def wait_for_validators(
        self,
    ) -> None:
        while True:
            self.drain_validation_results()

            all_done = all(
                validator_queue.unfinished_tasks
                == 0
                for validator_queue
                in self.validator_queues.values()
            )

            if all_done:
                break

            if (
                not self._active_validator_models()
            ):
                pending = (
                    self.total_validator_queue_size()
                )

                if pending > 0:
                    print(
                        (
                            "\n[QA BLOCKED] "
                            f"{pending} validation tasks "
                            "could not be executed because "
                            "all validator capacity is exhausted "
                            "or unavailable. Pending batches remain "
                            "on disk."
                        ),
                        flush=True,
                    )
                break

            self._interruptible_sleep(
                VALIDATOR_PRESSURE_POLL_SECONDS
            )

        self.drain_validation_results()

    def _interruptible_sleep(
        self,
        seconds: float,
    ) -> None:

        # ------------------------------------------------------------------
        # Plain time.sleep() cannot be relied on to stop early when a
        # custom SIGINT handler is installed (PEP 475 auto-restarts
        # interrupted sleeps in Python 3.5+), so a single long sleep can
        # keep the process "stuck" for minutes after Ctrl+C is pressed.
        # Sleeping in short ticks and rechecking stop_requested each time
        # is what actually makes Ctrl+C feel responsive.
        # ------------------------------------------------------------------

        remaining = max(
            0.0,
            float(
                seconds
                or 0.0
            ),
        )

        tick = 0.5

        while (
            remaining > 0
            and not self.stop_requested
        ):

            time.sleep(
                min(
                    tick,
                    remaining,
                )
            )

            remaining -= tick

    def _install_signal_handlers(
        self,
    ) -> None:

        def handler(
            signum,
            frame,
        ):

            self.stop_requested = True

            print(
                (
                    "\n[STOP] Stop requested. "
                    "The current generation step will "
                    "finish and queued validation will "
                    "then be drained safely."
                ),
                flush=True,
            )

        try:

            signal.signal(
                signal.SIGINT,
                handler,
            )

            signal.signal(
                signal.SIGTERM,
                handler,
            )

        except Exception:

            pass

    def _hydrate_runtime_from_state(
        self,
    ) -> None:

        for spec in MODEL_HIERARCHY:

            bucket = (
                self.state.model_bucket(
                    spec
                )
            )

            runtime = (
                self.runtime[
                    spec.key
                ]
            )

            runtime.daily_requests = int(
                bucket.get(
                    "daily_requests",
                    0,
                )
            )

            runtime.daily_tokens = int(
                bucket.get(
                    "daily_tokens",
                    0,
                )
            )

            runtime.question_count = int(
                bucket.get(
                    "question_count",
                    0,
                )
            )

            runtime.rejected_count = int(
                bucket.get(
                    "rejected_count",
                    0,
                )
            )

            runtime.duplicate_count = int(
                bucket.get(
                    "duplicate_count",
                    0,
                )
            )

            runtime.status = str(
                bucket.get(
                    "status"
                )
                or "ACTIVE"
            )

            runtime.exhausted_reason = (
                bucket.get(
                    "exhausted_reason"
                )
            )

            if (
                runtime.daily_requests
                >= spec.rpd
            ):

                runtime.status = (
                    "DAILY_EXHAUSTED"
                )

            if (
                spec.tpd
                and runtime.daily_tokens
                >= spec.tpd
            ):

                runtime.status = (
                    "DAILY_EXHAUSTED"
                )

    def print_header(
        self,
        mode: str,
    ) -> None:

        print(
            "=" * 84
        )

        print(
            "ELEVATE PRACTICE QUESTION BANK GENERATOR"
        )

        print(
            f"MODE: {mode.upper()}"
        )

        print(
            "Database: PostgreSQL / Supabase"
        )

        print(
            "Strategy: strict sequential model hierarchy"
        )

        print(
            "Validators: "
            + ", ".join(
                VALIDATOR_MODELS
            )
        )

        print(
            f"Validator workers: "
            f"{VALIDATOR_WORKER_COUNT}"
        )

        print(
            f"Validator batch distribution: "
            f"balanced across workers"
        )

        print(
            f"Validator batch size: "
            f"{self.validator_batch_size}"
        )

        print(
            "=" * 84
        )

        for spec in MODEL_HIERARCHY:

            print(
                f"{spec.rank:02d}. "
                f"{spec.provider:6s} | "
                f"{spec.model:34s} | "
                f"RPM={spec.rpm or 'provider'} "
                f"RPD={spec.rpd or 'provider'} "
                f"TPM={spec.tpm or 'provider'} "
                f"TPD={spec.tpd or 'provider'} | "
                f"{spec.quality_tier}"
            )

        print(
            "=" * 84
        )

        if PROVIDER_AUTHORITATIVE_LIMITS:
            print(
                (
                    "Quota mode: PROVIDER-AUTHORITATIVE. "
                    "Gemini API responses determine actual exhaustion; "
                    "local model limits are advisory/fallback values."
                )
            )

        if not RATE_LIMIT_OVERRIDE_FILE.exists():

            print(
                "Rate limits above are this script's built-in defaults. "
                "Provider free-tier limits change and vary per account -- "
                "confirm yours in Google AI Studio / the Groq console, and "
                f"if they differ, drop overrides in "
                f"{RATE_LIMIT_OVERRIDE_FILE} "
                "(see load_rate_limit_overrides() docstring for the format)."
            )

            print(
                "=" * 84
            )

    def next_combination(
        self,
    ) -> Dict[str, str]:

        candidates = (
            build_all_combinations(
                self.syllabus
            )
        )

        candidates.sort(
            key=lambda target: (
                self.combination_counts.get(
                    (
                        target["grade"],
                        target["subject"],
                        target["syllabus_topic"],
                        target["difficulty"],
                    ),
                    0,
                ),
                target["grade"],
                target["subject"],
                target["syllabus_topic"],
                target["difficulty"],
            )
        )

        return candidates[0]

    def _wait_for_local_limits(
        self,
        runtime: ModelRuntime,
        estimated_tokens: int,
    ) -> None:

        spec = runtime.spec

        while not self.stop_requested:

            runtime.trim_windows()

            now = time.time()

            if (
                runtime.rpm_used
                >= spec.rpm
            ):

                wait_for = max(
                    0.5,
                    60.0
                    - (
                        now
                        - runtime.requests_started[
                            0
                        ]
                    )
                    + 0.5,
                )

                print(
                    f"[WAIT] {spec.model}: "
                    "RPM window full; "
                    f"waiting {wait_for:.1f}s",
                    flush=True,
                )

                self._interruptible_sleep(
                    min(
                        wait_for,
                        90.0,
                    )
                )

                continue

            if (
                spec.tpm is not None
                and (
                    runtime.tpm_used
                    + estimated_tokens
                )
                > spec.tpm
            ):

                if runtime.token_events:

                    wait_for = max(
                        0.5,
                        60.0
                        - (
                            now
                            - runtime.token_events[
                                0
                            ][0]
                        )
                        + 0.5,
                    )

                    print(
                        f"[WAIT] {spec.model}: "
                        "TPM window near limit; "
                        f"waiting {wait_for:.1f}s",
                        flush=True,
                    )

                    self._interruptible_sleep(
                        min(
                            wait_for,
                            90.0,
                        )
                    )

                    continue

                raise ProviderError(
                    (
                        f"Configured batch is too large "
                        f"for {spec.model}'s TPM limit "
                        f"({spec.tpm})."
                    ),
                    temporary=False,
                )

            return

        # Loop exited because a stop was requested while waiting on a
        # local RPM/TPM window. Don't silently proceed to generate one
        # more batch after the user asked to stop.
        raise GenerationStopped()

    def _check_daily_limits(
        self,
        runtime: ModelRuntime,
    ) -> bool:
        """
        In provider-authoritative mode, local counters are informational only.

        A Gemini generation model is considered daily-exhausted only after
        Gemini itself reports that the daily quota has been exceeded.

        Local counters are still used for reporting/telemetry.
        """
        if (
            runtime.status
            in (
                "DAILY_EXHAUSTED",
                "UNAVAILABLE",
            )
        ):
            return False

        if PROVIDER_AUTHORITATIVE_LIMITS:
            return True

        spec = runtime.spec

        if (
            spec.rpd is not None
            and runtime.daily_requests
            >= spec.rpd
        ):
            runtime.status = (
                "DAILY_EXHAUSTED"
            )

            runtime.exhausted_reason = (
                "RPD"
            )

            self.state.mark_exhausted(
                spec,
                "RPD",
            )

            return False

        if (
            spec.tpd
            and runtime.daily_tokens
            >= spec.tpd
        ):
            runtime.status = (
                "DAILY_EXHAUSTED"
            )

            runtime.exhausted_reason = (
                "TPD"
            )

            self.state.mark_exhausted(
                spec,
                "TPD",
            )

            return False

        return True

    def _emit_quota_alerts(
        self,
        runtime: ModelRuntime,
    ) -> None:

        spec = runtime.spec

        ratios = [
            runtime.rpd_ratio
        ]

        if spec.tpd:

            ratios.append(
                runtime.tpd_ratio
            )

        ratio = max(
            ratios
        )

        crossed = max(
            (
                level
                for level
                in WARNING_LEVELS
                if ratio >= level
            ),
            default=0.0,
        )

        if (
            crossed
            and crossed
            > runtime.last_warning_level
        ):

            runtime.last_warning_level = (
                crossed
            )

            print(
                f"[ALERT] {spec.model}: "
                f"{crossed * 100:.0f}%+ daily quota consumed "
                f"(requests={runtime.daily_requests}/"
                f"{spec.rpd}, "
                f"tokens={runtime.daily_tokens}/"
                f"{spec.tpd or 'n/a'})."
            )

    def _estimated_tokens_for_batch(
        self,
        prompt: str,
        count: int,
    ) -> int:

        input_estimate = max(
            1,
            len(prompt)
            // 4,
        )

        output_estimate = max(
            600,
            count * 360,
        )

        return (
            input_estimate
            + output_estimate
        )

    def _call_model(
        self,
        runtime: ModelRuntime,
        target: Dict[str, str],
        count: int,
    ) -> Tuple[
        List[
            Dict[str, Any]
        ],
        int,
    ]:

        spec = runtime.spec

        prompt = build_prompt(
            target,
            count,
        )

        estimated_tokens = (
            self._estimated_tokens_for_batch(
                prompt,
                count,
            )
        )

        self._wait_for_local_limits(
            runtime,
            estimated_tokens,
        )

        runtime.requests_started.append(
            time.time()
        )

        runtime.request_count += 1

        generation_start = (
            time.time()
        )

        if (
            spec.provider
            == "gemini"
        ):

            raw, actual_tokens = (
                call_gemini(
                    spec,
                    prompt,
                    count,
                )
            )

        elif (
            spec.provider
            == "groq"
        ):

            raw, actual_tokens = (
                call_groq(
                    spec,
                    prompt,
                    count,
                )
            )

        else:

            raise ProviderError(
                (
                    "Unsupported provider: "
                    f"{spec.provider}"
                ),
                temporary=False,
            )

        elapsed = (
            time.time()
            - generation_start
        )

        actual_tokens = max(
            actual_tokens,
            0,
        )

        runtime.token_events.append(
            (
                time.time(),
                actual_tokens,
            )
        )

        runtime.daily_requests += 1
        runtime.daily_tokens += (
            actual_tokens
        )

        self.state.increment_usage(
            spec,
            1,
            actual_tokens,
        )

        self._emit_quota_alerts(
            runtime
        )

        questions = parse_questions(
            raw
        )
        
        runtime.question_count += len(
            questions
        )

        model_bucket = (
            self.state.model_bucket(
                spec
            )
        )

        model_bucket[
            "question_count"
        ] = runtime.question_count

        print(
            f"[OK] {spec.model} generated "
            f"{len(questions)} raw questions "
            f"in {elapsed:.1f}s "
            f"({actual_tokens:,} tokens).",
            flush=True,
        )

        return (
            questions,
            actual_tokens,
        )

    def _generate_with_retry(
        self,
        runtime: ModelRuntime,
        target: Dict[str, str],
        count: int,
    ) -> List[
        Dict[str, Any]
    ]:

        current_count = count

        last_error: Optional[
            Exception
        ] = None

        for attempt in range(
            1,
            GENERATION_PARSE_RETRY_LIMIT
            + 1,
        ):

            if self.stop_requested:

                raise GenerationStopped()

            try:

                raw_questions, _ = (
                    self._call_model(
                        runtime,
                        target,
                        current_count,
                    )
                )

                if len(
                    raw_questions
                ) != current_count:

                    raise ValueError(
                        (
                            f"Model returned "
                            f"{len(raw_questions)} "
                            "questions instead of "
                            f"{current_count}."
                        )
                    )

                return raw_questions

            except (
                json.JSONDecodeError,
                ValueError,
            ) as exc:

                last_error = exc

                if (
                    attempt < GENERATION_PARSE_RETRY_LIMIT
                    and not self.stop_requested
                ):

                    print(
                        (
                            f"[RETRY] "
                            f"{runtime.spec.model}: "
                            f"{exc} | "
                            "retrying SAME model "
                            f"with batch="
                            f"{current_count}"
                        ),
                        flush=True,
                    )

                    self._interruptible_sleep(
                        2.0
                    )

                    continue

                break

        if last_error:

            raise last_error

        raise ValueError(
            "Generation retry failed."
        )

    def validate_and_prepare(
        self,
        raw_questions: Iterable[
            Dict[str, Any]
        ],
        target: Dict[str, str],
        spec: ModelSpec,
    ) -> List[
        Dict[str, Any]
    ]:

        accepted: List[
            Dict[str, Any]
        ] = []

        runtime = (
            self.runtime[
                spec.key
            ]
        )

        seen_batch: set[
            str
        ] = set()

        for item in raw_questions:

            row, reason = (
                validate_question(
                    item,
                    target,
                )
            )

            if row is None:

                runtime.rejected_count += 1

                self.state.data[
                    "totals"
                ][
                    "rejected"
                ] += 1

                print(
                    f"[REJECT] {spec.model}: "
                    f"{reason}",
                    flush=True,
                )

                continue

            fp = row[
                "question_fingerprint"
            ]

            if fp in seen_batch:

                runtime.duplicate_count += 1

                self.state.data[
                    "totals"
                ][
                    "duplicates"
                ] += 1

                continue

            seen_batch.add(
                fp
            )

            row[
                "generation_model"
            ] = spec.model

            row[
                "generation_batch_id"
            ] = str(
                uuid.uuid4()
            )

            row[
                "generation_meta"
            ] = {
                "generator": (
                    "elevate.seed_questions"
                ),

                "provider": (
                    spec.provider
                ),

                "model": (
                    spec.model
                ),

                "quality_tier": (
                    spec.quality_tier
                ),

                "generated_at": (
                    now_iso()
                ),

                "mode": (
                    "demo"
                    if self.mode
                    == "demo"
                    else "production"
                ),
            }

            accepted.append(
                row
            )

        existing_fingerprints = (
            load_existing_fingerprints_for_batch(
                self.engine,
                [
                    row[
                        "question_fingerprint"
                    ]
                    for row in accepted
                ],
            )
        )

        if existing_fingerprints:
        
            filtered: List[
                Dict[str, Any]
            ] = []

            for row in accepted:
            
                fp = row[
                    "question_fingerprint"
                ]

                if fp in existing_fingerprints:
                
                    runtime.duplicate_count += 1

                    self.state.data[
                        "totals"
                    ][
                        "duplicates"
                    ] += 1

                    continue
                
                filtered.append(
                    row
                )

            accepted = filtered
        
        return accepted

    def queue_for_validation(
        self,
        rows: List[
            Dict[str, Any]
        ],
        target: Dict[str, str],
        source_model: str,
    ) -> None:

        if not rows:
            return

        assignments: Dict[
            str,
            List[
                ValidationItem
            ]
        ] = {
            model: []
            for model in VALIDATOR_MODELS
        }

        # --------------------------------------------------------------
        # Capacity-aware validator distribution.
        #
        # Never intentionally assign new work to a validator that has
        # already exhausted its daily quota.
        # --------------------------------------------------------------
        active_models = (
            self._active_validator_models()
        )

        if active_models:
            assignment_models = (
                active_models
            )
        else:
            print(
                (
                    "[QA HOLD] "
                    "No validator currently has confirmed usable "
                    "capacity; validation work remains unassigned."
                ),
                flush=True,
            )
            return

            print(
                (
                    "[QA HOLD] "
                    "No validator currently has usable daily "
                    "capacity; keeping new validation work pending."
                ),
                flush=True,
            )

        loads = {
            model: self.validator_queues[
                model
            ].unfinished_tasks
            for model in assignment_models
        }

        assignment_cursor = 0

        for row in rows:
            min_load = min(
                loads.values()
            )

            tied_models = [
                model
                for model in assignment_models
                if loads[model] == min_load
            ]

            validator_model = tied_models[
                assignment_cursor
                % len(tied_models)
            ]

            assignment_cursor += 1

            assignments[
                validator_model
            ].append(
                ValidationItem(
                    source_model=source_model,
                    target=target,
                    row=row,
                )
            )

            loads[
                validator_model
            ] += 1

        total_queued = 0

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # Put individual ValidationItem objects into the queues.
        #
        # The validator worker decides the actual batch size dynamically
        # when it starts processing.
        # --------------------------------------------------------------
        for (
            validator_model,
            items
        ) in assignments.items():

            if not items:
                continue

            validator_queue = (
                self.validator_queues[
                    validator_model
                ]
            )

            for item in items:
                validator_queue.put(
                    item
                )

            total_queued += len(
                items
            )

            print(
                (
                    f"[QA QUEUE] "
                    f"{validator_model}: "
                    f"+{len(items)} questions | "
                    f"queue="
                    f"{validator_queue.qsize()}"
                ),
                flush=True,
            )

        print(
            (
                f"[QA QUEUED] "
                f"total={total_queued} "
                f"from={source_model} | "
                f"total_pending="
                f"{self.total_validator_queue_size()}"
            ),
            flush=True,
        )

    def total_validator_queue_size(
        self,
    ) -> int:

        return sum(
            validator_queue.unfinished_tasks
            for validator_queue
            in self.validator_queues.values()
        )

    def drain_validation_results(
        self,
    ) -> None:

        while True:

            try:
                result = (
                    self.validation_results.get_nowait()
                )
                
            except queue.Empty:

                break
                
            validator_model = (
                result.validator_model
            )

            validator_stats = (
                self.state.data
                .setdefault(
                    "validators",
                    {}
                )
                .setdefault(
                    validator_model,
                    {
                        "batches": 0,
                        "validated_questions": 0,
                        "approved": 0,
                        "repaired": 0,
                        "rejected": 0,
                        "errors": 0,
                    },
                )
            )
            
            validator_stats[
                "batches"
            ] += 1
            
            validator_stats[
                "validated_questions"
            ] += len(
                result.items
            )
            
            validator_stats[
                "approved"
            ] += len(
                result.approved
            )
            
            validator_stats[
                "repaired"
            ] += int(
                result.repaired_count
            )
            
            validator_stats[
                "rejected"
            ] += int(
                result.rejected
            )
            
            if result.error:
                validator_stats[
                    "errors"
                ] += 1


            try:

                if result.error:

                    print(
                        (
                            f"[QA FAILED] "
                            f"batch={result.batch_id}: "
                            f"{result.error}"
                        ),
                        flush=True,
                    )

                    # Pending batch remains on disk.
                    continue

                inserted = (
                    insert_questions(
                        self.engine,
                        result.approved,
                    )
                )

                for row in result.approved:


                    key = (
                        row["grade"],
                        row["subject"],
                        row[
                            "syllabus_topic"
                        ],
                        row["difficulty"],
                    )

                    self.combination_counts[
                        key
                    ] = (
                        self.combination_counts.get(
                            key,
                            0,
                        )
                        + 1
                    )

                self.state.data[
                    "totals"
                ][
                    "generated"
                ] += len(
                    result.items
                )

                self.state.data[
                    "totals"
                ][
                    "inserted"
                ] += inserted

                self.state.data[
                    "totals"
                ][
                    "rejected"
                ] += result.rejected

                self.state.save()

                print(
                    (
                        f"[QA INSERT] "
                        f"batch={result.batch_id}: "
                        f"approved="
                        f"{len(result.approved)} "
                        f"inserted={inserted} "
                        f"rejected="
                        f"{result.rejected}"
                    ),
                    flush=True,
                )

                delete_pending_validation(
                    result.batch_id
                )

            finally:

                self.validation_results.task_done()
                
    def print_sample(self, rows: List[Dict[str, Any]]) -> None:
        print("\n" + "-" * 84)
        for idx, row in enumerate(rows[:3], start=1):
            print(f"SAMPLE {idx} | {row['grade']} / {row['subject']} / {row['syllabus_topic']} / {row['difficulty']}")
            print(f"Q: {row['question_text']}")
            labels = ["A", "B", "C", "D"]
            for option_idx, option in enumerate(row["options"]):
                marker = " <-- CORRECT" if option_idx == row["correct_index"] else ""
                print(f"  {labels[option_idx]}) {option}{marker}")
            print(f"Explanation: {row['explanation']}")
            print("-" * 84)

    # =========================================================================
    # DEMO
    # =========================================================================

    def run_demo(
        self,
        questions_per_model: int = (
            DEFAULT_DEMO_QUESTIONS_PER_MODEL
        ),
    ) -> None:

        self.mode = "demo"

        self.validator_batch_size = (
            VALIDATOR_BATCH_DEMO
        )

        self.print_header(
            "demo"
        )

        demo_targets = (
            build_demo_combinations(
                self.syllabus,
                len(
                    MODEL_HIERARCHY
                ),
            )
        )

        print(
            (
                f"Demo budget: "
                f"{questions_per_model} "
                "questions per model; "
                f"{len(MODEL_HIERARCHY)} models "
                f"=> up to "
                f"{questions_per_model * len(MODEL_HIERARCHY)} "
                "raw questions."
            ),
            flush=True,
        )

        print(
            (
                "Generation remains strictly sequential. "
                f"Questions are distributed across "
                f"{VALIDATOR_WORKER_COUNT} parallel validators."
            ),
            flush=True,
        )

        print()

        # --------------------------------------------------------------
        # STRICT MODEL ORDER: 1 -> 2 -> ... -> 8
        # --------------------------------------------------------------

        for index, spec in enumerate(
            MODEL_HIERARCHY
        ):

            if self.stop_requested:

                break

            runtime = (
                self.runtime[
                    spec.key
                ]
            )

            if not self._check_daily_limits(
                runtime
            ):

                print(
                    (
                        f"[SKIP] {spec.model}: "
                        "already exhausted "
                        "according to persisted state."
                    ),
                    flush=True,
                )

                continue

            target = demo_targets[
                index
            ]

            print(
                (
                    f"\n[MODEL {index + 1}/"
                    f"{len(MODEL_HIERARCHY)}] "
                    f"{spec.model}\n"
                    f"Target: "
                    f"{target['grade']} | "
                    f"{target['subject']} | "
                    f"{target['syllabus_topic']} | "
                    f"{target['difficulty']}"
                ),
                flush=True,
            )

            try:

                raw_questions = (
                    self._generate_with_retry(
                        runtime,
                        target,
                        questions_per_model,
                    )
                )

                prepared = (
                    self.validate_and_prepare(
                        raw_questions,
                        target,
                        spec,
                    )
                )

                self.queue_for_validation(
                    prepared,
                    target,
                    spec.model,
                )
                
                # Show the generated questions immediately, exactly as before.
                self.print_sample(prepared)

                # Validator continues independently.
                self.drain_validation_results()

                print(
                    (
                        f"[DEMO QUEUED] "
                        f"raw={len(raw_questions)} "
                        f"prepared={len(prepared)} "
                        f"queue="
                        f"{self.total_validator_queue_size()}"
                    ),
                    flush=True,
                )

                print(
                    (
                        f"rejected="
                        f"{runtime.rejected_count} "
                        f"duplicates="
                        f"{runtime.duplicate_count}"
                    ),
                    flush=True,
                )

                self.state.save()

            except GenerationStopped:

                # Stop was requested while waiting on a local
                # rate-limit window; nothing was generated this step.
                pass

            except ProviderError as exc:

                print(
                    (
                        f"[MODEL ERROR] "
                        f"{friendly_provider_message(exc, spec.model)}"
                    ),
                    flush=True,
                )

                if exc.daily_exhausted:

                    runtime.status = (
                        "DAILY_EXHAUSTED"
                    )

                    runtime.exhausted_reason = (
                        "provider reported "
                        "daily quota exhaustion"
                    )

                    self.state.mark_exhausted(
                        spec,
                        runtime.exhausted_reason,
                    )

                elif not exc.temporary:

                    runtime.status = (
                        "UNAVAILABLE"
                    )

                else:

                    print(
                        (
                            "[DEMO] Temporary error; "
                            "moving on so the demo can "
                            "still sample the remaining models."
                        ),
                        flush=True,
                    )

            except (
                json.JSONDecodeError,
                ValueError,
            ) as exc:

                runtime.rejected_count += (
                    questions_per_model
                )

                print(
                    (
                        f"[PARSE ERROR] "
                        f"{spec.model}: "
                        f"{' '.join(str(exc).split())[:180]}"
                    ),
                    flush=True,
                )

            except Exception as exc:

                print(
                    (
                        f"[UNEXPECTED ERROR] "
                        f"{spec.model}: "
                        f"{' '.join(str(exc).split())[:180]}"
                    ),
                    flush=True,
                )

        # --------------------------------------------------------------
        # NO MORE GENERATION
        # --------------------------------------------------------------

        self.generation_finished = True

        print(
            (
                "\n[GENERATION DONE] "
                "All demo generation models have finished."
            ),
            flush=True,
        )

        print(
            (
                "[QA WAIT] "
                "Validator will continue consuming "
                "the remaining queue."
            ),
            flush=True,
        )

        self.wait_for_validators()

        self.drain_validation_results()

        print(
            (
                "[QA DONE] "
                "Demo validation queue is empty."
            ),
            flush=True,
        )

        self._shutdown_validator()

        self.print_summary(
            "demo"
        )

    # =========================================================================
    # PRODUCTION
    # =========================================================================

    def run_production(
        self,
    ) -> None:

        self.mode = "production"

        self.validator_batch_size = (
            VALIDATOR_BATCH_PRODUCTION
        )

        self.print_header(
            "production"
        )

        print(
            (
                "Production mode: one model is used "
                "continuously until its daily quota "
                "is exhausted."
            ),
            flush=True,
        )

        print(
            (
                "Generation remains strictly sequential."
            ),
            flush=True,
        )

        print(
            (
                "Questions are distributed across "
                f"{VALIDATOR_WORKER_COUNT} parallel validators."
            ),
            flush=True,
        )

        print(
            (
                "Press Ctrl+C to stop safely."
            ),
            flush=True,
        )

        print()

        try:

            # ----------------------------------------------------------
            # ORIGINAL STRICT SEQUENTIAL GENERATION LOOP
            # ----------------------------------------------------------

            while not self.stop_requested:

                # ------------------------------------------------------
                # SMART VALIDATOR BACKPRESSURE
                #
                # Generation remains sequential, but it may pause so
                # validators can catch up or so expiring validator
                # quotas are not wasted while backlog is high.
                # ------------------------------------------------------
                if not self.wait_for_validator_pressure():
                    break

                active_runtime: Optional[
                    ModelRuntime
                ] = None

                for spec in MODEL_HIERARCHY:

                    runtime = (
                        self.runtime[
                            spec.key
                        ]
                    )

                    if (
                        runtime.status
                        == "ACTIVE"
                        and self._check_daily_limits(
                            runtime
                        )
                    ):

                        active_runtime = (
                            runtime
                        )

                        break

                if active_runtime is None:

                    print(
                        (
                            "\n[STOP] ALL CONFIGURED "
                            "MODELS ARE EXHAUSTED "
                            "OR UNAVAILABLE."
                        ),
                        flush=True,
                    )

                    break

                target = (
                    self.next_combination()
                )

                batch_size = (
                    active_runtime
                    .spec
                    .batch_size_production
                )

                if (
                    active_runtime.spec.tpm
                    and active_runtime.spec.tpm
                    < 15_000
                ):

                    batch_size = min(
                        batch_size,
                        6,
                    )

                elif (
                    active_runtime.spec.tpm
                    and active_runtime.spec.tpm
                    < 80_000
                ):

                    batch_size = min(
                        batch_size,
                        12,
                    )

                print(
                    (
                        f"\n[GENERATE] "
                        f"{active_runtime.spec.model}\n"
                        f"Target: "
                        f"{target['grade']} | "
                        f"{target['subject']} | "
                        f"{target['syllabus_topic']} | "
                        f"{target['difficulty']}\n"
                        f"Batch size: {batch_size}"
                    ),
                    flush=True,
                )

                try:

                    raw_questions = (
                        self._generate_with_retry(
                            active_runtime,
                            target,
                            batch_size,
                        )
                    )

                    prepared = (
                        self.validate_and_prepare(
                            raw_questions,
                            target,
                            active_runtime.spec,
                        )
                    )

                    self.queue_for_validation(
                        prepared,
                        target,
                        active_runtime.spec.model,
                    )

                    print(
                        (
                            f"[QUEUED FOR QA] "
                            f"{active_runtime.spec.model}: "
                            f"generated="
                            f"{len(raw_questions)} "
                            f"prepared="
                            f"{len(prepared)} "
                            f"queue="
                            f"{self.total_validator_queue_size()}/"
                            f"{VALIDATOR_WORKER_COUNT}"
                        ),
                        flush=True,
                    )

                    # Validator continues independently.
                    self.drain_validation_results()

                    self.state.save()

                except GenerationStopped:

                    # Stop was requested while waiting on a local
                    # rate-limit window; nothing was generated this step.
                    pass

                except ProviderError as exc:

                    message = str(
                        exc
                    )

                    print(
                        (
                            f"[PROVIDER] "
                            f"{friendly_provider_message(exc, active_runtime.spec.model)}"
                        ),
                        flush=True,
                    )

                    if exc.daily_exhausted:

                        active_runtime.status = (
                            "DAILY_EXHAUSTED"
                        )

                        active_runtime.exhausted_reason = (
                            "provider reported "
                            "daily quota exhaustion"
                        )

                        self.state.mark_exhausted(
                            active_runtime.spec,
                            active_runtime.exhausted_reason,
                        )

                        print(
                            (
                                f"[ADVANCE] Daily quota "
                                f"exhausted for "
                                f"{active_runtime.spec.model}. "
                                "Moving to next model."
                            ),
                            flush=True,
                        )

                        continue

                    if not exc.temporary:

                        active_runtime.status = (
                            "UNAVAILABLE"
                        )

                        active_runtime.exhausted_reason = (
                            message
                        )

                        bucket = (
                            self.state.model_bucket(
                                active_runtime.spec
                            )
                        )

                        bucket[
                            "status"
                        ] = "UNAVAILABLE"

                        bucket[
                            "exhausted_reason"
                        ] = message

                        self.state.save()

                        print(
                            (
                                f"[ADVANCE] "
                                f"{active_runtime.spec.model} "
                                "is unavailable/configuration-invalid. "
                                "Moving to next model."
                            ),
                            flush=True,
                        )

                        continue

                    delay = (
                        exc.retry_after
                        or 5.0
                    )

                    print(
                        (
                            f"[WAIT] Retrying SAME "
                            "model in "
                            f"{delay:.1f}s."
                        ),
                        flush=True,
                    )

                    self._interruptible_sleep(
                        min(
                            max(
                                delay,
                                1.0,
                            ),
                            300.0,
                        )
                    )

                except GenerationStopped:

                    # Stop was requested while this attempt was waiting
                    # on a local rate-limit window or retry backoff.
                    # Nothing was generated for this step, so there is
                    # nothing to save; fall through and let the outer
                    # while-loop exit on the next check.
                    pass

                except (
                    json.JSONDecodeError,
                    ValueError,
                ) as exc:

                    active_runtime.rejected_count += (
                        batch_size
                    )

                    print(
                        (
                            f"[PARSE] "
                            f"{active_runtime.spec.model}: "
                            f"{' '.join(str(exc).split())[:180]}"
                        ),
                        flush=True,
                    )

                    if not self.stop_requested:

                        print(
                            (
                                "[RETRY] Retrying SAME "
                                "model."
                            ),
                            flush=True,
                        )

                    self._interruptible_sleep(
                        2.0
                    )

                except Exception as exc:

                    print(
                        (
                            f"[ERROR] "
                            f"{active_runtime.spec.model}: "
                            f"{' '.join(str(exc).split())[:180]}"
                        ),
                        flush=True,
                    )

                    if not self.stop_requested:

                        print(
                            (
                                "[WAIT] Retrying the SAME "
                                "model in 5s. "
                                "Database/API failures do "
                                "not silently switch models."
                            ),
                            flush=True,
                        )

                    self._interruptible_sleep(
                        5.0
                    )

        except KeyboardInterrupt:

            # ----------------------------------------------------------
            # Graceful Ctrl+C
            # ----------------------------------------------------------

            self.stop_requested = True

            print(
                (
                    "\n[STOP] Ctrl+C received. "
                    "Stopping generation gracefully."
                ),
                flush=True,
            )

        finally:

            # ----------------------------------------------------------
            # IMPORTANT:
            # Generation has ended, but validator must be allowed to
            # finish every queued question.
            # ----------------------------------------------------------

            self.generation_finished = True

            print(
                (
                    "\n[GENERATION DONE] "
                    "No more generation jobs will be added."
                ),
                flush=True,
            )

            print(
                (
                    "[QA WAIT] "
                    "Draining remaining validation queue..."
                ),
                flush=True,
            )

            self.wait_for_validators()

            self.drain_validation_results()

            print(
                (
                    "[QA DONE] "
                    "All queued validation jobs "
                    "have finished."
                ),
                flush=True,
            )

            self._shutdown_validator()

            self.state.save()

            self.print_summary(
                "production"
            )

    # =========================================================================
    # VALIDATOR SHUTDOWN
    # =========================================================================

    def _shutdown_validator(
        self,
    ) -> None:

        #
        # Generation and validation queues have already been
        # drained before shutdown.
        #
        # Each validator exits naturally when:
        #   generation_finished == True
        #   and its own queue is empty.
        #

        for thread in self.validator_threads:

            if thread.is_alive():

                thread.join(
                    timeout=30
                )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    def print_summary(
        self,
        mode: str,
    ) -> None:

        try:

            inserted_counts = (
                load_inserted_counts_by_model(
                    self.engine
                )
            )

            inventory = (
                load_database_inventory(
                    self.engine,
                    self.syllabus,
                )
            )

        except Exception as exc:

            print(
                (
                    "[SUMMARY WARN] Could not "
                    "query final database state: "
                    f"{exc}"
                ),
                flush=True,
            )

            return

        database_total = int(
            inventory[
                "total_questions"
            ]
        )

        represented_total = int(
            inventory[
                "represented_total"
            ]
        )

        unclassified = int(
            inventory[
                "unclassified_questions"
            ]
        )

        # --------------------------------------------------------------
        # Persist an authoritative DB snapshot.
        # --------------------------------------------------------------

        self.state.data[
            "database_snapshot"
        ] = {
            **inventory,

            "generation_mode":
                mode,

            "by_model":
                inserted_counts,

            "validators":
                self.validator_stats,
        }

        self.state.data[
            "totals"
        ][
            "database_total"
        ] = database_total

        self.state.data[
            "totals"
        ][
            "database_represented_total"
        ] = represented_total

        self.state.data[
            "totals"
        ][
            "database_unclassified"
        ] = unclassified

        self.state.save()

        # --------------------------------------------------------------
        # Overall summary.
        # --------------------------------------------------------------

        totals = (
            self.state.data.get(
                "totals",
                {},
            )
        )

        print(
            "\n"
            + "=" * 120
        )

        print(
            f"{mode.upper()} SUMMARY"
        )

        print(
            "=" * 120
        )

        print(
            f"Requests:              "
            f"{int(totals.get('requests', 0))}"
        )

        print(
            f"Generated:             "
            f"{int(totals.get('generated', 0))}"
        )

        print(
            f"Tracked Inserted:      "
            f"{int(totals.get('inserted', 0))}"
        )

        print(
            f"Rejected:              "
            f"{int(totals.get('rejected', 0))}"
        )

        print(
            f"Duplicates:            "
            f"{int(totals.get('duplicates', 0))}"
        )

        print(
            f"DB Grand Total:        "
            f"{database_total}"
        )

        print(
            f"Curriculum Total:      "
            f"{represented_total}"
        )

        print(
            f"Unclassified DB rows:  "
            f"{unclassified}"
        )

        # --------------------------------------------------------------
        # Generation model database counts.
        # --------------------------------------------------------------

        print(
            "\nGENERATION MODEL DB COUNTS"
        )

        print(
            "-" * 120
        )

        for spec in MODEL_HIERARCHY:

            runtime = (
                self.runtime[
                    spec.key
                ]
            )

            actual_inserted = (
                inserted_counts.get(
                    spec.model,
                    0,
                )
            )

            print(
                (
                    f"{spec.rank:02d}. "
                    f"{spec.model:34s} | "
                    f"{runtime.status:16s} | "
                    f"RPD "
                    f"{runtime.daily_requests}/"
                    f"{spec.rpd} | "
                    f"TPD "
                    f"{runtime.daily_tokens}/"
                    f"{spec.tpd or 'n/a'} | "
                    f"DB inserted="
                    f"{actual_inserted}"
                )
            )

        # --------------------------------------------------------------
        # Validator statistics.
        # --------------------------------------------------------------

        print(
            "\nVALIDATOR STATUS"
        )

        print(
            "-" * 120
        )

        for validator_model in VALIDATOR_MODELS:
                
            stats = (
                self.state.data
                .get(
                    "validators",
                    {}
                )
                .get(
                    validator_model,
                    {
                        "batches": 0,
                        "validated_questions": 0,
                        "approved": 0,
                        "repaired": 0,
                        "rejected": 0,
                        "errors": 0,
                    },
                )
            )
        
            print(
                (
                    f"{validator_model:32s} | "
                    f"batches="
                    f"{stats['batches']:<4} | "
                    f"validated="
                    f"{stats['validated_questions']:<5} | "
                    f"approved="
                    f"{stats['approved']:<5} | "
                    f"repaired="
                    f"{stats['repaired']:<5} | "
                    f"rejected="
                    f"{stats['rejected']:<5} | "
                    f"errors="
                    f"{stats['errors']}"
                )
            )

        # --------------------------------------------------------------
        # Complete curriculum inventory.
        # --------------------------------------------------------------

        print(
            "\nDATABASE QUESTION INVENTORY"
        )

        print(
            "-" * 120
        )

        print(
            (
                f"{'Grade':12s} | "
                f"{'Subject':14s} | "
                f"{'Sub-topic':52s} | "
                f"{'Difficulty':10s} | "
                f"{'Questions':9s}"
            )
        )

        print(
            "-" * 120
        )

        for row in inventory[
            "combinations"
        ]:

            print(
                (
                    f"{row['grade']:12s} | "
                    f"{row['subject']:14s} | "
                    f"{row['syllabus_topic']:52s} | "
                    f"{row['difficulty']:10s} | "
                    f"{row['question_count']:9d}"
                )
            )

        print(
            "-" * 120
        )

        print(
            (
                f"{'GRAND TOTAL':92s} | "
                f"{database_total:9d}"
            )
        )

        print(
            "=" * 120
        )


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Elevate practice question-bank generator"
        )
    )

    parser.add_argument(
        "--mode",
        choices=(
            "demo",
            "production",
        ),
        default="demo",
    )

    parser.add_argument(
        "--demo-questions-per-model",
        type=int,
        default=DEFAULT_DEMO_QUESTIONS_PER_MODEL,
        help=(
            "Number of questions requested "
            "from each model during demo mode."
        ),
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    args = parse_args()

    if (
        args.demo_questions_per_model < 1
        or args.demo_questions_per_model > 10
    ):

        print(
            (
                "--demo-questions-per-model "
                "must be between 1 and 10."
            )
        )

        return 2

    try:

        apply_rate_limit_overrides()

        engine = build_db_engine()

        syllabus = load_syllabus()

        state = GenerationState(
            STATE_FILE
        )

        generator = QuestionGenerator(
            engine,
            state,
            syllabus,
        )

        if args.mode == "demo":

            generator.run_demo(
                args.demo_questions_per_model
            )

        else:

            generator.run_production()

        return 0

    except KeyboardInterrupt:

        print(
            "\nStopped safely."
        )

        return 130

    except Exception as exc:

        print(
            f"\n[FATAL] {exc}"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )