"""
Seed script to populate database with diverse questions for adaptive learning.
Run this script to add questions at all difficulty levels for testing.

Usage:
    python seed_questions.py
    python seed_questions.py --reset-questions
    python seed_questions.py --augment-large --per-topic 12
    python seed_questions.py --augment-large --skip-ai-augment --per-topic 12
    python seed_questions.py --reset-questions --strict-stem-rebuild --per-subtopic 20
"""

import sys, os
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend.app import create_app
from backend.models import db, Question
from backend.ai_topic_service import generate_topic_mcqs

# Create Flask app instance
app = create_app('development')

# Comprehensive question bank with 4 difficulty levels
QUESTIONS = [
    # MATHEMATICS - Easy
    {
        "subject": "mathematics",
        "grade": "elementary",
        "difficulty": "easy",
        "text": "What is 5 + 3?",
        "options": ["6", "7", "8", "9"],
        "correct_index": 2,
        "hint": "Count up from 5: 6, 7, 8",
        "explanation": "5 + 3 = 8",
        "tags": ["addition", "basic"],
        "syllabus_topic": "addition",
        "readability_level": "basic"
    },
    {
        "subject": "mathematics",
        "grade": "elementary",
        "difficulty": "easy",
        "text": "What is 10 - 4?",
        "options": ["5", "6", "7", "8"],
        "correct_index": 1,
        "hint": "Subtract 4 from 10",
        "explanation": "10 - 4 = 6",
        "tags": ["subtraction", "basic"],
        "syllabus_topic": "subtraction",
        "readability_level": "basic"
    },
    {
        "subject": "mathematics",
        "grade": "middle",
        "difficulty": "easy",
        "text": "What is 12 × 2?",
        "options": ["22", "24", "26", "28"],
        "correct_index": 1,
        "hint": "Double 12",
        "explanation": "12 × 2 = 24",
        "tags": ["multiplication", "basic"],
        "syllabus_topic": "multiplication",
        "readability_level": "basic"
    },
    
    # MATHEMATICS - Medium
    {
        "subject": "mathematics",
        "grade": "middle",
        "difficulty": "medium",
        "text": "Solve: 3x + 5 = 20. What is x?",
        "options": ["3", "4", "5", "6"],
        "correct_index": 2,
        "hint": "Subtract 5 from both sides first",
        "explanation": "3x = 15, so x = 5",
        "tags": ["algebra", "equations"],
        "syllabus_topic": "linear_equations",
        "readability_level": "intermediate"
    },
    {
        "subject": "mathematics",
        "grade": "middle",
        "difficulty": "medium",
        "text": "What is 15% of 200?",
        "options": ["20", "25", "30", "35"],
        "correct_index": 2,
        "hint": "Convert 15% to 0.15 and multiply",
        "explanation": "0.15 × 200 = 30",
        "tags": ["percentage", "decimals"]
    },
    {
        "subject": "mathematics",
        "grade": "high",
        "difficulty": "medium",
        "text": "What is the slope of a line passing through (2, 3) and (4, 7)?",
        "options": ["1", "2", "3", "4"],
        "correct_index": 1,
        "hint": "Use slope formula: (y2-y1)/(x2-x1)",
        "explanation": "(7-3)/(4-2) = 4/2 = 2",
        "tags": ["slope", "coordinate geometry"]
    },
    
    # MATHEMATICS - Hard
    {
        "subject": "mathematics",
        "grade": "high",
        "difficulty": "hard",
        "text": "Solve: x² - 5x + 6 = 0. What are the values of x?",
        "options": ["1 and 6", "2 and 3", "3 and 4", "4 and 5"],
        "correct_index": 1,
        "hint": "Factor the quadratic equation",
        "explanation": "(x-2)(x-3) = 0, so x = 2 or x = 3",
        "tags": ["quadratic", "factoring"]
    },
    {
        "subject": "mathematics",
        "grade": "high",
        "difficulty": "hard",
        "text": "What is the derivative of f(x) = 3x² + 2x?",
        "options": ["6x + 2", "3x + 2", "6x", "3x²"],
        "correct_index": 0,
        "hint": "Apply power rule: d/dx(xⁿ) = nxⁿ⁻¹",
        "explanation": "f'(x) = 6x + 2",
        "tags": ["calculus", "derivatives"]
    },
    
    # MATHEMATICS - Expert
    {
        "subject": "mathematics",
        "grade": "college",
        "difficulty": "expert",
        "text": "What is the integral of ∫(2x + 3)dx?",
        "options": ["x² + 3x + C", "2x² + 3x + C", "x² + 2x + C", "2x + 3x² + C"],
        "correct_index": 0,
        "hint": "Use power rule for integration",
        "explanation": "∫(2x + 3)dx = x² + 3x + C",
        "tags": ["calculus", "integration"]
    },
    {
        "subject": "mathematics",
        "grade": "college",
        "difficulty": "expert",
        "text": "Find the limit: lim(x→0) (sin(x)/x)",
        "options": ["0", "1", "∞", "undefined"],
        "correct_index": 1,
        "hint": "This is a standard limit",
        "explanation": "lim(x→0) (sin(x)/x) = 1",
        "tags": ["limits", "trigonometry"]
    },
    
    # SCIENCE - Easy
    {
        "subject": "science",
        "grade": "elementary",
        "difficulty": "easy",
        "text": "What do plants need to make food?",
        "options": ["Sunlight, water, and carbon dioxide", "Only water", "Only sunlight", "Only soil"],
        "correct_index": 0,
        "hint": "Think about photosynthesis",
        "explanation": "Plants use sunlight, water, and CO₂ for photosynthesis",
        "tags": ["photosynthesis", "plants"],
        "syllabus_topic": "photosynthesis",
        "readability_level": "basic"
    },
    {
        "subject": "science",
        "grade": "elementary",
        "difficulty": "easy",
        "text": "What are the three states of matter?",
        "options": ["Solid, liquid, gas", "Hot, cold, warm", "Big, medium, small", "Hard, soft, smooth"],
        "correct_index": 0,
        "hint": "Think of ice, water, and steam",
        "explanation": "Matter exists in solid, liquid, and gas states",
        "tags": ["matter", "states"],
        "syllabus_topic": "states_of_matter",
        "readability_level": "basic"
    },
    
    # SCIENCE - Medium
    {
        "subject": "science",
        "grade": "middle",
        "difficulty": "medium",
        "text": "What is the process by which water changes from liquid to gas?",
        "options": ["Condensation", "Evaporation", "Freezing", "Melting"],
        "correct_index": 1,
        "hint": "Think of puddles drying up",
        "explanation": "Evaporation is liquid turning to gas",
        "tags": ["water cycle", "phase change"]
    },
    {
        "subject": "science",
        "grade": "middle",
        "difficulty": "medium",
        "text": "What type of rock is formed from cooled lava?",
        "options": ["Sedimentary", "Metamorphic", "Igneous", "Limestone"],
        "correct_index": 2,
        "hint": "Think of volcanic activity",
        "explanation": "Igneous rocks form from cooled magma or lava",
        "tags": ["geology", "rocks"]
    },
    
    # SCIENCE - Hard
    {
        "subject": "science",
        "grade": "high",
        "difficulty": "hard",
        "text": "What is the powerhouse of the cell?",
        "options": ["Nucleus", "Ribosome", "Mitochondria", "Chloroplast"],
        "correct_index": 2,
        "hint": "This organelle produces ATP",
        "explanation": "Mitochondria generate energy (ATP) for cells",
        "tags": ["cell biology", "organelles"]
    },
    {
        "subject": "science",
        "grade": "high",
        "difficulty": "hard",
        "text": "What is the pH of a neutral solution?",
        "options": ["0", "7", "14", "1"],
        "correct_index": 1,
        "hint": "Think of pure water",
        "explanation": "A pH of 7 is neutral on the pH scale",
        "tags": ["chemistry", "pH"]
    },
    
    # SCIENCE - Expert
    {
        "subject": "science",
        "grade": "college",
        "difficulty": "expert",
        "text": "What is the first law of thermodynamics?",
        "options": [
            "Energy cannot be created or destroyed",
            "Entropy always increases",
            "Force equals mass times acceleration",
            "Every action has an equal and opposite reaction"
        ],
        "correct_index": 0,
        "hint": "Think about conservation",
        "explanation": "The first law states energy is conserved in a closed system",
        "tags": ["thermodynamics", "physics"]
    },
    
    # PHYSICS - Easy
    {
        "subject": "physics",
        "grade": "middle",
        "difficulty": "easy",
        "text": "What is the unit of force?",
        "options": ["Joule", "Newton", "Watt", "Pascal"],
        "correct_index": 1,
        "hint": "Named after a famous scientist",
        "explanation": "The Newton (N) is the SI unit of force",
        "tags": ["units", "force"],
        "syllabus_topic": "units_of_measure",
        "readability_level": "basic"
    },
    {
        "subject": "physics",
        "grade": "middle",
        "difficulty": "easy",
        "text": "What makes a rainbow?",
        "options": ["Light refraction in water droplets", "Magic", "Paint in the sky", "Chemical reaction"],
        "correct_index": 0,
        "hint": "Think about light and rain",
        "explanation": "Sunlight refracts through water droplets creating a spectrum",
        "tags": ["optics", "light"]
    },
    
    # PHYSICS - Medium
    {
        "subject": "physics",
        "grade": "high",
        "difficulty": "medium",
        "text": "What is the acceleration due to gravity on Earth?",
        "options": ["8.8 m/s²", "9.8 m/s²", "10.8 m/s²", "11.8 m/s²"],
        "correct_index": 1,
        "hint": "It's approximately 10 m/s²",
        "explanation": "g = 9.8 m/s² on Earth's surface",
        "tags": ["gravity", "acceleration"]
    },
    {
        "subject": "physics",
        "grade": "high",
        "difficulty": "medium",
        "text": "What is the formula for kinetic energy?",
        "options": ["KE = mv", "KE = ½mv²", "KE = mgh", "KE = Fd"],
        "correct_index": 1,
        "hint": "It involves mass and velocity squared",
        "explanation": "Kinetic energy = ½ × mass × velocity²",
        "tags": ["energy", "mechanics"]
    },
    
    # PHYSICS - Hard
    {
        "subject": "physics",
        "grade": "high",
        "difficulty": "hard",
        "text": "What is the speed of light in vacuum?",
        "options": ["3 × 10⁸ m/s", "3 × 10⁷ m/s", "3 × 10⁹ m/s", "3 × 10⁶ m/s"],
        "correct_index": 0,
        "hint": "It's approximately 300,000 km/s",
        "explanation": "c = 3 × 10⁸ m/s in vacuum",
        "tags": ["relativity", "constants"]
    },
    {
        "subject": "physics",
        "grade": "college",
        "difficulty": "hard",
        "text": "What is Ohm's Law?",
        "options": ["V = IR", "F = ma", "E = mc²", "P = VI"],
        "correct_index": 0,
        "hint": "It relates voltage, current, and resistance",
        "explanation": "Voltage = Current × Resistance",
        "tags": ["electricity", "circuits"]
    },
    
    # PHYSICS - Expert
    {
        "subject": "physics",
        "grade": "college",
        "difficulty": "expert",
        "text": "What does the Heisenberg Uncertainty Principle state?",
        "options": [
            "Position and momentum cannot both be precisely determined",
            "Energy is quantized",
            "Light behaves as both wave and particle",
            "Mass curves spacetime"
        ],
        "correct_index": 0,
        "hint": "It's a fundamental principle of quantum mechanics",
        "explanation": "You cannot simultaneously know exact position and momentum",
        "tags": ["quantum mechanics", "uncertainty"]
    },
    
    # CHEMISTRY - Easy
    {
        "subject": "chemistry",
        "grade": "middle",
        "difficulty": "easy",
        "text": "What is the chemical symbol for water?",
        "options": ["H₂O", "CO₂", "O₂", "NaCl"],
        "correct_index": 0,
        "hint": "Two hydrogen, one oxygen",
        "explanation": "Water is H₂O - two hydrogen atoms bonded to one oxygen",
        "tags": ["molecules", "basic"],
        "syllabus_topic": "molecules",
        "readability_level": "basic"
    },
    {
        "subject": "chemistry",
        "grade": "middle",
        "difficulty": "easy",
        "text": "What is table salt chemically?",
        "options": ["NaCl", "KCl", "CaCl₂", "MgCl₂"],
        "correct_index": 0,
        "hint": "Sodium chloride",
        "explanation": "Table salt is sodium chloride (NaCl)",
        "tags": ["compounds", "basic"]
    },
    
    # CHEMISTRY - Medium
    {
        "subject": "chemistry",
        "grade": "high",
        "difficulty": "medium",
        "text": "What is the atomic number of Carbon?",
        "options": ["4", "6", "8", "12"],
        "correct_index": 1,
        "hint": "Count the protons in the nucleus",
        "explanation": "Carbon has 6 protons, so atomic number is 6",
        "tags": ["periodic table", "elements"]
    },
    {
        "subject": "chemistry",
        "grade": "high",
        "difficulty": "medium",
        "text": "What type of bond shares electrons?",
        "options": ["Ionic", "Covalent", "Metallic", "Hydrogen"],
        "correct_index": 1,
        "hint": "Think of molecules like H₂O",
        "explanation": "Covalent bonds share electrons between atoms",
        "tags": ["bonding", "electrons"]
    },
    
    # CHEMISTRY - Hard
    {
        "subject": "chemistry",
        "grade": "high",
        "difficulty": "hard",
        "text": "What is Avogadro's number?",
        "options": ["6.02 × 10²³", "3.14 × 10⁸", "9.8 × 10⁹", "1.6 × 10⁻¹⁹"],
        "correct_index": 0,
        "hint": "Number of particles in one mole",
        "explanation": "Avogadro's number is 6.02 × 10²³ particles/mol",
        "tags": ["moles", "constants"]
    },
    {
        "subject": "chemistry",
        "grade": "college",
        "difficulty": "hard",
        "text": "What is the oxidation state of Mn in KMnO₄?",
        "options": ["+7", "+6", "+5", "+4"],
        "correct_index": 0,
        "hint": "K is +1, O is -2",
        "explanation": "Mn in KMnO₄ has +7 oxidation state",
        "tags": ["oxidation", "redox"]
    },
    
    # CHEMISTRY - Expert
    {
        "subject": "chemistry",
        "grade": "college",
        "difficulty": "expert",
        "text": "What is the Gibbs free energy equation?",
        "options": ["ΔG = ΔH - TΔS", "ΔG = ΔH + TΔS", "ΔG = ΔH/TΔS", "ΔG = TΔH - ΔS"],
        "correct_index": 0,
        "hint": "It relates enthalpy, entropy, and temperature",
        "explanation": "ΔG = ΔH - TΔS determines spontaneity",
        "tags": ["thermodynamics", "free energy"]
    },
    
    # BIOLOGY - Easy
    {
        "subject": "biology",
        "grade": "middle",
        "difficulty": "easy",
        "text": "What is the largest organ in the human body?",
        "options": ["Heart", "Liver", "Skin", "Brain"],
        "correct_index": 2,
        "hint": "It covers your entire body",
        "explanation": "The skin is the largest organ",
        "tags": ["anatomy", "organs"],
        "syllabus_topic": "anatomy",
        "readability_level": "basic"
    },
    {
        "subject": "biology",
        "grade": "middle",
        "difficulty": "easy",
        "text": "How many chromosomes do humans have?",
        "options": ["23", "46", "48", "92"],
        "correct_index": 1,
        "hint": "23 pairs = ?",
        "explanation": "Humans have 46 chromosomes (23 pairs)",
        "tags": ["genetics", "chromosomes"]
    },
    
    # BIOLOGY - Medium
    {
        "subject": "biology",
        "grade": "high",
        "difficulty": "medium",
        "text": "What is the function of red blood cells?",
        "options": [
            "Transport oxygen",
            "Fight infection",
            "Clot blood",
            "Produce hormones"
        ],
        "correct_index": 0,
        "hint": "Think about breathing",
        "explanation": "RBCs carry oxygen from lungs to tissues",
        "tags": ["blood", "circulatory system"]
    },
    {
        "subject": "biology",
        "grade": "high",
        "difficulty": "medium",
        "text": "What is the basic unit of life?",
        "options": ["Atom", "Molecule", "Cell", "Tissue"],
        "correct_index": 2,
        "hint": "All living things are made of these",
        "explanation": "The cell is the fundamental unit of life",
        "tags": ["cell biology", "basic"]
    },
    
    # BIOLOGY - Hard
    {
        "subject": "biology",
        "grade": "high",
        "difficulty": "hard",
        "text": "What process produces ATP in cells?",
        "options": ["Photosynthesis", "Cellular respiration", "Diffusion", "Osmosis"],
        "correct_index": 1,
        "hint": "It happens in mitochondria",
        "explanation": "Cellular respiration generates ATP energy",
        "tags": ["metabolism", "energy"]
    },
    {
        "subject": "biology",
        "grade": "college",
        "difficulty": "hard",
        "text": "What is the central dogma of molecular biology?",
        "options": [
            "DNA → RNA → Protein",
            "Protein → RNA → DNA",
            "RNA → DNA → Protein",
            "DNA → Protein → RNA"
        ],
        "correct_index": 0,
        "hint": "Flow of genetic information",
        "explanation": "DNA is transcribed to RNA, which is translated to protein",
        "tags": ["genetics", "molecular biology"]
    },
    
    # BIOLOGY - Expert
    {
        "subject": "biology",
        "grade": "college",
        "difficulty": "expert",
        "text": "What is the function of telomeres?",
        "options": [
            "Protect chromosome ends",
            "Code for proteins",
            "Replicate DNA",
            "Transcribe RNA"
        ],
        "correct_index": 0,
        "hint": "They're like caps on shoelaces",
        "explanation": "Telomeres protect chromosomes from degradation during replication",
        "tags": ["genetics", "chromosomes"]
    },
    
    # Additional MATHEMATICS questions for better testing
    {
        "subject": "mathematics",
        "grade": "middle",
        "difficulty": "easy",
        "text": "What is 100 ÷ 4?",
        "options": ["20", "25", "30", "35"],
        "correct_index": 1,
        "hint": "How many 4s in 100?",
        "explanation": "100 ÷ 4 = 25",
        "tags": ["division", "basic"]
    },
    {
        "subject": "mathematics",
        "grade": "high",
        "difficulty": "medium",
        "text": "What is sin(30°)?",
        "options": ["0.5", "0.707", "0.866", "1"],
        "correct_index": 0,
        "hint": "This is a standard angle",
        "explanation": "sin(30°) = 1/2 = 0.5",
        "tags": ["trigonometry", "angles"]
    },
    {
        "subject": "mathematics",
        "grade": "high",
        "difficulty": "hard",
        "text": "What is log₁₀(1000)?",
        "options": ["2", "3", "4", "5"],
        "correct_index": 1,
        "hint": "10 to what power equals 1000?",
        "explanation": "10³ = 1000, so log₁₀(1000) = 3",
        "tags": ["logarithms", "exponents"]
    },
    {
        "subject": "mathematics",
        "grade": "college",
        "difficulty": "expert",
        "text": "What is the Fourier transform used for?",
        "options": [
            "Converting time domain to frequency domain",
            "Solving differential equations",
            "Finding derivatives",
            "Calculating probabilities"
        ],
        "correct_index": 0,
        "hint": "Think about signal processing",
        "explanation": "Fourier transforms convert signals to frequency components",
        "tags": ["signal processing", "transforms"]
    },
    
    # Additional SCIENCE questions
    {
        "subject": "science",
        "grade": "middle",
        "difficulty": "easy",
        "text": "What planet is known as the Red Planet?",
        "options": ["Venus", "Mars", "Jupiter", "Saturn"],
        "correct_index": 1,
        "hint": "Fourth planet from the sun",
        "explanation": "Mars appears red due to iron oxide on its surface",
        "tags": ["astronomy", "planets"]
    },
    {
        "subject": "science",
        "grade": "high",
        "difficulty": "medium",
        "text": "What is the most abundant gas in Earth's atmosphere?",
        "options": ["Oxygen", "Nitrogen", "Carbon dioxide", "Argon"],
        "correct_index": 1,
        "hint": "About 78% of air",
        "explanation": "Nitrogen makes up 78% of the atmosphere",
        "tags": ["atmosphere", "chemistry"]
    },
    {
        "subject": "science",
        "grade": "high",
        "difficulty": "hard",
        "text": "What is the process of nuclear fusion in the sun?",
        "options": [
            "Hydrogen fuses into helium",
            "Helium fuses into hydrogen",
            "Carbon fuses into oxygen",
            "Uranium splits into smaller atoms"
        ],
        "correct_index": 0,
        "hint": "Smaller atoms combine to make bigger ones",
        "explanation": "The sun fuses hydrogen nuclei into helium, releasing energy",
        "tags": ["nuclear", "sun"]
    },
    
    # Additional PHYSICS questions
    {
        "subject": "physics",
        "grade": "middle",
        "difficulty": "easy",
        "text": "What happens when you rub a balloon on your hair?",
        "options": [
            "Static electricity builds up",
            "The balloon gets heavier",
            "Hair falls out",
            "Nothing"
        ],
        "correct_index": 0,
        "hint": "Think about charges",
        "explanation": "Friction transfers electrons, creating static electricity",
        "tags": ["electricity", "static"]
    },
    {
        "subject": "physics",
        "grade": "high",
        "difficulty": "medium",
        "text": "What is the relationship between wavelength and frequency?",
        "options": [
            "Inversely proportional",
            "Directly proportional",
            "No relationship",
            "Exponential"
        ],
        "correct_index": 0,
        "hint": "c = λν where c is constant",
        "explanation": "As frequency increases, wavelength decreases",
        "tags": ["waves", "optics"]
    },
    {
        "subject": "physics",
        "grade": "college",
        "difficulty": "hard",
        "text": "What is the equation for gravitational force?",
        "options": [
            "F = G(m₁m₂)/r²",
            "F = ma",
            "F = kx",
            "F = qE"
        ],
        "correct_index": 0,
        "hint": "Newton's law of gravitation",
        "explanation": "F = G(m₁m₂)/r² where G is gravitational constant",
        "tags": ["gravity", "forces"]
    },
    
    # Additional CHEMISTRY questions
    {
        "subject": "chemistry",
        "grade": "middle",
        "difficulty": "easy",
        "text": "What is the lightest element?",
        "options": ["Helium", "Hydrogen", "Lithium", "Carbon"],
        "correct_index": 1,
        "hint": "First element in the periodic table",
        "explanation": "Hydrogen has atomic number 1 and is the lightest",
        "tags": ["elements", "periodic table"]
    },
    {
        "subject": "chemistry",
        "grade": "high",
        "difficulty": "medium",
        "text": "What is produced when an acid reacts with a base?",
        "options": [
            "Salt and water",
            "Oxygen and hydrogen",
            "Carbon dioxide",
            "Ammonia"
        ],
        "correct_index": 0,
        "hint": "This is a neutralization reaction",
        "explanation": "Acid + Base → Salt + Water",
        "tags": ["reactions", "acids"]
    },
    {
        "subject": "chemistry",
        "grade": "college",
        "difficulty": "hard",
        "text": "What is the electron configuration of Oxygen?",
        "options": [
            "1s² 2s² 2p⁴",
            "1s² 2s² 2p⁶",
            "1s² 2s² 2p²",
            "1s² 2s² 2p³"
        ],
        "correct_index": 0,
        "hint": "Oxygen has 8 electrons",
        "explanation": "O: 1s² 2s² 2p⁴ (2 + 2 + 4 = 8 electrons)",
        "tags": ["atomic structure", "electrons"]
    },
    
    # Additional BIOLOGY questions
    {
        "subject": "biology",
        "grade": "middle",
        "difficulty": "easy",
        "text": "What do we call animals that eat only plants?",
        "options": ["Carnivores", "Herbivores", "Omnivores", "Decomposers"],
        "correct_index": 1,
        "hint": "Think of cows and rabbits",
        "explanation": "Herbivores eat only plant matter",
        "tags": ["ecology", "food chains"]
    },
    {
        "subject": "biology",
        "grade": "high",
        "difficulty": "medium",
        "text": "What is the main function of white blood cells?",
        "options": [
            "Carry oxygen",
            "Fight infection",
            "Clot blood",
            "Carry nutrients"
        ],
        "correct_index": 1,
        "hint": "They're part of the immune system",
        "explanation": "White blood cells defend against pathogens",
        "tags": ["immune system", "blood"]
    },
    {
        "subject": "biology",
        "grade": "college",
        "difficulty": "hard",
        "text": "What is the role of mRNA in protein synthesis?",
        "options": [
            "Carries genetic code from DNA to ribosomes",
            "Forms the ribosome structure",
            "Brings amino acids to ribosomes",
            "Breaks down proteins"
        ],
        "correct_index": 0,
        "hint": "Think about transcription and translation",
        "explanation": "mRNA carries the genetic code from nucleus to ribosomes",
        "tags": ["molecular biology", "protein synthesis"]
    },
]


def _topic_to_title(topic_slug: str) -> str:
    return " ".join((topic_slug or "").split("_"))


def _subject_to_title(subject_slug: str) -> str:
    return " ".join((subject_slug or "").split("_"))


def _readability_for_difficulty(difficulty: str) -> str:
    if difficulty == "easy":
        return "basic"
    if difficulty == "medium":
        return "intermediate"
    return "advanced"


CURATED_TOPIC_QUESTION_BANK = {
    ("technology", "ai_basics"): {
        "easy": [
            {
                "text": "What does AI stand for?",
                "options": ["Artificial Intelligence", "Automated Internet", "Analog Interface", "Applied Informatics"],
                "correct_index": 0,
                "hint": "It refers to machine-based intelligence.",
                "explanation": "AI stands for Artificial Intelligence.",
                "tags": ["technology", "ai_basics", "definition"],
            },
            {
                "text": "In machine learning, what is a model?",
                "options": ["A learned function that makes predictions", "A physical robot body", "A database table", "A type of monitor"],
                "correct_index": 0,
                "hint": "It maps input data to outputs.",
                "explanation": "A model is the learned function used for inference.",
                "tags": ["technology", "ai_basics", "model"],
            },
            {
                "text": "Which data is used in supervised learning?",
                "options": ["Labeled data", "Only encrypted data", "Only images", "Unsorted binary files"],
                "correct_index": 0,
                "hint": "Targets are known during training.",
                "explanation": "Supervised learning uses labeled examples.",
                "tags": ["technology", "ai_basics", "supervised_learning"],
            },
            {
                "text": "A classification task predicts:",
                "options": ["A category label", "A random password", "A continuous decimal only", "Database latency"],
                "correct_index": 0,
                "hint": "Think spam vs not spam.",
                "explanation": "Classification predicts discrete classes.",
                "tags": ["technology", "ai_basics", "classification"],
            },
            {
                "text": "What is the main purpose of a training dataset?",
                "options": ["To help the model learn patterns", "To run the web server", "To store final grades", "To replace evaluation"],
                "correct_index": 0,
                "hint": "Learning happens during training.",
                "explanation": "Training data is used by algorithms to learn relationships.",
                "tags": ["technology", "ai_basics", "training_data"],
            },
            {
                "text": "Which field is a common AI application area?",
                "options": ["Image recognition", "Concrete curing", "Analog radio tuning", "Manual filing"],
                "correct_index": 0,
                "hint": "Think computer vision.",
                "explanation": "Image recognition is a standard AI application.",
                "tags": ["technology", "ai_basics", "applications"],
            },
            {
                "text": "What is inference in AI?",
                "options": ["Using a trained model to make predictions", "Collecting data labels", "Deleting noisy data", "Compiling CSS"],
                "correct_index": 0,
                "hint": "It happens after training.",
                "explanation": "Inference is running a trained model on new inputs.",
                "tags": ["technology", "ai_basics", "inference"],
            },
            {
                "text": "Which metric directly uses true positives, false positives, and false negatives?",
                "options": ["Precision", "Frame rate", "Battery health", "Clock speed"],
                "correct_index": 0,
                "hint": "It focuses on prediction quality for positive class.",
                "explanation": "Precision depends on true positives and false positives.",
                "tags": ["technology", "ai_basics", "metrics"],
            },
        ],
        "medium": [
            {
                "text": "Why do we split data into training and validation sets?",
                "options": ["To estimate generalization during model tuning", "To reduce file size only", "To avoid storing labels", "To skip testing"],
                "correct_index": 0,
                "hint": "Validation checks performance on unseen samples during development.",
                "explanation": "Validation helps tune models while monitoring overfitting.",
                "tags": ["technology", "ai_basics", "validation"],
            },
            {
                "text": "What is overfitting?",
                "options": ["Model memorizes training data and performs poorly on new data", "Model trains too fast and always improves", "Model has too few parameters by design", "Model ignores all training examples"],
                "correct_index": 0,
                "hint": "High training accuracy with weak test accuracy is a clue.",
                "explanation": "Overfitting means poor generalization beyond seen data.",
                "tags": ["technology", "ai_basics", "overfitting"],
            },
            {
                "text": "When classes are imbalanced, which metric is often more informative than accuracy?",
                "options": ["F1-score", "CPU usage", "File count", "Uptime"],
                "correct_index": 0,
                "hint": "It combines precision and recall.",
                "explanation": "F1-score is useful for imbalanced classification.",
                "tags": ["technology", "ai_basics", "evaluation"],
            },
            {
                "text": "Which approach helps reduce overfitting in many models?",
                "options": ["Regularization", "Ignoring validation", "Removing all features", "Increasing label noise"],
                "correct_index": 0,
                "hint": "It penalizes overly complex models.",
                "explanation": "Regularization constrains model complexity.",
                "tags": ["technology", "ai_basics", "regularization"],
            },
            {
                "text": "What is a confusion matrix used for?",
                "options": ["Summarizing prediction outcomes by class", "Compressing model files", "Encrypting training labels", "Scheduling retraining"],
                "correct_index": 0,
                "hint": "It lists TP, FP, TN, and FN patterns.",
                "explanation": "A confusion matrix shows how predictions compare to actual labels.",
                "tags": ["technology", "ai_basics", "confusion_matrix"],
            },
            {
                "text": "In supervised learning, what is a feature?",
                "options": ["An input variable used for prediction", "The final report format", "A random seed only", "A GPU temperature"],
                "correct_index": 0,
                "hint": "Model inputs are built from these.",
                "explanation": "Features are input attributes used by the model.",
                "tags": ["technology", "ai_basics", "features"],
            },
            {
                "text": "Why should test data stay unseen until final evaluation?",
                "options": ["To provide an unbiased estimate of real-world performance", "To reduce training time only", "To increase model size", "To simplify UI"],
                "correct_index": 0,
                "hint": "Leakage can inflate reported performance.",
                "explanation": "Keeping test data untouched prevents evaluation bias.",
                "tags": ["technology", "ai_basics", "test_set"],
            },
            {
                "text": "What is data leakage in machine learning?",
                "options": ["Using information not available at prediction time", "Losing files after backup", "Running out of RAM", "Slow internet download"],
                "correct_index": 0,
                "hint": "It causes unrealistically high scores.",
                "explanation": "Leakage occurs when future/target-related info leaks into training features.",
                "tags": ["technology", "ai_basics", "data_leakage"],
            },
        ],
        "hard": [
            {
                "text": "A model has 99% accuracy but misses most fraud cases. What is the key issue?",
                "options": ["Class imbalance hides poor minority-class recall", "The model is too fast", "The UI is not responsive", "The training set is too small to measure"],
                "correct_index": 0,
                "hint": "Accuracy alone can mislead on rare events.",
                "explanation": "High accuracy can still fail minority classes when data is imbalanced.",
                "tags": ["technology", "ai_basics", "imbalance"],
            },
            {
                "text": "Which practice most improves fairness assessment in AI systems?",
                "options": ["Evaluate performance across demographic subgroups", "Use one global metric only", "Remove all validation data", "Train once and never audit"],
                "correct_index": 0,
                "hint": "Check disparities between groups.",
                "explanation": "Fairness auditing requires subgroup-level evaluation.",
                "tags": ["technology", "ai_basics", "fairness"],
            },
            {
                "text": "What does high variance usually indicate in model behavior?",
                "options": ["The model is sensitive to training data changes", "The model cannot represent patterns", "The labels are always perfect", "The feature count is exactly optimal"],
                "correct_index": 0,
                "hint": "Think overfitting tendency.",
                "explanation": "High variance means predictions change significantly with different training samples.",
                "tags": ["technology", "ai_basics", "bias_variance"],
            },
            {
                "text": "Which action helps diagnose whether a model is overfitting?",
                "options": ["Compare training and validation curves over epochs", "Only inspect deployment logs", "Increase batch size blindly", "Disable all metrics"],
                "correct_index": 0,
                "hint": "Learning curves reveal divergence.",
                "explanation": "Overfitting is often visible as training improves while validation worsens.",
                "tags": ["technology", "ai_basics", "learning_curves"],
            },
            {
                "text": "Why is feature scaling important for some algorithms?",
                "options": ["Different scales can distort distance or gradient-based optimization", "It changes labels directly", "It guarantees perfect accuracy", "It removes all outliers automatically"],
                "correct_index": 0,
                "hint": "Think k-NN and gradient descent.",
                "explanation": "Scale differences can dominate optimization and distance computations.",
                "tags": ["technology", "ai_basics", "feature_scaling"],
            },
            {
                "text": "What is the main purpose of cross-validation?",
                "options": ["Estimate model robustness across multiple data splits", "Reduce model parameters automatically", "Replace final testing entirely", "Encrypt inference outputs"],
                "correct_index": 0,
                "hint": "Repeated splits provide more stable estimates.",
                "explanation": "Cross-validation assesses consistency and reduces split-specific bias.",
                "tags": ["technology", "ai_basics", "cross_validation"],
            },
            {
                "text": "If precision is high and recall is low, what does that imply?",
                "options": ["Positive predictions are usually correct, but many positives are missed", "The model predicts positives too often", "The model is random", "The labels are noisy by definition"],
                "correct_index": 0,
                "hint": "Recall captures missed true positives.",
                "explanation": "High precision with low recall means strict positive prediction with many misses.",
                "tags": ["technology", "ai_basics", "precision_recall"],
            },
            {
                "text": "What risk appears when a model is trained on stale historical data only?",
                "options": ["Performance may degrade under data drift", "Model size always decreases", "Inference becomes impossible", "Validation becomes unnecessary"],
                "correct_index": 0,
                "hint": "Production data distribution can shift over time.",
                "explanation": "Data drift causes mismatch between training and current real-world inputs.",
                "tags": ["technology", "ai_basics", "data_drift"],
            },
        ],
        "expert": [
            {
                "text": "Which deployment strategy best reduces risk for a newly trained AI model?",
                "options": ["Canary rollout with monitored performance thresholds", "Immediate full rollout without monitoring", "Disable rollback mechanisms", "Skip baseline comparison"],
                "correct_index": 0,
                "hint": "Gradual release limits blast radius.",
                "explanation": "Canary releases allow controlled exposure and rapid rollback.",
                "tags": ["technology", "ai_basics", "deployment"],
            },
            {
                "text": "What is a key objective of model monitoring in production?",
                "options": ["Detect drift, quality regressions, and reliability issues over time", "Increase dataset size automatically", "Replace all offline evaluation", "Avoid retraining forever"],
                "correct_index": 0,
                "hint": "Monitoring tracks health after deployment.",
                "explanation": "Production monitoring catches failures that offline metrics miss.",
                "tags": ["technology", "ai_basics", "monitoring"],
            },
            {
                "text": "Why are reproducible training pipelines important in AI engineering?",
                "options": ["They enable reliable debugging, auditing, and controlled improvements", "They remove the need for testing", "They guarantee fairness automatically", "They prevent concept drift"],
                "correct_index": 0,
                "hint": "Repeatability is essential for trustworthy iteration.",
                "explanation": "Reproducibility allows teams to verify and safely evolve model behavior.",
                "tags": ["technology", "ai_basics", "reproducibility"],
            },
            {
                "text": "Which practice best supports responsible AI governance?",
                "options": ["Maintain documentation of data sources, limitations, and decision logic", "Hide model assumptions from stakeholders", "Skip failure-mode analysis", "Use one aggregate score only"],
                "correct_index": 0,
                "hint": "Transparency and traceability are core governance pillars.",
                "explanation": "Good governance requires documented lineage, risks, and operating boundaries.",
                "tags": ["technology", "ai_basics", "governance"],
            },
        ],
    }
}


def _get_curated_question(subject: str, topic: str, difficulty: str, variant_index: int):
    # Curated static bank disabled: we now prefer AI-generated batches.
    return None


def _build_synthetic_question(subject: str, grade: str, topic: str, difficulty: str, variant_index: int):
    """Build deterministic synthetic questions to guarantee broad coverage.

    These questions are template-driven and safe to generate at scale without external APIs.
    """
    topic_title = _topic_to_title(topic)
    subject_title = _subject_to_title(subject)

    curated = _get_curated_question(subject, topic, difficulty, variant_index)
    if curated:
        options = list(curated["options"])
        correct_text = options[curated["correct_index"]]
        shift = variant_index % len(options)
        options = options[shift:] + options[:shift]
        correct_index = options.index(correct_text)

        return {
            "subject": subject,
            "grade": grade,
            "difficulty": difficulty,
            "text": curated["text"],
            "options": options,
            "correct_index": correct_index,
            "hint": curated.get("hint", f"Think about {topic_title} fundamentals."),
            "explanation": curated.get("explanation", f"This is the correct concept for {topic_title}."),
            "tags": curated.get("tags", [subject, topic, difficulty, "curated"]),
            "syllabus_topic": topic,
            "readability_level": _readability_for_difficulty(difficulty),
        }
    grade_title = grade.capitalize()

    subject_context = {
        'technology': {
            'artifact': 'software system',
            'evidence': 'runtime behavior and logs',
            'goal': 'reliable digital solution',
        },
        'engineering': {
            'artifact': 'prototype system',
            'evidence': 'constraints and test measurements',
            'goal': 'safe and efficient design',
        },
        'science': {
            'artifact': 'scientific investigation',
            'evidence': 'observations and controlled variables',
            'goal': 'evidence-based explanation',
        },
        'mathematics': {
            'artifact': 'mathematical model',
            'evidence': 'derivations and calculations',
            'goal': 'correct quantitative reasoning',
        },
        'physics': {
            'artifact': 'physical model',
            'evidence': 'units, equations, and measurements',
            'goal': 'accurate prediction of motion/energy',
        },
        'chemistry': {
            'artifact': 'chemical process',
            'evidence': 'reaction behavior and composition',
            'goal': 'balanced and valid chemical reasoning',
        },
        'biology': {
            'artifact': 'biological system',
            'evidence': 'structure-function relationships',
            'goal': 'correct life-science interpretation',
        },
    }
    context = subject_context.get(subject, {
        'artifact': 'learning task',
        'evidence': 'relevant evidence',
        'goal': 'correct understanding',
    })

    task_contexts = [
        'practice quiz',
        'class project',
        'lab activity',
        'assessment worksheet',
        'peer-review task',
        'simulation exercise',
        'problem-solving drill',
        'capstone checkpoint',
    ]
    evidence_lenses = [
        'measurable outcomes',
        'validated assumptions',
        'error analysis',
        'benchmark comparisons',
        'constraint checks',
        'data quality signals',
        'domain rules',
        'observed system behavior',
    ]
    risk_patterns = [
        'unverified assumptions',
        'data leakage',
        'overfitting to familiar cases',
        'ignoring edge conditions',
        'premature optimization',
        'incomplete validation',
        'untracked bias effects',
        'drift in input data',
    ]

    task_ctx = task_contexts[variant_index % len(task_contexts)]
    evidence_ctx = evidence_lenses[(variant_index // 2) % len(evidence_lenses)]
    risk_ctx = risk_patterns[(variant_index // 3) % len(risk_patterns)]

    if difficulty == "easy":
        easy_stems = [
            f"Which option best defines the core idea of {topic_title} in {subject_title} for {grade_title}?",
            f"What is the most accurate basic description of {topic_title} in {subject_title}?",
            f"Which statement correctly introduces {topic_title} for {grade_title} {subject_title} learners?",
            f"In a {task_ctx}, which statement best explains {topic_title} in {subject_title}?",
        ]
        stem = easy_stems[variant_index % len(easy_stems)]
        correct = f"It explains {topic_title} using a clear {context['artifact']} example, correct terminology, and {evidence_ctx}"
        distractors = [
            f"It ignores {topic_title} and lists unrelated facts",
            f"It treats {risk_ctx} as acceptable and skips key checks",
            "It replaces understanding with memorizing one answer pattern",
        ]
    elif difficulty == "medium":
        medium_stems = [
            f"A learner is applying {topic_title} in {subject_title}. What should they do first?",
            f"During a {subject_title} task on {topic_title}, what is the best first step?",
            f"To solve a practical problem in {topic_title}, which approach should begin the workflow?",
            f"In a {task_ctx} focused on {topic_title}, what is the strongest first action?",
        ]
        stem = medium_stems[variant_index % len(medium_stems)]
        correct = f"Break the task into steps and justify each step using {context['evidence']} and {evidence_ctx}"
        distractors = [
            "Pick the first option that seems familiar and skip validation",
            f"Apply an unrelated method and ignore {risk_ctx}",
            "Rely on intuition only and ignore available evidence",
        ]
    elif difficulty == "hard":
        hard_stems = [
            f"In a challenging {subject_title} problem about {topic_title}, which method is most rigorous?",
            f"For a complex {topic_title} scenario, which approach gives the strongest technical justification?",
            f"Which method is most defensible when analyzing an advanced {topic_title} task?",
            f"In a {task_ctx} with strict constraints, which {topic_title} approach is most defensible?",
        ]
        stem = hard_stems[variant_index % len(hard_stems)]
        correct = f"Formulate constraints, compare alternatives, and verify outcomes with measurable criteria and {evidence_ctx}"
        distractors = [
            "Choose the first plausible method and stop evaluation",
            f"Treat assumptions as facts and leave {risk_ctx} unchecked",
            "Optimize speed only, even when accuracy is degraded",
        ]
    else:
        expert_stems = [
            f"For advanced work in {subject_title} on {topic_title}, which strategy best improves reliability?",
            f"Which expert practice most improves repeatability and trust in {topic_title} outcomes?",
            f"When operating at advanced level in {topic_title}, what most strengthens system reliability?",
            f"In a high-stakes {task_ctx}, which strategy makes {topic_title} outcomes most robust?",
        ]
        stem = expert_stems[variant_index % len(expert_stems)]
        correct = f"Use iterative validation, error analysis, and metric-based refinement to reach {context['goal']} while monitoring {evidence_ctx}"
        distractors = [
            "Freeze the first result and avoid reevaluation",
            "Optimize appearance only while ignoring measured outcomes",
            f"Skip documentation and leave {risk_ctx} unmanaged",
        ]

    options = [correct] + distractors
    shift = variant_index % len(options)
    options = options[shift:] + options[:shift]
    correct_index = options.index(correct)

    return {
        "subject": subject,
        "grade": grade,
        "difficulty": difficulty,
        "text": stem,
        "options": options,
        "correct_index": correct_index,
        "hint": f"Focus on first principles of {topic_title} before selecting the method.",
        "explanation": f"The best answer applies disciplined reasoning to {topic_title} in {subject_title}.",
        "tags": [subject, topic, difficulty, "synthetic", "coverage"],
        "syllabus_topic": topic,
        "readability_level": _readability_for_difficulty(difficulty),
    }


def _normalize_generated_item(item, subject: str, grade: str, topic: str, difficulty: str):
    if not isinstance(item, dict):
        return None
    text = str(item.get("text") or item.get("question") or "").strip()
    options = item.get("options") if isinstance(item.get("options"), list) else []
    options = [str(o).strip() for o in options if str(o).strip()]
    if not text or len(options) < 2:
        return None
    try:
        correct_index = int(item.get("correct_index", 0))
    except (TypeError, ValueError):
        correct_index = 0
    if correct_index < 0 or correct_index >= len(options):
        correct_index = 0
    return {
        "subject": subject,
        "grade": grade,
        "difficulty": difficulty,
        "text": text,
        "options": options[:4],
        "correct_index": correct_index,
        "hint": str(item.get("hint") or f"Use core {topic} reasoning.").strip(),
        "explanation": str(item.get("explanation") or f"This answer best matches {topic} concepts in {subject}.").strip(),
        "tags": [subject, topic, difficulty, "ai_seed"],
        "syllabus_topic": topic,
        "readability_level": _readability_for_difficulty(difficulty),
    }


def build_large_question_bank(manifest: dict, per_topic: int = 12, *, skip_ai: bool = False):
    """Create a large deterministic bank for all subject/grade/topic combinations.

    When ``skip_ai`` is True (recommended for CI / HF strict training), skip HTTP calls to
    the topic MCQ service and use synthetic stems only. Otherwise each (topic × difficulty)
    batch may block on ``generate_topic_mcqs`` (long timeouts when the service is down).
    """
    if not manifest:
        return []

    def _iter_manifest_topic_groups():
        for subject, payload in manifest.items():
            if isinstance(payload, dict):
                for grade, topics in payload.items():
                    if isinstance(topics, list) and topics:
                        yield subject, str(grade), topics
            elif isinstance(payload, list) and payload:
                # Backward-compatible manifest shape: subject -> [topics]
                yield subject, "high_school", payload

    generated = []
    combo_idx = 0
    for subject, grade, topics in _iter_manifest_topic_groups():
        difficulties = ["easy", "medium", "hard"]
        if str(grade).strip().lower() == "college":
            difficulties.append("expert")

        for topic in topics:
            for difficulty in difficulties:
                ai_seed_batch = []
                if not skip_ai:
                    service = generate_topic_mcqs(
                        subject=subject,
                        grade=grade,
                        difficulty=difficulty,
                        topic=topic,
                        count=per_topic,
                        generation_mode="standard",
                        seed=random.randint(1, 2_147_483_647),
                    )
                    if service.get("ok"):
                        for item in (service.get("questions") or []):
                            normalized = _normalize_generated_item(item, subject, grade, topic, difficulty)
                            if normalized:
                                ai_seed_batch.append(normalized)
                            if len(ai_seed_batch) >= per_topic:
                                break

                generated.extend(ai_seed_batch)
                for idx in range(max(0, per_topic - len(ai_seed_batch))):
                    generated.append(
                        _build_synthetic_question(subject, grade, topic, difficulty, idx)
                    )
                combo_idx += 1
                if combo_idx % 25 == 0:
                    print(
                        f"[seed_questions] coverage progress: combos={combo_idx} generated={len(generated)} "
                        f"(subject={subject!r} topic={topic!r} diff={difficulty})",
                        flush=True,
                    )

    return generated


def _split_count_across_difficulties(total: int, include_expert: bool):
    levels = ["easy", "medium", "hard"] + (["expert"] if include_expert else [])
    base = total // len(levels)
    remainder = total % len(levels)
    plan = {level: base for level in levels}
    for idx in range(remainder):
        plan[levels[idx]] += 1
    return plan


def build_strict_stem_bank(manifest: dict, per_subtopic: int = 20, *, skip_ai: bool = False):
    """Create exactly `per_subtopic` questions per (subject, grade, topic)."""
    if not manifest:
        return []

    def _iter_manifest_topic_groups():
        for subject, payload in manifest.items():
            if isinstance(payload, dict):
                for grade, topics in payload.items():
                    if isinstance(topics, list) and topics:
                        yield subject, str(grade), topics
            elif isinstance(payload, list) and payload:
                # Backward-compatible manifest shape: subject -> [topics]
                yield subject, "high_school", payload

    generated = []
    for subject, grade, topics in _iter_manifest_topic_groups():
        include_expert = str(grade).strip().lower() == "college"
        distribution = _split_count_across_difficulties(per_subtopic, include_expert)

        for topic in topics:
            variant = 0
            for difficulty, qty in distribution.items():
                ai_seed_batch = []
                if not skip_ai:
                    service = generate_topic_mcqs(
                        subject=subject,
                        grade=grade,
                        difficulty=difficulty,
                        topic=topic,
                        count=qty,
                        generation_mode="standard",
                        seed=random.randint(1, 2_147_483_647),
                    )
                    if service.get("ok"):
                        for item in (service.get("questions") or []):
                            normalized = _normalize_generated_item(item, subject, grade, topic, difficulty)
                            if normalized:
                                ai_seed_batch.append(normalized)
                            if len(ai_seed_batch) >= qty:
                                break

                generated.extend(ai_seed_batch)
                for _ in range(max(0, qty - len(ai_seed_batch))):
                    generated.append(
                        _build_synthetic_question(subject, grade, topic, difficulty, variant)
                    )
                    variant += 1

    return generated


def seed_database(
    reset_questions=False,
    augment_large=False,
    per_topic=12,
    strict_stem_rebuild=False,
    per_subtopic=20,
    skip_ai_augment=False,
):
    """Populate database with questions."""
    import json, os

    def _infer_syllabus_topic(sub: str | None, gr: str | None) -> str | None:
        if not manifest or not sub:
            return None

        subject_keys = [str(sub).strip(), str(sub).strip().lower(), str(sub).strip().title()]
        subject_payload = None
        for key in subject_keys:
            if key in manifest:
                subject_payload = manifest.get(key)
                break

        if isinstance(subject_payload, list):
            return subject_payload[0] if subject_payload else None

        if isinstance(subject_payload, dict):
            grade_keys = [str(gr).strip()] if gr is not None else []
            for key in grade_keys:
                topics = subject_payload.get(key)
                if isinstance(topics, list) and topics:
                    return topics[0]

            # Fallback to first non-empty grade/topic list.
            for topics in subject_payload.values():
                if isinstance(topics, list) and topics:
                    return topics[0]

        return None

    topics_manifest_path = os.path.join(os.path.dirname(__file__), 'data', 'syllabus_topics.json')
    manifest = None
    if os.path.exists(topics_manifest_path):
        try:
            with open(topics_manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except Exception as e:
            print('WARNING: Failed to load syllabus_topics manifest:', e)

    with app.app_context():
        # Check if questions already exist
        existing_count = Question.query.count()

        if strict_stem_rebuild and existing_count > 0 and not reset_questions:
            print("WARNING: strict STEM rebuild requires --reset-questions to avoid mixed banks.")
            return
        
        if existing_count > 0 and not reset_questions and not augment_large:
            print(f"INFO: Database already has {existing_count} questions. Skipping reseed to avoid data loss.")
            print("   Pass --reset-questions to delete/reseed, or --augment-large to append broad-coverage questions.")
            return

        if existing_count > 0 and reset_questions:
            # Clear existing questions only when explicitly requested
            Question.query.delete()
            db.session.commit()
            print("✓ Cleared existing questions (explicit reset requested)")
        
        question_bank = list(QUESTIONS)
        if strict_stem_rebuild:
            question_bank = build_strict_stem_bank(
                manifest, per_subtopic=per_subtopic, skip_ai=skip_ai_augment
            )
            print(f"INFO: Built strict STEM bank: {len(question_bank)} questions ({per_subtopic} per subject-grade-topic).", flush=True)
        elif augment_large:
            print(
                f"[seed_questions] building large bank per_topic={per_topic} skip_ai_augment={skip_ai_augment} …",
                flush=True,
            )
            generated = build_large_question_bank(
                manifest, per_topic=per_topic, skip_ai=skip_ai_augment
            )
            question_bank.extend(generated)
            print(
                f"INFO: Built {len(generated)} coverage questions ({per_topic} per topic per difficulty; "
                f"ai_calls={'off' if skip_ai_augment else 'on'}).",
                flush=True,
            )

        # Prevent duplicate inserts when augmenting existing banks.
        existing_texts = {
            row[0].strip().lower()
            for row in db.session.query(Question.text).all()
            if row[0]
        }

        # Add new questions
        added = 0
        skipped_duplicates = 0
        for q_data in question_bank:
            # if syllabus_topic missing, try to infer from manifest by subject/grade
            if manifest and not q_data.get('syllabus_topic'):
                inferred_topic = _infer_syllabus_topic(q_data.get('subject'), q_data.get('grade'))
                if inferred_topic:
                    q_data['syllabus_topic'] = inferred_topic

            text_key = (q_data.get('text') or '').strip().lower()
            if not text_key:
                continue
            if text_key in existing_texts:
                if not strict_stem_rebuild:
                    skipped_duplicates += 1
                    continue

            question = Question(**q_data)
            db.session.add(question)
            existing_texts.add(text_key)
            added += 1
        
        db.session.commit()
        
        print(f"\nSuccessfully added {added} questions!")
        if skipped_duplicates:
            print(f"Skipped {skipped_duplicates} duplicate questions")
        
        # Show statistics
        print("\nQuestion Distribution:")
        for difficulty in ['easy', 'medium', 'hard', 'expert']:
            count = Question.query.filter_by(difficulty=difficulty).count()
            print(f"  {difficulty.capitalize()}: {count}")
        
        print("\nSubject Distribution:")
        all_subjects = [row[0] for row in db.session.query(Question.subject).distinct().order_by(Question.subject).all()]
        for subject in all_subjects:
            count = Question.query.filter_by(subject=subject).count()
            print(f"  {subject.capitalize()}: {count}")


if __name__ == "__main__":
    print("Seeding question database...", flush=True)
    should_reset = "--reset-questions" in sys.argv
    should_augment = "--augment-large" in sys.argv
    should_strict_rebuild = "--strict-stem-rebuild" in sys.argv
    skip_ai_augment = "--skip-ai-augment" in sys.argv
    per_topic = 12
    per_subtopic = 20
    if "--per-topic" in sys.argv:
        try:
            per_topic_index = sys.argv.index("--per-topic") + 1
            per_topic = max(1, min(int(sys.argv[per_topic_index]), 50))
        except Exception:
            print("WARNING: Invalid --per-topic value. Using default 12.")

    if "--per-subtopic" in sys.argv:
        try:
            per_subtopic_index = sys.argv.index("--per-subtopic") + 1
            per_subtopic = max(1, min(int(sys.argv[per_subtopic_index]), 100))
        except Exception:
            print("WARNING: Invalid --per-subtopic value. Using default 20.")

    seed_database(
        reset_questions=should_reset,
        augment_large=should_augment,
        per_topic=per_topic,
        strict_stem_rebuild=should_strict_rebuild,
        per_subtopic=per_subtopic,
        skip_ai_augment=skip_ai_augment,
    )
