"""
config.py
---------
Central configuration for the CoT-Evo GA Medical Reasoning System.
All sensitive keys are loaded from your existing .env file.
"""

import os
from dotenv import load_dotenv
from kaggle_secrets import UserSecretsClient

# ── Load .env ──────────────────────────────────────────────────────────────────
load_dotenv()

# ── DeepSeek API ───────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY  = user_secrets.get_secret("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = user_secrets.get_secret("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL    = user_secrets.get_secret("DEEPSEEK_MODEL", "deepseek-chat")

# ── Embedding Model ────────────────────────────────────────────────────────────
# Medical-domain fine-tuned model from HuggingFace — downloaded & cached locally on first run.
# Much better than generic models for clinical terminology, diagnoses, drug names, symptoms.
EMBEDDING_MODEL = "abhinand/MedEmbed-base-v0.1"

# ── GA Hyperparameters ─────────────────────────────────────────────────────────
GA_POPULATION_SIZE   = 5    # initial candidates per question (one per thinking model)
GA_GENERATIONS       = 5   # number of evolution rounds
GA_MUTATION_RATE     = 0.3  # probability of mutating a selected offspring
GA_CROSSOVER_RATE    = 0.7  # probability of recombination
GA_KNN_K             = 2    # K for KNN novelty computation (as in diagram)
GA_ELITE_SIZE        = 1    # top candidates carried over unchanged each generation

# ── 5 Thinking Patterns (Stage 1: CoT Generation) ─────────────────────────────
THINKING_PATTERNS = [
    "chain_of_thought",
    "backward_think",
    "tree_of_thought",
    "reflective_reasoning",
    "analogical_reasoning",
]

# ── Fitness Criteria ───────────────────────────────────────────────────────────
FITNESS_CRITERIA = [
    "safety_utility",
    "clinical_validity",
    "clinical_relevance",
    "uncertainty_impact",
    "guideline_concordance",
]

# Evidence level → numeric weight
LEVEL_WEIGHTS = {
    "A": 5.0,  # Guidelines / systematic reviews / meta-analyses
    "B": 4.0,  # Observational / diagnostic studies
    "C": 2.0,  # Narrative reviews / expert opinion
    "D": 0.0,  # No strong evidence
}

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
OUTPUT_DIR  = "output"

# ── Parallelism ─────────────────────────────────────────────────────────────────
# Controls how many things run concurrently. Tune down if you hit API rate limits.
MAX_PARALLEL_EVALS     = 5   # concurrent DeepSeek fitness-eval calls per generation
MAX_PARALLEL_QUESTIONS = 3   # concurrent question evolutions per file
MAX_PARALLEL_FILES     = 2   # concurrent files processed at once
