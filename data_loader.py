"""
data_loader.py
--------------
Load and validate the medical QA dataset.

Expected JSON schema per record:
{
  "id": "q001",
  "question": "...",
  "answer": "...",
  "reasoning_chains": [
    {"model": "chain_of_thought",     "steps": ["Step 1: ...", ...]},
    {"model": "backward_think",       "steps": ["Step 1: ...", ...]},
    {"model": "tree_of_thought",      "steps": ["Step 1: ...", ...]},
    {"model": "reflective_reasoning", "steps": ["Step 1: ...", ...]},
    {"model": "analogical_reasoning", "steps": ["Step 1: ...", ...]}
  ]
}
"""

import json
import os
from typing import List, Dict, Any
from config import THINKING_PATTERNS


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load and validate the medical QA dataset from a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    validated = []
    for i, record in enumerate(data):
        try:
            record = _validate_record(record, i)
            validated.append(record)
        except ValueError as e:
            print(f"[DataLoader] Skipping record {i}: {e}")

    print(f"[DataLoader] Loaded {len(validated)} valid records from '{filepath}'")
    return validated


def _validate_record(record: Dict, idx: int) -> Dict:
    """Validate a single dataset record and normalize reasoning chain format."""
    required = ["id", "question", "answer", "reasoning_chains"]
    for field in required:
        if field not in record:
            raise ValueError(f"Missing required field '{field}'")

    chains = record["reasoning_chains"]
    if len(chains) != 5:
        raise ValueError(
            f"Expected 5 reasoning chains (one per thinking pattern), got {len(chains)}"
        )

    # Normalize: ensure each chain has 'model' and 'steps' keys
    for chain in chains:
        if "model" not in chain:
            raise ValueError("Each reasoning chain must have a 'model' key")
        if "steps" not in chain:
            raise ValueError("Each reasoning chain must have a 'steps' key (list of strings)")
        if not isinstance(chain["steps"], list):
            raise ValueError("'steps' must be a list of strings")

    return record


def flatten_chain(chain: Dict) -> str:
    """Convert a reasoning chain's steps list into a single readable string."""
    return "\n".join(
        f"Step {i+1}: {step}" for i, step in enumerate(chain["steps"])
    )


def get_all_chains_for_record(record: Dict) -> List[Dict]:
    """
    Returns a list of candidate dicts for a single QA record.
    Each candidate = {"model": ..., "steps": [...], "text": flat string}
    """
    candidates = []
    for chain in record["reasoning_chains"]:
        candidate = {
            "model":   chain["model"],
            "steps":   chain["steps"],
            "text":    flatten_chain(chain),
        }
        candidates.append(candidate)
    return candidates
