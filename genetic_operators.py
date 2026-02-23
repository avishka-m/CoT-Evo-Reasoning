"""
genetic_operators.py
--------------------
Implements Reflective Recombination and Mutation (panel c of the diagram).

RECOMBINATION:
  1. Identify Binding Point  — find the most semantically similar step 
     between Target CoT and Strategy Provider chain
  2. Cross-chain Recombination — merge steps up to binding point from 
     Target + steps after binding point from Strategy Provider

MUTATION (via DeepSeek API):
  1. Add      — ask DeepSeek to insert a missing reasoning step
  2. Delete   — ask DeepSeek to remove the weakest/redundant step
  3. Innovate — ask DeepSeek to rephrase/improve a random step
"""

import random
import time
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    GA_MUTATION_RATE,
)

# ── DeepSeek client (shared singleton) ────────────────────────────────────────
_client: OpenAI = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _client


# ══════════════════════════════════════════════════════════════════════════════
# RECOMBINATION
# ══════════════════════════════════════════════════════════════════════════════

def find_binding_point(
    target_steps: List[str],
    provider_steps: List[str],
) -> int:
    """
    Identify the binding point: index in target_steps where the two chains
    are most semantically similar (simple lexical overlap as fast heuristic).
    Returns the index in target_steps (0-indexed).
    """
    if not target_steps or not provider_steps:
        return 0

    best_score = -1
    best_idx   = len(target_steps) // 2  # default to midpoint

    for i, t_step in enumerate(target_steps):
        t_words = set(t_step.lower().split())
        for p_step in provider_steps:
            p_words = set(p_step.lower().split())
            if not t_words or not p_words:
                continue
            overlap = len(t_words & p_words) / len(t_words | p_words)
            if overlap > best_score:
                best_score = overlap
                best_idx   = i

    return best_idx


def recombine(
    target: Dict[str, Any],
    provider: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Cross-chain Recombination:
      offspring = target.steps[:binding_point] + provider.steps[binding_point:]

    Args:
        target:   the primary candidate (Target CoT)
        provider: the secondary candidate (Strategy Provider)

    Returns:
        New offspring candidate dict (without embedding/fitness, to be computed)
    """
    t_steps = target["steps"]
    p_steps = provider["steps"]

    binding = find_binding_point(t_steps, p_steps)

    # Ensure we don't create empty offspring
    front = t_steps[:binding + 1]
    back  = p_steps[binding + 1:] if (binding + 1) < len(p_steps) else []

    offspring_steps = front + back if (front or back) else t_steps

    offspring_text = "\n".join(
        f"Step {i+1}: {s}" for i, s in enumerate(offspring_steps)
    )

    return {
        "model":      f"recombined({target['model']}+{provider['model']})",
        "steps":      offspring_steps,
        "text":       offspring_text,
        "generation": target.get("generation", 0) + 1,
        "origin":     "recombination",
    }


# ══════════════════════════════════════════════════════════════════════════════
# MUTATION
# ══════════════════════════════════════════════════════════════════════════════

_MUTATION_SYSTEM = """You are a senior clinical reasoning expert. Your goal is to evolve reasoning chains toward the HIGHEST level of medical evidence (Level A: Guidelines/Meta-analysis).

FORBIDDEN:
1. NEVER introduce unsupported medical claims (hallucinations).
2. NEVER remove safety-critical reasoning steps.
3. NEVER add unnecessary verbosity.

ALLOWED:
1. Add evidence-backed steps.
2. Remove redundant or weak steps.
3. Refine vague medical phrasing to be precise and guideline-aligned.

Respond ONLY with a valid JSON array of strings representing the COMPLETE updated reasoning chain."""

def _call_deepseek_mutation(prompt: str, retries: int = 2) -> Optional[List[str]]:
    """Call DeepSeek to apply a mutation with medical constraints."""
    client = _get_client()
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": _MUTATION_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.4, # Lowered for better medical accuracy
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            # Handle both {"steps": [...]} and [...]
            if isinstance(data, dict):
                steps = data.get("steps", list(data.values())[0])
            else:
                steps = data
                
            if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
                return steps
        except Exception as e:
            print(f"[GeneticOperators] Mutation attempt {attempt} failed: {e}")
            time.sleep(1.0)
    return None


def mutate_add(
    candidate: Dict[str, Any],
    question: str,
    answer: str,
) -> Dict[str, Any]:
    """Mutation: ADD a critical guideline-based missing step."""
    steps_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(candidate["steps"]))
    prompt = (
        f"Medical Question: {question}\n"
        f"Correct Answer: {answer}\n\n"
        f"Current reasoning steps:\n{steps_str}\n\n"
        "Task: This reasoning chain lacks a critical evidence-based step. "
        "IDENTIFY and INSERT a step that aligns with gold-standard medical guidelines (Level A). "
        "Return the COMPLETE updated list of steps as a JSON array of strings."
    )
    new_steps = _call_deepseek_mutation(prompt)
    return _build_mutant(candidate, new_steps, "mutate_add")


def mutate_delete(
    candidate: Dict[str, Any],
    question: str,
    answer: str,
) -> Dict[str, Any]:
    """Mutation: DELETE weak or non-evidence-based steps."""
    if len(candidate["steps"]) <= 2:
        return candidate 
    steps_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(candidate["steps"]))
    prompt = (
        f"Medical Question: {question}\n"
        f"Correct Answer: {answer}\n\n"
        f"Current reasoning steps:\n{steps_str}\n\n"
        "Task: Identify any step that is weak, redundant, or lacks strong clinical grounding and REMOVE it. "
        "Ensure the surviving chain remains logically sound. "
        "Return the remaining steps as a JSON array of strings."
    )
    new_steps = _call_deepseek_mutation(prompt)
    return _build_mutant(candidate, new_steps, "mutate_delete")


def mutate_innovate(
    candidate: Dict[str, Any],
    question: str,
    answer: str,
) -> Dict[str, Any]:
    """Mutation: INNOVATE — upgrade a step to Level A evidence."""
    if not candidate["steps"]:
        return candidate
    target_step_idx = random.randint(0, len(candidate["steps"]) - 1)
    steps_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(candidate["steps"]))
    prompt = (
        f"Medical Question: {question}\n"
        f"Correct Answer: {answer}\n\n"
        f"Current reasoning steps:\n{steps_str}\n\n"
        f"Task: Redesign Step {target_step_idx+1} to be more clinically rigorous and "
        "directly aligned with Level A evidence (systematic reviews or guidelines). "
        "Do NOT just rephrase; fundamentally improve the clinical depth. "
        "Return the COMPLETE updated list of steps as a JSON array of strings."
    )
    new_steps = _call_deepseek_mutation(prompt)
    return _build_mutant(candidate, new_steps, "mutate_innovate")


def _build_mutant(
    original: Dict[str, Any],
    new_steps: Optional[List[str]],
    mutation_type: str,
) -> Dict[str, Any]:
    """Build a mutant candidate dict from new steps (falls back to original if mutation failed)."""
    if new_steps is None:
        print(f"[GeneticOperators] {mutation_type} failed, keeping original")
        return original

    text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(new_steps))
    return {
        "model":      f"{mutation_type}({original['model']})",
        "steps":      new_steps,
        "text":       text,
        "generation": original.get("generation", 0) + 1,
        "origin":     mutation_type,
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED OPERATOR
# ══════════════════════════════════════════════════════════════════════════════

MUTATION_OPS = [mutate_add, mutate_delete, mutate_innovate]


def apply_mutation(
    candidate: Dict[str, Any],
    question: str,
    answer: str,
    mutation_rate: float = GA_MUTATION_RATE,
) -> Dict[str, Any]:
    """
    Randomly select and apply one mutation operator with probability = mutation_rate.
    Returns the (potentially) mutated candidate.
    """
    if random.random() > mutation_rate:
        return candidate  # no mutation

    op = random.choice(MUTATION_OPS)
    print(f"[GeneticOperators] Applying {op.__name__} to {candidate['model']}")
    return op(candidate, question, answer)
