"""
fitness_evaluator.py
--------------------
High-precision medical reasoning evaluator.
Uses floating-point satisfaction levels (0.0-1.0) and efficiency penalties
to break fitness plateaus and prevent premature convergence.
"""

import json
import time
import math
from typing import Dict, Any, List
from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    FITNESS_CRITERIA,
    LEVEL_WEIGHTS,
)

# ── DeepSeek client (OpenAI-compatible) ───────────────────────────────────────
_client: OpenAI = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
    return _client


# ── Precision Evaluation Prompt ───────────────────────────────────────────────
EVAL_SYSTEM_PROMPT = """You are a senior medical evidence validator and clinical reasoning expert.
Evaluate the medical reasoning chain based on established physician criteria.

CRITICAL CRITERIA (Score 0-5 each):
1. Safety Utility: Does it detect life-threatening conditions early?
2. Clinical Validity: Is pathophysiology correct?
3. Clinical Relevance: Does it discriminate between diagnoses?
4. Uncertainty Impact: Does it reduce ambiguity?
5. Guideline Concordance: Is it consistent with clinical practice?

EVALUATION RULES:
- For each step, provide a 0-5 score for each criterion.
- For each step, determine the highest Evidence Level (A/B/C/D).
- Level A (Guidelines/Meta-analysis), Level B (Diagnostic studies), Level C (Expert opinion), Level D (No evidence).
- Be objective and medically rigorous.

Respond ONLY with a valid JSON object."""

EVAL_USER_TEMPLATE = """
--- QUESTION ---
{question}

--- EXPECTED ANSWER ---
{answer}

--- REASONING CHAIN ---
{reasoning}

Respond ONLY with this JSON structure:
{{
  "per_step": [
    {{
      "step_index": int,
      "clinical_scores": {{
        "safety_utility": float,
        "clinical_validity": float,
        "clinical_relevance": float,
        "uncertainty_impact": float,
        "guideline_concordance": float
      }},
      "evidence_level": "A/B/C/D",
      "justification": "..."
    }}
  ],
  "global_evidence_grade": "A/B/C/D"
}}
"""

def evaluate_reasoning(
    question: str,
    answer: str,
    reasoning_text: str,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> Dict[str, Any]:
    """Stage 3: Fitness Evaluation (Doctor Criteria)."""
    client = _get_client()
    prompt = EVAL_USER_TEMPLATE.format(
        question=question,
        answer=answer,
        reasoning=reasoning_text,
    )

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2500,
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)

            steps = parsed.get("per_step", [])
            processed_steps = []
            
            total_clinical_mean = 0.0
            total_evidence_score = 0.0
            
            for s in steps:
                # Robustly handle scores (LLM might return strings or even letters)
                scores_raw = s.get("clinical_scores", {})
                scores = {}
                for k, v in scores_raw.items():
                    if k in FITNESS_CRITERIA:
                        try:
                            if isinstance(v, str) and v.upper() in LEVEL_WEIGHTS:
                                scores[k] = LEVEL_WEIGHTS[v.upper()]
                            else:
                                scores[k] = float(v)
                        except (ValueError, TypeError):
                            scores[k] = 2.5 # Default to middle ground if unparseable
                
                # Clinical Criteria Mean (0-5)
                c_mean = sum(scores.values()) / 5.0 if scores else 0.0
                
                # Evidence Score (using weights from config)
                lvl = s.get("evidence_level", "D")
                e_score = LEVEL_WEIGHTS.get(lvl, 0.0)
                
                step_fitness = (c_mean * 0.6) + (e_score * 0.4)
                
                total_clinical_mean += c_mean
                total_evidence_score += e_score
                
                processed_steps.append({
                    "step_index": s.get("step_index"),
                    "clinical_scores": scores,
                    "evidence_level": lvl,
                    "score": round(step_fitness, 4),
                    "justification": s.get("justification")
                })

            n_steps = len(steps) if steps else 1
            avg_clinical = total_clinical_mean / n_steps
            avg_evidence = total_evidence_score / n_steps
            
            final_fitness = (avg_clinical * 0.6) + (avg_evidence * 0.4)

            return {
                "per_step": processed_steps,
                "clinical_criteria_mean": round(avg_clinical, 4),
                "evidence_score": round(avg_evidence, 4),
                "fitness_score": round(final_fitness, 4),
                "evidence_grade": parsed.get("global_evidence_grade", "D"),
                "raw_response": raw
            }

        except Exception as e:
            print(f"[FitnessEvaluator] Attempt {attempt} failed: {e}")
            time.sleep(retry_delay)

    return {
        "per_step": [],
        "fitness_score": 0.0,
        "evidence_grade": "D",
        "raw_response": ""
    }
