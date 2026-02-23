"""
knowledge_augmenter.py
----------------------
Implements Stage 2 of the CoT-Evo architecture: Knowledge Augmentation.
Validates clinical claims and annotates reasoning steps with evidence levels.
"""

import json
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

_client: OpenAI = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _client

# ── Step Annotation Prompt ──────────────────────────────────────────────────
ANNOTATE_SYSTEM_PROMPT = """You are a medical knowledge expert and evidence grader.
Your task is to validate factual claims in a medical reasoning step and annotate it with:
1. Guideline reference (if available)
2. Study type (Guideline / Meta-analysis / Diagnostic study / Expert opinion)
3. Evidence Level:
   - Level A: Guidelines / systematic reviews / meta-analyses
   - Level B: Observational / diagnostic studies
   - Level C: Narrative reviews / expert opinion
   - Level D: No strong evidence / physiological reasoning only

Respond ONLY with a JSON object."""

ANNOTATE_USER_TEMPLATE = """
Question: {question}
Expected Answer: {answer}
Reasoning Step: {step}

Validate this step and provide:
{{
  "is_factually_correct": bool,
  "guideline_reference": "string or null",
  "study_type": "Guideline / Meta-analysis / Diagnostic study / Expert opinion / None",
  "evidence_level": "A/B/C/D",
  "justification": "Brief clinical reasoning for this grade"
}}
"""

def annotate_step(step: str, question: str, answer: str, retries: int = 3) -> Dict[str, Any]:
    """Stage 2: Validate and annotate a single reasoning step with medical evidence."""
    client = _get_client()
    prompt = ANNOTATE_USER_TEMPLATE.format(question=question, answer=answer, step=step)
    
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": ANNOTATE_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"): raw = raw[4:]
            
            return json.loads(raw)
        except Exception as e:
            print(f"[KnowledgeAugmenter] Annotate attempt {attempt} failed: {e}")
            time.sleep(1.0)
            
    return {
        "is_factually_correct": True,
        "guideline_reference": None,
        "study_type": "Expert opinion",
        "evidence_level": "C",
        "justification": "Fallback due to API error."
    }

def generate_knowledge_snippets(question: str, answer: str, retries: int = 3) -> List[str]:
    """Generate high-level reflective knowledge snippets for the question."""
    client = _get_client()
    system_prompt = "You are a medical expert. Provide 3-4 key clinical pearls or guideline-based facts for this case as a JSON list of strings."
    prompt = f"Question: {question}\nAnswer: {answer}"
    
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content.strip())
            return data.get("snippets", list(data.values())[0])
        except Exception as e:
            print(f"[KnowledgeAugmenter] Snippet attempt {attempt} failed: {e}")
            time.sleep(1.0)
    return ["Standard clinical protocols apply."]

def augment_candidate_pool(record: Dict[str, Any], initial_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stage 2: Perform knowledge augmentation on initial candidates."""
    question = record["question"]
    answer = record["answer"]
    
    # 1. Generate Global Snippets
    print(f"    [Stage 2] Generating clinical pearls...")
    snippets = generate_knowledge_snippets(question, answer)
    
    augmented = []
    for c in initial_candidates:
        new_c = c.copy()
        # Initial candidates are already "Stage 1" chromosomes
        # We add snippets as "mental background"
        new_c["snippets"] = snippets
        augmented.append(new_c)
        
    return augmented
