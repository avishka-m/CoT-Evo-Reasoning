"""
Medical QA Reasoning Chain Generator
=====================================
Generates reasoning chains for medical questions using multiple AI models:
  - deepseek-reasoner          → chain_of_thought
  - llama-3.1-8b-instant       → reflective_reasoning
  - llama-4-scout-17b          → backward_think
  - deepseek-chat              → tree_of_thought
  - llama-3.3-70b-versatile    → analogical_reasoning

APIs used: Groq (for Llama models) and DeepSeek
Both API keys are read from a .env file.

Usage:
  pip install openai groq python-dotenv
  python generate_reasoning.py
"""

import json
import os
import time
import re
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq

# ── Load environment variables ───────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in .env file")

# ── Clients ──────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="DEEPSEEK_BASE_URL"
)

# ── Model → reasoning type mapping ───────────────────────────────────────────
MODELS = {
    "chain_of_thought": {
        "provider": "deepseek",
        "model": "deepseek-reasoner",
    },
    "reflective_reasoning": {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
    },
    "backward_think": {
        "provider": "groq",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    "tree_of_thought": {
        "provider": "deepseek",
        "model": "deepseek-chat",
    },
    "analogical_reasoning": {
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
    },
}

# ── Prompts for each reasoning style ─────────────────────────────────────────
PROMPTS = {
    "chain_of_thought": """You are a medical expert. Given the clinical scenario and answer below, generate a chain-of-thought reasoning with exactly 6 clear, numbered steps that logically progress from patient presentation to diagnosis and management.

Question: {question}
Answer: {answer}

Return ONLY a JSON array of 6 strings (the steps), no other text. Example format:
["Step 1 text", "Step 2 text", "Step 3 text", "Step 4 text", "Step 5 text", "Step 6 text"]""",

    "reflective_reasoning": """You are a medical expert. Given the clinical scenario and answer below, generate a reflective reasoning process with exactly 6 steps that shows self-correction, doubt, reconsideration, and final confirmation of the diagnosis and management.

Question: {question}
Answer: {answer}

Return ONLY a JSON array of 6 strings (the steps), no other text. Example format:
["Initial impression: ...", "Reconsidering: ...", "Reflection: ...", "Confirming: ...", "Reflection on management: ...", "Final plan: ..."]""",

    "backward_think": """You are a medical expert. Given the clinical scenario and answer below, generate a backward reasoning process with exactly 6 steps — start from the known answer/management goal and work backwards to explain why the clinical findings support it.

Question: {question}
Answer: {answer}

Return ONLY a JSON array of 6 strings (the steps), no other text. Start from the treatment goal and work backwards to the presenting features.""",

    "tree_of_thought": """You are a medical expert. Given the clinical scenario and answer below, generate a tree-of-thought reasoning with exactly 6 steps that explores multiple diagnostic branches before selecting the correct one, then branches into treatment options.

Question: {question}
Answer: {answer}

Return ONLY a JSON array of 6 strings (the steps), no other text. Use format like:
["Branch 1 — ...", "Branch 1a — ...", "Branch 1b — ...", "Branch 2 — ...", "Selected branch: ...", "Treatment tree: ..."]""",

    "analogical_reasoning": """You are a medical expert. Given the clinical scenario and answer below, generate an analogical reasoning process with exactly 6 steps that uses real-world analogies, metaphors, and comparisons to explain the pathophysiology, diagnosis, and treatment.

Question: {question}
Answer: {answer}

Return ONLY a JSON array of 6 strings (the steps), no other text. Each step should include a creative analogy or metaphor that illuminates the medical concept.""",
}

# ── Helper: parse steps from model response ───────────────────────────────────
def parse_steps(text: str) -> list[str]:
    """Extract a JSON array of strings from model output, with fallback."""
    text = text.strip()

    # Try to find a JSON array in the response
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            steps = json.loads(match.group())
            if isinstance(steps, list) and len(steps) > 0:
                return [str(s) for s in steps]
        except json.JSONDecodeError:
            pass

    # Fallback: split by newline and clean up
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    lines = [re.sub(r'^[\d\.\-\*]+\s*', '', l) for l in lines]
    lines = [l for l in lines if len(l) > 10]
    return lines[:6] if lines else [text]


# ── Call model ────────────────────────────────────────────────────────────────
def call_model(reasoning_type: str, question: str, answer: str, retries: int = 3) -> list[str]:
    config = MODELS[reasoning_type]
    prompt = PROMPTS[reasoning_type].format(question=question, answer=answer)
    
    for attempt in range(retries):
        try:
            if config["provider"] == "groq":
                response = groq_client.chat.completions.create(
                    model=config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1024,
                )
                content = response.choices[0].message.content

            elif config["provider"] == "deepseek":
                response = deepseek_client.chat.completions.create(
                    model=config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1024,
                )
                content = response.choices[0].message.content

            return parse_steps(content)

        except Exception as e:
            print(f"    ⚠ Attempt {attempt+1} failed for {reasoning_type} ({config['model']}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                return [f"Error generating {reasoning_type}: {str(e)}"]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    input_file  = r"C:/Users/LENOVO/Downloads/extracted_500_questions.json"
    output_file = r"C:/Users/LENOVO/Downloads/new/medical_qa_reasoning_20.json"

    print(f"Loading questions from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    questions = all_questions[:20]
    print(f"Processing first {len(questions)} questions...\n")

    results = []

    for idx, item in enumerate(questions, 1):
        question = item["question"]
        answer   = item["answer"]

        print(f"[{idx:02d}/20] Processing: {answer}")

        reasoning_chains = []

        for reasoning_type, config in MODELS.items():
            print(f"    → {reasoning_type} ({config['model']})...", end=" ", flush=True)
            steps = call_model(reasoning_type, question, answer)
            reasoning_chains.append({
                "model": reasoning_type,
                "steps": steps
            })
            print("✓")
            time.sleep(0.5)  # small delay between calls

        results.append({
            "id": f"q{idx:03d}",
            "question": question,
            "answer": answer,
            "reasoning_chains": reasoning_chains
        })

        print()

    print(f"Saving output to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! {len(results)} questions saved to {output_file}")


if __name__ == "__main__":
    main()