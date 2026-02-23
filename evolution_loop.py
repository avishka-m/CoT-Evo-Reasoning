"""
evolution_loop.py
-----------------
Main CoT-Evo Optimization Loop (panel d of the diagram):

  For each question:
    For each generation:
      1. Evaluate fitness  (DeepSeek: 5 criteria × A/B/C/D)
      2. Compute embeddings (sentence-transformer)
      3. Novelty & Reward Selection  (KNN N(t) + L(t) → Pareto front)
      4. Recombination + Mutation (cross-chain + Add/Delete/Innovate)
      5. Track best candidate per generation
    → Output: best evolved reasoning chain + full fitness breakdown
"""

import os
import json
import copy
import random
from typing import List, Dict, Any

from tqdm import tqdm

from config import (
    GA_GENERATIONS,
    GA_POPULATION_SIZE,
    GA_MUTATION_RATE,
    GA_CROSSOVER_RATE,
    GA_ELITE_SIZE,
    OUTPUT_DIR,
)
from data_loader import get_all_chains_for_record
from embeddings import embed_texts
from fitness_evaluator import evaluate_reasoning
from candidate_pool import CandidatePool
from novelty_selector import select_parents
from genetic_operators import recombine, apply_mutation
from knowledge_augmenter import augment_candidate_pool


def _evaluate_and_embed(
    candidates: List[Dict],
    question: str,
    answer: str,
) -> List[Dict]:
    """Evaluate fitness and compute embeddings for all candidates lacking them."""
    texts = []
    unevaluated_indices = []

    for i, c in enumerate(candidates):
        if "fitness_score" not in c:
            unevaluated_indices.append(i)
            texts.append(c["text"])

    # ── Fitness evaluation via DeepSeek ───────────────────────────────────────
    for i in unevaluated_indices:
        c = candidates[i]
        print(f"    [Eval] Evaluating steps (Doctor Criteria) for: {c['model']}")
        result = evaluate_reasoning(question, answer, c["text"])
        c["fitness_score"]           = result["fitness_score"]
        c["per_step"]                = result.get("per_step", [])
        c["clinical_criteria_mean"]  = result.get("clinical_criteria_mean", 0)
        c["evidence_score"]          = result.get("evidence_score", 0)
        c["evidence_grade"]          = result.get("evidence_grade", "D")

    # ── Behavioral embeddings ─────────────────────────────────────────────────
    all_texts = [c["text"] for c in candidates]
    embeddings = embed_texts(all_texts)
    for i, c in enumerate(candidates):
        c["embedding"] = embeddings[i]

    return candidates


def doctor_validation_gate(candidate: Dict[str, Any]) -> bool:
    """
    STAGE 6 — Final Doctor Validation Gate
    Checks all steps for safety and evidence level.
    REJECT if any step is Level D and safety-relevant.
    """
    per_step = candidate.get("per_step", [])
    if not per_step:
        return False
    """
    The first condition catches steps that are both evidence-free and safety-critical. evidence_level == "D" means the step has no real clinical evidence backing it (the lowest grade, essentially "expert opinion or nothing"). safety_utility > 2.0 means the step has a meaningful impact on patient safety. A step that affects patient safety but has no evidence behind it is dangerous, so the whole candidate is rejected immediately.
    The second condition catches steps with fundamentally poor clinical reasoning regardless of evidence level. If clinical_validity < 2.0, the step is considered clinically unsound on its own merits, and again the whole candidate is rejected.
    """
        
    for step in per_step:
        # Check for Level D + Safety-Relevant
        # Safety relevance is high if safety_utility score > 0 (it means it impacts patient safety)
        lvl = step.get("evidence_level", "D")
        clinical_scores = step.get("clinical_scores", {})
        safety_utility = clinical_scores.get("safety_utility", 0.0)
        
        if lvl == "D" and safety_utility > 2.0: # If it impacts safety significantly but has no evidence
            print(f"    [Gate] REJECTED: Step {step.get('step_index')} is Level D but safety-critical.")
            return False
            
        # Ensure minimum clinical validity for all steps
        if clinical_scores.get("clinical_validity", 0.0) < 2.0:
            print(f"    [Gate] REJECTED: Step {step.get('step_index')} has low clinical validity.")
            return False
            
    return True


def evolve_question(
    record: Dict[str, Any],
    generations: int = GA_GENERATIONS,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full CoT-Evo GA for a single QA record.
    Includes Stage 2: Knowledge Augmentation.
    """
    question   = record["question"]
    answer     = record["answer"]
    record_id  = record["id"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Evolving: {record_id}")
        print(f"  Q: {question[:80]}...")
        print(f"{'='*60}")

    # ── Stage 1: Initialize candidates ───────────────────────────────────────
    initial_candidates = get_all_chains_for_record(record)
    for c in initial_candidates:
        c["generation"] = 0
        c["origin"]     = "initial"

    # ── Stage 2: Knowledge Augmentation ──────────────────────────────────────
    if verbose:
        print(f"  [Stage 2] Augmenting pool with clinical knowledge...")
    augmented_candidates = augment_candidate_pool(record, initial_candidates)

    pool = CandidatePool()
    pool.add_many(augmented_candidates)

    history = []

    # ── GA Loop ───────────────────────────────────────────────────────────────
    for gen in range(generations):
        if verbose:
            print(f"\n  -- Generation {gen+1}/{generations} --")

        all_candidates = pool.get_all()

        # Step 1 & 2: Evaluate fitness + embed
        all_candidates = _evaluate_and_embed(all_candidates, question, answer)

        # Step 3 & 4: Novelty & Reward Selection → parents
        embeddings_arr = pool.get_embeddings()
        parents = select_parents(all_candidates, embeddings_arr, n_parents=2)

        if verbose:
            best = pool.get_best()
            print(f"    Pool: {pool.summary()}")
            if best:
                print(f"    Best so far: {best['model']} | fitness={best['fitness_score']:.4f}")

        # Track history
        best_now = sorted(all_candidates, key=lambda c: c["fitness_score"], reverse=True)[0]
        history.append({
            "generation":    gen + 1,
            "best_fitness":  best_now["fitness_score"],
            "best_model":    best_now["model"],
            "pool_size":     pool.size(),
        })

        # ── Elitism ────────────────────────────────────────────────────────────
        elite = pool.get_top_k(GA_ELITE_SIZE)

        # ── Step 5: Recombination + Mutation ──────────────────────────────────
        offspring = []

        if len(parents) >= 2 and random.random() < GA_CROSSOVER_RATE:
            child = recombine(parents[0], parents[1])
            child = apply_mutation(child, question, answer, GA_MUTATION_RATE)
            offspring.append(child)

            # Reverse recombination
            child2 = recombine(parents[1], parents[0])
            child2 = apply_mutation(child2, question, answer, GA_MUTATION_RATE)
            offspring.append(child2)
        else:
            for p in parents:
                mutant = apply_mutation(
                    copy.deepcopy(p), question, answer, mutation_rate=1.0
                )
                offspring.append(mutant)

        # ── Replace pool ──────────────────────────────────────────────────────
        new_pool = CandidatePool()
        new_pool.add_many(elite)
        new_pool.add_many(offspring)

        remaining_slots = max(0, GA_POPULATION_SIZE - new_pool.size())
        if remaining_slots > 0:
            non_elite = [c for c in all_candidates if c not in elite]
            non_elite_sorted = sorted(non_elite, key=lambda c: c.get("fitness_score", 0), reverse=True)
            new_pool.add_many(non_elite_sorted[:remaining_slots])

        pool = new_pool

    # Final selection: Get all candidates and find the best one that passes Stage 6 gate
    final_candidates = pool.get_all()
    final_candidates = _evaluate_and_embed(final_candidates, question, answer)
    sorted_final = sorted(final_candidates, key=lambda c: c["fitness_score"], reverse=True)
    
    best_candidate = None
    if verbose:
        print(f"  [Stage 6] Validating top candidates through Doctor Gate...")
        
    for cand in sorted_final:
        if doctor_validation_gate(cand):
            best_candidate = cand
            break
            
    if best_candidate is None:
        best_candidate = sorted_final[0] # Fallback if none pass (should be avoided)
        print("    [WARNING] No candidate passed Doctor Gate. Falling back to highest fitness.")

    return {
        "id":       record_id,
        "question": question,
        "answer":   answer,
        "best_reasoning": {
            "model":          best_candidate["model"],
            "steps":          best_candidate["steps"],
            "text":           best_candidate["text"],
            "fitness_score":  best_candidate["fitness_score"],
            "clinical_criteria_mean": best_candidate.get("clinical_criteria_mean", 0),
            "evidence_score": best_candidate.get("evidence_score", 0),
            "evidence_grade": best_candidate.get("evidence_grade", "D"),
            "per_step":       best_candidate.get("per_step", []),
            "generation":     best_candidate.get("generation", 0),
            "origin":         best_candidate.get("origin", "unknown"),
        },
        "evolution_history": history,
    }


def run_evolution(
    dataset: List[Dict[str, Any]],
    generations: int = GA_GENERATIONS,
    output_path: str = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run CoT-Evo GA Optimization."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "evolved_reasoning.json")

    results = []
    for record in tqdm(dataset, desc="Evolving questions"):
        result = evolve_question(record, generations=generations, verbose=verbose)
        results.append(result)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[EvolutionLoop] Done! Results saved to: {output_path}")
    print(_summary_stats(results))
    return results


def _summary_stats(results: List[Dict]) -> str:
    scores = [r["best_reasoning"]["fitness_score"] for r in results]
    if not scores:
        return "No results."
    return (
        f"\n{'='*50}\n"
        f"  EVOLUTION SUMMARY\n"
        f"  Questions evolved : {len(scores)}\n"
        f"  Best fitness      : {max(scores):.4f}\n"
        f"  Mean fitness      : {sum(scores)/len(scores):.4f}\n"
        f"  Worst fitness     : {min(scores):.4f}\n"
        f"{'='*50}"
    )
