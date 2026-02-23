"""
novelty_selector.py
-------------------
Implements Novelty-Driven Candidate Selection (panel b of the diagram):

  N(t) = Novelty Score  — avg distance to K nearest neighbors in embedding space
  L(t) = Local Competition Score — fraction of K neighbors outperformed on fitness
  V(t) = (N(t), L(t))  — Pareto-based selection vector

Higher N(t) → more novel/diverse reasoning
Higher L(t) → locally competitive (good fitness among neighbors)
Pareto front selection keeps candidates that are good on BOTH objectives.
"""

import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Any, Tuple

from config import GA_KNN_K


def compute_novelty_scores(
    embeddings: np.ndarray,
    fitness_scores: List[float],
    k: int = GA_KNN_K,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute N(t) and L(t) for all candidates in the pool.

    Args:
        embeddings:    (N, D) array of behavioral embeddings
        fitness_scores: list of N fitness values
        k:             number of nearest neighbors

    Returns:
        novelty_scores:   (N,) array — N(t) for each candidate
        local_comp_scores:(N,) array — L(t) for each candidate
    """
    n = len(embeddings)
    if n <= 1:
        return np.ones(n), np.ones(n)

    k_actual = min(k, n - 1)  # can't have more neighbors than pool size - 1
    fitness_arr = np.array(fitness_scores)

    nbrs = NearestNeighbors(n_neighbors=k_actual + 1, metric="cosine")
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Exclude self (index 0 is always the point itself)
    neighbor_distances = distances[:, 1:]   # shape (N, k)
    neighbor_indices   = indices[:, 1:]     # shape (N, k)

    # N(t): average distance to K nearest neighbors
    novelty_scores = neighbor_distances.mean(axis=1)

    # L(t): fraction of K neighbors outperformed on fitness
    local_comp_scores = np.zeros(n)
    for i in range(n):
        my_fitness = fitness_arr[i]
        neighbor_fitnesses = fitness_arr[neighbor_indices[i]]
        local_comp_scores[i] = np.mean(my_fitness > neighbor_fitnesses)

    return novelty_scores, local_comp_scores


def pareto_front(
    novelty_scores: np.ndarray,
    local_comp_scores: np.ndarray,
) -> List[int]:
    """
    Find indices of candidates on the Pareto front of (N(t), L(t)).
    A candidate is Pareto-dominant if no other candidate is better
    on BOTH objectives simultaneously.

    Returns:
        List of indices on the Pareto front.
    """
    n = len(novelty_scores)
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is >= on both and strictly > on at least one
            if (
                novelty_scores[j]    >= novelty_scores[i]
                and local_comp_scores[j] >= local_comp_scores[i]
                and (
                    novelty_scores[j]    > novelty_scores[i]
                    or local_comp_scores[j] > local_comp_scores[i]
                )
            ):
                dominated[i] = True
                break

    front_indices = [i for i in range(n) if not dominated[i]]
    return front_indices if front_indices else list(range(n))


def select_parents(
    candidates: List[Dict[str, Any]],
    embeddings: np.ndarray,
    n_parents: int = 2,
) -> List[Dict[str, Any]]:
    """
    Selection Stage:
    1. Compute Novelty Score N(t) (avg distance to KNN)
    2. Compute Selection Score V(t) = 0.7*Fitness + 0.3*Novelty
    3. Return top candidates based on V(t)
    """
    if len(candidates) <= n_parents:
        return candidates

    fitness_scores = [c.get("fitness_score", 0.0) for c in candidates]
    
    # Compute N(t)
    novelty_scores, _ = compute_novelty_scores(embeddings, fitness_scores)
    
    # Compute V(t)
    for i, c in enumerate(candidates):
        c["novelty_score"] = float(novelty_scores[i])
        # V(t) = (0.7 * Fitness Score) + (0.3 * Novelty Score)
        c["selection_score"] = round((0.7 * c["fitness_score"]) + (0.3 * c["novelty_score"]), 4)

    # Sort by selection score
    sorted_candidates = sorted(candidates, key=lambda x: x["selection_score"], reverse=True)
    
    # Return top n_parents
    return sorted_candidates[:n_parents]
