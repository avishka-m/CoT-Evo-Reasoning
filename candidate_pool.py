"""
candidate_pool.py
-----------------
Manages the population of CoT reasoning candidates for a single question.

Each candidate is a dict:
{
  "model":         str,          # origin thinking pattern
  "steps":         List[str],    # reasoning steps
  "text":          str,          # flattened steps string
  "embedding":     np.ndarray,   # behavioral embedding vector
  "fitness_score": float,        # 0.0 – 1.0 from fitness_evaluator
  "criteria":      dict,         # full 5-criterion breakdown
  "generation":    int,          # which GA generation created this
}
"""

from typing import List, Dict, Any, Optional
import numpy as np


class CandidatePool:
    """Manages a population of CoT reasoning candidates for one QA record."""

    def __init__(self):
        self._pool: List[Dict[str, Any]] = []

    # ── Population Management ──────────────────────────────────────────────────

    def add(self, candidate: Dict[str, Any]) -> None:
        """Add a candidate to the pool."""
        self._pool.append(candidate)

    def add_many(self, candidates: List[Dict[str, Any]]) -> None:
        for c in candidates:
            self.add(c)

    def size(self) -> int:
        return len(self._pool)

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self._pool)

    def get_top_k(self, k: int) -> List[Dict[str, Any]]:
        """Return top-k candidates sorted by fitness_score descending."""
        scored = [c for c in self._pool if "fitness_score" in c]
        return sorted(scored, key=lambda c: c["fitness_score"], reverse=True)[:k]

    def get_best(self) -> Optional[Dict[str, Any]]:
        """Return the single best candidate by fitness_score."""
        top = self.get_top_k(1)
        return top[0] if top else None

    def clear(self) -> None:
        self._pool.clear()

    # ── Embedding Helpers ──────────────────────────────────────────────────────

    def get_embeddings(self) -> np.ndarray:
        """
        Stack all candidate embeddings into a 2D array.
        Shape: (N, D) where D = embedding dimension.
        Only includes candidates that have an embedding.
        """
        vecs = [c["embedding"] for c in self._pool if "embedding" in c]
        if not vecs:
            return np.empty((0, 0))
        return np.stack(vecs, axis=0)

    def get_fitness_scores(self) -> List[float]:
        return [c.get("fitness_score", 0.0) for c in self._pool]

    # ── Representation ────────────────────────────────────────────────────────

    def summary(self) -> str:
        if not self._pool:
            return "CandidatePool(empty)"
        scores = self.get_fitness_scores()
        return (
            f"CandidatePool(size={self.size()}, "
            f"best={max(scores):.4f}, "
            f"mean={sum(scores)/len(scores):.4f}, "
            f"worst={min(scores):.4f})"
        )
