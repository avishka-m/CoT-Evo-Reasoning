"""
embeddings.py
-------------
Behavioral embedding for CoT reasoning chains using sentence-transformers.

Model: abhinand/MedEmbed-base-v0.1  (HuggingFace, cached locally after first download)
  → Medical-domain fine-tuned embedding model, far superior for clinical text
    (diagnoses, symptoms, drug names, lab results) vs generic models.

Used for computing KNN novelty scores in the Novelty-Driven Candidate Selection stage.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from config import EMBEDDING_MODEL

# Singleton model — loaded once, reused across all calls
_model: SentenceTransformer = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embeddings] Loading model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of reasoning-chain strings into a 2D numpy array.
    Shape: (len(texts), embedding_dim)
    """
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings  # shape: (N, D)


def embed_single(text: str) -> np.ndarray:
    """Embed a single reasoning-chain string. Returns shape (D,)."""
    return embed_texts([text])[0]
