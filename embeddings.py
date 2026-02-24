"""
embeddings.py
-------------
Behavioral embedding for CoT reasoning chains using sentence-transformers.

Model: abhinand/MedEmbed-base-v0.1  (HuggingFace, cached locally after first download)
  → Medical-domain fine-tuned embedding model, far superior for clinical text
    (diagnoses, symptoms, drug names, lab results) vs generic models.

Used for computing KNN novelty scores in the Novelty-Driven Candidate Selection stage.

Speed note: automatically uses CUDA (RTX 3050) when available; falls back to CPU.
The singleton is protected by a threading.Lock so concurrent threads can't double-load.
"""

import threading
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List
from config import EMBEDDING_MODEL

# ── Auto-detect best available device ─────────────────────────────────────────
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Singleton model + thread-safe lock
_model: SentenceTransformer = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:                         # fast path (no lock needed after init)
        with _model_lock:
            if _model is None:                 # double-checked locking
                print(f"[Embeddings] Loading model: {EMBEDDING_MODEL}  (device={DEVICE})")
                _model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
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
