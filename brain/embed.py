#!/usr/bin/env python3
"""
Minimal embedding module — wraps sentence-transformers for the brain graph.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string. Returns float32 numpy array."""
    model = _get_model()
    return model.encode(text, convert_to_numpy=True).astype(np.float32)


def embed_texts(texts: list) -> np.ndarray:
    """Batch embed multiple texts. Returns 2D float32 numpy array (N x dim)."""
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
