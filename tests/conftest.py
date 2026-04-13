"""Shared fixtures for Engram tests."""

import os
import sys
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension


def _random_embedding(seed=None):
    """Return a random unit vector of EMBEDDING_DIM dimensions."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _similar_embedding(base, similarity=0.95, seed=None):
    """Return an embedding with approximate cosine similarity to base."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(EMBEDDING_DIM).astype(np.float32)
    noise /= np.linalg.norm(noise)
    # Interpolate: high similarity = mostly base
    vec = base * similarity + noise * (1 - similarity)
    vec = vec.astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


@pytest.fixture
def tmp_brain(tmp_path, monkeypatch):
    """Set up a temporary brain directory and patch brain module globals.

    Returns the tmp_path for the brain directory.
    Patches BRAIN_DIR, DB_PATH, and mocks embed_text/embed_texts.
    """
    import brain

    brain_dir = tmp_path / "brain"
    brain_dir.mkdir()
    db_path = brain_dir / "brain.db"

    monkeypatch.setattr(brain, "BRAIN_DIR", brain_dir)
    monkeypatch.setattr(brain, "DB_PATH", db_path)

    # Mock embedding function to return random vectors deterministically
    call_count = [0]

    def mock_embed_text(text):
        call_count[0] += 1
        seed = hash(text) % (2**31)
        return _random_embedding(seed)

    def mock_embed_texts(texts):
        return np.stack([mock_embed_text(t) for t in texts])

    monkeypatch.setattr(brain, "embed_text", mock_embed_text)
    monkeypatch.setattr(brain, "embed_texts", mock_embed_texts)

    # Also patch in search module
    import brain.search
    monkeypatch.setattr(brain.search, "embed_text", mock_embed_text)

    return brain_dir


@pytest.fixture
def db(tmp_brain):
    """Return an initialized DB connection using the tmp brain."""
    import brain
    conn = brain.get_db()
    yield conn
    conn.close()


@pytest.fixture
def populated_db(db):
    """A DB with sample nodes and edges for testing search/sleep."""
    import brain

    nodes = [
        ("node_alpha", "Alpha concept in ML", "concept",
         "Machine learning is a subset of artificial intelligence", 0.9, "human"),
        ("node_beta", "Beta decision on deployment", "decision",
         "We decided to deploy using Kubernetes for scalability", 0.85, "second_brain"),
        ("node_gamma", "Gamma question about pricing", "question",
         "Should we switch to usage-based pricing model", 0.7, "extract"),
        ("node_delta", "Delta insight on team velocity", "insight",
         "Team velocity has increased 40% since adopting pair programming", 0.8, "extract"),
        ("node_epsilon", "Epsilon observation on user churn", "observation",
         "User churn correlates with onboarding completion rate", 0.75, "extract"),
    ]

    for nid, label, ntype, content, conf, source in nodes:
        db.execute(
            "INSERT INTO nodes (id, label, type, content, confidence, source, origin) "
            "VALUES (?, ?, ?, ?, ?, ?, 'self')",
            (nid, label, ntype, content, conf, source)
        )
        vec = brain.embed_text(content)
        db.execute(
            "INSERT INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
            (nid, vec.astype(np.float32).tobytes(), "test-model")
        )

    # Add edges
    edges = [
        ("node_alpha", "node_beta", "enables"),
        ("node_beta", "node_gamma", "relates_to"),
        ("node_gamma", "node_delta", "relates_to"),
        ("node_alpha", "node_delta", "depends_on"),
    ]
    for from_id, to_id, rel in edges:
        db.execute(
            "INSERT INTO edges (from_id, to_id, relation, source) VALUES (?, ?, ?, 'human')",
            (from_id, to_id, rel)
        )
    db.commit()
    return db


@pytest.fixture
def fixture_dir():
    """Create and return a temp directory for test fixtures."""
    d = Path("/tmp/engram_test_fixtures")
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
