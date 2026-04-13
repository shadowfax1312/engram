"""Schema integrity tests — verifies brain.db has no relation column, onyx_brain.db keeps it."""
import os
import sqlite3
import pytest

BRAIN_DB = os.path.expanduser("~/.openclaw/workspace/graph/brain.db")
ONYX_DB = os.path.expanduser("~/.openclaw/workspace/graph/onyx_brain.db")


def columns(db_path, table):
    conn = sqlite3.connect(db_path)
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    conn.close()
    return [row[1] for row in info]


def test_brain_edges_no_relation():
    cols = columns(BRAIN_DB, "edges")
    assert "relation" not in cols, f"relation column should be gone from brain.db edges, got: {cols}"


def test_brain_edges_required_columns():
    cols = columns(BRAIN_DB, "edges")
    for c in ("from_id", "to_id", "weight", "note", "source"):
        assert c in cols, f"Missing column {c} in brain.db edges"


def test_brain_edges_unique_constraint():
    """UNIQUE(from_id, to_id) — inserting duplicate should fail."""
    conn = sqlite3.connect(BRAIN_DB)
    # Get an existing edge pair
    row = conn.execute("SELECT from_id, to_id FROM edges LIMIT 1").fetchone()
    conn.close()
    if row is None:
        pytest.skip("No edges in brain.db")
    conn = sqlite3.connect(BRAIN_DB)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO edges (from_id, to_id) VALUES (?, ?)", (row[0], row[1])
        )
    conn.close()


def test_brain_nodes_required_columns():
    cols = columns(BRAIN_DB, "nodes")
    for c in ("id", "label", "content", "type", "origin"):
        assert c in cols, f"Missing column {c} in brain.db nodes"


def test_onyx_brain_edges_has_relation():
    if not os.path.exists(ONYX_DB):
        pytest.skip("onyx_brain.db not found")
    cols = columns(ONYX_DB, "edges")
    assert "relation" in cols, "onyx_brain.db edges should still have relation column"
