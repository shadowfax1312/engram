#!/usr/bin/env python3
"""
Engram — SQLite-backed relational knowledge graph.

Nodes = entities (people, theses, deals, decisions, questions, concepts)
Edges = relationships (enables, depends_on, tensions_with, relates_to, etc.)
Embeddings = cached vector representations for semantic search.

All paths are resolved via the BRAIN_DIR environment variable.
"""

import os
import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────
BRAIN_DIR = Path(os.environ.get("BRAIN_DIR", Path.home() / ".engram"))
DB_PATH = BRAIN_DIR / "brain.db"

# Embedding config — override via env vars
EMBEDDING_MODEL = os.environ.get("ENGRAM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_embed_model = None


def _get_embed_model():
    """Lazy-load the sentence-transformers model."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string. Returns float32 numpy array."""
    model = _get_embed_model()
    return model.encode(text, convert_to_numpy=True).astype(np.float32)


def embed_texts(texts: list) -> np.ndarray:
    """Batch embed multiple texts. Returns 2D float32 numpy array (N x dim)."""
    model = _get_embed_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def get_db():
    """Get a connection to the brain database with WAL mode and migrations."""
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=120)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Idempotent migrations — safe to run on every startup
    for col, defn in [
        ("relevance_score", "REAL DEFAULT 1.0"),
        ("last_accessed_at", "TEXT"),
        ("access_count", "INTEGER DEFAULT 0"),
        ("gc_flag", "INTEGER DEFAULT 0"),
        ("decayed", "INTEGER DEFAULT 0"),
        ("permanent", "INTEGER DEFAULT 0"),
        ("core_memory", "INTEGER DEFAULT 0"),
        ("origin", "TEXT DEFAULT 'self'"),
        ("session_id", "TEXT"),
        ("origin_date", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE nodes ADD COLUMN {col} {defn}")
            conn.commit()
        except sqlite3.OperationalError:
            pass

    # Auto-create schema if tables don't exist yet
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if 'nodes' not in tables:
        c = conn.cursor()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (
            id          TEXT PRIMARY KEY,
            label       TEXT NOT NULL,
            type        TEXT NOT NULL,
            content     TEXT,
            confidence  REAL DEFAULT 1.0,
            source      TEXT,
            session_id  TEXT,
            origin_date TEXT,
            created_at  TEXT DEFAULT (datetime('now')),
            updated_at  TEXT DEFAULT (datetime('now')),
            metadata    TEXT DEFAULT '{}',
            access_count INTEGER DEFAULT 0,
            gc_flag     INTEGER DEFAULT 0,
            decayed     INTEGER DEFAULT 0,
            permanent   INTEGER DEFAULT 0,
            core_memory INTEGER DEFAULT 0,
            origin      TEXT DEFAULT 'self',
            relevance_score REAL DEFAULT 1.0,
            last_accessed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS edges (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            from_id     TEXT NOT NULL REFERENCES nodes(id),
            to_id       TEXT NOT NULL REFERENCES nodes(id),
            relation    TEXT NOT NULL,
            weight      REAL DEFAULT 1.0,
            note        TEXT,
            source      TEXT DEFAULT 'human',
            created_at  TEXT DEFAULT (datetime('now')),
            UNIQUE(from_id, to_id, relation)
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            node_id     TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
            embedding   BLOB NOT NULL,
            model       TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
            created_at  TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS access_log (
            id          INTEGER PRIMARY KEY,
            node_id     TEXT,
            accessed_at TEXT DEFAULT (datetime('now')),
            source      TEXT
        );
        CREATE TABLE IF NOT EXISTS ruminate_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at      TEXT DEFAULT (datetime('now')),
            insight     TEXT NOT NULL,
            nodes_involved TEXT,
            confidence  REAL DEFAULT 0.7,
            promoted    INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
        CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_id);
        CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
        CREATE INDEX IF NOT EXISTS idx_nodes_origin_date ON nodes(origin_date);
        CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);
        CREATE INDEX IF NOT EXISTS idx_access_log_node ON access_log(node_id);
        """)
        conn.commit()
    return conn


def init_schema():
    """Initialize the database schema. Idempotent."""
    conn = get_db()
    conn.close()
    print(f"Schema initialized at {DB_PATH}")


def log_access(node_ids: list, source: str, conn=None):
    """Append access log entries. Increments access_count + last_accessed_at."""
    try:
        own_conn = conn is None
        if own_conn:
            conn = get_db()
        now = datetime.now().isoformat()
        conn.executemany(
            "INSERT INTO access_log (node_id, accessed_at, source) VALUES (?, ?, ?)",
            [(nid, now, source) for nid in node_ids]
        )
        conn.executemany(
            "UPDATE nodes SET access_count = COALESCE(access_count, 0) + 1, "
            "last_accessed_at = ? WHERE id = ?",
            [(now, nid) for nid in node_ids]
        )
        if own_conn:
            conn.commit()
            conn.close()
    except Exception:
        pass


def add_node(id, label, type, content=None, confidence=1.0, source='human',
             metadata=None, origin='self'):
    """Insert or update a node. Rejects garbage labels. Auto-embeds content."""
    if not label or label.strip() in ('', '...', 'null', 'None', 'undefined'):
        return False
    if len(label.strip()) < 3:
        return False

    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO nodes (id, label, type, content, confidence, source, metadata, origin)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            label=excluded.label,
            content=excluded.content,
            confidence=excluded.confidence,
            updated_at=datetime('now'),
            metadata=excluded.metadata,
            origin=excluded.origin
    """, (id, label, type, content, confidence, source,
          json.dumps(metadata or {}), origin))

    if content:
        embedding = embed_text(content)
        c.execute(
            "INSERT OR REPLACE INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
            (id, embedding.astype(np.float32).tobytes(), EMBEDDING_MODEL)
        )
    conn.commit()
    conn.close()


def add_edge(from_id, to_id, relation, note=None, weight=1.0, source='human'):
    """Insert an edge. Silently ignores duplicates."""
    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO edges (from_id, to_id, relation, note, weight, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (from_id, to_id, relation, note, weight, source))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()


def stats():
    """Print graph statistics to stdout."""
    conn = get_db()
    c = conn.cursor()
    nodes = c.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edges = c.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    types = c.execute(
        "SELECT type, COUNT(*) as n FROM nodes GROUP BY type ORDER BY n DESC"
    ).fetchall()
    relations = c.execute(
        "SELECT relation, COUNT(*) as n FROM edges GROUP BY relation ORDER BY n DESC"
    ).fetchall()
    conn.close()
    print(f"\nGraph Stats")
    print(f"   Nodes: {nodes} | Edges: {edges}")
    print(f"\n   Node types:")
    for t in types:
        print(f"     {t['type']}: {t['n']}")
    print(f"\n   Edge relations:")
    for r in relations:
        print(f"     {r['relation']}: {r['n']}")


def neighbors(node_id, depth=1):
    """Return immediate neighbors of a node."""
    conn = get_db()
    results = conn.execute("""
        SELECT n.id, n.label, n.type, e.relation, e.note
        FROM edges e
        JOIN nodes n ON (e.to_id = n.id OR e.from_id = n.id)
        WHERE (e.from_id = ? OR e.to_id = ?) AND n.id != ?
    """, (node_id, node_id, node_id)).fetchall()
    conn.close()
    return results


if __name__ == "__main__":
    init_schema()
    print("Engram graph engine ready.")
