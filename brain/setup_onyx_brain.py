#!/usr/bin/env python3
"""
setup_onyx_brain.py — Seed onyx_brain.db from brain.db

Copies:
  - All nodes WHERE origin='onyx'
  - All embeddings for those node_ids
  - All edges WHERE both from_id AND to_id are onyx nodes
  - Schema for: nodes, edges, embeddings, ruminate_log, access_log

Does NOT copy: curiosity_log, dopamine_weights, introspection_log
"""

import sqlite3
import sys
from pathlib import Path

GRAPH_DIR = Path(__file__).parent
BRAIN_DB = GRAPH_DIR / "brain.db"
ONYX_DB = GRAPH_DIR / "onyx_brain.db"


def main():
    if not BRAIN_DB.exists():
        print(f"ERROR: {BRAIN_DB} not found")
        sys.exit(1)

    if ONYX_DB.exists():
        print(f"WARNING: {ONYX_DB} already exists — removing and rebuilding")
        ONYX_DB.unlink()

    src = sqlite3.connect(BRAIN_DB, timeout=120)
    src.row_factory = sqlite3.Row

    dst = sqlite3.connect(ONYX_DB, timeout=120)
    dst.execute("PRAGMA journal_mode=WAL")
    dst.execute("PRAGMA synchronous=NORMAL")

    # ── Create schema ────────────────────────────────────────────
    dst.executescript("""
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
        gc_flag INTEGER DEFAULT 0,
        decayed INTEGER DEFAULT 0,
        permanent INTEGER DEFAULT 0,
        core_memory INTEGER DEFAULT 0,
        origin TEXT DEFAULT 'onyx',
        relevance_score REAL DEFAULT 1.0,
        last_accessed_at TEXT,
        fitness_score REAL DEFAULT 0.5,
        soft_decay_flagged INTEGER DEFAULT 0
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
    dst.commit()

    # ── Copy onyx nodes ──────────────────────────────────────────
    # Get column names from source (handles schema variations)
    src_cols = [r[1] for r in src.execute("PRAGMA table_info(nodes)").fetchall()]
    dst_cols = [r[1] for r in dst.execute("PRAGMA table_info(nodes)").fetchall()]
    common_cols = [c for c in src_cols if c in dst_cols]
    cols_str = ", ".join(common_cols)
    placeholders = ", ".join("?" for _ in common_cols)

    nodes = src.execute(
        f"SELECT {cols_str} FROM nodes WHERE origin = 'onyx'"
    ).fetchall()

    onyx_ids = set()
    for row in nodes:
        vals = [row[c] for c in common_cols]
        dst.execute(f"INSERT OR IGNORE INTO nodes ({cols_str}) VALUES ({placeholders})", vals)
        onyx_ids.add(row["id"])

    dst.commit()
    print(f"Nodes copied: {len(onyx_ids)}")

    # ── Copy embeddings ──────────────────────────────────────────
    emb_count = 0
    for nid in onyx_ids:
        row = src.execute(
            "SELECT node_id, embedding, model, created_at FROM embeddings WHERE node_id = ?",
            (nid,)
        ).fetchone()
        if row:
            dst.execute(
                "INSERT OR IGNORE INTO embeddings (node_id, embedding, model, created_at) VALUES (?, ?, ?, ?)",
                (row["node_id"], row["embedding"], row["model"], row["created_at"])
            )
            emb_count += 1

    dst.commit()
    print(f"Embeddings copied: {emb_count}")

    # ── Copy onyx-onyx edges ─────────────────────────────────────
    edge_count = 0
    edges = src.execute(
        "SELECT from_id, to_id, relation, weight, note, source, created_at FROM edges"
    ).fetchall()

    for e in edges:
        if e["from_id"] in onyx_ids and e["to_id"] in onyx_ids:
            dst.execute(
                "INSERT OR IGNORE INTO edges (from_id, to_id, relation, weight, note, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (e["from_id"], e["to_id"], e["relation"], e["weight"], e["note"], e["source"], e["created_at"])
            )
            edge_count += 1

    dst.commit()
    print(f"Edges copied: {edge_count}")

    src.close()
    dst.close()

    print(f"\nonyx_brain.db created at {ONYX_DB}")
    print(f"  nodes:      {len(onyx_ids)}")
    print(f"  embeddings: {emb_count}")
    print(f"  edges:      {edge_count}")


if __name__ == "__main__":
    main()
