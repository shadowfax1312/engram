#!/usr/bin/env python3
"""
Onyx Second Brain — Graph Engine
SQLite-backed relational knowledge graph.
Nodes = entities (people, theses, deals, decisions, questions)
Edges = relationships (enables, depends_on, tensions_with, relates_to, requires, resolved_by, led_by, etc.)
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "brain.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=120)
    conn.row_factory = sqlite3.Row
    # WAL mode for safe concurrent reads/writes
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    # Idempotent migrations — safe to run on every startup
    try:
        conn.execute("ALTER TABLE nodes ADD COLUMN relevance_score REAL DEFAULT 1.0")
    except sqlite3.OperationalError:
        pass  # column already exists
    try:
        conn.execute("ALTER TABLE nodes ADD COLUMN last_accessed_at TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    try:
        conn.execute("ALTER TABLE nodes ADD COLUMN access_count INTEGER DEFAULT 0")
        conn.commit()
        print("  + access_count column added")
    except Exception:
        pass  # Already exists
    for col, defn in [
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
            pass  # column already exists
    # Auto-create schema if tables don't exist yet (inline to avoid recursion)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
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
            gc_flag INTEGER DEFAULT 0,
            decayed INTEGER DEFAULT 0,
            permanent INTEGER DEFAULT 0,
            core_memory INTEGER DEFAULT 0,
            origin TEXT DEFAULT 'self',
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
    conn = get_db()
    c = conn.cursor()

    c.executescript("""
    CREATE TABLE IF NOT EXISTS nodes (
        id          TEXT PRIMARY KEY,
        label       TEXT NOT NULL,
        type        TEXT NOT NULL,  -- thesis, person, deal, decision, question, network, concept, event, org
        content     TEXT,           -- description / notes
        confidence  REAL DEFAULT 1.0,  -- 0.0-1.0 (machine-generated hypotheses < 1.0)
        source      TEXT,           -- 'human', 'ruminate', 'knowledge_base', 'gpt_export', 'whatsapp'
        session_id  TEXT,           -- original conversation/session ID from source
        origin_date TEXT,           -- ISO timestamp from source (when this knowledge was created)
        created_at  TEXT DEFAULT (datetime('now')),
        updated_at  TEXT DEFAULT (datetime('now')),
        metadata    TEXT DEFAULT '{}',  -- JSON blob for extra fields
        access_count INTEGER DEFAULT 0,
        gc_flag INTEGER DEFAULT 0,
        decayed INTEGER DEFAULT 0,
        permanent INTEGER DEFAULT 0,
        core_memory INTEGER DEFAULT 0,
        origin TEXT DEFAULT 'self'
    );

    CREATE TABLE IF NOT EXISTS edges (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        from_id     TEXT NOT NULL REFERENCES nodes(id),
        to_id       TEXT NOT NULL REFERENCES nodes(id),
        relation    TEXT NOT NULL,  -- enables, depends_on, tensions_with, relates_to, requires, resolved_by, led_by, opposes, supports, unknown
        weight      REAL DEFAULT 1.0,
        note        TEXT,
        source      TEXT DEFAULT 'human',
        created_at  TEXT DEFAULT (datetime('now')),
        UNIQUE(from_id, to_id, relation)
    );

    CREATE TABLE IF NOT EXISTS ruminate_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_at      TEXT DEFAULT (datetime('now')),
        insight     TEXT NOT NULL,
        nodes_involved TEXT,   -- JSON array of node IDs
        confidence  REAL DEFAULT 0.7,
        promoted    INTEGER DEFAULT 0  -- 1 if promoted to a node
    );

    CREATE TABLE IF NOT EXISTS embeddings (
        node_id     TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
        embedding   BLOB NOT NULL,      -- numpy float32 array, tobytes()
        model       TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
        created_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
    CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_id);
    CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
    CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);

    CREATE TABLE IF NOT EXISTS access_log (
        id          INTEGER PRIMARY KEY,
        node_id     TEXT,
        accessed_at TEXT DEFAULT (datetime('now')),
        source      TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_access_log_node ON access_log(node_id);
    """)

    conn.commit()
    conn.close()
    print(f"✅ Schema initialized at {DB_PATH}")


def log_access(node_ids: list, source: str, conn=None):
    """Append access log entries. Silent — never raises."""
    try:
        own_conn = conn is None
        if own_conn:
            conn = get_db()
        now = datetime.now().isoformat()
        conn.executemany(
            "INSERT INTO access_log (node_id, accessed_at, source) VALUES (?, ?, ?)",
            [(nid, now, source) for nid in node_ids]
        )
        # Increment access_count on nodes
        conn.executemany(
            "UPDATE nodes SET access_count = COALESCE(access_count, 0) + 1, last_accessed_at = ? WHERE id = ?",
            [(now, nid) for nid in node_ids]
        )
        if own_conn:
            conn.commit()
            conn.close()
    except Exception:
        pass

def backfill_access_counts():
    """Backfill access_count on nodes from access_log table."""
    conn = get_db()
    try:
        conn.execute("""
            UPDATE nodes SET access_count = (
                SELECT COUNT(*) FROM access_log WHERE access_log.node_id = nodes.id
            )
        """)
        conn.commit()
        updated = conn.execute("SELECT COUNT(*) FROM nodes WHERE access_count > 0").fetchone()[0]
        print(f"  Backfilled access_count: {updated} nodes have access_count > 0")
    except Exception as e:
        print(f"  backfill_access_counts error: {e}")
    finally:
        conn.close()


def add_node(id, label, type, content=None, confidence=1.0, source='human', metadata=None, origin='self'):
    # Garbage filter: reject nodes with empty/placeholder labels
    if not label or label.strip() in ('', '...', 'null', 'None', 'undefined'):
        print(f"  ⚠ Rejected garbage node: {id} (empty/placeholder label)")
        return False
    if len(label.strip()) < 3:
        print(f"  ⚠ Rejected garbage node: {id} (label too short: '{label}')")
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
    """, (id, label, type, content, confidence, source, json.dumps(metadata or {}), origin))
    # Auto-embed if content is present
    if content:
        from embed import embed_text, MODEL_NAME
        embedding = embed_text(content)
        c.execute(
            "INSERT OR REPLACE INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
            (id, embedding.astype(np.float32).tobytes(), MODEL_NAME)
        )
    conn.commit()
    conn.close()

def add_edge(from_id, to_id, relation, note=None, weight=1.0, source='human'):
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO edges (from_id, to_id, relation, note, weight, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (from_id, to_id, relation, note, weight, source))
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # edge already exists
    conn.close()

def stats():
    conn = get_db()
    c = conn.cursor()
    nodes = c.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edges = c.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    types = c.execute("SELECT type, COUNT(*) as n FROM nodes GROUP BY type ORDER BY n DESC").fetchall()
    relations = c.execute("SELECT relation, COUNT(*) as n FROM edges GROUP BY relation ORDER BY n DESC").fetchall()
    conn.close()
    print(f"\n📊 Graph Stats")
    print(f"   Nodes: {nodes} | Edges: {edges}")
    print(f"\n   Node types:")
    for t in types:
        print(f"     {t['type']}: {t['n']}")
    print(f"\n   Edge relations:")
    for r in relations:
        print(f"     {r['relation']}: {r['n']}")

def neighbors(node_id, depth=1):
    conn = get_db()
    c = conn.cursor()
    results = c.execute("""
        SELECT n.id, n.label, n.type, e.relation, e.note
        FROM edges e
        JOIN nodes n ON (e.to_id = n.id OR e.from_id = n.id)
        WHERE (e.from_id = ? OR e.to_id = ?) AND n.id != ?
    """, (node_id, node_id, node_id)).fetchall()
    conn.close()
    return results

if __name__ == "__main__":
    init_schema()
    print("Graph engine ready.")
