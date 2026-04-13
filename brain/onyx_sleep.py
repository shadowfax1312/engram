#!/usr/bin/env python3
"""
Onyx Sleep cycle — fitness scoring, dedup, cross-link, and GC for onyx_brain.db.

Same structural logic as sleep.py but operates on the Onyx research graph.
No prompt changes needed — sleep is structural, not framed.

Run: python3 onyx_sleep.py [--dry-run] [--threshold 0.05] [--skip-crosslink]
"""

import sys
import os
import json
import math
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# ── DB path override ─────────────────────────────────────────────
DB_PATH = os.environ.get("ONYX_DB", os.path.join(os.path.dirname(__file__), "onyx_brain.db"))

sys.path.insert(0, str(Path(__file__).parent))
import init_graph
init_graph.DB_PATH = Path(DB_PATH)

from init_graph import get_db


# ── Ensure signal1 columns exist ────────────────────────────────
def _ensure_columns(conn):
    for col, defn in [
        ("fitness_score", "REAL DEFAULT 0.5"),
        ("soft_decay_flagged", "INTEGER DEFAULT 0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE nodes ADD COLUMN {col} {defn}")
            conn.commit()
        except Exception:
            pass


# ── Embedding helpers ────────────────────────────────────────────

def _cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _load_embeddings(conn):
    rows = conn.execute(
        "SELECT e.node_id, e.embedding, n.confidence, n.label "
        "FROM embeddings e JOIN nodes n ON e.node_id = n.id"
    ).fetchall()
    result = []
    for r in rows:
        vec = np.frombuffer(r["embedding"], dtype=np.float32).copy()
        if np.linalg.norm(vec) > 0:
            result.append({
                "node_id": r["node_id"],
                "vector": vec,
                "confidence": r["confidence"],
                "label": r["label"],
            })
    return result


# ── Embedding Dedup ──────────────────────────────────────────────

def embedding_dedup(conn, threshold=0.88, dry_run=False):
    print(f"  START embedding_dedup (threshold={threshold}, dry_run={dry_run})", flush=True)

    data = _load_embeddings(conn)
    n = len(data)
    print(f"  Loaded {n} embeddings", flush=True)

    if n < 2:
        print(f"  embedding_dedup: 0 pairs found, 0 nodes deleted", flush=True)
        return 0

    mat = np.stack([d["vector"] for d in data]).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = mat / norms
    sim = normed @ normed.T

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                la = data[i]["label"]
                lb = data[j]["label"]
                if la.lower() == lb.lower():
                    continue
                if data[i]["node_id"].startswith("insight_") or data[j]["node_id"].startswith("insight_"):
                    continue
                pairs.append((i, j, float(sim[i, j])))

    print(f"  Found {len(pairs)} candidate duplicate pairs", flush=True)

    deleted = set()
    merge_count = 0

    for i, j, score in sorted(pairs, key=lambda x: -x[2]):
        a, b = data[i], data[j]
        if a["node_id"] in deleted or b["node_id"] in deleted:
            continue

        if a["confidence"] > b["confidence"]:
            keep, drop = a, b
        elif b["confidence"] > a["confidence"]:
            keep, drop = b, a
        elif len(a["node_id"]) <= len(b["node_id"]):
            keep, drop = a, b
        else:
            keep, drop = b, a

        print(f"  MERGE: {drop['label']} -> {keep['label']} (sim={score:.3f})", flush=True)

        if dry_run:
            deleted.add(drop["node_id"])
            merge_count += 1
            continue

        drop_id = drop["node_id"]
        keep_id = keep["node_id"]

        try:
            edges_from = conn.execute(
                "SELECT id, to_id, relation FROM edges WHERE from_id = ?", (drop_id,)
            ).fetchall()
            for e in edges_from:
                if e["to_id"] == keep_id:
                    conn.execute("DELETE FROM edges WHERE id = ?", (e["id"],))
                else:
                    exists = conn.execute(
                        "SELECT 1 FROM edges WHERE from_id = ? AND to_id = ? AND relation = ?",
                        (keep_id, e["to_id"], e["relation"]),
                    ).fetchone()
                    if exists:
                        conn.execute("DELETE FROM edges WHERE id = ?", (e["id"],))
                    else:
                        conn.execute("UPDATE edges SET from_id = ? WHERE id = ?", (keep_id, e["id"]))

            edges_to = conn.execute(
                "SELECT id, from_id, relation FROM edges WHERE to_id = ?", (drop_id,)
            ).fetchall()
            for e in edges_to:
                if e["from_id"] == keep_id:
                    conn.execute("DELETE FROM edges WHERE id = ?", (e["id"],))
                else:
                    exists = conn.execute(
                        "SELECT 1 FROM edges WHERE from_id = ? AND to_id = ? AND relation = ?",
                        (e["from_id"], keep_id, e["relation"]),
                    ).fetchone()
                    if exists:
                        conn.execute("DELETE FROM edges WHERE id = ?", (e["id"],))
                    else:
                        conn.execute("UPDATE edges SET to_id = ? WHERE id = ?", (keep_id, e["id"]))

            conn.execute("DELETE FROM embeddings WHERE node_id = ?", (drop_id,))
            conn.execute("DELETE FROM nodes WHERE id = ?", (drop_id,))
            deleted.add(drop_id)
            merge_count += 1

        except Exception as ex:
            print(f"  ERROR merging {drop_id}: {ex}", flush=True)
            conn.rollback()
            raise

    if not dry_run:
        conn.commit()

    print(f"  END embedding_dedup: {merge_count} nodes merged", flush=True)
    return merge_count


# ── Cross-Link ───────────────────────────────────────────────────

def cross_link(conn, threshold=0.68, max_new_edges=3000, dry_run=False, skip=False):
    if skip:
        print(f"  SKIP cross_link (--skip-crosslink flag set)", flush=True)
        return 0
    print(f"  START cross_link (threshold={threshold}, max_new_edges={max_new_edges}, dry_run={dry_run})", flush=True)

    data = _load_embeddings(conn)
    n = len(data)
    print(f"  Loaded {n} embeddings (post-dedup)", flush=True)

    if n < 2:
        print(f"  Cross-link: wrote 0 new edges (0 skipped as duplicates)", flush=True)
        return 0

    mat = np.stack([d["vector"] for d in data]).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = mat / norms
    sim = normed @ normed.T

    existing = set()
    for row in conn.execute("SELECT from_id, to_id FROM edges").fetchall():
        existing.add((row["from_id"], row["to_id"]))
        existing.add((row["to_id"], row["from_id"]))

    degree = defaultdict(int)
    for row in conn.execute("SELECT from_id, to_id FROM edges").fetchall():
        degree[row["from_id"]] += 1
        degree[row["to_id"]] += 1
    hub_set = {nid for nid, d in degree.items() if d > 50}
    print(f"  Hubs (>50 edges): {len(hub_set)}", flush=True)

    id_list = [d["node_id"] for d in data]
    hub_mask = np.array([nid in hub_set for nid in id_list], dtype=bool)

    candidates = []
    for i in range(n):
        if hub_mask[i]:
            continue
        row = sim[i, i + 1:]
        row_non_hub = ~hub_mask[i + 1:]
        above = np.where((row >= threshold) & row_non_hub)[0]
        for k in above:
            j = i + 1 + int(k)
            a_id, b_id = id_list[i], id_list[j]
            if (a_id, b_id) not in existing:
                candidates.append((a_id, b_id, float(sim[i, j])))

    print(f"  Found {len(candidates)} candidate edges", flush=True)

    candidates.sort(key=lambda x: -x[2])
    candidates = candidates[:max_new_edges]

    if dry_run:
        print(f"  Cross-link: would write up to {len(candidates)} edges (dry run)", flush=True)
        return len(candidates)

    written = 0
    skipped = 0
    try:
        for idx, (id_a, id_b, s) in enumerate(candidates):
            cur = conn.execute(
                "INSERT OR IGNORE INTO edges (from_id, to_id, relation, weight, note, source) "
                "VALUES (?, ?, 'relates_to', ?, ?, 'cross_link')",
                (id_a, id_b, s, f"cross_link sim={s:.3f}"),
            )
            if cur.rowcount > 0:
                written += 1
            else:
                skipped += 1

            if (idx + 1) % 500 == 0:
                print(f"    ... {idx + 1}/{len(candidates)} processed", flush=True)

        conn.commit()
    except Exception as ex:
        print(f"  ERROR in cross_link: {ex}", flush=True)
        conn.rollback()
        raise

    print(f"  Cross-link: wrote {written} new edges ({skipped} skipped as duplicates)", flush=True)
    return written


def compute_fitness(conn):
    cutoff = (datetime.now() - timedelta(days=30)).isoformat()

    rows = conn.execute("""
        SELECT
            n.id,
            n.source,
            COALESCE(ec.edge_count, 0) AS edge_count,
            COALESCE(ac.access_count, 0) AS access_count_30d,
            COALESCE(cl.cross_link_count, 0) AS cross_link_count
        FROM nodes n
        LEFT JOIN (
            SELECT node_id, COUNT(*) AS edge_count
            FROM (
                SELECT from_id AS node_id FROM edges
                UNION ALL
                SELECT to_id AS node_id FROM edges
            )
            GROUP BY node_id
        ) ec ON ec.node_id = n.id
        LEFT JOIN (
            SELECT node_id, COUNT(*) AS access_count
            FROM access_log
            WHERE accessed_at >= ?
            GROUP BY node_id
        ) ac ON ac.node_id = n.id
        LEFT JOIN (
            SELECT node_id, COUNT(*) AS cross_link_count
            FROM (
                SELECT from_id AS node_id FROM edges WHERE source = 'cross_link'
                UNION ALL
                SELECT to_id AS node_id FROM edges WHERE source = 'cross_link'
            )
            GROUP BY node_id
        ) cl ON cl.node_id = n.id
    """, (cutoff,)).fetchall()

    fitness = {}
    for r in rows:
        seed_bonus = 2.0 if r["source"] in ("human", "second_brain") else 0.0
        score = (
            r["edge_count"] * 1.0
            + r["access_count_30d"] * 0.5
            + r["cross_link_count"] * 0.3
            + seed_bonus
        )
        fitness[r["id"]] = score

    print(f"  Fitness computed for {len(fitness)} nodes", flush=True)
    return fitness


def run_sleep(conn, dry_run=False, threshold=0.05, full=False, skip_crosslink=False):
    print("=" * 60, flush=True)
    print("  ONYX SLEEP CYCLE" + (" [DRY RUN]" if dry_run else ""), flush=True)
    print(f"  DB: {DB_PATH}", flush=True)
    print("=" * 60, flush=True)

    _ensure_columns(conn)

    node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    print(f"  Graph: {node_count} nodes, {edge_count} edges", flush=True)

    # Step 0a: Embedding dedup
    print(f"\n{'—'*50}", flush=True)
    print("  Step 0a: Embedding dedup" + (" [DRY RUN]" if dry_run else ""), flush=True)
    print(f"{'—'*50}", flush=True)
    dedup_count = embedding_dedup(conn, dry_run=dry_run)

    # Step 0b: Cross-link
    print(f"\n{'—'*50}", flush=True)
    print("  Step 0b: Cross-link similar nodes" + (" [DRY RUN]" if dry_run else ""), flush=True)
    print(f"{'—'*50}", flush=True)
    cross_link_count = cross_link(conn, dry_run=dry_run, skip=skip_crosslink)

    # Step 1: Fitness scoring
    print(f"\n{'—'*50}", flush=True)
    print("  Step 1: Compute fitness scores", flush=True)
    print(f"{'—'*50}", flush=True)
    fitness_scores = compute_fitness(conn)

    # Step 1b: Core memory promotion via sqrt(N)
    print(f"\n{'—'*50}", flush=True)
    print("  Step 1b: Core memory promotion (sqrt N)", flush=True)
    print(f"{'—'*50}", flush=True)
    from sleep import promote_core_memories
    promoted, demoted = promote_core_memories(conn, fitness_scores)

    # Step 2: Signal 1 fitness computation
    print(f"\n{'—'*50}", flush=True)
    print("  Step 2: Signal 1 fitness computation" + (" [DRY RUN]" if dry_run else ""), flush=True)
    print(f"{'—'*50}", flush=True)
    from sleep import compute_all_fitness
    compute_all_fitness(conn, dry_run=dry_run)

    # Step 3: Signal 1 GC
    print(f"\n{'—'*50}", flush=True)
    print("  Step 3: Signal 1 GC" + (" [DRY RUN]" if dry_run else ""), flush=True)
    print(f"{'—'*50}", flush=True)
    from sleep import fitness_gc_v2
    gc_count = fitness_gc_v2(conn, dry_run=dry_run)

    stats = {
        "dedup_merged": dedup_count,
        "cross_linked": cross_link_count,
        "nodes_scored": len(fitness_scores),
        "core_promoted": len(promoted),
        "core_demoted": len(demoted),
        "gc_deleted": gc_count,
        "dry_run": dry_run,
    }

    print(f"\n{'=' * 60}", flush=True)
    print(f"  Onyx sleep complete: {stats['dedup_merged']} deduped, {stats['cross_linked']} cross-linked, "
          f"{stats['gc_deleted']} GC'd", flush=True)
    print(f"{'=' * 60}", flush=True)

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Onyx sleep cycle — fitness scoring, dedup, GC")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--skip-crosslink", action="store_true")
    args = parser.parse_args()
    conn = get_db()
    run_sleep(conn, dry_run=args.dry_run, threshold=args.threshold, full=args.full, skip_crosslink=args.skip_crosslink)
    conn.close()
