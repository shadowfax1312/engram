#!/usr/bin/env python3
"""
Engram — Sleep cycle: fitness scoring, core memory promotion, and GC.

Pipeline:
  1. embedding_dedup   — merge near-duplicate nodes (cosine >= 0.88)
  2. cross_link        — add relates_to edges between similar unlinked nodes
  3. compute_fitness   — structural fitness: edges + access + cross-links + seed_bonus
  4. promote_core_memories — top sqrt(N) nodes get core_memory=true
  5. decay_relevance   — log-scaled half-life decay on relevance_score
  6. soft_gc           — delete low-relevance nodes (7-day grace period)
  7. clustering        — hierarchical DBSCAN clustering + hotspot nodes

Fitness function:
  f(node) = edge_count * 1.0
           + access_count_30d * 0.5
           + cross_link_count * 0.3
           + seed_bonus (+2.0 if source IN ('human', 'second_brain'))

Run: python3 -m brain.sleep [--dry-run] [--threshold 0.05] [--full]
"""

import json
import math
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from brain import get_db


# ── Embedding helpers ────────────────────────────────────────────

def _cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _load_embeddings(conn):
    """Load all embeddings joined with node metadata."""
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
    """Merge near-duplicate nodes based on embedding cosine similarity."""
    print(f"  START embedding_dedup (threshold={threshold})", flush=True)
    data = _load_embeddings(conn)
    n = len(data)
    if n < 2:
        print(f"  embedding_dedup: 0 pairs found", flush=True)
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
                la, lb = data[i]["label"], data[j]["label"]
                if la.lower() == lb.lower():
                    continue
                if data[i]["node_id"].startswith("insight_") or \
                   data[j]["node_id"].startswith("insight_"):
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

        if dry_run:
            deleted.add(drop["node_id"])
            merge_count += 1
            continue

        drop_id, keep_id = drop["node_id"], keep["node_id"]
        try:
            for e in conn.execute(
                "SELECT id, to_id, relation FROM edges WHERE from_id = ?",
                (drop_id,)
            ).fetchall():
                if e["to_id"] == keep_id:
                    conn.execute("DELETE FROM edges WHERE id = ?", (e["id"],))
                else:
                    exists = conn.execute(
                        "SELECT 1 FROM edges WHERE from_id=? AND to_id=? AND relation=?",
                        (keep_id, e["to_id"], e["relation"])
                    ).fetchone()
                    if exists:
                        conn.execute("DELETE FROM edges WHERE id = ?", (e["id"],))
                    else:
                        conn.execute("UPDATE edges SET from_id=? WHERE id=?",
                                     (keep_id, e["id"]))

            for e in conn.execute(
                "SELECT id, from_id, relation FROM edges WHERE to_id = ?",
                (drop_id,)
            ).fetchall():
                if e["from_id"] == keep_id:
                    conn.execute("DELETE FROM edges WHERE id = ?", (e["id"],))
                else:
                    exists = conn.execute(
                        "SELECT 1 FROM edges WHERE from_id=? AND to_id=? AND relation=?",
                        (e["from_id"], keep_id, e["relation"])
                    ).fetchone()
                    if exists:
                        conn.execute("DELETE FROM edges WHERE id = ?", (e["id"],))
                    else:
                        conn.execute("UPDATE edges SET to_id=? WHERE id=?",
                                     (keep_id, e["id"]))

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

def cross_link(conn, threshold=0.68, max_new_edges=3000, dry_run=False):
    """Add relates_to edges between semantically similar but unlinked nodes."""
    print(f"  START cross_link (threshold={threshold})", flush=True)
    data = _load_embeddings(conn)
    n = len(data)
    if n < 2:
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

    candidates = []
    id_list = [d["node_id"] for d in data]
    for i in range(n):
        if id_list[i] in hub_set:
            continue
        for j in range(i + 1, n):
            if id_list[j] in hub_set:
                continue
            if (id_list[i], id_list[j]) in existing:
                continue
            if sim[i, j] >= threshold:
                candidates.append((id_list[i], id_list[j], float(sim[i, j])))

    candidates.sort(key=lambda x: -x[2])
    candidates = candidates[:max_new_edges]

    if dry_run:
        print(f"  Cross-link: would write {len(candidates)} edges (dry run)", flush=True)
        return len(candidates)

    written = 0
    for id_a, id_b, s in candidates:
        cur = conn.execute(
            "INSERT OR IGNORE INTO edges (from_id, to_id, relation, weight, note, source) "
            "VALUES (?, ?, 'relates_to', ?, ?, 'cross_link')",
            (id_a, id_b, s, f"cross_link sim={s:.3f}")
        )
        if cur.rowcount > 0:
            written += 1
    conn.commit()
    print(f"  Cross-link: wrote {written} new edges", flush=True)
    return written


# ── Fitness ──────────────────────────────────────────────────────

def compute_fitness(conn):
    """Compute structural fitness for every node.

    f(node) = edge_count + access_30d * 0.5 + cross_links * 0.3 + seed_bonus
    """
    cutoff = (datetime.now() - timedelta(days=30)).isoformat()
    rows = conn.execute("""
        SELECT
            n.id, n.source,
            COALESCE(ec.edge_count, 0) AS edge_count,
            COALESCE(ac.access_count, 0) AS access_count_30d,
            COALESCE(cl.cross_link_count, 0) AS cross_link_count
        FROM nodes n
        LEFT JOIN (
            SELECT node_id, COUNT(*) AS edge_count FROM (
                SELECT from_id AS node_id FROM edges
                UNION ALL SELECT to_id AS node_id FROM edges
            ) GROUP BY node_id
        ) ec ON ec.node_id = n.id
        LEFT JOIN (
            SELECT node_id, COUNT(*) AS access_count FROM access_log
            WHERE accessed_at >= ? GROUP BY node_id
        ) ac ON ac.node_id = n.id
        LEFT JOIN (
            SELECT node_id, COUNT(*) AS cross_link_count FROM (
                SELECT from_id AS node_id FROM edges WHERE source = 'cross_link'
                UNION ALL SELECT to_id AS node_id FROM edges WHERE source = 'cross_link'
            ) GROUP BY node_id
        ) cl ON cl.node_id = n.id
    """, (cutoff,)).fetchall()

    fitness = {}
    for r in rows:
        seed_bonus = 2.0 if r["source"] in ("human", "second_brain") else 0.0
        fitness[r["id"]] = (
            r["edge_count"] * 1.0
            + r["access_count_30d"] * 0.5
            + r["cross_link_count"] * 0.3
            + seed_bonus
        )
    print(f"  Fitness computed for {len(fitness)} nodes", flush=True)
    return fitness


# ── Core Memory Promotion ────────────────────────────────────────

def promote_core_memories(conn, fitness_scores):
    """Promote top sqrt(N) nodes to core_memory status."""
    total = len(fitness_scores)
    target_count = int(math.sqrt(total))
    ranked = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
    new_core = set(nid for nid, _ in ranked[:target_count])

    promoted, demoted = [], []
    rows = conn.execute("SELECT id, metadata FROM nodes").fetchall()
    for row in rows:
        nid = row["id"]
        try:
            meta = json.loads(row["metadata"] or "{}")
        except (json.JSONDecodeError, TypeError):
            meta = {}
        is_core = meta.get("core_memory", False)
        if nid in new_core and not is_core:
            meta["core_memory"] = True
            conn.execute("UPDATE nodes SET metadata=?, core_memory=1 WHERE id=?",
                         (json.dumps(meta), nid))
            promoted.append(nid)
        elif nid not in new_core and is_core:
            meta.pop("core_memory", None)
            conn.execute("UPDATE nodes SET metadata=?, core_memory=0 WHERE id=?",
                         (json.dumps(meta), nid))
            demoted.append(nid)
    conn.commit()
    print(f"  Core memories: target={target_count}, "
          f"promoted={len(promoted)}, demoted={len(demoted)}", flush=True)
    return promoted, demoted


# ── Relevance Decay ──────────────────────────────────────────────

def decay_relevance(conn, base_half_life=30, dry_run=False, full=False):
    """Recalculate relevance_score using log-scaled half-life.

    half_life = base * (1 + log(access_count + 1))
    score = 0.5 ^ (days_since_access / half_life)
    """
    now = datetime.now()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(nodes)").fetchall()}
    has_origin = "origin" in cols
    origin_col = "COALESCE(origin, 'self') AS origin" if has_origin else "'self' AS origin"

    query = (
        "SELECT id, last_accessed_at, updated_at, created_at, "
        "COALESCE(permanent, 0) AS permanent, "
        "COALESCE(access_count, 0) AS access_count, "
        f"COALESCE(confidence, 1.0) AS confidence, {origin_col} "
        "FROM nodes"
    )
    if not full:
        query += " ORDER BY RANDOM() LIMIT 500"
    nodes = conn.execute(query).fetchall()

    updated = 0
    for node in nodes:
        if node["permanent"] == 2:
            updated += 1
            continue

        if node["origin"] == "external":
            ac = node["access_count"]
            score = 0.05 if ac == 0 else min(1.0, 0.3 + (ac * 0.07))
            if not dry_run:
                conn.execute("UPDATE nodes SET relevance_score=? WHERE id=?",
                             (score, node["id"]))
            updated += 1
            continue

        ts_str = node["last_accessed_at"] or node["updated_at"] or node["created_at"]
        try:
            ts = datetime.fromisoformat(ts_str)
        except (TypeError, ValueError):
            ts = now
        days_since = max(0, (now - ts).total_seconds() / 86400)
        hl = base_half_life * (1 + math.log(max(1, node["access_count"]) + 1))
        score = max(0.01, 0.5 ** (days_since / hl))
        if not dry_run:
            conn.execute("UPDATE nodes SET relevance_score=? WHERE id=?",
                         (score, node["id"]))
        updated += 1

    if not dry_run:
        conn.commit()
    mode = "full" if full else f"sampled {len(nodes)}"
    print(f"  Decay: updated relevance_score for {updated} nodes ({mode})", flush=True)
    return updated


# ── Soft GC ──────────────────────────────────────────────────────

def soft_gc(conn, threshold=0.05, dry_run=False):
    """Delete nodes with relevance_score below threshold.

    Hard protections (never deleted):
      - source IN ('human', 'second_brain')
      - type IN ('person', 'org', 'decision', 'event')
      - created_at within last 7 days (grace period)
    """
    age_cutoff = (datetime.now() - timedelta(days=7)).isoformat()
    ext_age_cutoff = (datetime.now() - timedelta(days=90)).isoformat()

    candidates = conn.execute("""
        SELECT id, label, relevance_score FROM nodes
        WHERE relevance_score < ?
          AND created_at < ?
          AND source NOT IN ('human', 'second_brain')
          AND type NOT IN ('person', 'org', 'decision', 'event')
          AND (permanent IS NULL OR permanent < 2)
          AND NOT (COALESCE(origin, 'self') = 'external' AND COALESCE(access_count, 0) > 0)
          AND NOT (COALESCE(origin, 'self') = 'external' AND created_at >= ?)
    """, (threshold, age_cutoff, ext_age_cutoff)).fetchall()

    print(f"  Soft GC candidates: {len(candidates)} below threshold {threshold}", flush=True)

    if dry_run or not candidates:
        return len(candidates)

    ids = [r["id"] for r in candidates]
    ph = ",".join("?" for _ in ids)
    conn.execute(f"DELETE FROM edges WHERE from_id IN ({ph}) OR to_id IN ({ph})",
                 ids + ids)
    conn.execute(f"DELETE FROM embeddings WHERE node_id IN ({ph})", ids)
    conn.execute(f"DELETE FROM nodes WHERE id IN ({ph})", ids)
    conn.commit()
    print(f"  Soft GC deleted {len(candidates)} nodes", flush=True)
    return len(candidates)


# ── Main Sleep Entry Point ───────────────────────────────────────

def run_sleep(conn, dry_run=False, threshold=0.05, full=False):
    """Run the full sleep cycle. Returns stats dict."""
    print("=" * 60, flush=True)
    print("  SLEEP CYCLE" + (" [DRY RUN]" if dry_run else ""), flush=True)
    print("=" * 60, flush=True)

    node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    print(f"  Graph: {node_count} nodes, {edge_count} edges", flush=True)

    dedup_count = embedding_dedup(conn, dry_run=dry_run)
    cross_link_count = cross_link(conn, dry_run=dry_run)
    fitness_scores = compute_fitness(conn)
    promoted, demoted = promote_core_memories(conn, fitness_scores)
    decay_relevance(conn, dry_run=dry_run, full=full)
    gc_count = soft_gc(conn, threshold=threshold, dry_run=dry_run)

    stats = {
        "dedup_merged": dedup_count,
        "cross_linked": cross_link_count,
        "nodes_scored": len(fitness_scores),
        "promoted": len(promoted),
        "demoted": len(demoted),
        "gc_deleted": gc_count,
        "dry_run": dry_run,
    }
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Sleep complete: {dedup_count} deduped, {cross_link_count} cross-linked, "
          f"{len(promoted)} promoted, {gc_count} GC'd", flush=True)
    print(f"{'=' * 60}", flush=True)
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sleep cycle")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    conn = get_db()
    run_sleep(conn, dry_run=args.dry_run, threshold=args.threshold, full=args.full)
    conn.close()
