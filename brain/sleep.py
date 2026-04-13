#!/usr/bin/env python3
"""
Sleep cycle — fitness scoring, core memory promotion, and fitness-based GC.

Replaces naive confidence-only gc_decay with a structural fitness model:
  1. compute_fitness   — score every node by edges, access, cross-links, source
  2. promote_core_memories — top sqrt(N) nodes get core_memory=true
  3. fitness_gc        — delete low-fitness nodes (with hard protections)

Run: python3 sleep.py [--dry-run] [--threshold 1.5]
"""

import sys
import json
import math
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from init_graph import get_db


# ── Constants ────────────────────────────────────────────────────

SKIP_HUBS = {
    "Owner",  # customize for your use case
    "McKinsey", "IIM Ahmedabad", "IIT Hyderabad",
}


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


def _graph_state(conn, label=""):
    """Print current graph statistics."""
    nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    orphans = conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE id NOT IN "
        "(SELECT from_id FROM edges UNION SELECT to_id FROM edges)"
    ).fetchone()[0]
    ratio = edges / nodes if nodes > 0 else 0
    print(f"[{label}] nodes={nodes}  edges={edges}  orphans={orphans}  edge/node={ratio:.2f}", flush=True)


# ── Embedding Dedup ──────────────────────────────────────────────

def embedding_dedup(conn, threshold=0.88, dry_run=False):
    """Merge near-duplicate nodes based on embedding cosine similarity."""
    print(f"  START embedding_dedup (threshold={threshold}, dry_run={dry_run})", flush=True)

    data = _load_embeddings(conn)
    n = len(data)
    print(f"  Loaded {n} embeddings", flush=True)

    if n < 2:
        print(f"  embedding_dedup: 0 pairs found, 0 nodes deleted", flush=True)
        return 0

    # Build vectorised cosine sim matrix
    mat = np.stack([d["vector"] for d in data]).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = mat / norms
    sim = normed @ normed.T

    # Collect candidate pairs (upper triangle, above threshold)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                la = data[i]["label"]
                lb = data[j]["label"]
                # Only flag if labels differ (case-insensitive)
                if la.lower() == lb.lower():
                    continue
                # Skip insight_ nodes
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

        # Canonical = higher confidence; tie → shorter id
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
            # Redirect from_id edges
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

            # Redirect to_id edges
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

            # Delete duplicate node + embedding
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
    """Add relates_to edges between semantically similar but unlinked nodes."""
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

    # Vectorised cosine sim matrix
    mat = np.stack([d["vector"] for d in data]).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = mat / norms
    sim = normed @ normed.T

    # Existing edge set (bidirectional)
    existing = set()
    for row in conn.execute("SELECT from_id, to_id FROM edges").fetchall():
        existing.add((row["from_id"], row["to_id"]))
        existing.add((row["to_id"], row["from_id"]))

    # Hub detection: nodes with > 50 edges
    degree = defaultdict(int)
    for row in conn.execute("SELECT from_id, to_id FROM edges").fetchall():
        degree[row["from_id"]] += 1
        degree[row["to_id"]] += 1
    hub_set = {nid for nid, d in degree.items() if d > 50}
    print(f"  Hubs (>50 edges): {len(hub_set)}", flush=True)

    # Collect candidates — vectorized row-by-row to avoid O(N²) Python loop
    id_list = [d["node_id"] for d in data]
    hub_mask = np.array([nid in hub_set for nid in id_list], dtype=bool)

    candidates = []
    for i in range(n):
        if hub_mask[i]:
            continue
        # Only upper-triangle (j > i), skip hub columns
        row = sim[i, i + 1:]
        row_non_hub = ~hub_mask[i + 1:]
        above = np.where((row >= threshold) & row_non_hub)[0]
        for k in above:
            j = i + 1 + int(k)
            a_id, b_id = id_list[i], id_list[j]
            if (a_id, b_id) not in existing:
                candidates.append((a_id, b_id, float(sim[i, j])))

    print(f"  Found {len(candidates)} candidate edges", flush=True)

    # Sort descending, cap at max_new_edges
    candidates.sort(key=lambda x: -x[2])
    candidates = candidates[:max_new_edges]

    if dry_run:
        print(f"  Cross-link: would write up to {len(candidates)} edges (dry run)", flush=True)
        print(f"  END cross_link (dry run)", flush=True)
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
    print(f"  END cross_link: {written} edges written", flush=True)
    return written


def compute_fitness(conn):
    """Compute structural fitness score for every node.

    fitness = edge_count * 1.0
            + access_count_30d * 0.5
            + cross_link_count * 0.3
            + seed_bonus (+2.0 if source IN ('human', 'second_brain'))
    """
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


# ── Signal 1: Unified fitness (access/recency/origin) ──────────

def _parse_timestamp(ts):
    """Parse timestamp to unix seconds."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return dt.timestamp()
        except:
            return None
    return None


def compute_node_fitness(origin, access_count, last_accessed_at, now=None):
    """Unified fitness score: baseline + recency + frequency."""
    now = now or time.time()

    baseline = 0.5 if origin == 'self' else 0.3

    last_ts = _parse_timestamp(last_accessed_at)
    age_days = (now - last_ts) / 86400.0 if last_ts else 0
    recency = 2.0 ** (-age_days / 7.0)  # Half-life: 7 days

    frequency = min(0.5, math.log(max(1, access_count or 0) + 1) * 0.1)

    fitness = baseline + recency + frequency
    return min(1.0, max(0.0, fitness))


def compute_all_fitness(conn, dry_run=False):
    """Replacement for decay_relevance().
    Computes unified fitness_score for all nodes.
    Also sets soft_decay_flagged for nodes 90+ days old.
    """
    print("  START compute_all_fitness", flush=True)
    now = time.time()

    nodes = conn.execute(
        "SELECT id, origin, access_count, last_accessed_at FROM nodes"
    ).fetchall()

    updates = []
    for node in nodes:
        node_id = node['id']
        origin = node['origin']
        access_count = node['access_count'] or 0
        last_accessed_at = node['last_accessed_at']

        fit = compute_node_fitness(origin, access_count, last_accessed_at, now)

        last_ts = _parse_timestamp(last_accessed_at) or now
        age_days = (now - last_ts) / 86400.0
        soft_decay = 1 if age_days > 90 else 0

        updates.append((fit, soft_decay, node_id))

    if not dry_run:
        conn.executemany(
            "UPDATE nodes SET fitness_score = ?, soft_decay_flagged = ? WHERE id = ?",
            updates
        )
        conn.commit()

    fitness_vals = [u[0] for u in updates]
    soft_decay_count = sum(u[1] for u in updates)

    print(f"  Fitness: updated {len(updates)} nodes (avg={sum(fitness_vals)/len(fitness_vals):.3f}, soft_decay={soft_decay_count})", flush=True)
    return len(updates)


def fitness_gc_v2(conn, dry_run=False):
    """Only prune if: fitness < 0.15 AND soft_decay_flagged=1 AND age > 120 days."""
    print(f"  START fitness_gc_v2 (dry_run={dry_run})", flush=True)
    now = time.time()

    candidates = conn.execute("""
        SELECT id, label, fitness_score, soft_decay_flagged, last_accessed_at
        FROM nodes
        WHERE fitness_score < 0.15 AND soft_decay_flagged = 1
    """).fetchall()

    gc_list = []
    for c in candidates:
        last_ts = _parse_timestamp(c['last_accessed_at']) or now
        age_days = (now - last_ts) / 86400.0
        if age_days > 120:
            gc_list.append((c['id'], c['label'], c['fitness_score'], age_days))

    if not gc_list:
        print(f"  GC: 0 candidates (none met: fitness<0.15 + soft_decay + age>120d)", flush=True)
        return 0

    print(f"  GC: found {len(gc_list)} candidates", flush=True)
    for nid, label, fit, age in gc_list[:5]:
        print(f"    - {label[:40]} (fitness={fit:.3f}, age={age:.0f}d)", flush=True)
    if len(gc_list) > 5:
        print(f"    ... and {len(gc_list) - 5} more", flush=True)

    if dry_run:
        print(f"  GC (dry_run): would delete {len(gc_list)} nodes", flush=True)
        return len(gc_list)

    deleted = 0
    for nid, label, _, _ in gc_list:
        conn.execute("DELETE FROM edges WHERE from_id = ? OR to_id = ?", (nid, nid))
        conn.execute("DELETE FROM embeddings WHERE node_id = ?", (nid,))
        conn.execute("DELETE FROM nodes WHERE id = ?", (nid,))
        deleted += 1

    conn.commit()
    print(f"  GC: deleted {deleted} nodes", flush=True)
    return deleted


def promote_core_memories(conn, fitness_scores):
    """Promote top sqrt(N) nodes to core_memory status.

    Returns (promoted_ids, demoted_ids).
    """
    total = len(fitness_scores)
    target_count = int(math.sqrt(total))

    # Sort by fitness descending, take top target_count
    ranked = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
    new_core = set(nid for nid, _ in ranked[:target_count])

    promoted = []
    demoted = []

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
            conn.execute("UPDATE nodes SET metadata = ? WHERE id = ?",
                         (json.dumps(meta), nid))
            promoted.append(nid)
        elif nid not in new_core and is_core:
            meta.pop("core_memory", None)
            conn.execute("UPDATE nodes SET metadata = ? WHERE id = ?",
                         (json.dumps(meta), nid))
            demoted.append(nid)

    conn.commit()
    print(f"  Core memories: target={target_count}, promoted={len(promoted)}, demoted={len(demoted)}", flush=True)
    return promoted, demoted


def _calculate_half_life(access_count: int, base_half_life: int = 30) -> float:
    """Log-scaled half-life. High-access nodes decay slower, continuously."""
    return base_half_life * (1 + math.log(max(1, access_count) + 1))


def decay_relevance(conn, base_half_life=30, dry_run=False, full=False):
    """Recalculate relevance_score using log-scaled half-life based on access_count.

    half_life = base_half_life * (1 + log(access_count + 1))
    score = 0.5 ^ (days_since_access / half_life)

    Nodes with permanent=2 (hard-pinned): skip decay entirely.
    External nodes (origin="external"): access-count-only scoring.
    """
    now = datetime.now()

    # Check if origin column exists
    cols = {r[1] for r in conn.execute("PRAGMA table_info(nodes)").fetchall()}
    has_origin = "origin" in cols
    origin_col = "COALESCE(origin, 'self') AS origin" if has_origin else "'self' AS origin"

    if full:
        nodes = conn.execute(
            "SELECT id, last_accessed_at, updated_at, created_at, "
            "COALESCE(permanent, 0) AS permanent, "
            "COALESCE(access_count, 0) AS access_count, "
            f"COALESCE(confidence, 1.0) AS confidence, {origin_col} "
            "FROM nodes"
        ).fetchall()
    else:
        nodes = conn.execute(
            "SELECT id, last_accessed_at, updated_at, created_at, "
            "COALESCE(permanent, 0) AS permanent, "
            "COALESCE(access_count, 0) AS access_count, "
            f"COALESCE(confidence, 1.0) AS confidence, {origin_col} "
            "FROM nodes ORDER BY RANDOM() LIMIT 500"
        ).fetchall()

    updated = 0
    for node in nodes:
        perm = node["permanent"]
        ac = node["access_count"]
        conf = node["confidence"]
        node_origin = node["origin"]

        # Hard-pinned: skip decay entirely
        if perm == 2:
            updated += 1
            continue

        # External nodes: access-count-only decay, no timestamp component
        if node_origin == "external":
            if ac == 0:
                score = 0.05  # near-floor — fade if never accessed
            else:
                score = min(1.0, 0.3 + (ac * 0.07))
            if not dry_run:
                conn.execute(
                    "UPDATE nodes SET relevance_score = ? WHERE id = ?",
                    (score, node["id"])
                )
            updated += 1
            continue

        # Pick best available timestamp
        ts_str = node["last_accessed_at"] or node["updated_at"] or node["created_at"]
        try:
            ts = datetime.fromisoformat(ts_str)
            # Strip tzinfo to ensure naive datetime comparison
            if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
        except (TypeError, ValueError):
            ts = now  # fallback: treat as accessed now, score = 1.0

        days_since = max(0, (now - ts).total_seconds() / 86400)

        # Log-scaled half-life: high-access nodes decay slower, continuously
        hl = _calculate_half_life(ac, base_half_life)
        score = max(0.01, 0.5 ** (days_since / hl))

        if not dry_run:
            conn.execute(
                "UPDATE nodes SET relevance_score = ? WHERE id = ?",
                (score, node["id"])
            )
        updated += 1

    if not dry_run:
        conn.commit()

    mode_str = "full" if full else f"sampled {len(nodes)}"
    print(f"  Decay: updated relevance_score for {updated} nodes ({mode_str})", flush=True)
    return updated


def fitness_gc(conn, fitness_scores, threshold=1.5, dry_run=False):
    """Delete nodes below fitness threshold, with hard protections.

    Never deletes:
      - source IN ('human', 'second_brain')
      - type IN ('person', 'org', 'decision', 'event')
      - nodes accessed in the last 7 days

    Returns count of deleted nodes.
    """
    age_cutoff = (datetime.now() - timedelta(days=7)).isoformat()

    # Get recently accessed node IDs
    recent_access = set(
        r["node_id"] for r in conn.execute(
            "SELECT DISTINCT node_id FROM access_log WHERE accessed_at >= ?",
            (age_cutoff,)
        ).fetchall()
    )

    # Get recently created node IDs (grace period — new nodes haven't had time to accumulate fitness)
    recently_created = set(
        r["id"] for r in conn.execute(
            "SELECT id FROM nodes WHERE created_at >= ?",
            (age_cutoff,)
        ).fetchall()
    )

    # Get protected nodes
    protected = set(
        r["id"] for r in conn.execute(
            "SELECT id FROM nodes WHERE source IN ('human', 'second_brain') "
            "OR type IN ('person', 'org', 'decision', 'event')"
        ).fetchall()
    )

    candidates = []
    for nid, score in fitness_scores.items():
        if score >= threshold:
            continue
        if nid in protected:
            continue
        if nid in recent_access:
            continue
        if nid in recently_created:
            continue
        candidates.append((nid, score))

    print(f"  GC candidates: {len(candidates)} nodes below threshold {threshold}", flush=True)

    if dry_run:
        for nid, score in candidates[:15]:
            label = conn.execute("SELECT label FROM nodes WHERE id = ?", (nid,)).fetchone()
            lbl = label["label"] if label else "?"
            print(f"    [dry-run] {nid} ({lbl}, fitness={score:.2f})", flush=True)
        if len(candidates) > 15:
            print(f"    ... and {len(candidates) - 15} more", flush=True)
        return len(candidates)

    if not candidates:
        return 0

    try:
        ids = [c[0] for c in candidates]
        placeholders = ",".join("?" for _ in ids)

        # Cascade: delete edges, embeddings, then nodes
        conn.execute(
            f"DELETE FROM edges WHERE from_id IN ({placeholders}) OR to_id IN ({placeholders})",
            ids + ids
        )
        conn.execute(
            f"DELETE FROM embeddings WHERE node_id IN ({placeholders})", ids
        )
        conn.execute(
            f"DELETE FROM nodes WHERE id IN ({placeholders})", ids
        )
        conn.commit()
    except Exception as ex:
        print(f"  ERROR in fitness_gc: {ex}", flush=True)
        conn.rollback()
        raise

    print(f"  GC deleted {len(candidates)} nodes", flush=True)
    return len(candidates)


def soft_gc(conn, threshold=0.05, dry_run=False):
    """Delete nodes with relevance_score below threshold.

    Hard protections (never deleted):
      - source IN ('human', 'second_brain')
      - type IN ('person', 'org', 'decision', 'event')
      - created_at within last 7 days (grace period)

    Relies on decay_relevance() having been called first this cycle.
    """
    age_cutoff = (datetime.now() - timedelta(days=7)).isoformat()

    # External nodes with access_count > 0 are protected from GC.
    # External nodes with access_count = 0 AND created > 90 days ago are eligible.
    ext_age_cutoff = (datetime.now() - timedelta(days=90)).isoformat()

    candidates = conn.execute("""
        SELECT id, label, relevance_score FROM nodes
        WHERE relevance_score < ?
          AND created_at < ?
          AND source NOT IN ('human', 'second_brain')
          AND type NOT IN ('person', 'org', 'decision', 'event')
          AND (permanent IS NULL OR permanent < 2)
          AND NOT (
              COALESCE(origin, 'self') = 'external'
              AND COALESCE(access_count, 0) > 0
          )
          AND NOT (
              COALESCE(origin, 'self') = 'external'
              AND created_at >= ?
          )
    """, (threshold, age_cutoff, ext_age_cutoff)).fetchall()

    print(f"  Soft GC candidates: {len(candidates)} nodes below threshold {threshold}", flush=True)

    if dry_run:
        for r in candidates[:15]:
            print(f"    [dry-run] {r['id']} ({r['label']}, score={r['relevance_score']:.4f})", flush=True)
        if len(candidates) > 15:
            print(f"    ... and {len(candidates) - 15} more", flush=True)
        return len(candidates)

    if not candidates:
        return 0

    try:
        ids = [r["id"] for r in candidates]
        placeholders = ",".join("?" for _ in ids)
        conn.execute(f"DELETE FROM edges WHERE from_id IN ({placeholders}) OR to_id IN ({placeholders})", ids + ids)
        conn.execute(f"DELETE FROM embeddings WHERE node_id IN ({placeholders})", ids)
        conn.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", ids)
        conn.commit()
    except Exception as ex:
        print(f"  ERROR in soft_gc: {ex}", flush=True)
        conn.rollback()
        raise

    print(f"  Soft GC deleted {len(candidates)} nodes", flush=True)
    return len(candidates)


def run_sleep(conn, dry_run=False, threshold=0.05, full=False, skip_crosslink=False):
    """Entry point. Runs fitness scoring, core memory promotion, and GC.

    Returns stats dict.
    """
    print("=" * 60, flush=True)
    print("  SLEEP CYCLE" + (" [DRY RUN]" if dry_run else ""), flush=True)
    print("=" * 60, flush=True)

    node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    print(f"  Graph: {node_count} nodes, {edge_count} edges", flush=True)

    # Step 0a: Embedding dedup
    print(f"\n{chr(8212)*50}", flush=True)
    print("  Step 0a: Embedding dedup" + (" [DRY RUN]" if dry_run else ""), flush=True)
    print(f"{chr(8212)*50}", flush=True)
    dedup_count = embedding_dedup(conn, dry_run=dry_run)

    # Step 0b: Cross-link
    print(f"\n{chr(8212)*50}", flush=True)
    print("  Step 0b: Cross-link similar nodes" + (" [DRY RUN]" if dry_run else ""), flush=True)
    print(f"{chr(8212)*50}", flush=True)
    cross_link_count = cross_link(conn, dry_run=dry_run, skip=skip_crosslink)

    # Step 1: Fitness scoring
    print(f"\n{'─' * 50}", flush=True)
    print("  Step 1: Compute fitness scores", flush=True)
    print(f"{'─' * 50}", flush=True)
    fitness_scores = compute_fitness(conn)

    # Step 2: Core memory promotion (disabled — no longer writing core_memory flags)
    # print(f"\n{'─' * 50}", flush=True)
    # print("  Step 2: Promote core memories", flush=True)
    # print(f"{'─' * 50}", flush=True)
    # promoted, demoted = promote_core_memories(conn, fitness_scores)
    promoted, demoted = [], []

    # Step 3: Fitness computation (Signal 1 — replaces decay_relevance)
    print(f"\n{'─' * 50}", flush=True)
    print("  Step 3: Fitness computation" + (" [DRY RUN]" if dry_run else ""), flush=True)
    print(f"{'─' * 50}", flush=True)
    compute_all_fitness(conn, dry_run=dry_run)

    # Step 4: Soft GC (Signal 1 — updated thresholds)
    print(f"\n{'─' * 50}", flush=True)
    print("  Step 4: Soft GC" + (" [DRY RUN]" if dry_run else ""), flush=True)
    print(f"{'─' * 50}", flush=True)
    # Use Signal 1 GC logic (fitness < 0.15 + soft_decay + age > 120d)
    gc_count = fitness_gc_v2(conn, dry_run=dry_run)

    # (deprecated) old soft_gc and fitness_gc
    # gc_count = soft_gc(conn, threshold=threshold, dry_run=dry_run)
    # gc_count = fitness_gc(conn, fitness_scores, threshold=threshold, dry_run=dry_run)

    # Step 5: Hierarchical clustering (DEPRECATED 2026-04-04)
    # Hotspots removed — retrieval now uses recursive BFS + embedding index.
    # Hotspots had 0% auto-accuracy, required constant babysitting.
    # cluster.run_clustering() no longer called.
    cluster_stats = {"deprecated": True, "reason": "replaced by recursive BFS retrieval"}

    # Summary
    stats = {
        "dedup_merged": dedup_count,
        "cross_linked": cross_link_count,
        "nodes_scored": len(fitness_scores),
        "promoted": len(promoted),
        "demoted": len(demoted),
        "gc_deleted": gc_count,
        "clustering": cluster_stats,
        "dry_run": dry_run,
    }

    print(f"\n{'=' * 60}", flush=True)
    print(f"  Sleep complete: {stats['dedup_merged']} deduped, {stats['cross_linked']} cross-linked, "
          f"{stats['promoted']} promoted, {stats['demoted']} demoted, {stats['gc_deleted']} GC'd", flush=True)
    print(f"{'=' * 60}", flush=True)

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sleep cycle — fitness scoring, core memory promotion, GC")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="relevance_score threshold for soft GC (default 0.05)")
    parser.add_argument("--full", action="store_true", help="Full decay scan instead of 500-node sample")
    parser.add_argument("--skip-crosslink", action="store_true", help="Skip Step 0b cross-link (fast cron mode)")
    args = parser.parse_args()
    conn = get_db()
    run_sleep(conn, dry_run=args.dry_run, threshold=args.threshold, full=args.full, skip_crosslink=args.skip_crosslink)
    conn.close()
