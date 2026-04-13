#!/usr/bin/env python3
"""
Onyx Second Brain — Search Module

hybrid_search:    embedding similarity + recursive BFS graph walk (replaces fixed 2-hop)
semantic_search:  pure embedding cosine similarity
embed_text:       re-export from embed module
log_access:       re-export from init_graph (increments access_count + last_accessed_at)

2026-04-04: Upgraded hybrid_search from fixed 2-hop to recursive BFS with similarity pruning.
            Hotspots deprecated — set decayed=1, no longer created in sleep.py.
            Rationale: hotspots were structurally complex (0% auto-accuracy, needed babysitting).
            Recursive BFS + embedding index achieves better retrieval diversity without overhead.
"""

import sys
import os
import numpy as np
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))
from init_graph import get_db, log_access
from embed import embed_text, MODEL_NAME


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def semantic_search(query: str, top_k: int = 7,
                    include_external: bool = True,
                    origin: str = None) -> List[Dict]:
    """Pure embedding cosine-similarity search.

    origin= filters to a specific origin (e.g. 'onyx', 'self', 'external').
    Returns list of dicts with keys: id, label, type, content, score.
    Calls log_access on returned node IDs.
    """
    conn = get_db()
    try:
        query_vec = embed_text(query).astype(np.float32)

        if origin:
            origin_filter = f"AND n.origin = '{origin}'"
        elif not include_external:
            origin_filter = "AND (n.origin IS NULL OR n.origin = 'self')"
        else:
            origin_filter = ""
        rows = conn.execute(f"""
            SELECT e.node_id, e.embedding, n.label, n.type, n.content, n.origin
            FROM embeddings e
            JOIN nodes n ON e.node_id = n.id
            WHERE (n.decayed = 0 OR n.decayed IS NULL)
            {origin_filter}
        """).fetchall()

        scored = []
        for r in rows:
            vec = np.frombuffer(r["embedding"], dtype=np.float32)
            if np.linalg.norm(vec) == 0:
                continue
            sim = _cosine_sim(query_vec, vec)
            scored.append({
                "id": r["node_id"],
                "label": r["label"],
                "type": r["type"],
                "content": r["content"] or "",
                "origin": r["origin"] or "self",
                "score": sim,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:top_k]

        # Log access for returned results — log_access handles access_count + last_accessed_at
        if results:
            result_ids = [r["id"] for r in results]
            log_access(result_ids, source="search", conn=conn)

        # Curiosity log — same Signal 3 logic as hybrid_search
        max_score = results[0]["score"] if results else 0.0
        query_words = len(query.split())
        query_coherence = min(1.0, query_words / 8.0)
        surprise_score = query_coherence * (1.0 - max_score)
        flagged = 1 if surprise_score >= 0.35 else 0
        conn.execute("""
            INSERT INTO curiosity_log (query, max_score, result_count, surprise_score, flagged, source)
            VALUES (?, ?, ?, ?, ?, 'semantic_search')
        """, (query, max_score, len(results), round(surprise_score, 4), flagged))
        conn.commit()

        return results
    finally:
        conn.close()


def hybrid_search(query: str, top_k: int = 7,
                  include_external: bool = True,
                  max_depth: int = 3,
                  prune_threshold: float = 0.40,
                  origin: str = None) -> List[Dict]:
    """Embedding similarity + recursive BFS graph walk with similarity pruning.

    1. Embed query, score all nodes by cosine similarity
    2. Seed BFS from top-k embedding matches
    3. Recurse into neighbors — but only if their embedding score exceeds prune_threshold
    4. Depth-decay the graph bonus (closer hops score higher)
    5. Merge and re-rank by combined score

    Pruning ensures we follow semantically relevant threads deeper instead of
    exploding breadth indiscriminately. max_depth=3, prune_threshold=0.40 by default.

    Returns list of dicts with keys: id, label, type, content, score.
    """
    conn = get_db()
    try:
        query_vec = embed_text(query).astype(np.float32)

        if origin:
            origin_filter = f"AND n.origin = '{origin}'"
        elif not include_external:
            origin_filter = "AND (n.origin IS NULL OR n.origin = 'self')"
        else:
            origin_filter = ""
        rows = conn.execute(f"""
            SELECT e.node_id, e.embedding, n.label, n.type, n.content, n.origin
            FROM embeddings e
            JOIN nodes n ON e.node_id = n.id
            WHERE (n.decayed = 0 OR n.decayed IS NULL)
            AND n.type != 'hotspot'
            {origin_filter}
        """).fetchall()

        # Build embedding index — score every node against query
        node_data = {}
        for r in rows:
            vec = np.frombuffer(r["embedding"], dtype=np.float32)
            if np.linalg.norm(vec) == 0:
                continue
            sim = _cosine_sim(query_vec, vec)
            node_data[r["node_id"]] = {
                "id": r["node_id"],
                "label": r["label"],
                "type": r["type"],
                "content": r["content"] or "",
                "origin": r["origin"] or "self",
                "score": sim,
            }

        # Seed BFS from top embedding matches
        seed_k = max(top_k, 5)
        seed_nodes = sorted(node_data.values(), key=lambda x: x["score"], reverse=True)[:seed_k]

        # Recursive BFS with similarity pruning
        # combined[node_id] = best score seen (embedding + depth-decayed graph bonus)
        combined: Dict[str, float] = {}
        for n in seed_nodes:
            combined[n["id"]] = n["score"]

        visited = set(combined.keys())
        frontier = set(combined.keys())

        graph_bonus_base = 0.15  # bonus at hop 1, decays by depth
        for depth in range(1, max_depth + 1):
            if not frontier:
                break
            next_frontier = set()
            bonus = graph_bonus_base / depth  # 0.15 → 0.075 → 0.05

            # Batch fetch all neighbors of current frontier
            placeholders = ",".join("?" * len(frontier))
            frontier_list = list(frontier)
            neighbor_rows = conn.execute(f"""
                SELECT from_id, to_id FROM edges
                WHERE from_id IN ({placeholders}) OR to_id IN ({placeholders})
            """, frontier_list + frontier_list).fetchall()

            for row in neighbor_rows:
                for nid in (row["from_id"], row["to_id"]):
                    if nid in visited or nid not in node_data:
                        continue
                    base_score = node_data[nid]["score"]
                    # Prune: only follow nodes with meaningful semantic relevance
                    if base_score < prune_threshold:
                        continue
                    new_score = base_score + bonus
                    if new_score > combined.get(nid, 0):
                        combined[nid] = new_score
                    next_frontier.add(nid)
                    visited.add(nid)

            frontier = next_frontier

        # Build final results
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for nid, score in ranked:
            if nid in node_data:
                results.append({
                    "id": nid,
                    "label": node_data[nid]["label"],
                    "type": node_data[nid]["type"],
                    "content": node_data[nid]["content"],
                    "origin": node_data[nid]["origin"],
                    "score": score,
                })

        # Log access — log_access handles access_count + last_accessed_at
        if results:
            result_ids = [r["id"] for r in results]
            log_access(result_ids, source="search", conn=conn)

        # Signal 3: curiosity log — log query surprise
        # surprise_score = query_coherence proxy × (1 - max_retrieval_score)
        # A coherent query with weak results = prediction error = curiosity signal
        # query_coherence approximated by query length (longer = more specific = more expected)
        max_score = results[0]["score"] if results else 0.0
        query_words = len(query.split())
        query_coherence = min(1.0, query_words / 8.0)  # normalise: 8-word query = coherence 1.0
        surprise_score = query_coherence * (1.0 - max_score)
        SURPRISE_THRESHOLD = 0.35  # flag if surprise is high
        flagged = 1 if surprise_score >= SURPRISE_THRESHOLD else 0

        conn.execute("""
            INSERT INTO curiosity_log (query, max_score, result_count, surprise_score, flagged, source)
            VALUES (?, ?, ?, ?, ?, 'hybrid_search')
        """, (query, max_score, len(results), round(surprise_score, 4), flagged))
        conn.commit()

        return results
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Search the brain graph")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid search")
    parser.add_argument("--include-external", action="store_true")
    args = parser.parse_args()

    if args.hybrid:
        results = hybrid_search(args.query, top_k=args.top_k, include_external=args.include_external)
    else:
        results = semantic_search(args.query, top_k=args.top_k, include_external=args.include_external)

    for r in results:
        print(f"  [{r['score']:.3f}] {r['type']:12s} {r['label']}")
        if r["content"]:
            print(f"           {r['content'][:120]}")
