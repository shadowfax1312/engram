#!/usr/bin/env python3
"""
Engram — Semantic and hybrid search.

semantic_search:  Pure embedding cosine similarity — O(N) scan over cached embeddings.
hybrid_search:    Embedding similarity + 2-hop graph walk for context-aware retrieval.

Core memory nodes receive a +0.15 retrieval boost.
"""

import numpy as np
from typing import List, Dict

from brain import get_db, log_access, embed_text


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


CORE_MEMORY_BOOST = 0.15


def semantic_search(query: str, top_k: int = 7,
                    include_external: bool = False) -> List[Dict]:
    """Pure embedding cosine-similarity search.

    Returns list of dicts: id, label, type, content, score.
    Core memory nodes receive a retrieval boost.
    """
    conn = get_db()
    try:
        query_vec = embed_text(query).astype(np.float32)

        origin_filter = "" if include_external else \
            "AND (n.origin IS NULL OR n.origin = 'self')"
        rows = conn.execute(f"""
            SELECT e.node_id, e.embedding, n.label, n.type, n.content,
                   COALESCE(n.core_memory, 0) AS core_memory
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
            boost = CORE_MEMORY_BOOST if r["core_memory"] else 0.0
            scored.append({
                "id": r["node_id"],
                "label": r["label"],
                "type": r["type"],
                "content": r["content"] or "",
                "score": sim + boost,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:top_k]

        if results:
            log_access([r["id"] for r in results], source="search", conn=conn)
            conn.commit()

        return results
    finally:
        conn.close()


def hybrid_search(query: str, top_k: int = 7,
                  include_external: bool = False) -> List[Dict]:
    """Embedding similarity + 2-hop graph walk.

    1. Get top embedding matches
    2. Walk 2 hops out from top results
    3. Merge and re-rank by combined score
    """
    conn = get_db()
    try:
        query_vec = embed_text(query).astype(np.float32)

        origin_filter = "" if include_external else \
            "AND (n.origin IS NULL OR n.origin = 'self')"
        rows = conn.execute(f"""
            SELECT e.node_id, e.embedding, n.label, n.type, n.content,
                   COALESCE(n.core_memory, 0) AS core_memory
            FROM embeddings e
            JOIN nodes n ON e.node_id = n.id
            WHERE (n.decayed = 0 OR n.decayed IS NULL)
            {origin_filter}
        """).fetchall()

        node_data = {}
        for r in rows:
            vec = np.frombuffer(r["embedding"], dtype=np.float32)
            if np.linalg.norm(vec) == 0:
                continue
            sim = _cosine_sim(query_vec, vec)
            boost = CORE_MEMORY_BOOST if r["core_memory"] else 0.0
            node_data[r["node_id"]] = {
                "id": r["node_id"],
                "label": r["label"],
                "type": r["type"],
                "content": r["content"] or "",
                "score": sim + boost,
                "vec": vec,
            }

        seed_k = max(top_k, 5)
        seed_nodes = sorted(
            node_data.values(), key=lambda x: x["score"], reverse=True
        )[:seed_k]
        seed_ids = {n["id"] for n in seed_nodes}

        # 2-hop graph walk
        graph_ids = set()
        for nid in seed_ids:
            hop1 = conn.execute("""
                SELECT CASE WHEN from_id = ? THEN to_id ELSE from_id END AS neighbor
                FROM edges WHERE from_id = ? OR to_id = ?
            """, (nid, nid, nid)).fetchall()
            hop1_ids = {r["neighbor"] for r in hop1}
            graph_ids.update(hop1_ids)
            for h1id in hop1_ids:
                hop2 = conn.execute("""
                    SELECT CASE WHEN from_id = ? THEN to_id ELSE from_id END AS neighbor
                    FROM edges WHERE from_id = ? OR to_id = ?
                """, (h1id, h1id, h1id)).fetchall()
                graph_ids.update(r["neighbor"] for r in hop2)

        combined = {}
        for n in seed_nodes:
            combined[n["id"]] = n["score"]

        graph_bonus = 0.15
        for gid in graph_ids:
            if gid in node_data:
                base = node_data[gid]["score"]
                combined[gid] = max(combined.get(gid, 0), base + graph_bonus)

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for nid, score in ranked:
            if nid in node_data:
                results.append({
                    "id": nid,
                    "label": node_data[nid]["label"],
                    "type": node_data[nid]["type"],
                    "content": node_data[nid]["content"],
                    "score": score,
                })

        if results:
            log_access([r["id"] for r in results], source="search", conn=conn)
            conn.commit()

        return results
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Search the brain graph")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--include-external", action="store_true")
    args = parser.parse_args()

    fn = hybrid_search if args.hybrid else semantic_search
    results = fn(args.query, top_k=args.top_k,
                 include_external=args.include_external)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['type']:12s} {r['label']}")
        if r["content"]:
            print(f"           {r['content'][:120]}")
