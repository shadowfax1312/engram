#!/usr/bin/env python3
"""
Onyx Context Retrieval — injects existing graph reasoning into LLM prompts.

Used by ruminate to ground synthesis in what the graph already knows.
Adapted from cashew's context.py, using our schema (nodes/edges) and hybrid_search.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(__file__))
from init_graph import get_db
from search import hybrid_search


@dataclass
class RelevantNode:
    id: str
    label: str
    content: str
    node_type: str
    confidence: float
    relevance_score: float
    parent_chain: List[Dict] = field(default_factory=list)


def get_context(query: str, top_k: int = 5, include_external: bool = False, origin: str = None) -> List[RelevantNode]:
    """
    Retrieve most relevant nodes for a query, with derivation chain context.
    Uses hybrid_search (embedding + 2-hop graph walk) for quality retrieval.

    Args:
        query: The topic to retrieve context for.
        top_k: Max nodes to return.
        include_external: Whether to include corpus nodes (origin='external').

    Returns:
        List of RelevantNode ranked by relevance.
    """
    raw = hybrid_search(query, top_k=top_k, include_external=include_external, origin=origin)

    conn = get_db()
    c = conn.cursor()

    results = []
    for r in raw:
        node_id = r.get("id") or r.get("node_id", "")
        if not node_id:
            continue

        # Get derivation chain (1-hop parents via edges)
        parents = c.execute("""
            SELECT e.from_id, n.label, n.content, e.note
            FROM edges e
            JOIN nodes n ON n.id = e.from_id
            WHERE e.to_id = ?
            ORDER BY e.weight DESC
            LIMIT 3
        """, (node_id,)).fetchall()

        parent_chain = [
            {
                "parent_id": p[0],
                "label": p[1],
                "content": (p[2] or "")[:120],
                "note": p[3],
            }
            for p in parents
        ]

        results.append(RelevantNode(
            id=node_id,
            label=r.get("label", ""),
            content=r.get("content", ""),
            node_type=r.get("type", ""),
            confidence=float(r.get("confidence", 1.0)),
            relevance_score=float(r.get("score", 0.0)),
            parent_chain=parent_chain,
        ))

    conn.close()
    return results


def format_context(nodes: List[RelevantNode], max_chars: int = 2000) -> str:
    """
    Format retrieved nodes into a clean context block for LLM injection.
    Includes derivation chains where available.
    """
    if not nodes:
        return ""

    lines = ["Existing reasoning in the graph on this topic:", ""]
    total = 0

    for i, node in enumerate(nodes, 1):
        entry = f"{i}. [{node.node_type}] {node.label}\n   {node.content[:200]}"
        if node.parent_chain:
            parents_str = "; ".join(
                f"{p['label']}" for p in node.parent_chain[:2]
            )
            entry += f"\n   ← derived from: {parents_str}"
        entry += f"\n   confidence={node.confidence:.2f}, relevance={node.relevance_score:.3f}\n"

        total += len(entry)
        if total > max_chars:
            lines.append("   [truncated — more context available]")
            break
        lines.append(entry)

    return "\n".join(lines)


def get_context_string(query: str, top_k: int = 5, origin: str = None) -> str:
    """Convenience wrapper: returns formatted context string directly."""
    nodes = get_context(query, top_k=top_k, origin=origin)
    return format_context(nodes)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Onyx context retrieval")
    parser.add_argument("query", help="Topic to retrieve context for")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-external", action="store_true")
    args = parser.parse_args()

    nodes = get_context(args.query, top_k=args.top_k, include_external=args.include_external)
    print(format_context(nodes) or "No context found.")
