#!/usr/bin/env python3
"""
Engram — Export brain.db to dashboard-compatible JSON.

Generates dashboard/data/graph.json for the visualization frontend.
Optionally serves the dashboard locally.

Run: python3 -m dashboard.export [--serve] [--port 8080]
"""

import os
import sqlite3
import json
import sys
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

from brain import BRAIN_DIR, DB_PATH

DASHBOARD_DIR = Path(__file__).parent
OUTPUT_PATH = DASHBOARD_DIR / "data" / "graph.json"


def export():
    """Export nodes and edges to JSON for the dashboard."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("""
        SELECT id, label, type, content, confidence, source,
               relevance_score, access_count, created_at, origin
        FROM nodes
    """)

    nodes = []
    for row in c.fetchall():
        node_type = row["type"] or "concept"
        origin = row["origin"] or "self"
        domain = "agent" if (
            origin == "agent" or node_type == "hotspot"
            or (row["source"] or "").startswith("ruminate")
        ) else "user"

        nodes.append({
            "id": row["id"],
            "content": f"{row['label'] or ''}\n\n{row['content'] or ''}".strip(),
            "node_type": node_type,
            "confidence": row["confidence"] or 0.7,
            "source_file": row["source"] or "",
            "timestamp": row["created_at"] or "",
            "domain": domain,
            "access_count": row["access_count"] or 0,
            "relevance": row["relevance_score"] or 0
        })

    c.execute("SELECT from_id, to_id, relation, weight, note FROM edges")
    edges = []
    node_ids = {n["id"] for n in nodes}
    for row in c.fetchall():
        if row["from_id"] in node_ids and row["to_id"] in node_ids:
            edges.append({
                "source": row["from_id"],
                "target": row["to_id"],
                "relation": row["relation"] or "relates_to",
                "weight": row["weight"] or 1.0,
                "reasoning": row["note"] or ""
            })

    conn.close()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, indent=2)

    print(f"Exported {len(nodes)} nodes, {len(edges)} edges to {OUTPUT_PATH}")
    return len(nodes), len(edges)


def serve(port=8080):
    """Serve the dashboard locally."""
    os.chdir(DASHBOARD_DIR)
    handler = SimpleHTTPRequestHandler
    httpd = HTTPServer(("", port), handler)
    print(f"Dashboard serving at http://localhost:{port}")
    print(f"Press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export brain to dashboard")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    export()
    if args.serve:
        serve(port=args.port)
