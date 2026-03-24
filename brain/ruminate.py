#!/usr/bin/env python3
"""
Engram — Ruminate: generative synthesis engine.

Samples clusters from the knowledge graph, sends them to an LLM,
and generates cross-domain insights, contradictions, and patterns.

Key design:
- GENERATIVE not reactive — finds patterns in existing nodes
- Quality gate (confidence threshold) not quantity gate
- Terminates when no high-confidence insights found
- Each cycle's insights change the graph for the next cycle

Run: python3 -m brain.ruminate [--force] [--cycles N]
"""

import os
import json
import random
import re
import time
import urllib.request
import numpy as np
from pathlib import Path
from datetime import datetime

from brain import get_db, add_node, add_edge, log_access, embed_text, embed_texts, \
    EMBEDDING_MODEL, BRAIN_DIR

# ── Configuration ─────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.75
PROMOTE_THRESHOLD = 0.78
MAX_CLUSTER_SIZE = 150
SAMPLE_SIZE = 15
MIN_INSIGHTS_TO_CONTINUE = 1
MAX_CONSECUTIVE_EMPTY = 2
MAX_BATCH_NODES = 100

LLM_MODEL = os.environ.get("ENGRAM_RUMINATE_MODEL", "claude-sonnet-4-6")
LLM_ENDPOINT = os.environ.get("ENGRAM_LLM_ENDPOINT", "http://localhost:3456/v1/chat/completions")

STATE_FILE = BRAIN_DIR / "ruminate_state.json"


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_run": None, "consecutive_empty": 0, "total_runs": 0,
            "total_insights": 0, "cycle_history": []}


def save_state(state):
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── LLM Interface ─────────────────────────────────────────────────

def call_llm(prompt, retries=3, max_tokens=4096, model=None):
    """Call LLM via configurable endpoint (OpenAI-compatible)."""
    model = model or LLM_MODEL
    for attempt in range(retries):
        try:
            data = json.dumps({
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }).encode()
            req = urllib.request.Request(
                LLM_ENDPOINT, data=data,
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer not-needed"},
                method="POST"
            )
            r = urllib.request.urlopen(req, timeout=180)
            resp = json.loads(r.read())
            text = resp.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if text:
                return text
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"   LLM call failed attempt {attempt+1}: {e}", flush=True)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def parse_json_response(text):
    if not text:
        return None
    text = re.sub(r'```(?:json)?\n?', '', text).strip()
    start = text.find('{')
    end = text.rfind('}') + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except Exception:
        return None


def parse_json_array_response(text):
    if not text:
        return None
    text = re.sub(r'```(?:json)?\n?', '', text).strip()
    start = text.find('[')
    end = text.rfind(']') + 1
    if start == -1 or end == 0:
        return None
    try:
        result = json.loads(text[start:end])
        return result if isinstance(result, list) else None
    except Exception:
        return None


# ── Graph Helpers ─────────────────────────────────────────────────

def get_graph_stats():
    conn = get_db()
    nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    conn.close()
    return nodes, edges


def get_full_graph():
    conn = get_db()
    nodes = conn.execute("SELECT * FROM nodes").fetchall()
    edges = conn.execute("SELECT * FROM edges").fetchall()
    conn.close()
    return nodes, edges


def build_adjacency(nodes, edges):
    adj = {n['id']: {'node': dict(n), 'neighbors': []} for n in nodes}
    for e in edges:
        if e['from_id'] in adj and e['to_id'] in adj:
            adj[e['from_id']]['neighbors'].append({
                'id': e['to_id'], 'relation': e['relation'],
                'note': e['note'], 'direction': 'out'
            })
            adj[e['to_id']]['neighbors'].append({
                'id': e['from_id'], 'relation': e['relation'],
                'note': e['note'], 'direction': 'in'
            })
    return adj


def find_clusters(adj, min_size=3):
    """Find nodes with 3+ connections."""
    clusters = []
    for node_id, data in adj.items():
        if len(data['neighbors']) >= min_size:
            clusters.append({
                'hub_id': node_id,
                'hub': data['node'],
                'members': [adj[n['id']]['node'] for n in data['neighbors']
                            if n['id'] in adj],
                'relations': data['neighbors']
            })
    return sorted(clusters, key=lambda x: len(x['members']), reverse=True)


def get_periphery_nodes(adj, n=2):
    """Get peripheral nodes (1-2 connections) for diversity sampling."""
    periphery = [(nid, data['node'], len(data['neighbors']))
                 for nid, data in adj.items()
                 if 1 <= len(data['neighbors']) <= 2]
    if not periphery:
        return []
    sample = random.sample(periphery, min(n, len(periphery)))
    return [{'hub_id': nid, 'hub': node,
             'members': [adj[nb['id']]['node'] for nb in adj[nid]['neighbors']
                         if nb['id'] in adj],
             'relations': adj[nid]['neighbors']}
            for nid, node, _ in sample]


# ── Synthesis ─────────────────────────────────────────────────────

BATCH_SYNTHESIS_PROMPT = """You are analyzing multiple knowledge clusters for patterns, evolutions, and contradictions.

Look for:
1. **Evolutions** — same topic appearing at different times with different conclusions
2. **Contradictions** — nodes that tension with each other
3. **Cross-domain connections** — unexpected relationships between different knowledge types
4. **Emergent patterns** — meta-insights visible only when viewing multiple nodes together
5. **Oscillations** — patterns that cycle back
6. **Open questions** — unresolved tensions

IMPORTANT: Every insight MUST connect to existing nodes using EXACT node IDs.

Return ONLY a valid JSON array:
[
  {{
    "cluster_id": "<EXACT_HUB_ID>",
    "insights": [
      {{
        "id": "snake_case_id",
        "label": "Human Readable Label",
        "type": "evolution|contradiction|connection|pattern|oscillation|question",
        "content": "1-3 sentence description",
        "confidence": 0.0-1.0,
        "connects_to": ["NODE_ID_1", "NODE_ID_2"]
      }}
    ],
    "proposed_edges": [
      {{
        "from_id": "EXACT_NODE_ID",
        "to_id": "EXACT_NODE_ID",
        "relation": "evolved_to|tensions_with|enables|derived_from",
        "note": "why connected"
      }}
    ]
  }}
]

{clusters_section}
{prior_section}
Analyze all clusters for patterns. Only return insights with confidence >= 0.7."""


def synthesize_clusters_batch(clusters, adj):
    """Send all clusters in a single LLM call."""
    total_members = sum(len(c['members']) for c in clusters)
    effective_sample = SAMPLE_SIZE
    if total_members > MAX_BATCH_NODES and len(clusters) > 0:
        effective_sample = max(5, MAX_BATCH_NODES // len(clusters))

    all_node_ids = []
    cluster_blocks = []
    for idx, cluster in enumerate(clusters):
        hub = cluster['hub']
        members = cluster['members']
        hub_id = cluster['hub_id']
        if len(members) > effective_sample:
            members = random.sample(members, effective_sample)
        node_ids = [hub_id] + [m.get('id', '') for m in members if m.get('id')]
        all_node_ids.extend(node_ids)
        log_access(node_ids, source="ruminate")

        members_text = "".join(
            f"- [{m.get('type', '?')}] {m.get('label', '?')} (ID: {m.get('id', '?')}): "
            f"{(m.get('content', '') or '')[:150]}\n"
            for m in members
        )
        block = (
            f"--- CLUSTER {idx+1} ---\n"
            f"Node IDs: {', '.join(node_ids[:50])}\n"
            f"HUB: [{hub.get('type', '?')}] {hub.get('label', '?')} (ID: {hub_id})\n"
            f"{(hub.get('content', '') or '')[:300]}\n"
            f"MEMBERS ({len(members)}):\n{members_text[:3000]}\n"
        )
        cluster_blocks.append(block)

    # Prior insights
    prior_section = ""
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT insight, confidence FROM ruminate_log
            WHERE confidence >= 0.75 ORDER BY run_at DESC LIMIT 10
        """).fetchall()
        conn.close()
        if rows:
            lines = [f"- [{r['confidence']:.2f}] {r['insight'][:200]}" for r in rows]
            prior_section = (
                "\nPRIOR INSIGHTS (avoid duplicates):\n" + "\n".join(lines) + "\n"
            )
    except Exception:
        pass

    prompt = BATCH_SYNTHESIS_PROMPT.format(
        clusters_section="\n".join(cluster_blocks),
        prior_section=prior_section
    )
    response = call_llm(prompt)
    return parse_json_array_response(response)


def insert_insights(insights, cluster_hub_id):
    """Insert high-confidence insights as nodes."""
    conn = get_db()
    c = conn.cursor()
    added = 0
    for ins in insights:
        if ins.get('confidence', 0) < CONFIDENCE_THRESHOLD:
            continue
        label = ins.get('label', 'Unknown Insight')
        content = ins.get('content', '')
        ins_type = ins.get('type', 'pattern')
        node_id = f"insight_{ins.get('id', 'unknown')}_{datetime.now().strftime('%Y%m%d%H%M')}"

        c.execute('''INSERT OR IGNORE INTO nodes (id, label, type, content, confidence, source)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (node_id, label, f"insight_{ins_type}", content,
                   ins.get('confidence', 0.7), 'ruminate'))
        c.execute('''INSERT OR IGNORE INTO edges (from_id, to_id, relation, note, source)
                     VALUES (?, ?, ?, ?, ?)''',
                  (node_id, cluster_hub_id, 'derived_from', 'synthesized from cluster', 'ruminate'))

        connected = ins.get('connects_to', []) or []
        for src in connected[:5]:
            if c.execute("SELECT 1 FROM nodes WHERE id=?", (src,)).fetchone():
                c.execute('''INSERT OR IGNORE INTO edges (from_id, to_id, relation, note, source)
                             VALUES (?, ?, ?, ?, ?)''',
                          (node_id, src, 'synthesizes', 'insight connects nodes', 'ruminate'))
        try:
            c.execute("INSERT INTO ruminate_log (insight, nodes_involved, confidence) VALUES (?, ?, ?)",
                      (content, json.dumps([cluster_hub_id] + (connected or [])),
                       ins.get('confidence', 0.7)))
        except Exception:
            pass
        added += 1
    conn.commit()
    conn.close()
    return added


def insert_proposed_edges(edges):
    """Insert proposed edges between existing nodes."""
    conn = get_db()
    c = conn.cursor()
    added = 0
    for e in edges:
        fid, tid = e.get('from_id'), e.get('to_id')
        if not fid or not tid:
            continue
        if not c.execute("SELECT 1 FROM nodes WHERE id=?", (fid,)).fetchone():
            continue
        if not c.execute("SELECT 1 FROM nodes WHERE id=?", (tid,)).fetchone():
            continue
        c.execute('''INSERT OR IGNORE INTO edges (from_id, to_id, relation, note, source)
                     VALUES (?, ?, ?, ?, ?)''',
                  (fid, tid, e.get('relation', 'relates_to'), e.get('note', ''), 'ruminate'))
        added += 1
    conn.commit()
    conn.close()
    return added


def embed_new_nodes():
    """Batch embed all nodes that don't yet have an embedding."""
    conn = get_db()
    c = conn.cursor()
    unembedded = c.execute("""
        SELECT n.id, n.content FROM nodes n
        LEFT JOIN embeddings e ON n.id = e.node_id
        WHERE e.node_id IS NULL AND n.content IS NOT NULL AND n.content != ''
    """).fetchall()
    if not unembedded:
        conn.close()
        return 0
    ids = [n['id'] for n in unembedded]
    texts = [n['content'] for n in unembedded]
    vectors = embed_texts(texts)
    c.executemany(
        "INSERT OR REPLACE INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
        [(nid, vec.astype(np.float32).tobytes(), EMBEDDING_MODEL)
         for nid, vec in zip(ids, vectors)]
    )
    conn.commit()
    conn.close()
    return len(unembedded)


# ── Main Cycle ────────────────────────────────────────────────────

def run_cycle(state, force=False):
    """Run one ruminate cycle. Returns (insights_found, should_continue)."""
    node_count, edge_count = get_graph_stats()
    print(f"\n  Graph: {node_count} nodes, {edge_count} edges")

    if not force and state.get('consecutive_empty', 0) >= MAX_CONSECUTIVE_EMPTY:
        print(f"  Pausing: {state['consecutive_empty']} consecutive empty cycles.")
        return 0, False

    nodes, edges = get_full_graph()
    adj = build_adjacency(nodes, edges)
    clusters = find_clusters(adj, min_size=3)
    print(f"  Found {len(clusters)} clusters with 3+ connections")
    if not clusters:
        return 0, False

    n = min(6, len(clusters))
    clusters_to_process = random.sample(clusters, n)
    periphery = get_periphery_nodes(adj, n=2)
    if periphery:
        clusters_to_process.extend(periphery)

    cluster_map = {c['hub_id']: c for c in clusters_to_process}
    print(f"  Batching {len(clusters_to_process)} clusters into single LLM call...")

    batch_results = synthesize_clusters_batch(clusters_to_process, adj)
    total_insights, total_edges = 0, 0

    if batch_results:
        for entry in batch_results:
            if not isinstance(entry, dict):
                continue
            cluster_id = entry.get('cluster_id')
            if not cluster_id or cluster_id not in cluster_map:
                continue
            insights = [i for i in entry.get('insights', [])
                        if i.get('confidence', 0) >= CONFIDENCE_THRESHOLD]
            if insights:
                added = insert_insights(insights, cluster_id)
                total_insights += added
            proposed = entry.get('proposed_edges', [])
            if proposed:
                total_edges += insert_proposed_edges(proposed)

    embed_new_nodes()
    print(f"  Cycle complete: {total_insights} insights, {total_edges} edges")
    return total_insights, total_insights >= MIN_INSIGHTS_TO_CONTINUE


def run(force=False, cycles=1):
    """Run multiple ruminate cycles."""
    state = load_state()
    print(f"\n  Ruminate — Generative Think Cycle")
    print("-" * 60)

    cycle_num = 0
    total = 0
    while cycle_num < cycles:
        cycle_num += 1
        print(f"\n{'=' * 60}\nCYCLE {cycle_num}/{cycles}\n{'=' * 60}")
        insights, should_continue = run_cycle(state, force=(force and cycle_num == 1))
        total += insights
        state['last_run'] = datetime.now().isoformat()
        state['total_runs'] = state.get('total_runs', 0) + 1
        state['total_insights'] = state.get('total_insights', 0) + insights
        state['consecutive_empty'] = 0 if insights else state.get('consecutive_empty', 0) + 1
        history = state.get('cycle_history', [])
        history.append({'timestamp': datetime.now().isoformat(), 'insights': insights})
        state['cycle_history'] = history[-20:]
        save_state(state)
        if not should_continue and cycle_num < cycles:
            print(f"\n  Early stop: no high-confidence insights.")
            break

    node_count, edge_count = get_graph_stats()
    print(f"\n{'-' * 60}")
    print(f"Run complete. {cycle_num} cycles, {total} insights.")
    print(f"Graph: {node_count} nodes, {edge_count} edges")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ruminate — generative think cycle")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cycles", type=int, default=1)
    args = parser.parse_args()
    run(force=args.force, cycles=args.cycles)
