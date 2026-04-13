#!/usr/bin/env python3
"""
Ruminate cycle — Onyx's generative think cycle.

Key design (from cashew architecture):
- GENERATIVE not reactive — finds patterns in existing nodes, not just new inputs
- Quality gate (confidence threshold) not quantity gate (new node count)
- Sends cluster content to LLM asking for patterns, evolutions, contradictions
- Terminates when no high-confidence insights found, not when "no new material"
- Each cycle's synthesized insights change the graph for next cycle

Run: python3 ruminate.py [--promote] [--force] [--cycles N]
"""

import sys
import sqlite3
import json
import random
import subprocess
import re
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))
from init_graph import get_db, add_node, add_edge, log_access
import numpy as np
from embed import embed_text, embed_texts, MODEL_NAME
from search import semantic_search
from gateway import call_llm as _gateway_call_llm

# ── Configuration ─────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.75     # Min confidence to accept insight
PROMOTE_THRESHOLD = 0.78        # Min confidence to promote to node
COSINE_THRESHOLD = 0.38         # Below this, treat as "no result"
MAX_CLUSTER_SIZE = 150          # Send whole cluster if under this
SAMPLE_SIZE = 8                 # Sample if cluster exceeds max (reduced for prompt size)
MIN_INSIGHTS_TO_CONTINUE = 1    # Need this many to justify another cycle
MAX_CONSECUTIVE_EMPTY = 2       # Stop after this many empty cycles
MAX_BATCH_NODES = 120           # Max total nodes across all clusters in one batch

# ── State tracking ────────────────────────────────────────────────
STATE_FILE = Path(__file__).parent / "ruminate_state.json"

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "last_run": None,
        "consecutive_empty": 0,
        "total_runs": 0,
        "total_insights": 0,
        "cycle_history": []
    }

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

# ── LLM Interface ─────────────────────────────────────────────────
def call_haiku(prompt, retries=3, max_tokens=4096):
    """Call LLM via OpenClaw gateway (uses gateway.py for correct model routing)."""
    return _gateway_call_llm(prompt, max_tokens=max_tokens, retries=retries)

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
    except:
        return None

def parse_json_array_response(text):
    """Parse a JSON array response from the LLM."""
    if not text:
        return None
    # Strip markdown code fences
    text = re.sub(r'```(?:json)?\n?', '', text).strip()
    text = re.sub(r'```\s*$', '', text).strip()
    start = text.find('[')
    end = text.rfind(']') + 1
    if start == -1 or end == 0:
        return None
    try:
        result = json.loads(text[start:end])
        if not isinstance(result, list):
            return None
        # Reject degenerate responses: array of strings (Qwen copying the example)
        valid = [e for e in result if isinstance(e, dict) and 'cluster_id' in e]
        return valid if valid else None
    except:
        return None


def repair_json_array_response(prose_response, cluster_ids):
    """Fallback: ask Qwen to convert its prose response to JSON.

    Called when Qwen ignored the JSON-only instruction and returned markdown/prose.
    """
    ids_str = ", ".join(f'"{cid}"' for cid in cluster_ids)
    repair_prompt = f"""Convert the following analysis into a valid JSON array. Output ONLY the JSON array starting with [. No prose, no markdown.

Required format for each element:
{{"cluster_id": "<one of: {ids_str}>", "insights": [{{"id": "snake_id", "label": "Label", "type": "pattern", "content": "...", "confidence": 0.8, "connects_to": []}}], "proposed_edges": []}}

Analysis to convert:
{prose_response[:3000]}

Output the JSON array now, starting with [:"""
    from gateway import call_llm_ollama
    result = call_llm_ollama(repair_prompt, model="qwen2.5:7b", max_tokens=2048, retries=2)
    if result:
        return parse_json_array_response(result)
    return None

# ── Core Graph Functions ──────────────────────────────────────────
def get_full_graph():
    conn = get_db()
    c = conn.cursor()
    nodes = c.execute("SELECT * FROM nodes").fetchall()
    edges = c.execute("SELECT * FROM edges").fetchall()
    conn.close()
    return nodes, edges

def get_graph_stats():
    conn = get_db()
    c = conn.cursor()
    nodes = c.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edges = c.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    conn.close()
    return nodes, edges

def build_adjacency(nodes, edges):
    adj = {n['id']: {'node': dict(n), 'neighbors': []} for n in nodes}
    for e in edges:
        if e['from_id'] in adj and e['to_id'] in adj:
            adj[e['from_id']]['neighbors'].append({
                'id': e['to_id'],
                'relation': e['relation'],
                'note': e['note'],
                'direction': 'out'
            })
            adj[e['to_id']]['neighbors'].append({
                'id': e['from_id'],
                'relation': e['relation'],
                'note': e['note'],
                'direction': 'in'
            })
    return adj

def find_clusters(adj, min_size=3):
    """Find nodes with 3+ connections — potential insight clusters.

    NOTE: No log_access here — cluster discovery is read-only exploration.
    Access logging happens in synthesize_cluster/synthesize_clusters_batch
    when nodes are actually sent to the LLM.
    """
    clusters = []
    for node_id, data in adj.items():
        if len(data['neighbors']) >= min_size:
            cluster = {
                'hub_id': node_id,
                'hub': data['node'],
                'members': [adj[n['id']]['node'] for n in data['neighbors'] if n['id'] in adj],
                'relations': data['neighbors']
            }
            clusters.append(cluster)

    return sorted(clusters, key=lambda x: len(x['members']), reverse=True)


def get_periphery_nodes(adj, n=2):
    """Get peripheral nodes for stratified sampling.

    Returns n nodes with lowest degree (1-2 connections) to inject
    diversity into rumination — prevents rich-get-richer bias.
    """
    periphery = []
    for node_id, data in adj.items():
        degree = len(data['neighbors'])
        if 1 <= degree <= 2:
            periphery.append((node_id, data['node'], degree))

    if not periphery:
        return []

    # Sample from periphery
    sample = random.sample(periphery, min(n, len(periphery)))
    return [{'hub_id': nid, 'hub': node, 'members': [adj[nb['id']]['node'] for nb in adj[nid]['neighbors'] if nb['id'] in adj], 'relations': adj[nid]['neighbors']} for nid, node, _ in sample]

# ── The Generative Core ───────────────────────────────────────────
SYNTHESIS_PROMPT = """You are analyzing a knowledge cluster for patterns, evolutions, and contradictions.

This cluster has a hub node and connected member nodes. Look for:
1. **Evolutions** — same topic/pattern appearing at different times or with different conclusions
2. **Contradictions** — nodes that tension with each other, unresolved conflicts
3. **Cross-domain connections** — unexpected relationships between different types of knowledge
4. **Emergent patterns** — meta-insights that only appear when viewing multiple nodes together
5. **Oscillations** — patterns that cycle back (growth that's illusory, recurring conflicts)
6. **Open questions/tensions** — unresolved questions, things that remain uncertain or in tension

IMPORTANT: Every insight MUST connect to existing nodes. Use EXACT node IDs from the cluster.

Return ONLY valid JSON:
{{
  "insights": [
    {{
      "id": "snake_case_id",
      "label": "Human Readable Label",
      "type": "evolution|contradiction|connection|pattern|oscillation|question",
      "content": "1-3 sentence description of the insight",
      "confidence": 0.0-1.0,
      "connects_to": ["EXACT_NODE_ID_FROM_CLUSTER_1", "EXACT_NODE_ID_FROM_CLUSTER_2"]
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

Node IDs in this cluster:
{node_ids}

If no high-confidence insights exist, return {{"insights": [], "proposed_edges": []}}.
Better to return nothing than low-quality noise. Every insight MUST reference at least 2 valid node IDs from the list above.

HUB: [{hub_type}] {hub_label} (ID: {hub_id})
{hub_content}

MEMBERS ({member_count}):
{members_text}

Analyze for patterns. Only return insights with confidence >= 0.7."""

def _get_prior_insights(node_ids):
    """Fetch recent high-confidence insights involving any of these nodes."""
    try:
        conn = get_db()
        c = conn.cursor()
        rows = c.execute("""
            SELECT insight, nodes_involved, confidence, run_at
            FROM ruminate_log
            WHERE confidence >= 0.75
            ORDER BY run_at DESC
            LIMIT 50
        """).fetchall()
        conn.close()

        node_set = set(node_ids)
        matching = []
        for row in rows:
            try:
                involved = json.loads(row['nodes_involved'] or '[]')
            except Exception:
                involved = []
            if node_set & set(involved):
                matching.append({
                    'insight': row['insight'],
                    'confidence': row['confidence'],
                    'run_at': row['run_at']
                })
            if len(matching) >= 5:
                break
        return matching
    except Exception:
        return []


def _get_cycle_history_summary():
    """Pull last 3 cycle_history entries from state."""
    try:
        state = load_state()
        history = state.get('cycle_history', [])[-3:]
        if not history:
            return ""
        lines = []
        for entry in history:
            lines.append(f"- {entry.get('timestamp', '?')}: {entry.get('insights', 0)} insights")
        return "\n".join(lines)
    except Exception:
        return ""


def synthesize_cluster(cluster, adj):
    """Send cluster to LLM, ask for patterns/evolutions/contradictions."""
    hub = cluster['hub']
    members = cluster['members']
    hub_id = cluster['hub_id']

    # Sample if too large
    if len(members) > MAX_CLUSTER_SIZE:
        members = random.sample(members, SAMPLE_SIZE)

    # Collect all node IDs for reference
    node_ids = [hub_id] + [m.get('id', '') for m in members if m.get('id')]
    node_ids_text = ", ".join(node_ids[:50])  # Cap to avoid prompt bloat

    # Log access for cluster nodes
    log_access(node_ids, source="ruminate")

    # Format members with IDs
    members_text = ""
    for m in members:
        members_text += f"- [{m.get('type', '?')}] {m.get('label', '?')} (ID: {m.get('id', '?')}): {(m.get('content', '') or '')[:150]}\n"

    # Build prior insights section
    prior_section = ""
    prior_insights = _get_prior_insights(node_ids)
    cycle_summary = _get_cycle_history_summary()
    if prior_insights or cycle_summary:
        prior_lines = []
        if prior_insights:
            for pi in prior_insights:
                prior_lines.append(f"- [{pi['confidence']:.2f}] {pi['insight'][:200]}")
        if cycle_summary:
            prior_lines.append(f"\nRecent cycle stats:\n{cycle_summary}")
        prior_section = (
            "\nPRIOR INSIGHTS FROM PREVIOUS CYCLES (use these to avoid duplicates and build on existing patterns):\n"
            + "\n".join(prior_lines)
            + "\n\nDo not re-derive insights already listed in prior cycles. Build on them or find contradictions.\n"
        )

    # Inject graph context — what we already know about this cluster's topic
    graph_context_section = ""
    try:
        from context import get_context_string
        cluster_topic = f"{hub.get('label', '')} {hub.get('content', '')[:100]}"
        graph_ctx = get_context_string(cluster_topic, top_k=4)
        if graph_ctx:
            graph_context_section = (
                "\nGRAPH CONTEXT (existing reasoning on this topic — build on or challenge these):\n"
                + graph_ctx + "\n"
            )
    except Exception:
        pass  # Context injection is best-effort — never block synthesis

    prompt = SYNTHESIS_PROMPT.format(
        hub_type=hub.get('type', '?'),
        hub_label=hub.get('label', '?'),
        hub_id=hub_id,
        hub_content=(hub.get('content', '') or '')[:300],
        member_count=len(members),
        members_text=members_text[:6000],  # Cap total prompt size
        node_ids=node_ids_text
    )

    # Inject graph context before prior insights
    if graph_context_section:
        prompt = prompt.replace(
            "Analyze for patterns. Only return insights with confidence >= 0.7.",
            graph_context_section + "\nAnalyze for patterns. Only return insights with confidence >= 0.7."
        )

    # Inject prior insights after members, before final instruction
    if prior_section:
        # Insert before the "Analyze for patterns" line
        prompt = prompt.replace(
            "Analyze for patterns. Only return insights with confidence >= 0.7.",
            prior_section + "\nAnalyze for patterns. Only return insights with confidence >= 0.7."
        )

    response = call_haiku(prompt)
    return parse_json_response(response)


# ── Batch Synthesis ──────────────────────────────────────────────
BATCH_SYNTHESIS_PROMPT = """You are analyzing multiple knowledge clusters simultaneously. For each cluster, identify patterns, evolutions, contradictions, connections, and oscillations.

Look for:
1. **Evolutions** — same topic/pattern appearing at different times or with different conclusions
2. **Contradictions** — nodes that tension with each other, unresolved conflicts
3. **Cross-domain connections** — unexpected relationships between different types of knowledge
4. **Emergent patterns** — meta-insights that only appear when viewing multiple nodes together
5. **Oscillations** — patterns that cycle back (growth that's illusory, recurring conflicts)
6. **Open questions/tensions** — unresolved questions, things that remain uncertain or in tension

IMPORTANT: Every insight MUST connect to existing nodes. Use EXACT node IDs from the cluster.

Return ONLY a valid JSON array, one entry per cluster. No markdown. No prose. Start with [.
CRITICAL: "cluster_id" MUST be the exact UUID shown after "(ID: ...)" in the cluster header.

Each entry has this shape:
  cluster_id: string (exact UUID from cluster header)
  insights: array of objects with keys: id, label, type, content, confidence, connects_to
  proposed_edges: array of objects with keys: from_id, to_id, relation, note
  type values: evolution, contradiction, connection, pattern, oscillation, question
  confidence: float 0.0-1.0

If no high-confidence insights exist for a cluster, return empty arrays for that entry.
Better to return nothing than low-quality noise. Every insight MUST reference at least 2 valid node IDs from its cluster.

{clusters_section}
{prior_section}
Analyze all clusters for patterns. Only return insights with confidence >= 0.7."""


def synthesize_clusters_batch(clusters, adj):
    """Send all clusters in a single LLM call, return list of per-cluster results."""
    # Context size guard: reduce per-cluster sample if total nodes exceed limit
    total_members = sum(len(c['members']) for c in clusters)
    effective_sample = SAMPLE_SIZE
    if total_members > MAX_BATCH_NODES and len(clusters) > 0:
        effective_sample = max(5, MAX_BATCH_NODES // len(clusters))
        print(f"   ⚠ {total_members} total nodes exceeds {MAX_BATCH_NODES} limit, sampling {effective_sample} per cluster")

    # Build combined cluster section
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
        node_ids_text = ", ".join(node_ids[:50])

        log_access(node_ids, source="ruminate")

        members_text = ""
        for m in members:
            members_text += f"- [{m.get('type', '?')}] {m.get('label', '?')} (ID: {m.get('id', '?')}): {(m.get('content', '') or '')[:150]}\n"

        block = (
            f"--- CLUSTER {idx+1} ---\n"
            f"Node IDs: {node_ids_text}\n"
            f"HUB: [{hub.get('type', '?')}] {hub.get('label', '?')} (ID: {hub_id})\n"
            f"{(hub.get('content', '') or '')[:300]}\n"
            f"MEMBERS ({len(members)}):\n"
            f"{members_text[:3000]}\n"
        )
        cluster_blocks.append(block)

    clusters_section = "\n".join(cluster_blocks)

    # Build prior insights section
    prior_section = ""
    prior_insights = _get_prior_insights(all_node_ids)
    cycle_summary = _get_cycle_history_summary()
    if prior_insights or cycle_summary:
        prior_lines = []
        if prior_insights:
            for pi in prior_insights:
                prior_lines.append(f"- [{pi['confidence']:.2f}] {pi['insight'][:200]}")
        if cycle_summary:
            prior_lines.append(f"\nRecent cycle stats:\n{cycle_summary}")
        prior_section = (
            "\nPRIOR INSIGHTS FROM PREVIOUS CYCLES (avoid duplicates, build on existing patterns):\n"
            + "\n".join(prior_lines)
            + "\n\nDo not re-derive insights already listed. Build on them or find contradictions.\n"
        )

    # Append a hard JSON reminder at the end (Qwen pays more attention to end of prompt)
    json_footer = (
        "\n\nCRITICAL FINAL INSTRUCTION: Your ENTIRE response must be a valid JSON array."
        " Start with [ and end with ]. No markdown headers. No prose. No explanations."
        " If you output anything other than a JSON array, it will be discarded."
    )

    prompt = BATCH_SYNTHESIS_PROMPT.format(
        clusters_section=clusters_section,
        prior_section=prior_section
    ) + json_footer

    print(f"   DEBUG: Prompt length = {len(prompt)} chars", flush=True)

    response = call_haiku(prompt)

    if not response:
        print(f"   DEBUG: call_haiku returned None", flush=True)
        return None

    print(f"   DEBUG: Response length = {len(response)} chars", flush=True)
    first_char = response.lstrip()[:1]
    print(f"   DEBUG: Response first char = {repr(first_char)} (want '[')", flush=True)

    result = parse_json_array_response(response)
    if result is not None:
        print(f"   DEBUG: Parsed {len(result)} cluster entries", flush=True)
        return result

    # Fallback: Qwen returned prose — ask it to convert
    print(f"   DEBUG: JSON parse failed, trying repair pass...", flush=True)
    cluster_ids = [c['hub_id'] for c in clusters]
    result = repair_json_array_response(response, cluster_ids)
    if result is not None:
        print(f"   DEBUG: Repair pass yielded {len(result)} cluster entries", flush=True)
    else:
        print(f"   DEBUG: Repair pass also failed. Dropping batch.", flush=True)
    return result


def _novelty_check(content, threshold=0.78):
    """Check if an insight is novel enough vs existing embeddings.

    Returns (is_novel, best_match_id, best_sim).
    Fails open — if embedding fails, treat as novel.
    """
    try:
        new_vec = embed_text(content).astype(np.float32)
        conn = get_db()
        rows = conn.execute("""
            SELECT e.node_id, e.embedding FROM embeddings e
            JOIN nodes n ON e.node_id = n.id
            WHERE n.type LIKE 'insight_%'
        """).fetchall()
        conn.close()

        best_sim = 0.0
        best_id = None
        for r in rows:
            vec = np.frombuffer(r["embedding"], dtype=np.float32)
            na = np.linalg.norm(new_vec)
            nb = np.linalg.norm(vec)
            if na == 0 or nb == 0:
                continue
            sim = float(np.dot(new_vec, vec) / (na * nb))
            if sim > best_sim:
                best_sim = sim
                best_id = r["node_id"]

        return best_sim < threshold, best_id, best_sim
    except Exception:
        return True, None, 0.0  # Fail open


def _compound_insight(content_a, label_a, content_b, label_b):
    """Synthesize a compound insight from two similar insights via Haiku."""
    prompt = f"""Two insights are very similar. Synthesize them into a single, deeper compound insight.

Insight A: "{label_a}" — {content_a}
Insight B: "{label_b}" — {content_b}

Return ONLY valid JSON:
{{"label": "Compound Insight Label", "content": "1-3 sentence compound insight that captures both", "confidence": 0.0-1.0}}"""

    response = call_haiku(prompt)
    return parse_json_response(response)


def insert_insights(insights, cluster_hub_id):
    """Insert high-confidence insights as nodes.

    Includes novelty gate: checks cosine similarity against existing insight
    embeddings. If similarity >= 0.78, either compounds via Haiku or skips.
    """
    conn = get_db()
    c = conn.cursor()
    added = 0

    for ins in insights:
        conf = ins.get('confidence', 0)
        if conf < CONFIDENCE_THRESHOLD:
            continue

        label = ins.get('label', 'Unknown Insight')
        content = ins.get('content', '')
        ins_type = ins.get('type', 'pattern')

        # Novelty gate — check against existing insights
        # TEMP DISABLED: causing DB lock issues
        is_novel, match_id, match_sim = True, None, 0.0  # _novelty_check(content)
        if not is_novel and match_id:
            print(f"      ~ Novelty gate: '{label[:30]}...' sim={match_sim:.3f} with {match_id}")
            # Attempt compound insight
            match_row = c.execute("SELECT label, content FROM nodes WHERE id = ?", (match_id,)).fetchone()
            if match_row:
                compound = _compound_insight(content, label, match_row["content"], match_row["label"])
                if compound:
                    content = compound.get("content", content)
                    label = compound.get("label", label)
                    conf = compound.get("confidence", conf)
                    ins_type = "pattern"  # compounds are patterns
                    print(f"        -> Compounded: '{label[:40]}...'")
                else:
                    print(f"        -> Skipping duplicate")
                    continue
            else:
                continue

        node_id = f"insight_{ins.get('id', 'unknown')}_{datetime.now().strftime('%Y%m%d%H%M')}"

        # Insert node — tagged as system inference
        c.execute('''INSERT OR IGNORE INTO nodes (id, label, type, content, confidence, source, origin)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (node_id, label, f"insight_{ins_type}", content, conf, 'ruminate', 'onyx'))

        # Edge to cluster hub
        c.execute('''INSERT OR IGNORE INTO edges (from_id, to_id, relation, note, source)
                     VALUES (?, ?, ?, ?, ?)''',
                  (node_id, cluster_hub_id, 'derived_from', 'synthesized from cluster', 'ruminate'))

        # Edges to connected nodes (new format: connects_to)
        connected = ins.get('connects_to', []) or ins.get('source_nodes', [])
        edges_added = 0
        for src in connected[:5]:
            if c.execute("SELECT 1 FROM nodes WHERE id=?", (src,)).fetchone():
                c.execute('''INSERT OR IGNORE INTO edges (from_id, to_id, relation, note, source)
                             VALUES (?, ?, ?, ?, ?)''',
                          (node_id, src, 'synthesizes', 'insight connects nodes', 'ruminate'))
                edges_added += 1

        if edges_added < 2:
            print(f"      ⚠ Low connectivity for {label[:30]}... ({edges_added} edges)")

        # Log to ruminate_log (inline to avoid opening a second DB connection)
        import json as _json
        try:
            c.execute("""
                INSERT INTO ruminate_log (insight, nodes_involved, confidence)
                VALUES (?, ?, ?)
            """, (content, _json.dumps([cluster_hub_id] + (connected or [])), conf))
        except Exception:
            pass  # log failure is non-fatal

        added += 1

    conn.commit()
    conn.close()
    return added

def insert_proposed_edges(edges):
    """Insert high-quality proposed edges."""
    conn = get_db()
    c = conn.cursor()
    added = 0
    
    for e in edges:
        fid = e.get('from_id')
        tid = e.get('to_id')
        if not fid or not tid:
            continue
        
        # Verify both nodes exist
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

def log_insight(insight, nodes_involved, confidence=0.7):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO ruminate_log (insight, nodes_involved, confidence)
        VALUES (?, ?, ?)
    """, (insight, json.dumps(nodes_involved), confidence))
    conn.commit()
    conn.close()

def embed_new_nodes():
    """Batch embed all nodes that don't yet have an embedding."""
    conn = get_db()
    c = conn.cursor()
    unembedded = c.execute("""
        SELECT n.id, n.label, n.content FROM nodes n
        LEFT JOIN embeddings e ON n.id = e.node_id
        WHERE e.node_id IS NULL
          AND n.content IS NOT NULL
          AND n.content != ''
    """).fetchall()

    if not unembedded:
        conn.close()
        return 0

    print(f"\n🧬 Embedding {len(unembedded)} new nodes...")
    ids = [n['id'] for n in unembedded]
    texts = [n['content'] for n in unembedded]

    vectors = embed_texts(texts)

    c.executemany(
        "INSERT OR REPLACE INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
        [(nid, vec.astype(np.float32).tobytes(), MODEL_NAME) for nid, vec in zip(ids, vectors)]
    )
    conn.commit()
    conn.close()
    print(f"   ✅ {len(unembedded)} nodes embedded.")
    return len(unembedded)

# ── Main Cycle ────────────────────────────────────────────────────
def run_cycle(state, force=False):
    """Run one ruminate cycle. Returns (insights_found, should_continue)."""
    
    node_count, edge_count = get_graph_stats()
    ratio = edge_count / node_count if node_count > 0 else 0
    
    print(f"\n📊 Graph: {node_count} nodes, {edge_count} edges (ratio {ratio:.2f})")
    
    # Check consecutive empty cycles
    if not force and state.get('consecutive_empty', 0) >= MAX_CONSECUTIVE_EMPTY:
        print(f"⏸️  Pausing: {state['consecutive_empty']} consecutive empty cycles.")
        print("   Run with --force to override, or add new material.")
        return 0, False
    
    nodes, edges = get_full_graph()
    adj = build_adjacency(nodes, edges)
    clusters = find_clusters(adj, min_size=3)
    
    print(f"🔍 Found {len(clusters)} clusters with 3+ connections")
    
    if not clusters:
        print("   No clusters to analyze.")
        return 0, False
    
    # Process random sample of clusters (avoid rich-get-richer bias)
    total_insights = 0
    total_edges = 0
    n = min(4, len(clusters))
    clusters_to_process = random.sample(clusters, n)

    # Inject 2 peripheral nodes for stratified sampling
    periphery = get_periphery_nodes(adj, n=2)
    if periphery:
        clusters_to_process.extend(periphery)
        print(f"   + Injected {len(periphery)} peripheral nodes for diversity")

    # Build cluster_id -> cluster lookup for routing
    cluster_map = {c['hub_id']: c for c in clusters_to_process}

    for c in clusters_to_process:
        print(f"   • {c['hub']['label']} ({len(c['members'])} members)")

    print(f"\n   Batching {len(clusters_to_process)} clusters into single LLM call...")
    batch_results = synthesize_clusters_batch(clusters_to_process, adj)

    if not batch_results:
        print(f"       ✗ No response from batch call")
    else:
        for entry in batch_results:
            if not isinstance(entry, dict):
                continue
            cluster_id = entry.get('cluster_id')
            if not cluster_id or cluster_id not in cluster_map:
                print(f"       ⚠ Unknown cluster_id: {cluster_id}, skipping")
                continue

            cluster = cluster_map[cluster_id]
            insights = entry.get('insights', [])
            proposed_edges = entry.get('proposed_edges', [])

            # Filter by confidence
            high_conf_insights = [ins for ins in insights if ins.get('confidence', 0) >= CONFIDENCE_THRESHOLD]

            if high_conf_insights:
                added_nodes = insert_insights(high_conf_insights, cluster['hub_id'])
                total_insights += added_nodes
                print(f"       ✓ {cluster['hub']['label'][:40]}: {added_nodes} insights")
                for ins in high_conf_insights[:2]:
                    print(f"         • [{ins.get('type')}] {ins.get('label')}")

            if proposed_edges:
                added_edges = insert_proposed_edges(proposed_edges)
                total_edges += added_edges
                if added_edges:
                    print(f"       + {cluster['hub']['label'][:40]}: {added_edges} edges")
    
    # Embed new nodes
    embed_new_nodes()
    
    # Log summary
    print(f"\n✅ Cycle complete: {total_insights} insights, {total_edges} edges")
    
    # Decide if we should continue
    should_continue = total_insights >= MIN_INSIGHTS_TO_CONTINUE
    
    return total_insights, should_continue


def run(promote=False, force=False, cycles=1):
    state = load_state()
    
    print(f"\n🧠 Ruminate — Generative Think Cycle")
    print("─" * 60)
    print(f"Runs: {state.get('total_runs', 0)} | Insights: {state.get('total_insights', 0)}")
    print(f"Mode: {'forced' if force else 'normal'} | Target cycles: {cycles}")
    
    cycle_num = 0
    total_insights_this_run = 0
    
    while cycle_num < cycles:
        cycle_num += 1
        print(f"\n{'═' * 60}")
        print(f"CYCLE {cycle_num}/{cycles}")
        print(f"{'═' * 60}")
        
        insights, should_continue = run_cycle(state, force=(force and cycle_num == 1))
        total_insights_this_run += insights
        
        # Update state
        state['last_run'] = datetime.now().isoformat()
        state['total_runs'] = state.get('total_runs', 0) + 1
        state['total_insights'] = state.get('total_insights', 0) + insights
        
        if insights == 0:
            state['consecutive_empty'] = state.get('consecutive_empty', 0) + 1
        else:
            state['consecutive_empty'] = 0
        
        # Track cycle history
        history = state.get('cycle_history', [])
        history.append({
            'timestamp': datetime.now().isoformat(),
            'insights': insights
        })
        state['cycle_history'] = history[-20:]  # Keep last 20
        
        save_state(state)
        
        if not should_continue and cycle_num < cycles:
            print(f"\n⏹️  Early stop: no high-confidence insights found.")
            break
    
    # Final stats
    node_count, edge_count = get_graph_stats()
    print(f"\n{'─' * 60}")
    print(f"Run complete. {cycle_num} cycles, {total_insights_this_run} insights added.")
    print(f"Graph now: {node_count} nodes, {edge_count} edges (ratio {edge_count/node_count:.2f})")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    promote = "--promote" in sys.argv
    force = "--force" in sys.argv
    
    # Parse --cycles N
    cycles = 1
    for i, arg in enumerate(sys.argv):
        if arg == "--cycles" and i + 1 < len(sys.argv):
            try:
                cycles = int(sys.argv[i + 1])
            except ValueError:
                pass
    
    run(promote=promote, force=force, cycles=cycles)
