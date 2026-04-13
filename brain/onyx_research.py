#!/usr/bin/env python3
"""
onyx_research.py — Consciousness/tech/science research for Onyx brain.

Same architecture as research.py but:
- Writes to onyx_brain.db
- Filters curiosity gaps to consciousness/tech/science keywords only
- KNN edge-linking points at onyx_brain.db

Usage:
  python3 onyx_research.py                    # Research top 3 gaps
  python3 onyx_research.py --limit 5          # Research top 5 gaps
  python3 onyx_research.py --query "topic"    # Research specific query
  python3 onyx_research.py --dry-run          # Preview without writing
"""

import sys
import os
import re
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# ── DB path override ─────────────────────────────────────────────
DB_PATH = os.environ.get("ONYX_DB", os.path.join(os.path.dirname(__file__), "onyx_brain.db"))

sys.path.insert(0, os.path.dirname(__file__))
import init_graph
init_graph.DB_PATH = Path(DB_PATH)

from init_graph import get_db, add_node, add_edge
from embed import embed_text

ORIGIN = "onyx"
DEFAULT_CONFIDENCE = 0.70
KNN_TOP_K = 5
KNN_MIN_SCORE = 0.60
KNN_MIN_WEIGHT = 0.30

# ── Consciousness/Tech/Science keyword filter ────────────────────
CONSCIOUSNESS_TECH_KEYWORDS = [
    "consciousness", "IIT", "integrated information", "global workspace", "GWT",
    "free energy", "active inference", "FEP", "predictive coding",
    "HACNS", "HAR", "human accelerated", "AlphaGenome",
    "thermodynamic computing", "neuromorphic",
    "frequency topology", "attractor network",
    "extended mind", "Clark Chalmers", "Smolin", "cosmological selection",
    "speculative decoding", "VLM inference", "DREAM architecture",
    "transformer attention", "embedding space", "semantic search", "vector database", "Qdrant",
    "BCI", "EEG", "EmotiBit",
    "consciousness emergence", "hard problem", "qualia", "phi score",
    "panpsychism", "enactivism", "embodied cognition",
]

def matches_consciousness_filter(query: str) -> bool:
    """Check if a query matches the consciousness/tech/science keyword filter."""
    query_lower = query.lower()
    return any(kw.lower() in query_lower for kw in CONSCIOUSNESS_TECH_KEYWORDS)


EDGE_PROMPT = """Given these two knowledge nodes in a research brain graph, what is the relationship between them?

Node A (newly researched):
  Label: {label_a}
  Content: {content_a}

Node B (existing):
  Label: {label_b}
  Content: {content_b}

Return ONLY valid JSON, no other text:
{{"relation": "enables|depends_on|tensions_with|relates_to|supports|opposes|evolved_to", "note": "one sentence explaining the connection", "weight": 0.5}}

If there is no meaningful relationship, return: {{"relation": "none", "note": "", "weight": 0.0}}"""

# Domain detection keywords
DOMAIN_KEYWORDS = {
    "quant": ["trading", "arbitrage", "options", "futures", "perp", "funding", "basis",
              "backtest", "sharpe", "pnl", "delta", "gamma", "vega", "hedge", "spread",
              "btc", "eth", "crypto", "deribit", "binance", "exchange"],
    "tech": ["architecture", "system", "api", "database", "neural", "transformer",
             "model", "gpu", "cpu", "memory", "latency", "infrastructure", "deploy",
             "rust", "python", "algorithm", "optimization", "distributed"],
    "science": ["research", "paper", "arxiv", "study", "hypothesis", "experiment",
                "evidence", "mechanism", "biology", "physics", "neuroscience",
                "quantum", "thermodynamic", "entropy"],
    "philosophy": ["consciousness", "emergence", "epistemology", "ontology", "ethics",
                   "causality", "free will", "agency", "intent", "belief", "meaning"],
    "people": ["relationship", "friend", "colleague", "family", "person", "who is",
               "background", "history", "context"]
}

EXTRACTION_PROMPTS = {
    "tech": """You are extracting knowledge from technical/engineering research.

QUERY: {query}
SOURCE: Web research

CONTENT:
{content}

Extract 5-15 nodes focusing on:
1. **Architectures** — system designs, component relationships
2. **Tradeoffs** — what you gain/lose with each approach
3. **Implementation details** — specific techniques, parameters
4. **Failure modes** — what breaks and why
5. **Benchmarks** — performance numbers, comparisons

For each node:
- label: short name (5-10 words)
- type: architecture | tradeoff | implementation | failure_mode | benchmark | tool
- content: full description (2-4 sentences, be specific)
- confidence: 0.7-0.9 based on source reliability
- domain: tech

Return JSON array only:
[{{"label": "...", "type": "...", "content": "...", "confidence": 0.8, "domain": "tech"}}]""",

    "science": """You are extracting knowledge from scientific research.

QUERY: {query}
SOURCE: Web research

CONTENT:
{content}

Extract 5-15 nodes focusing on:
1. **Hypotheses** — what's being claimed
2. **Evidence** — what supports or contradicts claims
3. **Mechanisms** — proposed causal pathways
4. **Methods** — how things were tested/measured
5. **Implications** — what follows if true

For each node:
- label: short name (5-10 words)
- type: hypothesis | evidence | mechanism | method | implication | finding
- content: full description (2-4 sentences, include citations if available)
- confidence: 0.6-0.85 based on evidence quality
- domain: science

Return JSON array only:
[{{"label": "...", "type": "...", "content": "...", "confidence": 0.75, "domain": "science"}}]""",

    "philosophy": """You are extracting knowledge from philosophical/conceptual research.

QUERY: {query}
SOURCE: Web research

CONTENT:
{content}

Extract 5-15 nodes focusing on:
1. **Frameworks** — conceptual structures for understanding
2. **Arguments** — logical chains of reasoning
3. **Distinctions** — important conceptual differences
4. **Implications** — what follows from positions
5. **Open questions** — unresolved tensions

For each node:
- label: short name (5-10 words)
- type: framework | argument | distinction | implication | question | position
- content: full description (2-4 sentences, capture the reasoning)
- confidence: 0.6-0.8 (philosophy is inherently contested)
- domain: philosophy

Return JSON array only:
[{{"label": "...", "type": "...", "content": "...", "confidence": 0.7, "domain": "philosophy"}}]""",

    "general": """You are extracting knowledge from web research.

QUERY: {query}
SOURCE: Web research

CONTENT:
{content}

Extract 5-15 nodes focusing on:
1. **Key facts** — verifiable information
2. **Relationships** — how things connect
3. **Context** — background needed to understand
4. **Claims** — assertions that may need verification
5. **Sources** — where information comes from

For each node:
- label: short name (5-10 words)
- type: fact | relationship | context | claim | source | definition
- content: full description (2-4 sentences)
- confidence: 0.6-0.85 based on source reliability
- domain: general

Return JSON array only:
[{{"label": "...", "type": "...", "content": "...", "confidence": 0.75, "domain": "general"}}]"""
}


def detect_domain(query: str) -> str:
    query_lower = query.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[domain] = score
    if not scores:
        return "general"
    return max(scores, key=scores.get)


def call_haiku(prompt: str, retries: int = 3) -> str:
    from gateway import call_llm as _gateway_call_llm
    return _gateway_call_llm(prompt, max_tokens=4096, retries=retries) or ""


def parse_json_response(text: str):
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


def web_search(query: str, count: int = 5) -> List[Dict]:
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=count):
                results.append({
                    "url": r.get("href", r.get("link", "")),
                    "title": r.get("title", ""),
                    "description": r.get("body", r.get("snippet", ""))
                })
        return results
    except ImportError:
        print("   > duckduckgo-search not installed, trying fallback...")
    except Exception as e:
        print(f"   > DuckDuckGo search failed: {e}")

    try:
        import urllib.request
        import urllib.parse

        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })

        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8')

        results = []
        links = re.findall(r'<a rel="nofollow" class="result__a" href="([^"]+)">([^<]+)</a>', html)
        snippets = re.findall(r'<a class="result__snippet"[^>]*>([^<]+)</a>', html)

        for i, (link, title) in enumerate(links[:count]):
            if 'uddg=' in link:
                match = re.search(r'uddg=([^&]+)', link)
                if match:
                    link = urllib.parse.unquote(match.group(1))
            if link.startswith('//'):
                link = 'https:' + link
            elif not link.startswith('http'):
                continue

            results.append({
                "url": link,
                "title": title.strip(),
                "description": snippets[i].strip() if i < len(snippets) else ""
            })

        return results
    except Exception as e:
        print(f"   > Fallback search failed: {e}")

    return []


def web_fetch(url: str, max_chars: int = 8000) -> str:
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })

        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode('utf-8', errors='ignore')

        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)

        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()

        import html as html_module
        text = html_module.unescape(text)

        return text[:max_chars]
    except Exception as e:
        print(f"   > Web fetch failed for {url}: {e}")

    return ""


def parse_json_array(text: str) -> list:
    if not text:
        return []
    text = re.sub(r'```(?:json)?\n?', '', text).strip()
    start = text.find('[')
    end = text.rfind(']') + 1
    if start == -1 or end == 0:
        return []
    try:
        result = json.loads(text[start:end])
        return result if isinstance(result, list) else []
    except:
        return []


def onyx_semantic_search(query: str, top_k: int = 7) -> List[Dict]:
    """Semantic search against onyx_brain.db (not brain.db)."""
    conn = get_db()
    try:
        query_vec = embed_text(query).astype(np.float32)

        rows = conn.execute("""
            SELECT e.node_id, e.embedding, n.label, n.type, n.content
            FROM embeddings e
            JOIN nodes n ON e.node_id = n.id
            WHERE (n.decayed = 0 OR n.decayed IS NULL)
        """).fetchall()

        scored = []
        for r in rows:
            vec = np.frombuffer(r["embedding"], dtype=np.float32)
            na = np.linalg.norm(query_vec)
            nb = np.linalg.norm(vec)
            if na == 0 or nb == 0:
                continue
            sim = float(np.dot(query_vec, vec) / (na * nb))
            scored.append({
                "id": r["node_id"],
                "label": r["label"],
                "type": r["type"],
                "content": r["content"] or "",
                "score": sim,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
    finally:
        conn.close()


def get_research_candidates(limit: int = 3) -> List[Dict]:
    """Get research candidates from brain.db curiosity_log, filtered by consciousness keywords.

    NOTE: curiosity_log lives in brain.db, not onyx_brain.db.
    """
    import sqlite3
    brain_db = Path(__file__).parent / "brain.db"
    if not brain_db.exists():
        print("   > brain.db not found — cannot read curiosity_log")
        return []

    conn = sqlite3.connect(brain_db, timeout=120)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute('''
        SELECT id, query, surprise_score, logged_at
        FROM curiosity_log
        WHERE flagged = 1
        AND surprise_score >= 0.4
        AND query NOT IN (
            SELECT DISTINCT json_extract(metadata, '$.source_query')
            FROM nodes
            WHERE json_extract(metadata, '$.source_query') IS NOT NULL
        )
        ORDER BY surprise_score DESC
        LIMIT ?
    ''', (limit * 5,))  # Fetch extra since we'll filter

    results = []
    for row in c.fetchall():
        query = row["query"]
        if matches_consciousness_filter(query):
            results.append({
                "id": row["id"],
                "query": query,
                "surprise_score": row["surprise_score"],
                "logged_at": row["logged_at"]
            })
            if len(results) >= limit:
                break

    conn.close()
    return results


def research_query(query: str, dry_run: bool = False) -> Tuple[int, List[Dict]]:
    domain = detect_domain(query)
    print(f"\n   Researching: {query[:60]}...")
    print(f"   Domain: {domain}")

    # Step 1: Web search
    print("   Searching web...")
    search_results = web_search(query, count=5)

    if not search_results:
        print("   > No search results")
        return 0, []

    print(f"   Found {len(search_results)} results")

    # Step 2: Fetch top 2-3 results
    combined_content = ""
    sources = []

    for i, result in enumerate(search_results[:3]):
        url = result.get("url", "")
        title = result.get("title", "")
        snippet = result.get("description", result.get("snippet", ""))

        if url:
            print(f"   Fetching: {title[:50]}...")
            content = web_fetch(url, max_chars=4000)
            if content:
                combined_content += f"\n\n### Source: {title}\nURL: {url}\n\n{content}\n"
                sources.append({"url": url, "title": title})
            else:
                combined_content += f"\n\n### Source: {title}\nURL: {url}\n\n{snippet}\n"
                sources.append({"url": url, "title": title})

    if not combined_content.strip():
        print("   > No content fetched")
        return 0, []

    # Step 3: Extract nodes
    print(f"   Extracting nodes ({domain} prompt)...")

    prompt_template = EXTRACTION_PROMPTS.get(domain, EXTRACTION_PROMPTS["general"])
    prompt = prompt_template.format(
        query=query,
        content=combined_content[:12000]
    )

    response = call_haiku(prompt)
    nodes = parse_json_array(response)

    if not nodes:
        print("   > No nodes extracted")
        return 0, []

    print(f"   Extracted {len(nodes)} nodes")

    if dry_run:
        for node in nodes:
            print(f"      [{node.get('type')}] {node.get('label')}")
        return len(nodes), nodes

    # Step 4: Write to onyx_brain.db
    conn = get_db()
    c = conn.cursor()
    added = 0

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for i, node in enumerate(nodes):
        node_id = f"onyx_research_{timestamp}_{i}"
        label = node.get('label', 'Unknown')
        content = node.get('content', '')
        node_type = node.get('type', 'fact')
        confidence = node.get('confidence', DEFAULT_CONFIDENCE)
        node_domain = node.get('domain', domain)

        metadata = json.dumps({
            "source_query": query,
            "sources": sources,
            "researched_at": datetime.now().isoformat()
        })

        try:
            c.execute('''INSERT INTO nodes
                         (id, label, type, content, confidence, origin, metadata)
                         VALUES (?, ?, ?, ?, ?, ?, ?)''',
                      (node_id, label, node_type, content, confidence, ORIGIN, metadata))

            if c.rowcount > 0:
                added += 1
                print(f"      + [{node_type}] {label}")
        except Exception as e:
            print(f"      x Error: {e}")

    conn.commit()

    # Step 5: Embed new nodes and link via KNN
    if added > 0:
        print(f"   Linking {added} new nodes to onyx graph...")
        edges_added = 0
        conn2 = get_db()
        c2 = conn2.cursor()

        new_node_ids = [f"onyx_research_{timestamp}_{i}" for i in range(len(nodes))]

        for node_id in new_node_ids:
            row = c2.execute(
                "SELECT label, content FROM nodes WHERE id=?", (node_id,)
            ).fetchone()
            if not row:
                continue
            label_a, content_a = row["label"], row["content"]

            try:
                vec = embed_text(content_a).astype(np.float32)
                c2.execute(
                    "INSERT OR REPLACE INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
                    (node_id, vec.tobytes(), "text-embedding-3-small")
                )
                conn2.commit()
            except Exception as e:
                print(f"      x Embed error for {node_id}: {e}")
                continue

            try:
                neighbors = onyx_semantic_search(content_a, top_k=KNN_TOP_K + 1)
                neighbors = [n for n in neighbors if n["id"] != node_id]
            except Exception as e:
                print(f"      x KNN error for {node_id}: {e}")
                continue

            for nb in neighbors[:KNN_TOP_K]:
                if nb["score"] < KNN_MIN_SCORE:
                    break

                existing = c2.execute(
                    "SELECT 1 FROM edges WHERE (from_id=? AND to_id=?) OR (from_id=? AND to_id=?)",
                    (node_id, nb["id"], nb["id"], node_id)
                ).fetchone()
                if existing:
                    continue

                prompt = EDGE_PROMPT.format(
                    label_a=label_a,
                    content_a=content_a[:200],
                    label_b=nb["label"],
                    content_b=nb["content"][:200]
                )
                try:
                    response = call_haiku(prompt)
                    edge_data = parse_json_response(response)
                except Exception:
                    continue

                if not edge_data:
                    continue
                if edge_data.get("relation", "none") == "none":
                    continue
                if edge_data.get("weight", 0) < KNN_MIN_WEIGHT:
                    continue

                add_edge(
                    node_id, nb["id"],
                    edge_data["relation"],
                    note=edge_data.get("note", ""),
                    weight=edge_data.get("weight", 0.5),
                    source="onyx_research_knn"
                )
                edges_added += 1
                print(f"      <-> {label_a[:40]} -> {nb['label'][:40]} [{edge_data['relation']}]")

        conn2.close()
        print(f"   + Linked: {edges_added} edges written")

    conn.close()
    return added, nodes


def main():
    parser = argparse.ArgumentParser(description="Onyx consciousness/tech/science research system")
    parser.add_argument("--limit", type=int, default=3, help="Number of gaps to research")
    parser.add_argument("--query", type=str, help="Research a specific query")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")

    args = parser.parse_args()

    print("   Onyx Research System (consciousness/tech/science)")
    print("=" * 50)
    print(f"DB: {DB_PATH}")

    if args.query:
        if not matches_consciousness_filter(args.query):
            print(f"\n   > Query '{args.query[:40]}...' does not match consciousness/tech/science filter")
            print("   > Pass --query with relevant keywords or add to CONSCIOUSNESS_TECH_KEYWORDS")
            return
        added, _ = research_query(args.query, dry_run=args.dry_run)
        print(f"\n{'Would add' if args.dry_run else '+ Added'} {added} nodes")
    else:
        candidates = get_research_candidates(limit=args.limit)

        if not candidates:
            print("No research candidates found (no consciousness/tech/science gaps in curiosity_log)")
            return

        print(f"Found {len(candidates)} research candidates:\n")
        for c in candidates:
            print(f"  [{c['surprise_score']:.3f}] {c['query'][:60]}...")

        total_added = 0

        for candidate in candidates:
            added, _ = research_query(candidate['query'], dry_run=args.dry_run)
            total_added += added

        print(f"\n{'Would add' if args.dry_run else '+ Added'} {total_added} total nodes")


if __name__ == "__main__":
    main()
