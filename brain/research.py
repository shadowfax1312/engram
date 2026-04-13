#!/usr/bin/env python3
"""
research.py — Autonomous research system for Onyx brain.

Pulls high-surprise gaps from curiosity_log, researches them via web search,
extracts findings, and writes to brain.db as origin=external nodes.

The extraction prompt varies by detected domain:
- quant/trading → formulas, mechanisms, edge cases
- tech/engineering → architectures, tradeoffs, implementations
- science/research → hypotheses, evidence, methods
- people/relationships → context, history, connections
- philosophy/meta → frameworks, arguments, implications

Usage:
  python3 research.py                    # Research top 3 gaps
  python3 research.py --limit 5          # Research top 5 gaps
  python3 research.py --query "topic"    # Research specific query
  python3 research.py --dry-run          # Preview without writing
"""

import sys
import os
import re
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from init_graph import get_db, add_node, add_edge
from embed import embed_text
from search import semantic_search

ORIGIN = "onyx"  # Research is Onyx's self-directed learning, not external import
DEFAULT_CONFIDENCE = 0.70  # Research findings start at 0.70, can be promoted via ruminate
KNN_TOP_K = 5           # Neighbors to consider for edge linking
KNN_MIN_SCORE = 0.60    # Min semantic similarity to attempt edge
KNN_MIN_WEIGHT = 0.30   # Min edge weight to actually write

EDGE_PROMPT = """Given these two knowledge nodes in a personal brain graph, what is the relationship between them?

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

# Domain-specific extraction prompts
EXTRACTION_PROMPTS = {
    "quant": """You are extracting knowledge from quant/trading research.

QUERY: {query}
SOURCE: Web research

CONTENT:
{content}

Extract 5-15 nodes focusing on:
1. **Formulas and models** — actual math, parameters, not prose
2. **Mechanisms** — how markets/systems work
3. **Edge cases and failure modes** — where strategies break
4. **Parameter relationships** — what inputs drive what outputs
5. **Actionable strategies** — specific approaches with conditions

For each node:
- label: short name (5-10 words)
- type: formula | mechanism | edge_case | strategy | parameter | market_structure
- content: full description (2-4 sentences, include numbers if present)
- confidence: 0.7-0.9 based on source reliability
- domain: quant

Return JSON array only:
[{{"label": "...", "type": "...", "content": "...", "confidence": 0.8, "domain": "quant"}}]""",

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
    """Detect the domain of a query based on keywords."""
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
    """Call Claude Haiku via OpenClaw local proxy."""
    import urllib.request
    for attempt in range(retries):
        try:
            data = json.dumps({
                "model": "claude-haiku-4",
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}]
            }).encode()
            req = urllib.request.Request(
                "http://localhost:3456/v1/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer not-needed"
                },
                method="POST"
            )
            r = urllib.request.urlopen(req, timeout=180)
            resp = json.loads(r.read())
            return resp.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"   ⚠ Haiku call failed attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return ""


def web_search(query: str, count: int = 5) -> List[Dict]:
    """Search web via DuckDuckGo (no API key needed)."""
    try:
        # Use ddgs Python package
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
        print("   ⚠ duckduckgo-search not installed, trying fallback...")
    except Exception as e:
        print(f"   ⚠ DuckDuckGo search failed: {e}")
    
    # Fallback: use requests to scrape DuckDuckGo HTML (basic)
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
        
        # Basic parsing of DuckDuckGo HTML results
        results = []
        import re
        
        # Extract redirect links and decode them
        links = re.findall(r'<a rel="nofollow" class="result__a" href="([^"]+)">([^<]+)</a>', html)
        snippets = re.findall(r'<a class="result__snippet"[^>]*>([^<]+)</a>', html)
        
        for i, (link, title) in enumerate(links[:count]):
            # Decode DuckDuckGo redirect URLs
            if 'uddg=' in link:
                import urllib.parse
                match = re.search(r'uddg=([^&]+)', link)
                if match:
                    link = urllib.parse.unquote(match.group(1))
            
            # Ensure proper URL format
            if link.startswith('//'):
                link = 'https:' + link
            elif not link.startswith('http'):
                continue  # Skip invalid URLs
            
            results.append({
                "url": link,
                "title": title.strip(),
                "description": snippets[i].strip() if i < len(snippets) else ""
            })
        
        return results
    except Exception as e:
        print(f"   ⚠ Fallback search failed: {e}")
    
    return []


def web_fetch(url: str, max_chars: int = 8000) -> str:
    """Fetch and extract content from a URL."""
    try:
        import urllib.request
        from html.parser import HTMLParser
        
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })
        
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
        
        # Simple HTML to text extraction
        # Remove script and style content
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Decode HTML entities
        import html as html_module
        text = html_module.unescape(text)
        
        return text[:max_chars]
    except Exception as e:
        print(f"   ⚠ Web fetch failed for {url}: {e}")
    
    return ""


def parse_json_array(text: str) -> list:
    """Extract JSON array from LLM response."""
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


def get_research_candidates(limit: int = 3) -> List[Dict]:
    """Get top research candidates from curiosity_log."""
    conn = get_db()
    c = conn.cursor()
    
    # Get high-surprise queries that haven't been researched yet
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
    ''', (limit,))
    
    results = []
    for row in c.fetchall():
        results.append({
            "id": row[0],
            "query": row[1],
            "surprise_score": row[2],
            "logged_at": row[3]
        })
    
    conn.close()
    return results


def research_query(query: str, dry_run: bool = False) -> Tuple[int, List[Dict]]:
    """Research a single query and extract nodes."""
    
    domain = detect_domain(query)
    print(f"\n🔍 Researching: {query[:60]}...")
    print(f"   Domain: {domain}")
    
    # Step 1: Web search
    print("   Searching web...")
    search_results = web_search(query, count=5)
    
    if not search_results:
        print("   ⚠ No search results")
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
                # Use snippet as fallback
                combined_content += f"\n\n### Source: {title}\nURL: {url}\n\n{snippet}\n"
                sources.append({"url": url, "title": title})
    
    if not combined_content.strip():
        print("   ⚠ No content fetched")
        return 0, []
    
    # Step 3: Extract nodes
    print(f"   Extracting nodes ({domain} prompt)...")
    
    prompt_template = EXTRACTION_PROMPTS.get(domain, EXTRACTION_PROMPTS["general"])
    prompt = prompt_template.format(
        query=query,
        content=combined_content[:12000]  # Cap content length
    )
    
    response = call_haiku(prompt)
    nodes = parse_json_array(response)
    
    if not nodes:
        print("   ⚠ No nodes extracted")
        return 0, []
    
    print(f"   Extracted {len(nodes)} nodes")
    
    if dry_run:
        for node in nodes:
            print(f"      [{node.get('type')}] {node.get('label')}")
        return len(nodes), nodes
    
    # Step 4: Write to brain
    conn = get_db()
    c = conn.cursor()
    added = 0
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for i, node in enumerate(nodes):
        node_id = f"research_{timestamp}_{i}"
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
                         (id, label, type, content, confidence, origin, domain, metadata)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                      (node_id, label, node_type, content, confidence, ORIGIN, node_domain, metadata))
            
            if c.rowcount > 0:
                added += 1
                print(f"      ✓ [{node_type}] {label}")
        except Exception as e:
            print(f"      ✗ Error: {e}")
    
    conn.commit()

    # Step 5: Embed new nodes and link to graph via KNN
    if added > 0:
        print(f"   Linking {added} new nodes to graph...")
        edges_added = 0
        conn2 = get_db()
        c2 = conn2.cursor()

        # Get the node_ids we just wrote
        new_node_ids = [
            f"research_{timestamp}_{i}"
            for i in range(len(nodes))
        ]

        for node_id in new_node_ids:
            row = c2.execute(
                "SELECT label, content FROM nodes WHERE id=?", (node_id,)
            ).fetchone()
            if not row:
                continue
            label_a, content_a = row["label"], row["content"]

            # Embed and store
            try:
                vec = embed_text(content_a).astype(np.float32)
                c2.execute(
                    "INSERT OR REPLACE INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
                    (node_id, vec.tobytes(), "text-embedding-3-small")
                )
                conn2.commit()
            except Exception as e:
                print(f"      ✗ Embed error for {node_id}: {e}")
                continue

            # Find nearest neighbors (excluding self)
            try:
                neighbors = semantic_search(content_a, top_k=KNN_TOP_K + 1)
                neighbors = [n for n in neighbors if n["id"] != node_id]
            except Exception as e:
                print(f"      ✗ KNN error for {node_id}: {e}")
                continue

            for nb in neighbors[:KNN_TOP_K]:
                if nb["score"] < KNN_MIN_SCORE:
                    break

                # Skip if edge already exists
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
                    source="research_knn"
                )
                edges_added += 1
                print(f"      ↔ {label_a[:40]} → {nb['label'][:40]} [{edge_data['relation']}]")

        conn2.close()
        print(f"   ✓ Linked: {edges_added} edges written")

    conn.close()

    return added, nodes


def mark_query_researched(query: str):
    """Update curiosity_log to mark query as researched."""
    conn = get_db()
    c = conn.cursor()
    
    # Decay surprise score for researched queries
    c.execute('''
        UPDATE curiosity_log 
        SET surprise_score = surprise_score * 0.5,
            flagged = 0
        WHERE query = ?
    ''', (query,))
    
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Autonomous research system for Onyx brain")
    parser.add_argument("--limit", type=int, default=3, help="Number of gaps to research")
    parser.add_argument("--query", type=str, help="Research a specific query")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--skip-decay", action="store_true", help="Don't decay curiosity scores")
    
    args = parser.parse_args()
    
    print("🧠 Onyx Research System")
    print("=" * 50)
    
    if args.query:
        # Research specific query
        added, _ = research_query(args.query, dry_run=args.dry_run)
        print(f"\n{'Would add' if args.dry_run else '✅ Added'} {added} nodes")
    else:
        # Research top gaps from curiosity_log
        candidates = get_research_candidates(limit=args.limit)
        
        if not candidates:
            print("No research candidates found (all high-surprise queries already researched)")
            return
        
        print(f"Found {len(candidates)} research candidates:\n")
        for c in candidates:
            print(f"  [{c['surprise_score']:.3f}] {c['query'][:60]}...")
        
        total_added = 0
        
        for candidate in candidates:
            added, _ = research_query(candidate['query'], dry_run=args.dry_run)
            total_added += added
            
            if not args.dry_run and not args.skip_decay and added > 0:
                mark_query_researched(candidate['query'])
        
        print(f"\n{'Would add' if args.dry_run else '✅ Added'} {total_added} total nodes")


if __name__ == "__main__":
    main()
