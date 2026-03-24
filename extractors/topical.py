#!/usr/bin/env python3
"""
Engram — Focused topical extraction from dedicated research threads.

Unlike other extractors that use Haiku for speed, the model here is
configurable (defaults to Sonnet) for higher quality extraction from
deep-dive conversations about specific topics.

Configuration:
  ENGRAM_TOPICAL_DIR      — path to topical thread exports
  ENGRAM_LLM_ENDPOINT     — LLM API endpoint
  ENGRAM_TOPICAL_MODEL    — model for extraction (default: claude-sonnet-4-6)
  ENGRAM_TOPICAL_DOMAIN   — domain tag for extracted nodes (default: topical)
  ENGRAM_BATCH_SIZE       — conversations per batch (default: 5)

Run: python3 -m extractors.topical [--dry-run] [--domain DOMAIN] [--model MODEL]
"""

import os
import re
import json
import time
import hashlib
import urllib.request
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from brain import get_db, embed_text, EMBEDDING_MODEL, BRAIN_DIR

TOPICAL_DIR = Path(os.environ.get("ENGRAM_TOPICAL_DIR", Path.home() / "research"))
LLM_ENDPOINT = os.environ.get("ENGRAM_LLM_ENDPOINT", "http://localhost:3456/v1/chat/completions")
LLM_MODEL = os.environ.get("ENGRAM_TOPICAL_MODEL", "claude-sonnet-4-6")
DOMAIN_TAG = os.environ.get("ENGRAM_TOPICAL_DOMAIN", "topical")
BATCH_SIZE = int(os.environ.get("ENGRAM_BATCH_SIZE", "5"))
CHECKPOINT_FILE = BRAIN_DIR / "topical_extract_checkpoint.json"


def _load_checkpoint() -> Dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"processed_files": {}}


def _save_checkpoint(ckpt: Dict):
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(ckpt, indent=2))


def call_llm(prompt: str, model: str = None, retries: int = 3) -> Optional[str]:
    model = model or LLM_MODEL
    for attempt in range(retries):
        try:
            data = json.dumps({
                "model": model, "max_tokens": 4096,
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
        except Exception as e:
            print(f"   LLM call failed attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def parse_json_array(text: str) -> List[Dict]:
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
    except Exception:
        return []


def insert_node(conn, node_id, label, ntype, content, confidence=0.85,
                source="topical_extract", origin_date=None, origin="self"):
    c = conn.cursor()
    c.execute("""
        INSERT INTO nodes (id, label, type, content, confidence, source, origin_date, origin, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}')
        ON CONFLICT(id) DO UPDATE SET
            label=excluded.label, content=excluded.content,
            confidence=excluded.confidence, updated_at=datetime('now')
    """, (node_id, label, ntype, content, confidence, source, origin_date, origin))
    if content:
        vec = embed_text(content).astype(np.float32)
        c.execute(
            "INSERT OR REPLACE INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
            (node_id, vec.tobytes(), EMBEDDING_MODEL)
        )


TOPICAL_PROMPT = """You are performing deep knowledge extraction from a focused research thread.

Domain: {domain}
Source: {filename}

Content:
{text}

Extract 3-8 knowledge nodes. This is a deep-dive topic — extract:
- Core theses and arguments
- Supporting evidence and data points
- Contradictions or open tensions
- Decisions and their reasoning
- Cross-domain connections
- Emergent patterns

Each node must be self-contained and reference specific claims, not vague summaries.

Return JSON array:
[
  {{
    "label": "specific, descriptive label (5-12 words)",
    "type": "thesis|concept|fact|decision|question|pattern|contradiction",
    "content": "detailed description (2-4 sentences) with specific claims",
    "confidence": 0.7,
    "tags": ["tag1", "tag2"]
  }}
]

If nothing substantive, return: []"""


def extract_from_file(filepath: Path, domain: str, model: str = None,
                      dry_run: bool = False) -> int:
    """Extract knowledge from a single topical document."""
    content = filepath.read_text(encoding="utf-8", errors="replace")
    if len(content.strip()) < 200:
        return 0

    prompt = TOPICAL_PROMPT.format(
        domain=domain, filename=filepath.name, text=content[:12000]
    )
    response = call_llm(prompt, model=model)
    nodes = parse_json_array(response)
    if not nodes:
        return 0

    if dry_run:
        print(f"   Would extract {len(nodes)} nodes from {filepath.name}")
        for n in nodes[:3]:
            print(f"     - [{n.get('type')}] {n.get('label')}")
        return len(nodes)

    conn = get_db()
    added = 0
    stem = filepath.stem

    for node in nodes:
        label = node.get("label", "")
        node_content = node.get("content", "")
        if not label or not node_content:
            continue
        tags = node.get("tags", [])
        node_id = f"topic_{domain}_{hashlib.md5(f'{stem}_{label}'.encode()).hexdigest()[:12]}"
        metadata = json.dumps({"tags": tags, "domain": domain})

        c = conn.cursor()
        c.execute("""
            INSERT INTO nodes (id, label, type, content, confidence, source, origin_date, origin, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                label=excluded.label, content=excluded.content,
                confidence=excluded.confidence, updated_at=datetime('now'),
                metadata=excluded.metadata
        """, (node_id, label, node.get("type", "concept"), node_content,
              node.get("confidence", 0.85), f"topical_{domain}",
              datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
              "self", metadata))

        if node_content:
            vec = embed_text(node_content).astype(np.float32)
            c.execute(
                "INSERT OR REPLACE INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
                (node_id, vec.tobytes(), EMBEDDING_MODEL)
            )
        added += 1

    conn.commit()
    conn.close()
    return added


def run(dry_run: bool = False, domain: str = None, model: str = None):
    """Process all topical documents."""
    domain = domain or DOMAIN_TAG
    model = model or LLM_MODEL

    if not TOPICAL_DIR.exists():
        print(f"Topical directory not found: {TOPICAL_DIR}")
        print(f"Set ENGRAM_TOPICAL_DIR to your research threads directory.")
        return

    ckpt = _load_checkpoint()
    processed = ckpt.get("processed_files", {})

    files = sorted(
        f for f in TOPICAL_DIR.rglob("*")
        if f.suffix in {".md", ".txt", ".org", ".rst"} and f.is_file()
    )

    to_process = []
    for f in files:
        key = str(f.relative_to(TOPICAL_DIR))
        mtime = f.stat().st_mtime
        if key in processed and processed[key].get("mtime", 0) >= mtime:
            continue
        to_process.append(f)

    print(f"Found {len(to_process)} topical documents (domain: {domain}, model: {model})")
    total = 0

    for i, f in enumerate(to_process):
        print(f"\n[{i+1}/{len(to_process)}] {f.name}...")
        try:
            added = extract_from_file(f, domain=domain, model=model, dry_run=dry_run)
            total += added
            if not dry_run:
                key = str(f.relative_to(TOPICAL_DIR))
                processed[key] = {
                    "mtime": f.stat().st_mtime, "nodes": added,
                    "processed_at": datetime.now().isoformat()
                }
                _save_checkpoint({"processed_files": processed})
        except Exception as e:
            print(f"   Error: {e}")

    print(f"\n{'DRY RUN ' if dry_run else ''}Topical extraction done: {total} nodes (domain: {domain})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract from topical research threads")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--domain", default=None, help="Domain tag")
    parser.add_argument("--model", default=None, help="LLM model override")
    args = parser.parse_args()
    run(dry_run=args.dry_run, domain=args.domain, model=args.model)
