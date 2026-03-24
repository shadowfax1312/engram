#!/usr/bin/env python3
"""
Engram — Extract knowledge from daily memory files (markdown notes).

Two-pass extraction:
  Pass 1: Extract facts/entities as nodes
  Pass 2: KNN placement (k=3, threshold 0.72) so new nodes never orphan

Checkpointed by byte offset — only new content is processed.

Configuration:
  ENGRAM_MEMORY_DIR   — path to markdown notes directory
  ENGRAM_LLM_ENDPOINT — LLM API endpoint
  ENGRAM_EXTRACT_MODEL — model for extraction (default: claude-haiku-4-5)

Run: python3 -m extractors.memory [--dry-run]
"""

import os
import json
import re
import time
import urllib.request
import numpy as np
from pathlib import Path
from datetime import datetime

from brain import get_db, add_node, add_edge, embed_text, BRAIN_DIR

MEMORY_DIR = Path(os.environ.get("ENGRAM_MEMORY_DIR", Path.home() / "notes"))
CHECKPOINT_FILE = BRAIN_DIR / "extract_memory_checkpoint.json"
LLM_ENDPOINT = os.environ.get("ENGRAM_LLM_ENDPOINT", "http://localhost:3456/v1/chat/completions")
LLM_MODEL = os.environ.get("ENGRAM_EXTRACT_MODEL", "claude-haiku-4-5")
KNN_K = 3
KNN_THRESHOLD = 0.72
BATCH_SIZE = int(os.environ.get("ENGRAM_BATCH_SIZE", "20"))


def _load_checkpoint():
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {}


def _save_checkpoint(ckpt):
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(ckpt, indent=2))


def call_llm(prompt, retries=3):
    """Call LLM for extraction."""
    for attempt in range(retries):
        try:
            data = json.dumps({
                "model": LLM_MODEL,
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}]
            }).encode()
            req = urllib.request.Request(
                LLM_ENDPOINT, data=data,
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer not-needed"},
                method="POST"
            )
            r = urllib.request.urlopen(req, timeout=120)
            resp = json.loads(r.read())
            text = resp.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if text:
                return text
        except Exception as e:
            print(f"   LLM call failed attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def _parse_json_array(text):
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


def _cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def knn_placement(node_id, node_embedding):
    """Connect a new node to its k nearest neighbors above threshold."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT e.node_id, e.embedding FROM embeddings e WHERE e.node_id != ?",
            (node_id,)
        ).fetchall()
        scored = []
        for r in rows:
            vec = np.frombuffer(r["embedding"], dtype=np.float32)
            if np.linalg.norm(vec) == 0:
                continue
            sim = _cosine_sim(node_embedding, vec)
            if sim >= KNN_THRESHOLD:
                scored.append((r["node_id"], sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        for neighbor_id, sim in scored[:KNN_K]:
            add_edge(from_id=node_id, to_id=neighbor_id,
                     relation="relates_to",
                     note=f"knn_placement sim={sim:.3f}",
                     source="extract_memory")
    finally:
        conn.close()


EXTRACT_PROMPT = """Extract knowledge nodes from this text. Each node should capture a discrete fact, insight, decision, question, or pattern.

Return ONLY a valid JSON array:
[
  {{
    "id": "snake_case_unique_id",
    "label": "Short Human Label",
    "type": "thesis|concept|fact|decision|question|person|org|event",
    "content": "1-3 sentence description capturing the full meaning"
  }}
]

If nothing worth extracting, return [].

TEXT:
{text}"""


def extract_from_file(filepath, dry_run=False):
    """Extract nodes from a single file, respecting byte offset checkpoint."""
    ckpt = _load_checkpoint()
    fpath_str = str(filepath)
    last_offset = ckpt.get(fpath_str, 0)

    content = filepath.read_text(encoding="utf-8", errors="replace")
    if len(content.encode("utf-8")) <= last_offset:
        return 0

    new_content = content.encode("utf-8")[last_offset:].decode("utf-8", errors="replace")
    if len(new_content.strip()) < 50:
        return 0

    print(f"  Processing {filepath.name} ({len(new_content)} new bytes)...")

    prompt = EXTRACT_PROMPT.format(text=new_content[:8000])
    response = call_llm(prompt)
    nodes = _parse_json_array(response)

    if dry_run:
        print(f"    Would extract {len(nodes)} nodes")
        return len(nodes)

    added = 0
    for node in nodes:
        nid = node.get("id")
        if not nid:
            continue
        nid = f"mem_{filepath.stem}_{nid}"
        content_text = node.get("content", "")
        add_node(id=nid, label=node.get("label", "Unknown"),
                 type=node.get("type", "concept"),
                 content=content_text, confidence=0.85,
                 source="second_brain")
        if content_text:
            vec = embed_text(content_text)
            knn_placement(nid, vec)
        added += 1

    ckpt[fpath_str] = len(content.encode("utf-8"))
    _save_checkpoint(ckpt)
    print(f"    Extracted {added} nodes from {filepath.name}")
    return added


def run(dry_run=False):
    """Scan memory directory and extract from all .md files."""
    if not MEMORY_DIR.exists():
        print(f"Memory directory not found: {MEMORY_DIR}")
        print(f"Set ENGRAM_MEMORY_DIR to your notes directory.")
        return

    files = sorted(MEMORY_DIR.glob("*.md"))
    if not files:
        print("No .md files found in memory directory")
        return

    total = 0
    for f in files:
        try:
            total += extract_from_file(f, dry_run=dry_run)
        except Exception as e:
            print(f"  Error processing {f.name}: {e}")

    print(f"\nExtraction complete: {total} nodes from {len(files)} files")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract from daily memory files")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
