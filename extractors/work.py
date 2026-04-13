#!/usr/bin/env python3
"""
Engram — Extract knowledge from work/professional context.

Processes emails, meeting notes, and work chat exports.
Extracts professional decisions, project context, and relationship data.

Configuration:
  ENGRAM_WORK_DIR      — path to work documents directory
  ENGRAM_LLM_ENDPOINT  — LLM API endpoint
  ENGRAM_EXTRACT_MODEL — model for extraction (default: claude-haiku-4-5)
  ENGRAM_BATCH_SIZE    — files per batch (default: 10)

Run: python3 -m extractors.work [--dry-run] [--source emails|meetings|chats]
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

WORK_DIR = Path(os.environ.get("ENGRAM_WORK_DIR", Path.home() / "work-notes"))
LLM_ENDPOINT = os.environ.get("ENGRAM_LLM_ENDPOINT", "http://localhost:3456/v1/chat/completions")
LLM_MODEL = os.environ.get("ENGRAM_EXTRACT_MODEL", "claude-haiku-4-5")
BATCH_SIZE = int(os.environ.get("ENGRAM_BATCH_SIZE", "10"))
CHECKPOINT_FILE = BRAIN_DIR / "work_extract_checkpoint.json"


def _load_checkpoint() -> Dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"processed_files": {}}


def _save_checkpoint(ckpt: Dict):
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(ckpt, indent=2))


def call_llm(prompt: str, retries: int = 3) -> Optional[str]:
    for attempt in range(retries):
        try:
            data = json.dumps({
                "model": LLM_MODEL, "max_tokens": 2048,
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
                source="work_extract", origin_date=None, origin="self"):
    """Insert a node with embedding."""
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


WORK_EXTRACTION_PROMPT = """Extract professional knowledge from this work document.

Source: {source_type}
File: {filename}

Content:
{text}

Extract 2-6 knowledge nodes about:
- Decisions made and their reasoning
- Project context and status
- Professional relationships and roles
- Technical or business insights
- Open questions or unresolved issues

Return JSON array only:
[
  {{"label": "short label (5-10 words)", "type": "decision|insight|observation|question|fact|project", "content": "full description (1-3 sentences)", "confidence": 0.7}}
]

If nothing substantive, return: []"""


def extract_from_file(filepath: Path, source_type: str = "document",
                      dry_run: bool = False) -> int:
    """Extract knowledge from a single work document."""
    content = filepath.read_text(encoding="utf-8", errors="replace")
    if len(content.strip()) < 100:
        return 0

    prompt = WORK_EXTRACTION_PROMPT.format(
        source_type=source_type,
        filename=filepath.name,
        text=content[:8000]
    )
    response = call_llm(prompt)
    nodes = parse_json_array(response)
    if not nodes:
        return 0

    if dry_run:
        print(f"   Would extract {len(nodes)} nodes from {filepath.name}")
        return len(nodes)

    conn = get_db()
    added = 0
    stem = filepath.stem

    for node in nodes:
        label = node.get("label", "")
        node_content = node.get("content", "")
        if not label or not node_content:
            continue
        node_id = "work_" + hashlib.md5(
            f"{stem}_{label}".encode()
        ).hexdigest()[:12]
        insert_node(
            conn, node_id, label,
            node.get("type", "observation"),
            node_content,
            confidence=node.get("confidence", 0.8),
            source=f"work_{source_type}",
            origin_date=datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
        )
        added += 1

    conn.commit()
    conn.close()
    return added


def run(dry_run: bool = False, source: str = None):
    """Process all work documents."""
    if not WORK_DIR.exists():
        print(f"Work directory not found: {WORK_DIR}")
        print(f"Set ENGRAM_WORK_DIR to your work documents directory.")
        return

    ckpt = _load_checkpoint()
    processed = ckpt.get("processed_files", {})

    # Find all text-like files
    extensions = {".md", ".txt", ".eml", ".org", ".rst"}
    files = sorted(
        f for f in WORK_DIR.rglob("*")
        if f.suffix in extensions and f.is_file()
    )

    to_process = []
    for f in files:
        key = str(f.relative_to(WORK_DIR))
        mtime = f.stat().st_mtime
        if key in processed and processed[key].get("mtime", 0) >= mtime:
            continue
        to_process.append(f)

    print(f"Found {len(to_process)} work documents to process")
    total = 0

    for i, f in enumerate(to_process):
        print(f"\n[{i+1}/{len(to_process)}] {f.name}...")
        try:
            source_type = "meeting" if "meeting" in f.name.lower() else \
                          "email" if f.suffix == ".eml" else "document"
            added = extract_from_file(f, source_type=source_type, dry_run=dry_run)
            total += added
            if not dry_run:
                key = str(f.relative_to(WORK_DIR))
                processed[key] = {
                    "mtime": f.stat().st_mtime,
                    "nodes": added,
                    "processed_at": datetime.now().isoformat()
                }
                _save_checkpoint({"processed_files": processed})
        except Exception as e:
            print(f"   Error: {e}")

    print(f"\n{'DRY RUN ' if dry_run else ''}Work extraction done: {total} nodes")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract from work documents")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source", choices=["emails", "meetings", "chats"])
    args = parser.parse_args()
    run(dry_run=args.dry_run, source=args.source)
