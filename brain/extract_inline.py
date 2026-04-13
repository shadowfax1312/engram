#!/usr/bin/env python3
"""
extract_inline.py — Real-time conversation extractor.

Reads a staging markdown file, sends it to Haiku via gateway,
extracts knowledge nodes, writes them directly to brain.db.
Called by Onyx during/after conversations — no batch lag.

Usage:
  python3 extract_inline.py                         # reads /tmp/onyx-extract.md
  python3 extract_inline.py --input /tmp/myfile.md  # custom staging file
  python3 extract_inline.py --dry-run               # print nodes, don't write

Staging file format: free-form markdown. Just dump signal.
After successful extraction, staging file is cleared (not deleted).
"""

import sys
import os
import json
import re
import argparse
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from gateway import call_llm
from init_graph import get_db
from embed import embed_text, MODEL_NAME

STAGING_FILE = Path("/tmp/onyx-extract.md")

VALID_TYPES = {
    "decision", "belief", "observation", "insight",
    "fact", "correction", "preference", "pattern", "relationship"
}

EXTRACT_PROMPT = """You are extracting knowledge nodes from a conversation staging log.

Extract every discrete fact, decision, belief, insight, correction, or pattern worth remembering.
Be aggressive — if it would change how an AI assistant responds in a future session, extract it.

Two origin types:
- "self" = facts about the user (decisions they made, things they stated, preferences, events)
- "onyx" = Onyx's own synthesis (architectures explained, patterns identified, corrections made, frameworks built)

Return ONLY a valid JSON array:
[
  {{
    "label": "Short label (5-10 words)",
    "type": "decision|belief|observation|insight|fact|correction|preference|pattern|relationship",
    "origin": "self|onyx",
    "content": "1-3 sentence description capturing full meaning and context",
    "confidence": 0.7-1.0
  }}
]

Rules:
- Minimum confidence 0.70 — if uncertain, drop it
- Skip pleasantries, filler, meta-commentary about the conversation itself
- Capture the substance: what was decided, learned, corrected, or synthesized
- If nothing worth extracting: return []

STAGING LOG:
{text}"""


def _parse_nodes(text):
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


def _write_node(label, content, node_type, origin, confidence):
    """Write a single node + embedding to brain.db. Returns node_id."""
    if node_type not in VALID_TYPES:
        node_type = "observation"
    if origin not in ("self", "onyx", "external"):
        origin = "onyx"

    date_part = datetime.now().strftime("%Y-%m-%d")
    node_id = "rt_" + hashlib.md5(f"{date_part}_{label}".encode()).hexdigest()[:12]

    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO nodes (id, label, type, content, confidence, source,
                           origin_date, origin, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}')
        ON CONFLICT(id) DO UPDATE SET
            content=excluded.content,
            confidence=MAX(nodes.confidence, excluded.confidence),
            updated_at=datetime('now')
    """, (node_id, label, node_type, content, confidence,
          "extract_inline", datetime.now().isoformat(), origin))

    vec = embed_text(content).astype(np.float32)
    c.execute(
        "INSERT OR REPLACE INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
        (node_id, vec.tobytes(), MODEL_NAME)
    )
    conn.commit()
    conn.close()
    return node_id


def run(input_path: Path, dry_run: bool = False) -> int:
    if not input_path.exists():
        print(f"Staging file not found: {input_path}")
        return 0

    text = input_path.read_text(encoding="utf-8").strip()
    if not text or len(text) < 30:
        print("Staging file empty or too short — nothing to extract")
        return 0

    print(f"Extracting from {input_path} ({len(text)} chars)...")

    # Use sonnet for higher-quality extraction — cashew principle: quality over quantity
    # Haiku misses nuance, drops complex nodes, oversimplifies architecture/synthesis nodes
    prompt = EXTRACT_PROMPT.format(text=text[:16000])
    response = call_llm(prompt, model="anthropic/claude-sonnet-4-6", max_tokens=4000)

    if not response:
        print("LLM call failed — staging file preserved")
        return 0

    nodes = _parse_nodes(response)
    if not nodes:
        print("No nodes extracted")
        input_path.write_text("")  # clear staging
        return 0

    written = 0
    for node in nodes:
        conf = node.get("confidence", 0.7)
        if conf < 0.7:
            continue
        label = node.get("label", "").strip()
        content = node.get("content", "").strip()
        if not label or not content:
            continue

        if dry_run:
            origin = node.get("origin", "onyx")
            ntype = node.get("type", "observation")
            print(f"  [{origin}/{ntype}] {label} (conf={conf:.2f})")
            print(f"    {content[:120]}")
        else:
            nid = _write_node(
                label=label,
                content=content,
                node_type=node.get("type", "observation"),
                origin=node.get("origin", "onyx"),
                confidence=conf
            )
            print(f"  ✓ {nid} [{node.get('origin','onyx')}] {label[:60]}")
            written += 1

    if not dry_run:
        input_path.write_text("")  # clear staging after successful write
        print(f"\n{written} nodes written. Staging file cleared.")
    else:
        print(f"\n[dry-run] Would write {len([n for n in nodes if n.get('confidence',0.7) >= 0.7])} nodes")

    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract conversation signal direct to brain.db")
    parser.add_argument("--input", default=str(STAGING_FILE), help="Staging markdown file")
    parser.add_argument("--dry-run", action="store_true", help="Print nodes without writing")
    args = parser.parse_args()

    run(Path(args.input), dry_run=args.dry_run)
