#!/usr/bin/env python3
"""
Onyx Second Brain — Ingestion Pipeline

Extracts knowledge nodes from source data and populates the brain graph.

Usage:
  python3 ingest.py --source gpt
  python3 ingest.py --source whatsapp
  python3 ingest.py --source granola
  python3 ingest.py --source md-scaffold
  python3 ingest.py --all
  python3 ingest.py --source gpt --reset   # clear checkpoint first
"""

import sys
import os
import re
import json
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import subprocess
import argparse
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from init_graph import get_db, add_edge
from embed import embed_text, embed_texts, MODEL_NAME
from search import semantic_search
from ruminate import call_haiku, parse_json_array_response, parse_json_response

# ── Paths ────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "brain.db"
STATE_DIR = Path(__file__).parent / "state"
BACKUP_DIR = Path(__file__).parent / "backups"

GPT_DIR = Path("/path/to/second-brain/"
               "300683393621579184dc03cdf7443328896d3a4bb8727ca3b6ee583e5e006186-"
               "2026-02-23-07-03-09-e23cf9276e424e2eb82bdcffbf561fa1")
WHATSAPP_DIR = Path("/path/to/second-brain/Source data/Personal")
MD_DIR = Path("/path/to/second-brain")


# ── Backup ───────────────────────────────────────────────────────
def backup_db():
    """WAL-safe backup before any writes."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = BACKUP_DIR / f"graph-{ts}.db"
    subprocess.run(
        ["sqlite3", str(DB_PATH), f".backup {backup_path}"],
        check=True
    )
    print(f"Backup: {backup_path}")


# ── Checkpoint ───────────────────────────────────────────────────
def _checkpoint_path(name):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / f"{name}_progress.json"


def load_checkpoint(name):
    p = _checkpoint_path(name)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_checkpoint(name, data):
    data["last_run"] = datetime.now().isoformat()
    _checkpoint_path(name).write_text(json.dumps(data, indent=2))


def reset_checkpoint(name):
    p = _checkpoint_path(name)
    if p.exists():
        p.unlink()
        print(f"Checkpoint reset: {name}")


# ── Node insertion (with session_id + origin_date) ───────────────
def insert_node(conn, node_id, label, ntype, content, confidence=0.85,
                source="gpt_export", session_id=None, origin_date=None,
                origin="self"):
    """Insert node with full provenance fields + embedding."""
    c = conn.cursor()
    c.execute("""
        INSERT INTO nodes (id, label, type, content, confidence, source,
                           session_id, origin_date, origin, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}')
        ON CONFLICT(id) DO UPDATE SET
            label=excluded.label,
            content=excluded.content,
            confidence=excluded.confidence,
            updated_at=datetime('now'),
            session_id=excluded.session_id,
            origin_date=excluded.origin_date,
            origin=excluded.origin
    """, (node_id, label, ntype, content, confidence, source,
          session_id, origin_date, origin))

    # Embed
    if content:
        vec = embed_text(content).astype(np.float32)
        c.execute(
            "INSERT OR REPLACE INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
            (node_id, vec.tobytes(), MODEL_NAME)
        )


# ── GPT Extraction ──────────────────────────────────────────────
GPT_EXTRACTION_PROMPT = """Extract knowledge from this conversation.

Title: {title}
Date: {date}

Transcript:
{text}

Extract 2-6 distinct pieces of knowledge — what this person thinks, knows, decided, experienced, or is wrestling with.

Rules:
- Be specific: not "cares about X" but "believes X because Y"
- Each node must stand alone as a complete thought
- Skip greetings, small talk, trivial requests
- CRITICAL: Only extract facts about the USER (the human in this conversation)
- Do NOT extract facts the user is researching for someone else, or questions asked on behalf of others
- If the user says "my friend wants to know X" — skip it, that's not about the user

Return JSON array only:
[
  {{"label": "short label (5-10 words)", "type": "belief|decision|insight|observation|question|fact", "content": "full description (1-3 sentences)", "confidence": 0.7}}
]"""


def _load_gpt_conversations():
    """Load all GPT conversation JSON files, merge and sort by create_time."""
    all_convs = []

    # conversations-00*.json
    for path in sorted(GPT_DIR.glob("conversations-00*.json")):
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                # Filter: only valid conversation dicts
                all_convs.extend([c for c in data if isinstance(c, dict) and "id" in c and "mapping" in c])

    # shared_conversations.json
    shared = GPT_DIR / "shared_conversations.json"
    if shared.exists():
        with open(shared) as f:
            data = json.load(f)
            if isinstance(data, list):
                all_convs.extend([c for c in data if isinstance(c, dict) and "id" in c and "mapping" in c])

    # Sort chronologically
    all_convs.sort(key=lambda c: c.get("create_time", 0))
    return all_convs


def _extract_user_text(conv):
    """Extract concatenated user message text from a GPT conversation."""
    mapping = conv.get("mapping", {})
    user_texts = []
    for node in mapping.values():
        msg = node.get("message")
        if not msg:
            continue
        if msg.get("author", {}).get("role") != "user":
            continue
        content = msg.get("content", {})
        if content.get("content_type") != "text":
            continue
        parts = content.get("parts", [])
        for part in parts:
            if isinstance(part, str) and part.strip():
                user_texts.append(part.strip())

    combined = "\n\n".join(user_texts)
    return combined[:3000]


def ingest_gpt():
    """P0a: Extract knowledge from GPT conversation exports."""
    print("\n═══ GPT Extraction ═══")

    checkpoint = load_checkpoint("gpt")
    processed_ids = set(checkpoint.get("processed_ids", []))
    print(f"Checkpoint: {len(processed_ids)} already processed")

    convs = _load_gpt_conversations()
    print(f"Loaded {len(convs)} conversations")

    conn = get_db()
    added_total = 0
    skipped = 0
    errors = 0
    batch_size = 20  # commit every N conversations

    for i, conv in enumerate(convs):
        conv_id = conv.get("id", "")
        if conv_id in processed_ids:
            continue

        title = conv.get("title", "Untitled")
        create_time = conv.get("create_time", 0)

        text = _extract_user_text(conv)
        if len(text) < 50:
            processed_ids.add(conv_id)
            skipped += 1
            continue

        try:
            dt = datetime.fromtimestamp(create_time)
        except (ValueError, TypeError, OSError):
            dt = datetime.now()

        date_str = dt.strftime("%Y-%m-%d")
        origin_date = dt.isoformat()

        prompt = GPT_EXTRACTION_PROMPT.format(
            title=title, date=date_str, text=text
        )
        response = call_haiku(prompt)
        nodes = parse_json_array_response(response)

        if not nodes:
            processed_ids.add(conv_id)
            errors += 1
            print(f"  ✗ [{i+1}/{len(convs)}] No extraction: {title[:40]}", flush=True)
            continue

        for node in nodes:
            label = node.get("label", "")
            content = node.get("content", "")
            ntype = node.get("type", "observation")  # LLM-assigned type
            confidence = node.get("confidence", 0.8)

            if not label or not content:
                continue

            # Generate stable node ID
            node_id = "gpt_" + hashlib.md5(
                f"{conv_id}_{label}".encode()
            ).hexdigest()[:12]

            insert_node(conn, node_id, label, ntype, content,
                        confidence=confidence, source="gpt_export",
                        session_id=conv_id, origin_date=origin_date,
                        origin="self")
            added_total += 1

        processed_ids.add(conv_id)

        print(f"  ✓ [{i+1}/{len(convs)}] {title[:45]} → {len(nodes)} nodes", flush=True)
        # Progress
        if (i + 1) % batch_size == 0:
            conn.commit()
            save_checkpoint("gpt", {"processed_ids": list(processed_ids)})
            total_done = len(processed_ids)
            print(f"  [{total_done}/{len(convs)}] +{added_total} nodes, "
                  f"{skipped} skipped, {errors} errors")

    conn.commit()
    conn.close()
    save_checkpoint("gpt", {"processed_ids": list(processed_ids)})
    print(f"\n✓ GPT done: {added_total} nodes added, "
          f"{skipped} skipped, {errors} errors")


# ── WhatsApp Extraction ──────────────────────────────────────────
WHATSAPP_EXTRACTION_PROMPT = """You are extracting structured knowledge from a personal WhatsApp conversation.
Source file: {filename}
Date: {date}

Conversation:
{messages_text}

Extract 1-4 knowledge nodes about the person whose archive this is.
Focus on: his beliefs, decisions, emotional states, recurring concerns, relationships.

CRITICAL ATTRIBUTION RULES:
- Only extract facts the user states about HIMSELF — his own actions, decisions, beliefs, experiences
- Do NOT extract news/facts the user is relaying about OTHER people (friends' jobs, others' deals, etc.)
- If the user says "my friend got a job at X" — that's about the friend, not the user. Skip it.
- If the user says "I got a job at X" — that's about the user. Extract it.
- When in doubt, skip. Better to miss a fact than misattribute.

Skip days with only logistics/small talk.

Return JSON array only:
[{{"label": "...", "type": "belief|decision|insight|observation|question|fact", "content": "...", "confidence": 0.7}}]"""

# Flexible WhatsApp message regex
WA_PATTERN = re.compile(
    r"[\[\[]?(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})[,\s]+"
    r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\]?\s*[-–]?\s*"
    r"([^:]+):\s(.+)"
)


def _parse_whatsapp_date(date_str):
    """Parse various WhatsApp date formats into a date object."""
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y",
                "%d-%m-%Y", "%d-%m-%y", "%d.%m.%Y", "%d.%m.%y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _parse_whatsapp_file(filepath):
    """Parse a WhatsApp export file into list of (date, sender, message) tuples."""
    messages = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = WA_PATTERN.match(line.strip())
            if m:
                date_str, time_str, sender, text = m.groups()
                dt = _parse_whatsapp_date(date_str)
                if dt:
                    messages.append((dt, sender.strip(), text.strip()))
    return messages


def _identify_user(messages):
    """Heuristic: find the user's sender name from first 50 messages."""
    exclude = {"life boss", "shailaja", "life boss❤️"}
    counts = defaultdict(int)
    for _, sender, _ in messages[:50]:
        name_lower = sender.lower().strip()
        if name_lower not in exclude:
            counts[name_lower] += 1

    if not counts:
        return None

    # Most common non-excluded sender
    best = max(counts, key=counts.get)
    # Return the original-case version
    for _, sender, _ in messages[:50]:
        if sender.lower().strip() == best:
            return sender
    return best


def ingest_whatsapp():
    """P0b: Extract knowledge from WhatsApp conversation exports."""
    print("\n═══ WhatsApp Extraction ═══")

    checkpoint = load_checkpoint("whatsapp")
    processed_files = checkpoint.get("processed_files", {})

    txt_files = sorted(WHATSAPP_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} WhatsApp files")

    conn = get_db()
    added_total = 0

    for filepath in txt_files:
        filename = filepath.name
        stem = filepath.stem

        messages = _parse_whatsapp_file(filepath)
        if not messages:
            print(f"  ✗ No parseable messages: {filename}")
            continue

        user_name = _identify_user(messages)
        if not user_name:
            print(f"  ✗ Can't identify the user in: {filename}")
            continue

        print(f"  • {filename} ({len(messages)} msgs, user='{user_name}')")

        # Last processed date for this file
        last_date_str = processed_files.get(filename)
        last_date = None
        if last_date_str:
            try:
                last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
            except ValueError:
                pass

        # Group messages by calendar day
        days = defaultdict(list)
        for dt, sender, text in messages:
            days[dt].append((sender, text))

        max_date = None
        file_added = 0

        # Prepare days to process
        days_to_process = []
        for day in sorted(days.keys()):
            if last_date and day <= last_date:
                continue
            day_msgs = days[day]
            if len(day_msgs) < 30:
                continue
            total_text = " ".join(text for _, text in day_msgs)
            if len(total_text) < 100:
                continue
            days_to_process.append((day, day_msgs))

        # Batch 5 days per LLM call (similar topics, latency is round-trip not tokens)
        BATCH_SIZE = 5
        for batch_start in range(0, len(days_to_process), BATCH_SIZE):
            batch = days_to_process[batch_start:batch_start + BATCH_SIZE]
            
            # Format all days in batch
            batch_text_parts = []
            for day, day_msgs in batch:
                lines = []
                for sender, text in day_msgs:
                    role = "the user" if sender == user_name else sender
                    lines.append(f"{role}: {text}")
                day_text = "\n".join(lines)[:2000]  # limit per day
                batch_text_parts.append(f"=== {day.isoformat()} ===\n{day_text}")
            
            batch_text = "\n\n".join(batch_text_parts)[:8000]  # total limit
            date_range = f"{batch[0][0]} to {batch[-1][0]}"
            
            prompt = f"""Analyze these WhatsApp messages from {filename} ({date_range}).

{batch_text}

Extract 1-4 knowledge nodes about the person whose archive this is.
Focus on: his emotional state, decisions, relationships, recurring concerns.
Only extract from the user's perspective — what does this reveal about HIM.
Include the date in each node's content when relevant.

Return JSON array:
[{{"label": "...", "content": "...", "type": "belief|decision|insight|observation|question", "confidence": 0.7, "date": "YYYY-MM-DD"}}]"""

            response = call_haiku(prompt)
            nodes = parse_json_array_response(response)

            if not nodes:
                print(f"    ✗ {date_range} — no extraction", flush=True)
                continue
            
            print(f"    ✓ {date_range} → {len(nodes)} nodes", flush=True)

            for node in nodes:
                label = node.get("label", "")
                content = node.get("content", "")
                ntype = node.get("type", "observation")
                confidence = node.get("confidence", 0.8)
                node_date = node.get("date", batch[-1][0].isoformat())

                if not label or not content:
                    continue

                node_id = "wa_" + hashlib.md5(
                    f"{stem}_{node_date}_{label}".encode()
                ).hexdigest()[:12]

                insert_node(conn, node_id, label, ntype, content,
                            confidence=confidence,
                            source=f"whatsapp_{stem}",
                            session_id=f"{stem}_{node_date}",
                            origin_date=f"{node_date}T00:00:00",
                            origin="self")
                file_added += 1

            max_date = batch[-1][0]

        if max_date:
            processed_files[filename] = max_date.isoformat()

        added_total += file_added
        if file_added:
            print(f"    +{file_added} nodes")

        conn.commit()
        save_checkpoint("whatsapp", {"processed_files": processed_files})

    conn.close()
    print(f"\n✓ WhatsApp done: {added_total} nodes added")


# ── Granola (stub) ───────────────────────────────────────────────
def ingest_granola():
    """P1: Granola extraction — not yet built."""
    print("\nGranola extraction: use granola skill (P1, not yet built)")


# ── MD Edge Scaffolding ──────────────────────────────────────────
MD_EDGE_PROMPT = """Given these two knowledge nodes from a personal brain graph, what is the relationship between them?

Node A:
  Label: {label_a}
  Content: {content_a}

Node B:
  Label: {label_b}
  Content: {content_b}

Section context (from source document):
{section_text}

Return ONLY valid JSON, no other text:
{{"relation": "enables|depends_on|tensions_with|relates_to|supports|opposes|evolved_to", "note": "one sentence explaining the connection", "weight": 0.5}}

If there is no meaningful relationship, return: {{"relation": "none", "note": "", "weight": 0.0}}"""


def ingest_md_scaffold():
    """P2: Use MD files to discover edges between existing nodes."""
    print("\n═══ MD Edge Scaffolding ═══")

    conn = get_db()
    node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    if node_count < 100:
        print(f"  ⚠ Only {node_count} nodes in graph (need 100+). "
              f"Run gpt/whatsapp ingestion first.")
        conn.close()
        return
    conn.close()

    md_files = sorted(MD_DIR.glob("[0-9][0-9]-*.md"))
    print(f"Found {len(md_files)} MD files, graph has {node_count} nodes")

    edges_added = 0

    for md_path in md_files:
        print(f"  • {md_path.name}")
        text = md_path.read_text(encoding="utf-8", errors="replace")

        # Split into sections by ## headers or double newlines
        sections = re.split(r"(?:^##\s+.+$|\n\n)", text, flags=re.MULTILINE)
        sections = [s.strip() for s in sections if 50 <= len(s.strip()) <= 500]

        for section in sections:
            results = semantic_search(section, top_k=3)
            if len(results) < 2:
                continue
            if results[0]["score"] < 0.6:
                continue

            # Try to connect top 2 results
            a, b = results[0], results[1]
            if a["id"] == b["id"]:
                continue

            prompt = MD_EDGE_PROMPT.format(
                label_a=a["label"], content_a=a["content"][:200],
                label_b=b["label"], content_b=b["content"][:200],
                section_text=section[:300]
            )
            response = call_haiku(prompt)
            edge_data = parse_json_response(response)

            if not edge_data:
                continue
            if edge_data.get("relation", "none") == "none":
                continue
            if edge_data.get("weight", 0) < 0.3:
                continue

            add_edge(a["id"], b["id"],
                     edge_data["relation"],
                     note=edge_data.get("note", ""),
                     weight=edge_data.get("weight", 0.5),
                     source="md_scaffold")
            edges_added += 1

    print(f"\n✓ MD scaffold done: {edges_added} edges added")


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Onyx ingestion pipeline")
    parser.add_argument("--source", choices=["gpt", "whatsapp", "granola", "md-scaffold"])
    parser.add_argument("--all", action="store_true", help="Run all sources in order")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint before run")
    args = parser.parse_args()

    if not args.source and not args.all:
        parser.print_help()
        sys.exit(1)

    # Step 0: backup
    backup_db()

    sources = []
    if args.all:
        sources = ["gpt", "whatsapp", "granola", "md-scaffold"]
    else:
        sources = [args.source]

    if args.reset:
        for src in sources:
            name = src.replace("-", "_").replace("md_scaffold", "md_scaffold")
            reset_checkpoint(name)

    dispatch = {
        "gpt": ingest_gpt,
        "whatsapp": ingest_whatsapp,
        "granola": ingest_granola,
        "md-scaffold": ingest_md_scaffold,
    }

    for src in sources:
        dispatch[src]()

    # Final stats
    conn = get_db()
    nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    conn.close()
    print(f"\n{'─' * 50}")
    print(f"Graph: {nodes} nodes, {edges} edges")
    print(f"{'─' * 50}")


if __name__ == "__main__":
    main()
