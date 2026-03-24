#!/usr/bin/env python3
"""
Engram — Extract knowledge from personal chat exports (Telegram/WhatsApp DMs).

Processes plain-text chat exports, groups by calendar day, and extracts
knowledge about the user's beliefs, decisions, and emotional state.

Configuration:
  ENGRAM_CHATS_DIR     — path to chat export directory
  ENGRAM_LLM_ENDPOINT  — LLM API endpoint
  ENGRAM_EXTRACT_MODEL — model for extraction (default: claude-haiku-4-5)
  ENGRAM_USER_NAME     — the user's display name in chats (for attribution)
  ENGRAM_BATCH_SIZE    — days per LLM batch call (default: 5)

Run: python3 -m extractors.chats [--dry-run] [--source whatsapp|telegram]
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
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

from brain import get_db, embed_text, EMBEDDING_MODEL, BRAIN_DIR

CHATS_DIR = Path(os.environ.get("ENGRAM_CHATS_DIR", Path.home() / "chats"))
LLM_ENDPOINT = os.environ.get("ENGRAM_LLM_ENDPOINT", "http://localhost:3456/v1/chat/completions")
LLM_MODEL = os.environ.get("ENGRAM_EXTRACT_MODEL", "claude-haiku-4-5")
USER_NAME = os.environ.get("ENGRAM_USER_NAME", "User")
BATCH_SIZE = int(os.environ.get("ENGRAM_BATCH_SIZE", "5"))
CHECKPOINT_FILE = BRAIN_DIR / "chats_extract_checkpoint.json"

# WhatsApp message regex (flexible date/time formats)
WA_PATTERN = re.compile(
    r"[\[\[]?(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})[,\s]+"
    r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\]?\s*[-–]?\s*"
    r"([^:]+):\s(.+)"
)


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


def _parse_date(date_str: str):
    """Parse various chat date formats."""
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y",
                "%d-%m-%Y", "%d-%m-%y", "%d.%m.%Y", "%d.%m.%y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def parse_whatsapp_file(filepath: Path) -> List[Tuple]:
    """Parse a WhatsApp export file into (date, sender, message) tuples."""
    messages = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = WA_PATTERN.match(line.strip())
            if m:
                date_str, time_str, sender, text = m.groups()
                dt = _parse_date(date_str)
                if dt:
                    messages.append((dt, sender.strip(), text.strip()))
    return messages


def identify_user(messages: List[Tuple], user_name: str = None) -> Optional[str]:
    """Identify the user's sender name from message frequency."""
    if user_name:
        for _, sender, _ in messages[:50]:
            if sender.lower().strip() == user_name.lower():
                return sender
    counts = defaultdict(int)
    for _, sender, _ in messages[:50]:
        counts[sender.lower().strip()] += 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def insert_node(conn, node_id, label, ntype, content, confidence=0.85,
                source="chat_extract", origin_date=None, origin="self"):
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


EXTRACTION_PROMPT = """Analyze these personal chat messages.

{batch_text}

Extract 1-4 knowledge nodes about the user ({user_name}).
Focus on: beliefs, decisions, emotional states, recurring concerns, relationships.

CRITICAL: Only extract facts the user states about THEMSELVES.
Skip logistics, small talk, and facts about other people.

Return JSON array:
[{{"label": "...", "content": "...", "type": "belief|decision|insight|observation|question", "confidence": 0.7, "date": "YYYY-MM-DD"}}]"""


def run(dry_run: bool = False, source: str = "whatsapp"):
    """Process all chat export files."""
    if not CHATS_DIR.exists():
        print(f"Chats directory not found: {CHATS_DIR}")
        print(f"Set ENGRAM_CHATS_DIR to your chat exports directory.")
        return

    ckpt = _load_checkpoint()
    processed_files = ckpt.get("processed_files", {})

    txt_files = sorted(CHATS_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} chat files in {CHATS_DIR}")

    conn = get_db()
    added_total = 0

    for filepath in txt_files:
        filename = filepath.name
        stem = filepath.stem
        messages = parse_whatsapp_file(filepath)
        if not messages:
            continue

        user = identify_user(messages, USER_NAME)
        if not user:
            print(f"  Could not identify user in: {filename}")
            continue

        print(f"  {filename} ({len(messages)} msgs, user='{user}')")

        last_date_str = processed_files.get(filename)
        last_date = None
        if last_date_str:
            try:
                last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
            except ValueError:
                pass

        days = defaultdict(list)
        for dt, sender, text in messages:
            days[dt].append((sender, text))

        days_to_process = [
            (day, msgs) for day, msgs in sorted(days.items())
            if (not last_date or day > last_date) and len(msgs) >= 30
        ]

        max_date = None
        file_added = 0

        for batch_start in range(0, len(days_to_process), BATCH_SIZE):
            batch = days_to_process[batch_start:batch_start + BATCH_SIZE]
            parts = []
            for day, day_msgs in batch:
                lines = [f"{USER_NAME if s == user else s}: {t}"
                         for s, t in day_msgs]
                parts.append(f"=== {day.isoformat()} ===\n" + "\n".join(lines)[:2000])
            batch_text = "\n\n".join(parts)[:8000]

            prompt = EXTRACTION_PROMPT.format(
                batch_text=batch_text, user_name=USER_NAME
            )
            response = call_llm(prompt)
            nodes = parse_json_array(response)
            if not nodes:
                continue

            if dry_run:
                print(f"    Would extract {len(nodes)} nodes")
                continue

            for node in nodes:
                label = node.get("label", "")
                content = node.get("content", "")
                if not label or not content:
                    continue
                node_date = node.get("date", batch[-1][0].isoformat())
                node_id = "chat_" + hashlib.md5(
                    f"{stem}_{node_date}_{label}".encode()
                ).hexdigest()[:12]
                insert_node(conn, node_id, label, node.get("type", "observation"),
                            content, confidence=node.get("confidence", 0.8),
                            source=f"chat_{stem}",
                            origin_date=f"{node_date}T00:00:00")
                file_added += 1
            max_date = batch[-1][0]

        if max_date:
            processed_files[filename] = max_date.isoformat()

        added_total += file_added
        conn.commit()
        _save_checkpoint({"processed_files": processed_files})

    conn.close()
    print(f"\n{'DRY RUN ' if dry_run else ''}Chat extraction done: {added_total} nodes added")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract from chat exports")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source", default="whatsapp", choices=["whatsapp", "telegram"])
    args = parser.parse_args()
    run(dry_run=args.dry_run, source=args.source)
