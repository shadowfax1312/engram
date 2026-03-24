#!/usr/bin/env python3
"""
Engram — Extract knowledge from agent session logs.

Processes JSONL session files (OpenClaw/Claude format).
Extracts beliefs, decisions, insights from conversations.
Checkpointed by file + modification time.

Configuration:
  ENGRAM_SESSIONS_DIR  — path to session logs directory
  ENGRAM_LLM_ENDPOINT  — LLM API endpoint
  ENGRAM_EXTRACT_MODEL — model for extraction (default: claude-haiku-4-5)

Run: python3 -m extractors.sessions [--dry-run] [--limit N]
"""

import os
import json
import re
import time
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from brain import get_db, BRAIN_DIR

SESSIONS_DIR = Path(os.environ.get(
    "ENGRAM_SESSIONS_DIR",
    Path.home() / ".openclaw" / "agents" / "main" / "sessions"
))
CHECKPOINT_FILE = BRAIN_DIR / "session_extract_checkpoint.json"
LLM_ENDPOINT = os.environ.get("ENGRAM_LLM_ENDPOINT", "http://localhost:3456/v1/chat/completions")
LLM_MODEL = os.environ.get("ENGRAM_EXTRACT_MODEL", "claude-haiku-4-5")
MIN_CONVERSATION_CHARS = 1500
BATCH_SIZE = int(os.environ.get("ENGRAM_BATCH_SIZE", "10"))


def load_checkpoint() -> Dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"processed_files": {}, "last_run": None}


def save_checkpoint(ckpt: Dict):
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    ckpt["last_run"] = datetime.now().isoformat()
    CHECKPOINT_FILE.write_text(json.dumps(ckpt, indent=2))


def call_llm(prompt: str, retries: int = 3) -> Optional[str]:
    """Call LLM via configurable endpoint."""
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


def extract_messages_from_session(session_path: Path) -> List[Dict]:
    """Extract user/assistant message pairs from a JSONL session file."""
    messages = []
    with open(session_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get('type') != 'message':
                    continue
                msg = entry.get('message', {})
                role = msg.get('role')
                content = msg.get('content', [])
                timestamp = entry.get('timestamp', '')
                text = ""
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get('type') == 'text':
                            text += c.get('text', '') + "\n"
                elif isinstance(content, str):
                    text = content
                if text.strip() and role in ('user', 'assistant'):
                    if role == 'user':
                        parts = text.split('```\n\n', 1)
                        if len(parts) > 1:
                            text = parts[-1]
                    messages.append({
                        'role': role,
                        'text': text.strip()[:2000],
                        'timestamp': timestamp
                    })
            except json.JSONDecodeError:
                continue
    return messages


def format_conversation(messages: List[Dict], max_chars: int = 8000) -> str:
    """Format messages into a conversation transcript."""
    lines = []
    total = 0
    for msg in messages:
        role = "User" if msg['role'] == 'user' else "Agent"
        line = f"[{role}]: {msg['text'][:500]}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n\n".join(lines)


EXTRACTION_PROMPT = """You are extracting knowledge from an agent conversation.

Session timestamp: {timestamp}

Conversation:
{conversation}

Extract 2-5 knowledge nodes about what the user thinks, knows, decided, or is working on.

CRITICAL RULES:
- Only extract facts the user states about THEMSELVES
- Skip logistics, greetings, debugging details
- Focus on: beliefs, decisions, insights, patterns, questions

Return JSON array only:
[
  {{"label": "short label (5-10 words)", "type": "belief|decision|insight|observation|question", "content": "full description (1-3 sentences)", "confidence": 0.7}}
]

If no substantive knowledge found, return empty array: []"""


def extract_from_session(session_path: Path, dry_run: bool = False) -> int:
    """Extract knowledge nodes from a single session."""
    messages = extract_messages_from_session(session_path)
    total_chars = sum(len(m['text']) for m in messages)
    if total_chars < MIN_CONVERSATION_CHARS:
        return 0

    conversation = format_conversation(messages)
    if len(conversation) < 500:
        return 0

    timestamp = messages[0]['timestamp'][:10] if messages else "unknown"
    prompt = EXTRACTION_PROMPT.format(timestamp=timestamp, conversation=conversation)
    response = call_llm(prompt)
    nodes = parse_json_array(response)
    if not nodes:
        return 0

    if dry_run:
        print(f"   Would extract {len(nodes)} nodes")
        return len(nodes)

    conn = get_db()
    c = conn.cursor()
    added = 0
    source = f"session_{session_path.stem[:8]}"

    for node in nodes:
        conf = node.get('confidence', 0.7)
        if conf < 0.6:
            continue
        node_id = f"sess_{session_path.stem[:8]}_{added}_{datetime.now().strftime('%H%M')}"
        try:
            c.execute('''INSERT OR IGNORE INTO nodes (id, label, type, content, confidence, source, origin)
                         VALUES (?, ?, ?, ?, ?, ?, ?)''',
                      (node_id, node.get('label', 'Unknown'),
                       node.get('type', 'observation'),
                       node.get('content', ''), conf, source, 'self'))
            if c.rowcount > 0:
                added += 1
        except Exception as e:
            print(f"   Error inserting: {e}")

    conn.commit()
    conn.close()
    return added


def run(dry_run: bool = False, limit: int = None):
    """Process all unprocessed session files."""
    ckpt = load_checkpoint()
    processed = ckpt.get("processed_files", {})

    if not SESSIONS_DIR.exists():
        print(f"Sessions directory not found: {SESSIONS_DIR}")
        print(f"Set ENGRAM_SESSIONS_DIR to your sessions directory.")
        return 0

    session_files = sorted(SESSIONS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    to_process = []
    for sf in session_files:
        file_key = sf.name
        file_mtime = sf.stat().st_mtime
        if file_key in processed and processed[file_key].get("mtime", 0) >= file_mtime:
            continue
        to_process.append(sf)

    if limit:
        to_process = to_process[:limit]

    print(f"Found {len(to_process)} sessions to process")
    total_nodes = 0

    for i, sf in enumerate(to_process):
        print(f"\n[{i+1}/{len(to_process)}] {sf.name[:20]}...")
        try:
            nodes_added = extract_from_session(sf, dry_run=dry_run)
            total_nodes += nodes_added
            if nodes_added > 0:
                print(f"   + {nodes_added} nodes")
            if not dry_run:
                processed[sf.name] = {
                    "mtime": sf.stat().st_mtime,
                    "nodes": nodes_added,
                    "processed_at": datetime.now().isoformat()
                }
                save_checkpoint({"processed_files": processed})
        except Exception as e:
            print(f"   Error: {e}")

    print(f"\n{'DRY RUN ' if dry_run else ''}Complete: {total_nodes} nodes from {len(to_process)} sessions")
    return total_nodes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract from session logs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(dry_run=args.dry_run, limit=args.limit)
