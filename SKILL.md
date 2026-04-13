---
name: engram
version: 1.0.0
description: Persistent knowledge graph for OpenClaw agents. Extracts insights from your conversations and memory files, synthesizes patterns via rumination cycles, and makes everything semantically searchable. A second brain that remembers, synthesizes, and forgets intelligently.
author: shadowfax1312
homepage: https://github.com/shadowfax1312/engram
license: MIT
---

# Engram

A persistent knowledge graph skill for OpenClaw. Extracts insights from your conversations, daily notes, and session logs — synthesizes them into a searchable graph that grows smarter over time.

## Architecture

- **Extract** — 5 extractors (memory, sessions, chats, work, topical) → nodes + edges in brain.db
- **Ruminate** — LLM synthesis cycles that find patterns across unrelated nodes
- **Sleep** — Fitness scoring, core memory promotion (top √N nodes), graceful decay/GC
- **Search** — O(log N) semantic search with +0.15 boost for core memories

## Install

```bash
npx clawhub install engram
```

## Setup

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and `BRAIN_DIR`
2. Initialize the graph: `python3 brain/init.py`
3. Run your first extraction: `python3 extractors/memory.py`
4. Start the think+sleep cycle: `bash scripts/run_cycles.sh`

## Crons

See `crons.md` for copy-paste ready OpenClaw cron configs:
- Hourly memory extraction
- 4h session extraction
- 2x daily ruminate + sleep
- 6h brain.db backup

## Templates

- `soul-template.md` — drop-in agent soul file with brain access instructions + safety parameters
- `agents-template.md` — session startup sequence, memory discipline, heartbeat config
