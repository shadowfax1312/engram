# Engram — OpenClaw Skill Manifest

**Name:** engram
**Version:** 1.0.0
**Description:** Persistent knowledge graph with extraction, rumination, and semantic search. A second brain that remembers, synthesizes, and forgets.

## Capabilities

- **Extract** knowledge from notes, sessions, chats, work docs, and research threads
- **Search** semantically with O(N) cosine similarity + 2-hop graph walk
- **Ruminate** — generative synthesis that finds patterns, contradictions, and cross-domain connections
- **Sleep** — fitness scoring, core memory promotion, duplicate merging, and garbage collection
- **Compact** — decay, prune orphans, merge duplicates
- **Dashboard** — D3.js graph visualization with search

## Requirements

- Python 3.10+
- `sentence-transformers` (for local embeddings)
- `numpy`, `scikit-learn`
- An OpenAI-compatible LLM endpoint (default: `http://localhost:3456/v1/chat/completions`)

## Quick Start

```bash
# Set your brain directory
export BRAIN_DIR=~/.engram

# Copy .env.example to .env and configure
cp .env.example .env

# Initialize the database
python3 -c "from brain import init_schema; init_schema()"

# Extract from your notes
export ENGRAM_MEMORY_DIR=~/notes
python3 -m extractors.memory

# Run a think→sleep cycle
bash scripts/run_cycles.sh 5

# Search
python3 -m brain.search "your query here"

# Dashboard
python3 -m dashboard.export --serve
```

## Environment Variables

See `.env.example` for full list. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `BRAIN_DIR` | `~/.engram` | Database and state directory |
| `ENGRAM_LLM_ENDPOINT` | `http://localhost:3456/v1/chat/completions` | OpenAI-compatible API |
| `ENGRAM_EXTRACT_MODEL` | `claude-haiku-4-5` | Model for extraction |
| `ENGRAM_RUMINATE_MODEL` | `claude-sonnet-4-6` | Model for rumination |
| `ENGRAM_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |

## Cron Setup

See `crons.md` for copy-paste ready OpenClaw cron configurations.
