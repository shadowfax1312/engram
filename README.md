# Engram

A persistent knowledge graph for AI agents. Extracts, synthesizes, and forgets — like biological memory.

## What is Engram?

Engram is a SQLite-backed knowledge graph that gives AI agents persistent, structured memory. It extracts knowledge from your notes, conversations, and work, then runs generative synthesis cycles to find patterns, contradictions, and cross-domain connections.

Named after the neuroscience term for a memory trace stored in neural tissue.

## Architecture

```
Source Data → Extractors → brain.db → Ruminate → Sleep → Consolidated Memory
     ↑                        ↓
  notes, chats,          Search API
  sessions, work         (semantic + hybrid)
```

### Core Components

| Component | Purpose | Model |
|-----------|---------|-------|
| `extractors/memory.py` | Daily notes → nodes | Haiku |
| `extractors/sessions.py` | Agent session logs → nodes | Haiku |
| `extractors/chats.py` | Personal chat exports → nodes | Haiku |
| `extractors/work.py` | Work documents → nodes | Haiku |
| `extractors/topical.py` | Deep research threads → nodes | Configurable |
| `brain/ruminate.py` | Generative synthesis | Sonnet |
| `brain/sleep.py` | Fitness scoring + GC | Local |
| `brain/search.py` | Semantic + hybrid search | Local embeddings |
| `brain/compact.py` | Decay + pruning | Local |

### Fitness Function

Every node has a structural fitness score:

```
f(node) = edge_count × 1.0
         + access_count_30d × 0.5
         + cross_link_count × 0.3
         + seed_bonus (+2.0 if human-authored)
```

- **Core memories**: Top √N nodes by fitness are promoted to core memory status
- **Decay**: Relevance decays with log-scaled half-life based on access frequency
- **GC**: Nodes below relevance threshold are deleted after 7-day grace period
- **Protection**: Human-authored nodes, people, orgs, decisions, and events are never GC'd

## Setup

### 1. Install dependencies

```bash
pip install sentence-transformers numpy scikit-learn
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your paths and preferences
```

### 3. Initialize

```bash
export BRAIN_DIR=~/.engram
python3 -c "from brain import init_schema; init_schema()"
```

### 4. Extract

```bash
# Point extractors at your data
export ENGRAM_MEMORY_DIR=~/notes
python3 -m extractors.memory

export ENGRAM_SESSIONS_DIR=~/.openclaw/agents/main/sessions
python3 -m extractors.sessions
```

### 5. Think & Sleep

```bash
# Single cycle
python3 -m brain.ruminate --cycles 1
python3 -m brain.sleep

# Or use the orchestration script
bash scripts/run_cycles.sh 10
```

### 6. Search

```bash
python3 -m brain.search "your query"
python3 -m brain.search "your query" --hybrid  # includes graph walk
```

### 7. Dashboard

```bash
python3 -m dashboard.export --serve
# Open http://localhost:8080
```

## Directory Structure

```
engram/
├── SKILL.md              # OpenClaw skill manifest
├── README.md             # This file
├── soul-template.md      # Agent soul template
├── agents-template.md    # Agent bootstrap template
├── .env.example          # Configuration template
├── .gitignore
├── brain/
│   ├── __init__.py       # DB setup, schema, core functions
│   ├── search.py         # Semantic + hybrid search
│   ├── sleep.py          # Fitness, core memory, GC
│   ├── compact.py        # Decay + pruning wrapper
│   └── ruminate.py       # Generative synthesis engine
├── extractors/
│   ├── memory.py         # Daily notes → nodes
│   ├── sessions.py       # Session logs → nodes
│   ├── chats.py          # Personal chats → nodes
│   ├── work.py           # Work context → nodes
│   └── topical.py        # Focused topic threads → nodes
├── scripts/
│   └── run_cycles.sh     # Think→sleep orchestration
├── dashboard/
│   ├── export.py         # Generate graph JSON
│   └── index.html        # D3.js visualization
└── crons.md              # Ready-to-paste cron configs
```

## Agent Integration

See `soul-template.md` for how agents should interact with the brain, and
`agents-template.md` for the session startup sequence and operational discipline.

Key principles:
- **Query before answering** — check the brain before guessing about preferences or history
- **Write immediately** — decisions, corrections, and preferences should be stored on the spot
- **Mental notes don't survive restarts** — if it matters, write it to the brain
- **Sub-agents are read-only** — only the primary agent writes

## Cron Schedule

See `crons.md` for complete, copy-paste ready cron configurations:

| Frequency | Task |
|-----------|------|
| Hourly | Extract from notes |
| Every 4h | Extract from sessions + chats |
| 2x daily | Ruminate + sleep cycles |
| Every 6h | WAL-safe database backup |
| Weekly | Full audit + compact |

## License

Private use. Not for redistribution.
