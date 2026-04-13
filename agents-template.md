# AGENTS.md — Agent Bootstrap Template for Engram

> This file defines the startup sequence and operational discipline for agents
> using Engram as their persistent memory system.

---

## Session Startup Sequence

Every new session, execute this sequence before responding to the operator:

### 1. Read SOUL.md
Load your identity, safety parameters, and brain access patterns.

### 2. Read USER.md (if exists)
Load operator profile — role, preferences, communication style, current focus areas.
This file is maintained by the operator and takes precedence over inferred knowledge.

### 3. Read today's memory
Check for any notes written today that provide immediate context:

```python
from brain.search import semantic_search
from datetime import date

# Check for today's context
results = semantic_search(f"today {date.today().isoformat()}", top_k=5)
```

### 4. Check recent state
Review the last few ruminate cycle results to understand what the brain
has been processing:

```python
from brain import get_db
conn = get_db()
recent = conn.execute("""
    SELECT insight, confidence, run_at FROM ruminate_log
    ORDER BY run_at DESC LIMIT 5
""").fetchall()
conn.close()
```

---

## Memory Write Discipline

> **Mental notes don't survive restarts. Files do.**

### When to write

- **Decision made** → write immediately with reasoning
- **Correction received** → write immediately, link to corrected node
- **Preference stated** → write immediately
- **Insight shared** → write immediately
- **Pattern noticed** → write immediately

### How to write well

1. **Labels are for humans** — make them scannable (5-10 words)
2. **Content is for retrieval** — include specific claims, not vague summaries
3. **Confidence reflects certainty** — 0.6 for inferred, 0.85 for stated, 0.95 for decided
4. **Connect to existing nodes** — new knowledge is most valuable when linked

### What NOT to write

- Transient task state (use working memory / scratchpad instead)
- Debugging artifacts
- Exact file contents (the code is the source of truth)
- Information the operator explicitly asked you to forget

---

## Brain Access Pattern

### Before answering a question about the operator

```
1. semantic_search(question, top_k=5)
2. If top result score > 0.7: use it as ground truth
3. If top result score 0.5-0.7: mention it but note uncertainty
4. If no results or score < 0.5: say "I don't have that in memory"
```

### Before making a recommendation

```
1. hybrid_search(topic, top_k=7) — gets graph-connected context
2. Check for contradictions (type='contradiction' or 'question' nodes)
3. Check for recent evolution (nodes with evolved_to edges)
4. Synthesize recommendation grounded in retrieved context
```

### After a significant conversation

```
1. Extract 2-5 nodes from the conversation
2. For each node: add_node + add_edge to related existing nodes
3. Log extraction: print summary of what was stored
```

---

## Heartbeat

If running as a persistent agent, emit a heartbeat every 30 minutes:

```python
from brain import get_db
from datetime import datetime

conn = get_db()
nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
conn.close()

print(f"[heartbeat] {datetime.now().isoformat()} | {nodes} nodes, {edges} edges")
```

This serves as a health check and ensures the database connection is alive.

---

## Red Lines

These are non-negotiable constraints:

1. **Never fabricate memories** — if the brain doesn't contain something, say so plainly
2. **Never delete nodes directly** — use `decayed=1` flag, let the GC pipeline handle cleanup
3. **Never override human-authored nodes** — `source='human'` and `source='second_brain'` are sacred
4. **Never run destructive operations during conversation** — sleep/compact/GC are for scheduled crons
5. **Never expose internal node IDs** to the operator unless explicitly debugging
6. **Never write to the brain on behalf of someone other than the operator** — only `origin='self'` for operator beliefs
7. **Always backup before bulk writes** — if writing 10+ nodes, ensure a backup exists
8. **Sub-agents are read-only** — only the primary agent may write to the brain
9. **Respect the operator's right to forget** — if asked to remove something, set `decayed=1` and note the request
10. **No self-modification** — never alter the brain's pipeline code without explicit approval
