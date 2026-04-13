# DEPRECATED: functions merged into sleep.py. This file will be removed.
#!/usr/bin/env python3
"""
Signal 1 Sleep Functions
Replace decay_relevance() with compute_all_fitness()
Replace fitness_gc() with new GC logic

Import into sleep.py or call directly for testing.
"""

import sys
import math
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from init_graph import get_db


def parse_timestamp(ts):
    """Parse timestamp to unix seconds"""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return dt.timestamp()
        except:
            return None
    return None


def compute_fitness(origin, access_count, last_accessed_at, now=None):
    """Unified fitness score: baseline + recency + frequency"""
    now = now or time.time()
    
    baseline = 0.5 if origin == 'self' else 0.3
    
    last_ts = parse_timestamp(last_accessed_at)
    age_days = (now - last_ts) / 86400.0 if last_ts else 0
    recency = 2.0 ** (-age_days / 7.0)  # Half-life: 7 days
    
    frequency = min(0.5, math.log(max(1, access_count or 0) + 1) * 0.1)
    
    fitness = baseline + recency + frequency
    return min(1.0, max(0.0, fitness))


def compute_all_fitness(conn, dry_run=False):
    """
    Replacement for decay_relevance().
    Computes unified fitness_score for all nodes.
    Also sets soft_decay_flagged for nodes 90+ days old.
    """
    print("  START compute_all_fitness", flush=True)
    now = time.time()
    
    nodes = conn.execute(
        "SELECT id, origin, access_count, last_accessed_at FROM nodes"
    ).fetchall()
    
    updates = []
    for node in nodes:
        node_id = node['id']
        origin = node['origin']
        access_count = node['access_count'] or 0
        last_accessed_at = node['last_accessed_at']
        
        fitness = compute_fitness(origin, access_count, last_accessed_at, now)
        
        last_ts = parse_timestamp(last_accessed_at) or now
        age_days = (now - last_ts) / 86400.0
        soft_decay = 1 if age_days > 90 else 0
        
        updates.append((fitness, soft_decay, node_id))
    
    if not dry_run:
        conn.executemany(
            "UPDATE nodes SET fitness_score = ?, soft_decay_flagged = ? WHERE id = ?",
            updates
        )
        conn.commit()
    
    # Stats
    fitness_vals = [u[0] for u in updates]
    soft_decay_count = sum(u[1] for u in updates)
    
    print(f"  Fitness: updated {len(updates)} nodes (avg={sum(fitness_vals)/len(fitness_vals):.3f}, soft_decay={soft_decay_count})", flush=True)
    return len(updates)


def fitness_gc_v2(conn, dry_run=False):
    """
    Replacement for fitness_gc().
    Only prune if: fitness < 0.15 AND soft_decay_flagged=1 AND age > 120 days
    """
    print(f"  START fitness_gc_v2 (dry_run={dry_run})", flush=True)
    now = time.time()
    
    # Find candidates
    candidates = conn.execute("""
        SELECT id, label, fitness_score, soft_decay_flagged, last_accessed_at
        FROM nodes
        WHERE fitness_score < 0.15 AND soft_decay_flagged = 1
    """).fetchall()
    
    # Filter by age > 120 days
    gc_list = []
    for c in candidates:
        last_ts = parse_timestamp(c['last_accessed_at']) or now
        age_days = (now - last_ts) / 86400.0
        if age_days > 120:
            gc_list.append((c['id'], c['label'], c['fitness_score'], age_days))
    
    if not gc_list:
        print(f"  GC: 0 candidates (none met: fitness<0.15 + soft_decay + age>120d)", flush=True)
        return 0
    
    print(f"  GC: found {len(gc_list)} candidates", flush=True)
    for nid, label, fitness, age in gc_list[:5]:
        print(f"    - {label[:40]} (fitness={fitness:.3f}, age={age:.0f}d)", flush=True)
    if len(gc_list) > 5:
        print(f"    ... and {len(gc_list) - 5} more", flush=True)
    
    if dry_run:
        print(f"  GC (dry_run): would delete {len(gc_list)} nodes", flush=True)
        return len(gc_list)
    
    deleted = 0
    for nid, label, _, _ in gc_list:
        conn.execute("DELETE FROM edges WHERE from_id = ? OR to_id = ?", (nid, nid))
        conn.execute("DELETE FROM embeddings WHERE node_id = ?", (nid,))
        conn.execute("DELETE FROM nodes WHERE id = ?", (nid,))
        deleted += 1
    
    conn.commit()
    print(f"  GC: deleted {deleted} nodes", flush=True)
    return deleted


if __name__ == '__main__':
    """Test run"""
    dry_run = '--live' not in sys.argv
    
    print("\n" + "="*60)
    print(f"SIGNAL 1 SLEEP TEST ({'DRY-RUN' if dry_run else 'LIVE'})")
    print("="*60 + "\n")
    
    conn = get_db()
    
    compute_all_fitness(conn, dry_run=dry_run)
    fitness_gc_v2(conn, dry_run=dry_run)
    
    conn.close()
    print("\n" + "="*60 + "\n")
