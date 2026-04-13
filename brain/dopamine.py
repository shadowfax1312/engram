"""
dopamine.py — Intrinsic reward signal for Engram's curiosity engine.

Two-stage reward model (validated by Bunny's feedback):

IMMEDIATE REWARD (on ingest):
- structural_novelty only — cluster bridges from sleep.py
- This is ground truth: did the graph topology actually change?

LAGGED REWARD (after 7 days):
- retrieval_frequency — how often was the node pulled in future queries?
- This validates whether the knowledge was actually useful

KL divergence proxy and prediction error demoted to tiebreakers only.
Reason: KL proxy measures graph sparsity (not uncertainty), and prediction 
error measures model confidence (notoriously uncalibrated).
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = Path(__file__).parent / "brain.db"

# Weights
STRUCTURAL_WEIGHT = 0.70       # Primary: real graph bridges (ground truth)
TIEBREAKER_WEIGHT = 0.30       # Secondary: sparsity + confidence proxy

LAGGED_REWARD_DAYS = 7         # When to compute retrieval-based dopamine
DECAY_RATE = 0.85              # How fast explored gaps decay in surprise_score
WEIGHT_LEARNING_RATE = 0.1     # How fast gap_type weights update


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def compute_immediate_dopamine(cluster_bridges: int, nodes_produced: int) -> float:
    """
    Immediate reward based on structural change only.
    
    cluster_bridges: new cross-cluster edges created by sleep.py
    nodes_produced: raw count (secondary signal)
    """
    if cluster_bridges == 0 and nodes_produced == 0:
        return 0.0

    # Primary: cluster bridges (ground truth signal)
    structural = min(cluster_bridges / 3.0, 1.0)
    
    # Tiebreaker: node count (if same bridges, more nodes = slight bonus)
    tiebreaker = min(nodes_produced / 5.0, 1.0)

    dopamine = (structural * STRUCTURAL_WEIGHT) + (tiebreaker * TIEBREAKER_WEIGHT)
    return round(min(dopamine, 1.0), 4)


def compute_lagged_dopamine(node_ids: list[str]) -> float:
    """
    Lagged reward based on retrieval frequency over past 7 days.
    
    Called by a cron job 7 days after node ingestion.
    node_ids: list of node IDs ingested in that introspection cycle
    """
    if not node_ids:
        return 0.0
    
    conn = get_db()
    
    # Count how many times these nodes appeared in search results
    placeholders = ','.join(['?' for _ in node_ids])
    
    # Check curiosity_log for queries that would have matched these nodes
    # This is a proxy — real implementation would log actual search hits
    retrieval_count = 0
    for node_id in node_ids:
        # Check if node was created and has been accessed
        node = conn.execute(
            "SELECT access_count FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if node:
            retrieval_count += node['access_count']
    
    conn.close()
    
    # Normalize: 5+ retrievals in a week = max reward
    lagged_score = min(retrieval_count / (len(node_ids) * 5.0), 1.0)
    return round(lagged_score, 4)


def give_chocolate(
    curiosity_log_id: int,
    query: str,
    gap_type: str,
    nodes_produced: int,
    novelty_score: float,  # kept for backward compat
    status: str,
    summary: str,
    cluster_bridges: int = 0,
    prior_uncertainty: float = 0.5,
    node_ids: list[str] = None,
) -> float:
    """
    Stage 1: Give immediate reward based on structural novelty.
    
    Schedules lagged reward computation for 7 days later.
    Returns immediate dopamine score.
    """
    dopamine = compute_immediate_dopamine(cluster_bridges, nodes_produced)
    
    conn = get_db()
    c = conn.cursor()
    
    # Log the investigation with immediate dopamine
    c.execute("""
        INSERT INTO introspection_log 
        (curiosity_log_id, query, gap_type, nodes_produced, novelty_score, dopamine, status, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (curiosity_log_id, query, gap_type, nodes_produced, novelty_score, dopamine, status, summary))
    
    introspection_id = c.lastrowid
    
    # Store node_ids for lagged reward computation
    if node_ids:
        c.execute("""
            UPDATE introspection_log 
            SET summary = summary || '\n[node_ids: ' || ? || ']'
            WHERE id = ?
        """, (','.join(node_ids), introspection_id))
    
    # Update gap_type weights based on immediate reward
    c.execute("""
        INSERT INTO dopamine_weights (gap_type, weight, total_investigations, total_novel_nodes, avg_dopamine)
        VALUES (?, 1.0, 1, ?, ?)
        ON CONFLICT(gap_type) DO UPDATE SET
            total_investigations = total_investigations + 1,
            total_novel_nodes = total_novel_nodes + ?,
            avg_dopamine = avg_dopamine + (? - avg_dopamine) * ?,
            weight = MAX(0.05, MIN(3.0, weight + (? - 0.5) * ?)),
            last_updated = datetime('now')
    """, (
        gap_type, nodes_produced, dopamine,
        nodes_produced, dopamine, WEIGHT_LEARNING_RATE,
        dopamine, WEIGHT_LEARNING_RATE
    ))
    
    # Decay the curiosity_log surprise_score for this gap (it's been explored)
    if status in ('filled', 'duplicate'):
        c.execute("""
            UPDATE curiosity_log 
            SET surprise_score = surprise_score * ?,
                flagged = CASE WHEN surprise_score * ? < 0.3 THEN 0 ELSE flagged END
            WHERE id = ?
        """, (DECAY_RATE, DECAY_RATE, curiosity_log_id))
    
    conn.commit()
    conn.close()
    
    return dopamine


def update_lagged_rewards():
    """
    Cron job: update dopamine weights based on 7-day retrieval frequency.
    
    Find introspection_log entries from ~7 days ago, compute lagged reward,
    update dopamine_weights with the validated signal.
    """
    conn = get_db()
    
    target_date = (datetime.now() - timedelta(days=LAGGED_REWARD_DAYS)).isoformat()[:10]
    
    # Find introspections from 7 days ago that haven't had lagged reward applied
    rows = conn.execute("""
        SELECT id, gap_type, summary, dopamine
        FROM introspection_log
        WHERE DATE(investigated_at) = ?
          AND summary NOT LIKE '%[lagged_applied]%'
    """, (target_date,)).fetchall()
    
    for row in rows:
        # Extract node_ids from summary
        import re
        match = re.search(r'\[node_ids: ([^\]]+)\]', row['summary'])
        if match:
            node_ids = match.group(1).split(',')
            lagged = compute_lagged_dopamine(node_ids)
            
            # ASYMMETRIC UPDATE (per Bunny's feedback):
            # High retrieval = boost weight (knowledge proved useful)
            # Low retrieval = don't punish (might be useful later, just not yet)
            if lagged > 0.3:  # Only boost if retrieval was meaningful
                conn.execute("""
                    UPDATE dopamine_weights 
                    SET avg_dopamine = avg_dopamine + (? - avg_dopamine) * 0.2,
                        weight = MIN(3.0, weight + (? - 0.3) * 0.2),
                        last_updated = datetime('now')
                    WHERE gap_type = ?
                """, (lagged, lagged, row['gap_type']))
            # Low lagged scores (< 0.3) are NOT penalized — asymmetric update
            
            # Mark as applied
            conn.execute("""
                UPDATE introspection_log 
                SET summary = summary || '\n[lagged_applied: ' || ? || ']'
                WHERE id = ?
            """, (lagged, row['id']))
    
    conn.commit()
    conn.close()
    
    return len(rows)


def get_weights() -> dict:
    """Return current gap_type weights for curiosity prioritization."""
    conn = get_db()
    rows = conn.execute("SELECT gap_type, weight, avg_dopamine, total_investigations FROM dopamine_weights").fetchall()
    conn.close()
    return {r['gap_type']: {
        'weight': r['weight'],
        'avg_dopamine': r['avg_dopamine'],
        'total_investigations': r['total_investigations']
    } for r in rows}


def get_prioritized_gaps(limit: int = 5) -> list:
    """
    Return high-priority curiosity gaps, weighted by:
    - surprise_score (raw signal)
    - gap_type dopamine weight (learned signal)
    - not recently investigated
    """
    conn = get_db()
    
    rows = conn.execute("""
        SELECT 
            cl.id,
            cl.query,
            cl.surprise_score,
            cl.logged_at,
            COALESCE(dw.weight, 1.0) as type_weight,
            (SELECT MAX(investigated_at) FROM introspection_log il 
             WHERE il.curiosity_log_id = cl.id) as last_investigated
        FROM curiosity_log cl
        LEFT JOIN dopamine_weights dw ON (
            CASE 
                WHEN cl.query LIKE '%quant%' OR cl.query LIKE '%trading%' OR cl.query LIKE '%volatility%' THEN 'quant'
                WHEN cl.query LIKE '%research%' OR cl.query LIKE '%paper%' OR cl.query LIKE '%study%' THEN 'research'
                WHEN cl.query LIKE '%infra%' OR cl.query LIKE '%graph%' OR cl.query LIKE '%brain%' THEN 'infra'
                WHEN cl.query LIKE '%PF2%' OR cl.query LIKE '%game%' OR cl.query LIKE '%spell%' THEN 'hobby'
                ELSE 'research'
            END = dw.gap_type
        )
        WHERE cl.flagged = 1 
          AND cl.surprise_score >= 0.35
          AND (last_investigated IS NULL OR last_investigated < datetime('now', '-6 hours'))
          -- Deduplicate by query text: only take the highest-scoring row per unique query
          AND cl.id = (
              SELECT id FROM curiosity_log cl2
              WHERE cl2.query = cl.query AND cl2.flagged = 1
              ORDER BY cl2.surprise_score DESC
              LIMIT 1
          )
        ORDER BY (cl.surprise_score * COALESCE(dw.weight, 1.0)) DESC
        LIMIT ?
    """, (limit,)).fetchall()
    
    conn.close()
    return [dict(r) for r in rows]


def print_dopamine_stats():
    """Print current dopamine weights and recent investigation history."""
    weights = get_weights()
    print("\n🧠 Dopamine Weights (gap_type → learned curiosity value):")
    for gap_type, stats in sorted(weights.items(), key=lambda x: -x[1]['weight']):
        bar = "█" * int(stats['weight'] * 10)
        print(f"  {gap_type:12} {bar:30} {stats['weight']:.3f} (n={stats['total_investigations']}, avg={stats['avg_dopamine']:.3f})")
    
    conn = get_db()
    recent = conn.execute("""
        SELECT query, gap_type, nodes_produced, dopamine, status, investigated_at
        FROM introspection_log
        ORDER BY investigated_at DESC
        LIMIT 10
    """).fetchall()
    conn.close()
    
    if recent:
        print("\n🍫 Recent Investigations (immediate reward):")
        for r in recent:
            emoji = "🟢" if r['dopamine'] > 0.5 else "🟡" if r['dopamine'] > 0.2 else "⚫"
            print(f"  {emoji} [{r['gap_type']:10}] {r['query'][:50]:50} → {r['nodes_produced']} nodes, dopamine={r['dopamine']:.3f}")


if __name__ == "__main__":
    print_dopamine_stats()
