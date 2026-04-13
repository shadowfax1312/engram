#!/usr/bin/env python3
"""
evolve.py — Mechanical architecture proposal engine.

Reads hard metrics from onyx_brain.db, infers architectural implications via
structured LLM prompt, scores proposals by confidence formula, sends
high-confidence proposals to Ganesh via Telegram.

Usage:
  python3 evolve.py              # Run full cycle: extract → infer → score → deliver
  python3 evolve.py --dry-run    # Print metrics + proposals without storing or sending
"""

import sys
import os
import json
import argparse
import sqlite3
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from gateway import call_llm

ONYX_DB = Path(__file__).parent / "onyx_brain.db"
BRAIN_DB = Path(__file__).parent / "brain.db"

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
OWNER_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

CONFIDENCE_THRESHOLD = 0.82


# ── DB helpers ──────────────────────────────────────────────────────

def get_db(db_path):
    conn = sqlite3.connect(db_path, timeout=120)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_evolve_log(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evolve_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            trigger_metric TEXT,
            trigger_value TEXT,
            implication TEXT,
            proposal TEXT,
            proposal_type TEXT,
            effort TEXT,
            confidence REAL,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT (datetime('now')),
            reviewed_at TEXT,
            executed_at TEXT,
            telegram_msg_id TEXT
        )
    """)
    conn.commit()


# ── Dopamine metrics from brain.db ────────────────────────────────

def get_dopamine_metrics() -> dict:
    """Read dopamine signal from brain.db — the only cross-db read in evolve.py"""
    try:
        conn = sqlite3.connect(str(BRAIN_DB))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Top gap_types by avg_dopamine
        top_gaps = c.execute("""
            SELECT gap_type, avg_dopamine, total_investigations, total_novel_nodes
            FROM dopamine_weights
            WHERE total_investigations > 0
            ORDER BY avg_dopamine DESC LIMIT 5
        """).fetchall()

        # Recent introspection dopamine by gap_type (last 30 days)
        recent = c.execute("""
            SELECT gap_type, AVG(dopamine) as avg_d, COUNT(*) as count,
                   SUM(CASE WHEN nodes_produced = 0 THEN 1 ELSE 0 END) as zero_yield
            FROM introspection_log
            WHERE investigated_at > datetime("now", "-30 days")
            AND gap_type IS NOT NULL
            GROUP BY gap_type
            ORDER BY avg_d DESC
        """).fetchall()

        conn.close()
        return {
            "top_dopamine_gaps": [dict(r) for r in top_gaps],
            "recent_gap_performance": [dict(r) for r in recent],
            "avg_dopamine_all": sum(r["avg_dopamine"] for r in top_gaps) / len(top_gaps) if top_gaps else 0.0,
        }
    except Exception as e:
        return {"top_dopamine_gaps": [], "recent_gap_performance": [], "avg_dopamine_all": 0.0}


def get_dopamine_weighted_unresolved(top_dopamine_gaps) -> float:
    """For each high-dopamine gap_type, count unresolved insight_question nodes in onyx_brain.db
    and weight by avg_dopamine."""
    if not top_dopamine_gaps:
        return 0.0
    try:
        onyx = get_db(ONYX_DB)
        score = 0.0
        for gap in top_dopamine_gaps:
            gap_type = gap["gap_type"]
            avg_d = gap["avg_dopamine"]
            unresolved = onyx.execute("""
                SELECT COUNT(*) FROM nodes n
                WHERE n.type = 'insight_question'
                AND n.domain = ?
                AND NOT EXISTS (
                    SELECT 1 FROM edges e
                    WHERE e.from_id = n.id
                    AND e.relation IN ('resolved_by', 'evolved_to', 'enables')
                )
            """, (gap_type,)).fetchone()[0]
            score += avg_d * unresolved
        onyx.close()
        return round(score, 4)
    except Exception:
        return 0.0


# ── Metric extraction (hardcoded SQL, no LLM) ──────────────────────

def extract_metrics():
    metrics = {}

    onyx = get_db(ONYX_DB)

    metrics["onyx_brain_nodes"] = onyx.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    metrics["onyx_brain_edges"] = onyx.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    metrics["onyx_oscillation_count"] = onyx.execute(
        "SELECT COUNT(*) FROM nodes WHERE type LIKE '%oscillation%'"
    ).fetchone()[0]

    pattern_conn_count = onyx.execute("""
        SELECT COUNT(*) FROM nodes
        WHERE type LIKE '%pattern%' OR type LIKE '%connection%'
    """).fetchone()[0]

    denom = metrics["onyx_oscillation_count"] + pattern_conn_count
    metrics["onyx_unresolved_ratio"] = (
        metrics["onyx_oscillation_count"] / denom if denom > 0 else 0.0
    )

    # Unresolved question density: insight_question nodes with no resolving outbound edges
    metrics["unresolved_question_density"] = onyx.execute("""
        SELECT COUNT(*) FROM nodes n
        WHERE n.type = 'insight_question'
        AND NOT EXISTS (
            SELECT 1 FROM edges e
            WHERE e.from_id = n.id
            AND e.relation IN ('resolved_by', 'evolved_to', 'enables')
        )
    """).fetchone()[0]

    # Ruminate rehit rate: cluster hub_ids appearing in ruminate_log > 3 times
    try:
        metrics["ruminate_rehit_rate"] = onyx.execute("""
            SELECT COUNT(*) FROM (
                SELECT nodes_involved, COUNT(*) as hits
                FROM ruminate_log
                GROUP BY nodes_involved
                HAVING hits > 3
            )
        """).fetchone()[0]
    except Exception:
        metrics["ruminate_rehit_rate"] = 0

    onyx.close()

    # Dopamine signals from brain.db
    dopamine = get_dopamine_metrics()
    metrics["top_dopamine_gaps"] = dopamine["top_dopamine_gaps"]
    metrics["recent_gap_performance"] = dopamine["recent_gap_performance"]
    metrics["avg_dopamine_all"] = dopamine["avg_dopamine_all"]
    metrics["dopamine_weighted_unresolved_score"] = get_dopamine_weighted_unresolved(
        dopamine["top_dopamine_gaps"]
    )

    return metrics


# ── Inference prompt ────────────────────────────────────────────────

INFERENCE_PROMPT = """You are an architecture analyst reading metrics from an AI knowledge graph system.
Based on ONLY the metrics below, generate 3-7 architecture proposals.

Each proposal MUST:
1. Name a specific metric that triggered it (include the actual number)
2. State the architectural implication in one sentence
3. Propose a concrete, buildable change (script, cron, prompt modification, new table)
4. Assign a proposal_type from: [new_script, cron_change, prompt_update, research_direction, data_model_change]
5. Assign effort: [low, medium, high]
6. Assign confidence: 0.0-1.0 based on signal strength and specificity

DO NOT generate proposals that are vague ("improve X"). Every proposal must name a specific file, function, table, or cron job to change.
DO NOT generate proposals about collecting more data unless the data gap directly causes a named metric to be unmeasurable.

DOPAMINE PRIORITY RULE: High avg_dopamine gap_types that also have high unresolved question counts are the highest-priority architectural targets. A gap_type with avg_dopamine > 0.6 and unresolved_count > 5 should generate at least one proposal.

METRICS:
{metrics_json}

Return ONLY a JSON array:
[{{"title": "short title", "trigger_metric": "metric_name", "trigger_value": "actual value that triggered this", "implication": "one sentence: what this metric implies architecturally", "proposal": "concrete change: what file/function/cron to create or modify and how", "proposal_type": "new_script|cron_change|prompt_update|research_direction|data_model_change", "effort": "low|medium|high", "confidence": 0.0}}]"""


def infer_proposals(metrics):
    prompt = INFERENCE_PROMPT.format(metrics_json=json.dumps(metrics, indent=2, default=str))
    response = call_llm(prompt, model="anthropic/claude-sonnet-4-5", max_tokens=4096)
    if not response:
        print("  ⚠ LLM returned empty response")
        return []
    return parse_json_array(response)


def parse_json_array(text):
    if not text:
        return []
    import re
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


# ── Confidence scoring (override LLM confidence with formula) ──────

def compute_confidence(proposal, metrics):
    base = float(proposal.get("confidence", 0.5))

    effort_boost = {"low": 0.05, "medium": 0.0, "high": -0.05}
    effort = proposal.get("effort", "medium")
    boost = effort_boost.get(effort, 0.0)

    vague_words = ["improve", "better", "enhance", "more", "optimize"]
    vague_penalty = -0.15 if any(w in proposal.get("title", "").lower() for w in vague_words) else 0

    hallucinated_penalty = -0.3 if proposal.get("trigger_metric") not in metrics else 0

    return min(1.0, max(0.0, base + boost + vague_penalty + hallucinated_penalty))


# ── Telegram delivery ──────────────────────────────────────────────

def send_telegram_proposal(proposal_id, proposal):
    text = (
        f"🧠 Evolve #{proposal_id} [{proposal['confidence']:.0%}]\n"
        f"{proposal['title']}\n\n"
        f"Signal: {proposal['trigger_metric']} = {proposal['trigger_value']}\n"
        f"Implication: {proposal['implication']}\n"
        f"Proposed: {proposal['proposal']}\n"
        f"Type: {proposal['proposal_type']} | Effort: {proposal['effort']}"
    )

    data = json.dumps({
        "chat_id": OWNER_CHAT_ID,
        "text": text,
    }).encode()

    req = urllib.request.Request(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                return str(result["result"]["message_id"])
            else:
                print(f"  ⚠ Telegram error: {result}")
    except Exception as e:
        print(f"  ⚠ Telegram send failed: {e}")
    return None


# ── Main cycle ──────────────────────────────────────────────────────

def run(dry_run=False):
    print("🧠 Evolve — Architecture Proposal Engine")
    print("=" * 50)

    # Step 1: Extract metrics
    print("\n📊 Extracting metrics...")
    metrics = extract_metrics()
    print(f"   Onyx brain nodes: {metrics['onyx_brain_nodes']} | Edges: {metrics['onyx_brain_edges']}")
    print(f"   Oscillation count: {metrics['onyx_oscillation_count']}")
    print(f"   Unresolved ratio: {metrics['onyx_unresolved_ratio']:.2f}")
    print(f"   Unresolved question density: {metrics['unresolved_question_density']}")
    print(f"   Ruminate rehit rate: {metrics['ruminate_rehit_rate']}")
    print(f"   Avg dopamine (all): {metrics['avg_dopamine_all']:.3f}")
    print(f"   Dopamine-weighted unresolved: {metrics['dopamine_weighted_unresolved_score']:.4f}")
    if metrics['top_dopamine_gaps']:
        print(f"   Top dopamine gaps: {[g['gap_type'] for g in metrics['top_dopamine_gaps'][:3]]}")

    if dry_run:
        print("\n📋 Full metrics dump:")
        print(json.dumps(metrics, indent=2, default=str))

    # Step 2: Infer proposals
    print("\n🔍 Inferring proposals via LLM...")
    proposals = infer_proposals(metrics)

    if not proposals:
        print("  ⚠ No proposals generated")
        return

    print(f"   Got {len(proposals)} raw proposals")

    # Step 3: Score
    for p in proposals:
        p["confidence"] = compute_confidence(p, metrics)

    proposals.sort(key=lambda p: p["confidence"], reverse=True)

    print("\n📋 Scored proposals:")
    for i, p in enumerate(proposals):
        marker = "🟢" if p["confidence"] >= CONFIDENCE_THRESHOLD else "⚪"
        print(f"   {marker} [{p['confidence']:.0%}] {p['title']}")
        print(f"      Signal: {p['trigger_metric']} = {p.get('trigger_value', '?')}")
        print(f"      Type: {p['proposal_type']} | Effort: {p['effort']}")

    if dry_run:
        print("\n🏁 Dry run complete — nothing stored or sent.")
        return

    # Step 4: Store all proposals
    onyx = get_db(ONYX_DB)
    ensure_evolve_log(onyx)

    stored = []
    for p in proposals:
        c = onyx.execute("""
            INSERT INTO evolve_log (title, trigger_metric, trigger_value, implication,
                                    proposal, proposal_type, effort, confidence, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        """, (
            p.get("title", ""),
            p.get("trigger_metric", ""),
            str(p.get("trigger_value", "")),
            p.get("implication", ""),
            p.get("proposal", ""),
            p.get("proposal_type", ""),
            p.get("effort", "medium"),
            p["confidence"],
        ))
        onyx.commit()
        row_id = c.lastrowid
        stored.append((row_id, p))

    # Step 5: Deliver high-confidence proposals via Telegram
    sent_count = 0
    for row_id, p in stored:
        if p["confidence"] >= CONFIDENCE_THRESHOLD:
            print(f"\n📤 Sending proposal #{row_id}: {p['title']}")
            msg_id = send_telegram_proposal(row_id, p)
            if msg_id:
                onyx.execute(
                    "UPDATE evolve_log SET telegram_msg_id=? WHERE id=?",
                    (msg_id, row_id)
                )
                onyx.commit()
                sent_count += 1
                print(f"   ✓ Sent (msg_id={msg_id})")
            else:
                print(f"   ✗ Failed to send")

    onyx.close()

    print(f"\n🏁 Done: {len(stored)} proposals stored, {sent_count} sent to Telegram")


def main():
    parser = argparse.ArgumentParser(description="Architecture proposal engine")
    parser.add_argument("--dry-run", action="store_true", help="Print metrics + proposals without storing or sending")
    args = parser.parse_args()
    run(dry_run=args.dry_run)

    print("""
# Add these crons manually:
# evolve.py — run after each ruminate+sleep cycle (daily ~11pm GST)
# evolve_review.py — run weekly Monday 9am GST
""")


if __name__ == "__main__":
    main()
