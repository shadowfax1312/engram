#!/usr/bin/env python3
"""
Engram — Compact: decay, merge duplicates, prune orphans.

Wrapper around sleep.py for cron compatibility.
Run: python3 -m brain.compact [--dry-run] [--threshold 0.05] [--full]
"""

from brain import get_db
from brain.sleep import run_sleep


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compact (sleep cycle wrapper)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    conn = get_db()
    run_sleep(conn, dry_run=args.dry_run, threshold=args.threshold, full=args.full)
    conn.close()
