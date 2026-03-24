#!/bin/bash
# run_cycles.sh — alternating ruminate → sleep × N
#
# Usage: bash scripts/run_cycles.sh [cycles=10]
#
# Requires: BRAIN_DIR env var set, or defaults to ~/.engram

set -euo pipefail

CYCLES=${1:-10}
LOG="/tmp/engram_cycles_$(date +%Y%m%d_%H%M%S).log"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

echo "Starting $CYCLES think-sleep cycles" | tee "$LOG"
echo "  BRAIN_DIR: ${BRAIN_DIR:-~/.engram}" | tee -a "$LOG"
echo "  Log: $LOG" | tee -a "$LOG"

for i in $(seq 1 "$CYCLES"); do
    echo "" | tee -a "$LOG"
    echo "──────────────────────────────────────────" | tee -a "$LOG"
    echo "  Cycle $i / $CYCLES — $(date '+%H:%M:%S')" | tee -a "$LOG"
    echo "──────────────────────────────────────────" | tee -a "$LOG"

    echo "  Think..." | tee -a "$LOG"
    python3 -m brain.ruminate --cycles 1 --force 2>&1 | tee -a "$LOG"

    echo "  Sleep..." | tee -a "$LOG"
    python3 -m brain.sleep 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "All $CYCLES cycles complete" | tee -a "$LOG"

# Print final stats
python3 -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from brain import get_db
conn = get_db()
nodes = conn.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]
edges = conn.execute('SELECT COUNT(*) FROM edges').fetchone()[0]
print(f'  Final graph: {nodes} nodes, {edges} edges')
conn.close()
" 2>&1 | tee -a "$LOG"

echo "Log saved to: $LOG"
