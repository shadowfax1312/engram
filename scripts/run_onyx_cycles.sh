#!/bin/bash
# run_onyx_cycles.sh — alternating onyx_ruminate → onyx_sleep × N
# Usage: bash run_onyx_cycles.sh [cycles=10]

CYCLES=${1:-10}
LOG=/tmp/run_onyx_cycles.log
cd "$(dirname "$0")"

echo "Onyx: Starting $CYCLES think->sleep cycles" | tee $LOG

for i in $(seq 1 $CYCLES); do
    echo "" | tee -a $LOG
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a $LOG
    echo "  Onyx Cycle $i / $CYCLES" | tee -a $LOG
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a $LOG

    echo "Think..." | tee -a $LOG
    python3 onyx_ruminate.py --cycles 1 --force 2>&1 | tee -a $LOG

    echo "Sleep..." | tee -a $LOG
    python3 onyx_sleep.py 2>&1 | tee -a $LOG
done

echo "" | tee -a $LOG
echo "All $CYCLES onyx cycles complete" | tee -a $LOG
openclaw system event --text "Done: $CYCLES onyx think->sleep cycles complete. Onyx graph consolidated." --mode now
