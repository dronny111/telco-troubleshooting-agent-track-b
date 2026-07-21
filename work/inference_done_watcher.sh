#!/bin/bash
# Waits for the Track A and Track B fresh-inference PIDs to exit,
# then writes a clear "DONE" marker with final row counts + timestamps.
#
# Re-edit PIDs below if they change. Run with:
#   nohup bash work/inference_done_watcher.sh > /tmp/done_watcher.log 2>&1 &

set -u

TRACK_A_PID=${TRACK_A_PID:-78627}
TRACK_B_PID=${TRACK_B_PID:-52330}

TRACK_A_CSV="/Users/ronnypolle/Desktop/telco_itu/telco_data/Track A/results_phase2_fresh_topk16/result.csv"
TRACK_B_CSV="/Users/ronnypolle/Desktop/telco_itu/work/submission_fresh_b/result.csv"
DONE_FILE="/Users/ronnypolle/Desktop/telco_itu/work/INFERENCE_DONE.txt"
HEARTBEAT_FILE="/Users/ronnypolle/Desktop/telco_itu/work/INFERENCE_HEARTBEAT.txt"

START_TS=$(date "+%Y-%m-%d %H:%M:%S %Z")

# Heartbeat every 5 minutes so you can see the watcher is alive
heartbeat() {
  while true; do
    {
      echo "watcher heartbeat: $(date '+%Y-%m-%d %H:%M:%S %Z')"
      echo "  watcher started: $START_TS"
      echo "  Track A PID $TRACK_A_PID alive: $(ps -p $TRACK_A_PID 2>/dev/null | grep -c python)"
      echo "  Track A rows: $(wc -l < "$TRACK_A_CSV" 2>/dev/null || echo NA)"
      echo "  Track B PID $TRACK_B_PID alive: $(ps -p $TRACK_B_PID 2>/dev/null | grep -c python)"
      echo "  Track B rows: $(wc -l < "$TRACK_B_CSV" 2>/dev/null || echo NA)"
    } > "$HEARTBEAT_FILE"
    sleep 300
  done
}
heartbeat &
HEARTBEAT_PID=$!

# Wait for both PIDs to exit
wait_for_pid() {
  local pid=$1
  while ps -p "$pid" > /dev/null 2>&1; do
    sleep 30
  done
}

wait_for_pid "$TRACK_A_PID"
TRACK_A_DONE_TS=$(date "+%Y-%m-%d %H:%M:%S %Z")
wait_for_pid "$TRACK_B_PID"
TRACK_B_DONE_TS=$(date "+%Y-%m-%d %H:%M:%S %Z")

kill "$HEARTBEAT_PID" 2>/dev/null || true

A_ROWS=$(wc -l < "$TRACK_A_CSV" 2>/dev/null || echo "NA")
B_ROWS=$(wc -l < "$TRACK_B_CSV" 2>/dev/null || echo "NA")
END_TS=$(date "+%Y-%m-%d %H:%M:%S %Z")

# Best-effort macOS notification (requires not-locked screen to display; harmless if it fails)
osascript -e 'display notification "Phase 2 fresh inference: Track A + Track B finished" with title "ITU Telco Inference Done"' 2>/dev/null || true

# Write the marker file
{
  echo "=============================================="
  echo "  PHASE 2 FRESH INFERENCE — DONE"
  echo "=============================================="
  echo "watcher started:  $START_TS"
  echo "watcher finished: $END_TS"
  echo ""
  echo "Track A (PID $TRACK_A_PID)"
  echo "  finished: $TRACK_A_DONE_TS"
  echo "  rows:     $A_ROWS  (of expected 501)"
  echo "  csv:      $TRACK_A_CSV"
  echo "  log:      /tmp/track_a_fresh.log"
  echo ""
  echo "Track B rerun (PID $TRACK_B_PID)"
  echo "  finished: $TRACK_B_DONE_TS"
  echo "  rows:     $B_ROWS  (of expected 101)"
  echo "  csv:      $TRACK_B_CSV"
  echo "  log:      /tmp/track_b_rerun.log"
  echo ""
  echo "Next steps:"
  echo "  1. Audit results: python3 work/run_track_a_inference.py is not needed; data is already in the CSVs above"
  echo "  2. Build final submission combining fresh + _07 fallback"
  echo "  3. Restore sleep: sudo pmset -a disablesleep 0"
  echo "=============================================="
} > "$DONE_FILE"

# Also touch a simple zero-byte marker for easy grep/find detection
touch "/Users/ronnypolle/Desktop/telco_itu/work/INFERENCE_DONE.flag"
