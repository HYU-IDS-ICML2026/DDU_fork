#!/usr/bin/env bash
set -u

EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="vit_adamw_sam_3seed"

echo "=== Screen session ==="
screen -ls | grep "[.]$SESSION_NAME" || echo "No active session named $SESSION_NAME"

echo "=== Sweep status ==="
if [[ -f "$EXPERIMENT_DIR/status.tsv" ]]; then
    cat "$EXPERIMENT_DIR/status.tsv"
else
    echo "Not started: status.tsv does not exist"
fi

echo "=== Current log tail ==="
if [[ -e "$EXPERIMENT_DIR/current.log" ]]; then
    tail -n 40 "$EXPERIMENT_DIR/current.log"
else
    echo "No current run log"
fi
