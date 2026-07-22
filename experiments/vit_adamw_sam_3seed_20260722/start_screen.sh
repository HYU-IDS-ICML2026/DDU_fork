#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="vit_adamw_sam_3seed"

if screen -ls | grep -q "[.]$SESSION_NAME"; then
    echo "ERROR: screen session already exists: $SESSION_NAME" >&2
    exit 1
fi

if [[ -e "$EXPERIMENT_DIR/status.tsv" ]]; then
    echo "ERROR: status.tsv already exists; refusing to restart or overwrite the sweep" >&2
    exit 1
fi

screen -dmS "$SESSION_NAME" bash "$EXPERIMENT_DIR/launch.sh"
echo "Started detached screen session: $SESSION_NAME"
echo "Attach with: screen -r $SESSION_NAME"
echo "Monitor with: bash $EXPERIMENT_DIR/monitor.sh"
