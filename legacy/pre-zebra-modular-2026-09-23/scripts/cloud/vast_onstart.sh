#!/usr/bin/env bash
# Invoke from the template's existing startup hook; do not replace that hook.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
CONFIG="$REPO_DIR/.cache/recovery/launch.json"
if [[ ! -f "$CONFIG" ]]; then
  echo "No saved recovery configuration. Run lightning_h100.sh arm first." >&2
  exit 2
fi
mkdir -p "$REPO_DIR/logs" "$REPO_DIR/.cache/runtime/recovery/tmp"
export TMPDIR="$REPO_DIR/.cache/runtime/recovery/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
ulimit -c 0
# This supervisor is stdlib-only. It launches training with the exact interpreter
# recorded by arm, even when the startup hook has not activated conda.
nohup python3 "$REPO_DIR/scripts/cloud/supervise_training.py" run --config "$CONFIG" \
  >> "$REPO_DIR/logs/recovery-supervisor.log" 2>&1 < /dev/null &
echo "Requested background recovery supervisor. Log: $REPO_DIR/logs/recovery-supervisor.log"
