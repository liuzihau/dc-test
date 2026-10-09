#!/usr/bin/env bash
# Sudoku/Zebra x vanilla/mdm/mdm_aux, sequentially on ONE H100.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
PYTHON_BIN="$(command -v "${DCACHE_PYTHON:-python}")"
ulimit -c 0
exec "$PYTHON_BIN" "$REPO_DIR/scripts/reasoning/run_baseline_queue.py" "$@"
