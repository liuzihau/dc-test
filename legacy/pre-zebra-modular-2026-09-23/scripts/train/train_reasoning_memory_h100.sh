#!/usr/bin/env bash
# Sudoku/Zebra x corrected merged both/both_aux, sequentially on ONE H100.
# Shared data and execution engine with the baseline queue; separate run paths.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
PYTHON_BIN="$(command -v "${DCACHE_PYTHON:-python}")"
for ARG in "$@"; do
  case "$ARG" in
    --suite|--suite=*) echo 'This wrapper fixes --suite memory; use the baseline wrapper for controls.' >&2; exit 2 ;;
  esac
done
ulimit -c 0
exec "$PYTHON_BIN" "$REPO_DIR/scripts/reasoning/run_baseline_queue.py" "$@" --suite memory
