#!/usr/bin/env bash
# Twelve fresh jobs: six split-suite mechanisms x Zebra and Sudoku-Puzzle.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${DCACHE_CUDA_VISIBLE_DEVICES:-2,3}"
export TMPDIR="$PWD/.cache/runtime/reasoning-split/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export MPLCONFIGDIR="$PWD/.cache/runtime/reasoning-split/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/runtime/reasoning-split/inductor"
export TRITON_CACHE_DIR="$PWD/.cache/runtime/reasoning-split/triton"
export XDG_CACHE_HOME="$PWD/.cache"
export CUDA_CACHE_PATH="$PWD/.cache/cuda"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
ulimit -c 0
action="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
exec "${DCACHE_PYTHON:-python}" -u -m reasoning.full_epoch_queue "$action" \
  --suite split --gpu "$CUDA_VISIBLE_DEVICES" --devices "${DCACHE_DEVICES:-2}" \
  --micro-batch "${DCACHE_MICRO_BATCH:-8}" --eval-batch "${DCACHE_EVAL_BATCH:-8}" \
  --task-order "${DCACHE_TASK_ORDER:-sudoku-first}" \
  --output "${DCACHE_RUN_DIR:-$PWD/outputs/reasoning/full-epoch-split-six-2x3090}" "$@"
