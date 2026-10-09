#!/usr/bin/env bash
# Twelve full-state continuations, after ALL first-epoch evaluations finish.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${DCACHE_CUDA_VISIBLE_DEVICES:-2,3}"
export TMPDIR="$PWD/.cache/runtime/reasoning-second-split/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export MPLCONFIGDIR="$PWD/.cache/runtime/reasoning-second-split/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/runtime/reasoning-second-split/inductor"
export TRITON_CACHE_DIR="$PWD/.cache/runtime/reasoning-second-split/triton"
export XDG_CACHE_HOME="$PWD/.cache"
export CUDA_CACHE_PATH="$PWD/.cache/runtime/reasoning-second-split/cuda"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
ulimit -c 0
action="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
run_dir="${DCACHE_SECOND_EPOCH_DIR:-$PWD/outputs/reasoning/second-epoch-split-six-2x3090-mb32}"
socket="$PWD/.cache/runtime/reasoning-second-split/tmux.sock"
session="reasoning-epoch2"
if [[ "$action" == tmux || "$action" == attach ]]; then
  tmux_cmd=(env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/tmux -S "$socket")
  if [[ "$action" == attach ]]; then
    exec "${tmux_cmd[@]}" attach-session -t "$session"
  fi
  if "${tmux_cmd[@]}" has-session -t "$session" 2>/dev/null; then
    echo "Queue session already exists; no duplicate launched."
  else
    mkdir -p logs
    printf -v command '%q ' bash "$PWD/scripts/reasoning/run_split_second_epoch_2x3090.sh" run "$@"
    printf -v log_path '%q' "$PWD/logs/split-second-epoch-mb32.log"
    "${tmux_cmd[@]}" new-session -d -s "$session" "set -o pipefail; ${command} 2>&1 | tee -a ${log_path}"
  fi
  echo "Attach: bash scripts/reasoning/run_split_second_epoch_2x3090.sh attach"
  echo "Status: $run_dir/status.json"
  echo "Console: $PWD/logs/split-second-epoch-mb32.log"
  exit 0
fi
exec "${DCACHE_PYTHON:-python}" -u -m reasoning.second_epoch_queue "$action" \
  --suite split --gpu "$CUDA_VISIBLE_DEVICES" --devices "${DCACHE_DEVICES:-2}" \
  --micro-batch "${DCACHE_MICRO_BATCH:-32}" --eval-batch "${DCACHE_EVAL_BATCH:-32}" \
  --task-order "${DCACHE_TASK_ORDER:-sudoku-first}" \
  --after "${DCACHE_FIRST_EPOCH_DIR:-$PWD/outputs/reasoning/full-epoch-split-six-2x3090-mb32}" \
  --output "$run_dir" "$@"
