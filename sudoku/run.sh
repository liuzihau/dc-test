#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="${DCACHE_GPUS:-2,3}"
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$repo_root/.tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export TMUX_TMPDIR="$repo_root/.tmux"
export MPLCONFIGDIR="$repo_root/.cache/runtime/sudoku-modular/matplotlib"
export TRITON_CACHE_DIR="$repo_root/.cache/runtime/sudoku-modular/triton"
export TORCHINDUCTOR_CACHE_DIR="$repo_root/.cache/runtime/sudoku-modular/inductor"
export CUDA_CACHE_PATH="$repo_root/.cache/runtime/sudoku-modular/cuda"
export HF_HOME="$repo_root/.cache/runtime/sudoku-modular/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$TMUX_TMPDIR" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME" logs
python_bin="${DCACHE_PYTHON:-/home/tliu0205/miniconda3/envs/dcache/bin/python}"
action="${1:-check}"
shift || true
if [ "$action" = launch ]; then
  session="${DCACHE_SUDOKU_SESSION:-sudoku-mdm-np-20ep}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Already running: $session" >&2
    exit 1
  fi
  printf -v command '%q ' bash "$repo_root/sudoku/run.sh" run "$@"
  tmux new-session -d -s "$session" "$command >> '$repo_root/logs/sudoku-mdm-np-20ep.log' 2>&1"
  echo "Started $session; log: logs/sudoku-mdm-np-20ep.log"
else
  exec "$python_bin" -m sudoku.schedule "$action" "$@"
fi
