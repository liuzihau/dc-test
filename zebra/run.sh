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
export MPLCONFIGDIR="$repo_root/.cache/runtime/zebra-modular/matplotlib"
export TRITON_CACHE_DIR="$repo_root/.cache/runtime/zebra-modular/triton"
export TORCHINDUCTOR_CACHE_DIR="$repo_root/.cache/runtime/zebra-modular/inductor"
export CUDA_CACHE_PATH="$repo_root/.cache/runtime/zebra-modular/cuda"
export HF_HOME="$repo_root/.cache/runtime/zebra-modular/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$TMUX_TMPDIR" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME" logs
python_bin="${DCACHE_PYTHON:-/home/tliu0205/miniconda3/envs/dcache/bin/python}"
action="${1:-check}"
shift || true
if [ "$action" = launch ]; then
  session="${DCACHE_ZEBRA_SESSION:-zebra-mdm-np-40ep}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Already running: $session" >&2
    exit 1
  fi
  printf -v command '%q ' bash "$repo_root/zebra/run.sh" run "$@"
  tmux new-session -d -s "$session" "$command >> '$repo_root/logs/zebra-mdm-np-40ep.log' 2>&1"
  echo "Started $session; log: logs/zebra-mdm-np-40ep.log"
else
  exec "$python_bin" -m zebra.schedule "$action" "$@"
fi
