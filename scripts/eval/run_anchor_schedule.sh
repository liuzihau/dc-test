#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
export TMPDIR="${REPO_DIR}/.cache/runtime/anchor-schedule/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export MPLCONFIGDIR="${REPO_DIR}/.cache/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="${REPO_DIR}/.cache/torchinductor"
export TRITON_CACHE_DIR="${REPO_DIR}/.cache/triton"
export HF_HOME="${REPO_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export XDG_CACHE_HOME="${REPO_DIR}/.cache"
export CUDA_CACHE_PATH="${REPO_DIR}/.cache/cuda"
export TORCH_HOME="${REPO_DIR}/.cache/torch"
export NUMBA_CACHE_DIR="${REPO_DIR}/.cache/numba"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${DCACHE_AUDIT_CPU_THREADS:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$TORCH_HOME" "$NUMBA_CACHE_DIR"
# Avoid a multi-gigabyte core dump if a native library aborts.
ulimit -c 0
PYTHON_BIN="${DCACHE_PYTHON:-/home/tliu0205/miniconda3/envs/dcache/bin/python}"
exec "$PYTHON_BIN" -u scripts/eval/run_anchor_schedule.py "$@"
