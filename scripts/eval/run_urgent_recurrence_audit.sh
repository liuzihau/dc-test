#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# Keep temporary files, matplotlib, HF and kernel caches off the root disk.
export TMPDIR="${REPO_DIR}/.cache/runtime/recurrence-audit/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export MPLCONFIGDIR="${REPO_DIR}/.cache/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="${REPO_DIR}/.cache/torchinductor"
export TRITON_CACHE_DIR="${REPO_DIR}/.cache/triton"
export HF_HOME="${REPO_DIR}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${DCACHE_AUDIT_CPU_THREADS:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

PYTHON_BIN="${DCACHE_PYTHON:-/home/tliu0205/miniconda3/envs/dcache/bin/python}"
exec "$PYTHON_BIN" -u scripts/eval/run_recurrence_audit.py "$@"
