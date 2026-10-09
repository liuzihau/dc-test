#!/usr/bin/env bash
# Bounded validation-only diagnostics; no optimizer or training queue.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ulimit -c 0
case "${1:-}" in
  gpu2) physical_gpu=2; encoding=answer_relative ;;
  gpu3) physical_gpu=3; encoding=typed_coordinates ;;
  *) echo "Usage: bash $0 {gpu2|gpu3}" >&2; exit 2 ;;
esac
runtime_root="$PWD/.cache/runtime/reasoning-zebra-trace"
export TMPDIR="$runtime_root/tmp" TMP="$runtime_root/tmp" TEMP="$runtime_root/tmp"
export MPLCONFIGDIR="$runtime_root/matplotlib" XDG_CACHE_HOME="$PWD/.cache"
export CUDA_CACHE_PATH="$runtime_root/cuda" TORCHINDUCTOR_CACHE_DIR="$runtime_root/inductor"
export TRITON_CACHE_DIR="$runtime_root/triton" OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
exec 9>"$runtime_root/gpu${physical_gpu}.lock"
flock -n 9 || { echo "Audit worker already running on GPU $physical_gpu" >&2; exit 1; }
python_bin="${DCACHE_PYTHON:-/home/tliu0205/miniconda3/envs/dcache/bin/python}"
"$python_bin" - "$physical_gpu" <<'PY'
import subprocess
import sys
gpu = sys.argv[1]
uuid = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
if any(row.split(',')[0].strip() == uuid for row in apps.splitlines()):
    raise SystemExit('GPU is occupied; refusing to compete with an existing process')
PY
export CUDA_VISIBLE_DEVICES="$physical_gpu"
exec "$python_bin" -u -m reasoning.zebra_trace_audit \
  --checkpoint "$PWD/outputs/reasoning/zebra-encoding-${encoding}-gpu${physical_gpu}/checkpoints/last.pt" \
  --data-dir "$PWD/.cache/reasoning/zebra-benchmark-full-v1" \
  --output-dir "$PWD/outputs/reasoning/zebra-trace-audit-${encoding}" \
  --examples 1000 --batch-size 32 --seed 2026 --device cuda
