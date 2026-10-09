#!/usr/bin/env bash
# No background training beyond the bounded worker below; no shared /tmp caches.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ulimit -c 0
runtime_root="$PWD/.cache/runtime/reasoning-zebra-binding"
export TMPDIR="$runtime_root/tmp" TMP="$runtime_root/tmp" TEMP="$runtime_root/tmp"
export MPLCONFIGDIR="$runtime_root/matplotlib" XDG_CACHE_HOME="$PWD/.cache"
export CUDA_CACHE_PATH="$runtime_root/cuda" TORCHINDUCTOR_CACHE_DIR="$runtime_root/inductor"
export TRITON_CACHE_DIR="$runtime_root/triton" OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
python_bin="${DCACHE_PYTHON:-/home/tliu0205/miniconda3/envs/dcache/bin/python}"
case "${1:-}" in
  prepare)
    exec 9>"$runtime_root/prepare.lock"
    flock -n 9 || exit 1
    exec "$python_bin" -u -m reasoning.zebra_binding prepare \
      --data-dir "$PWD/.cache/reasoning/zebra-clue-binding-v1"
    ;;
  gpu2) physical_gpu=2 ;;
  gpu3) physical_gpu=3 ;;
  *) echo "Usage: bash $0 {prepare|gpu2|gpu3}" >&2; exit 2 ;;
esac
exec 9>"$runtime_root/gpu${physical_gpu}.lock"
flock -n 9 || { echo "Worker already running" >&2; exit 1; }
"$python_bin" - "$physical_gpu" <<'PY'
import subprocess
import sys
uuid = subprocess.check_output(['nvidia-smi', '-i', sys.argv[1], '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
if any(row.split(',')[0].strip() == uuid for row in apps.splitlines()):
    raise SystemExit('GPU occupied; refusing to compete with another process')
PY
export CUDA_VISIBLE_DEVICES="$physical_gpu"
exec "$python_bin" -u -m reasoning.zebra_binding_queue --gpu "$physical_gpu"
