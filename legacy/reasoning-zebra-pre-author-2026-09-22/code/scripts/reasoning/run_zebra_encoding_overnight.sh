#!/usr/bin/env bash
# Do not run in parallel with another copy. Each queue holds a project-local lock.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ulimit -c 0
runtime_root="$PWD/.cache/runtime/reasoning-zebra-encoding"
export TMPDIR="$runtime_root/tmp" TMP="$runtime_root/tmp" TEMP="$runtime_root/tmp"
export MPLCONFIGDIR="$runtime_root/matplotlib" XDG_CACHE_HOME="$PWD/.cache"
export CUDA_CACHE_PATH="$runtime_root/cuda" TORCHINDUCTOR_CACHE_DIR="$runtime_root/inductor"
export TRITON_CACHE_DIR="$runtime_root/triton" OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
python_bin="${DCACHE_PYTHON:-python}"
case "${1:-plan}" in
  plan)
    printf '%s\n' 'GPU2: fixed answer position IDs; GPU3: fixed positions + typed public coordinates.' \
      'Wait for current 20013-step runs, final evaluations, and an empty GPU.' \
      'Then smoke/save/resume -> TRAIN-only memorization gate -> fresh 40026 updates (six epochs).' \
      'Same original Bernoulli corruption, LR3e-4, global128/micro32; no solver or new answer labels.'
    ;;
  gpu2)
    exec "$python_bin" -u -m reasoning.zebra_encoding_queue --gpu 2 --encoding answer_relative \
      --predecessor "$PWD/outputs/reasoning/zebra-tfw-hardstart-control-from5000-gpu2"
    ;;
  gpu3)
    exec "$python_bin" -u -m reasoning.zebra_encoding_queue --gpu 3 --encoding typed_coordinates \
      --predecessor "$PWD/outputs/reasoning/zebra-tfw-hardstart-fullmask50-from5000-gpu3"
    ;;
  *) echo "Usage: bash $0 {plan|gpu2|gpu3}" >&2; exit 2 ;;
esac
