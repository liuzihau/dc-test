#!/usr/bin/env bash
# Isolated paper-informed control. Never stops or joins other experiment queues.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
action="${1:-plan}"
export CUDA_VISIBLE_DEVICES="${DCACHE_TFW_GPU:-2}"
runtime_root="${DCACHE_TFW_RUNTIME_ROOT:-$PWD/.cache/runtime/reasoning-tfw}"
export TMPDIR="$runtime_root/tmp" TMP="$runtime_root/tmp"
export TEMP="$TMPDIR" MPLCONFIGDIR="$runtime_root/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache" CUDA_CACHE_PATH="$PWD/.cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$runtime_root/inductor"
export TRITON_CACHE_DIR="$runtime_root/triton"
export OMP_NUM_THREADS=4
ulimit -c 0
python_bin="${DCACHE_PYTHON:-python}"
run="${DCACHE_TFW_RUN_DIR:-$PWD/outputs/reasoning/zebra-tfw-gpt2-defaults-19m-gpu2}"
probe_epochs="${DCACHE_TFW_PROBE_EPOCHS:-3}"
base=("$python_bin" -u -m reasoning.tfw_runner
  --data-dir "$PWD/.cache/reasoning/zebra-benchmark-full-v1"
  --global-batch 128 --micro-batch 32 --lr 0.001 --schedule-epochs 300
  --logit-shift "${DCACHE_TFW_LOGIT_SHIFT:-1}"
  --probe-epochs "$probe_epochs" --precision bf16
  --val-every 500 --save-every 500 --save-seconds 1200
  --validation-examples 1000 --eval-batch-size 32 --generation-examples 1000)
case "$action" in
  plan) printf '%q ' "${base[@]}" --run-dir "$run"; printf '\n' ;;
  smoke)
    mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
    smoke_run=$(mktemp -d "$runtime_root/smoke.XXXXXXXX")
    "${base[@]}" --run-dir "$smoke_run" --stop-after-steps 1 --validation-examples 4 --log-every 1
    # Also check full-state restart, not only a fresh optimizer step.
    "${base[@]}" --run-dir "$smoke_run" --stop-after-steps 2 --validation-examples 4 --log-every 1
    ;;
  run)
    mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
    "${base[@]}" --run-dir "$run"
    ;;
  *) echo "Usage: bash $0 {plan|smoke|run}" >&2; exit 2 ;;
esac
