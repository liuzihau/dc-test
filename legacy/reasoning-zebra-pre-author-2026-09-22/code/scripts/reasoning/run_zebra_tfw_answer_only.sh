#!/usr/bin/env bash
# Separate, bounded padding-repair trial. Historical controls are untouched.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
action="${1:-plan}"
export CUDA_VISIBLE_DEVICES="${DCACHE_TFW_GPU:-3}"
runtime_root="$PWD/.cache/runtime/reasoning-tfw-answer-only"
export TMPDIR="$runtime_root/tmp" TMP="$runtime_root/tmp" TEMP="$runtime_root/tmp"
export MPLCONFIGDIR="$runtime_root/matplotlib" XDG_CACHE_HOME="$PWD/.cache"
export CUDA_CACHE_PATH="$runtime_root/cuda" TORCHINDUCTOR_CACHE_DIR="$runtime_root/inductor"
export TRITON_CACHE_DIR="$runtime_root/triton" OMP_NUM_THREADS=4
ulimit -c 0
python_bin="${DCACHE_PYTHON:-python}"
run="${DCACHE_TFW_ANSWER_RUN_DIR:-$PWD/outputs/reasoning/zebra-tfw-answer-only-no-shift-lr3e4-gpu3}"
sanity_run="$PWD/outputs/reasoning/zebra-tfw-answer-only-no-shift-lr3e4-overfit32-gpu3-v2"
base=("$python_bin" -u -m reasoning.tfw_runner
  --data-dir "$PWD/.cache/reasoning/zebra-benchmark-full-v1"
  --global-batch 128 --micro-batch 32 --lr 0.0003 --schedule-epochs 300
  --logit-shift 0 --target-region answer --padding-attention masked --precision bf16
  --val-every 500 --save-every 500 --save-seconds 1200 --log-every 10
  --validation-examples 1000 --eval-batch-size 32 --generation-examples 1000)
case "$action" in
  plan) printf '%q ' "${base[@]}" --run-dir "$run" --stop-after-steps 5000 --generation-every 1000 --final-generation; printf '\n' ;;
  smoke)
    mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
    smoke_run=$(mktemp -d "$runtime_root/smoke.XXXXXXXX")
    "${base[@]}" --run-dir "$smoke_run" --stop-after-steps 1 --validation-examples 4 --log-every 1
    "${base[@]}" --run-dir "$smoke_run" --stop-after-steps 2 --validation-examples 4 --log-every 1
    ;;
  overfit)
    mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
    "${base[@]}" --run-dir "$sanity_run" --overfit-examples 32 --stop-after-steps 3000 \
      --sanity-every 250 --validation-examples 128 --generation-every 0
    ;;
  run)
    mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
    # Do not spend the full probe budget before a real memorization check.
    "$python_bin" - "$sanity_run" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
c = json.loads((p / 'contract.json').read_text())
expected = dict(target_region='answer', padding_attention='masked', logit_shift=0,
                lr=.0003, global_batch=128, micro_batch=32, seed=1,
                overfit_cursor_version=2, lr_schedule_full_data_epochs=300)
if any(c.get(k) != v for k, v in expected.items()) or len(c['overfit_train_indices']) != 32:
    raise SystemExit('Memorization diagnostic configuration differs; refusing to start.')
manifest = Path('.cache/reasoning/zebra-benchmark-full-v1/manifest.json')
if c['data_sha256'] != hashlib.sha256(manifest.read_bytes()).hexdigest():
    raise SystemExit('Memorization diagnostic uses a different dataset.')
for step in (2500, 2750, 3000):
    r = json.loads((p / f'sanity/step-{step:09d}.json').read_text())
    if r['greedy_rollout_exact'] < .95 or r['greedy_rollout_pad_fraction'] != 0:
        raise SystemExit(f'Memorization check failed at {step}; inspect before longer training.')
print('Train-only memorization gate passed; this is NOT evidence of held-out reasoning performance.')
PY
    "${base[@]}" --run-dir "$run" --stop-after-steps 5000 --generation-every 1000 --final-generation
    ;;
  *) echo "Usage: bash $0 {plan|smoke|overfit|run}" >&2; exit 2 ;;
esac
