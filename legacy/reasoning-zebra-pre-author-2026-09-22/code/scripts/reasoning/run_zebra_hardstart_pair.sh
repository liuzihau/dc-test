#!/usr/bin/env bash
# Paired continuation of the repaired step-5000 baseline. Never changes old runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
action="${1:-plan}"
runtime_root="$PWD/.cache/runtime/reasoning-tfw-hardstart"
export TMPDIR="$runtime_root/tmp" TMP="$runtime_root/tmp" TEMP="$runtime_root/tmp"
export MPLCONFIGDIR="$runtime_root/matplotlib" XDG_CACHE_HOME="$PWD/.cache"
export CUDA_CACHE_PATH="$runtime_root/cuda" TORCHINDUCTOR_CACHE_DIR="$runtime_root/inductor"
export TRITON_CACHE_DIR="$runtime_root/triton" OMP_NUM_THREADS=4
ulimit -c 0
python_bin="${DCACHE_PYTHON:-python}"
data="$PWD/.cache/reasoning/zebra-benchmark-full-v1"
parent="$PWD/outputs/reasoning/zebra-tfw-answer-only-no-shift-lr3e4-gpu3/checkpoints/last.pt"
base=("$python_bin" -u -m reasoning.tfw_runner --data-dir "$data"
  --global-batch 128 --micro-batch 32 --lr 0.0003 --schedule-epochs 300
  --logit-shift 0 --target-region answer --padding-attention masked --precision bf16
  --val-every 500 --save-every 500 --save-seconds 1200 --log-every 10
  --validation-examples 1000 --eval-batch-size 32 --generation-examples 1000)
report() {
  mkdir -p "$runtime_root" "$MPLCONFIGDIR"
  flock "$runtime_root/report.lock" "$python_bin" scripts/reasoning/report_zebra_hardstart.py
}
launch() {
  local arm="$1" gpu="$2" probability="$3" run
  export CUDA_VISIBLE_DEVICES="$gpu"
  mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
  run="$PWD/outputs/reasoning/zebra-tfw-hardstart-${arm}-from5000-gpu${gpu}"
  "$python_bin" - "$parent" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]).resolve(strict=True)
if json.loads(p.with_suffix('.pt.json').read_text())['step'] != 5000:
    raise SystemExit('Parent must be the frozen step-5000 checkpoint; refusing a moving source.')
PY
  "${base[@]}" --run-dir "$run" --fork-from "$parent" \
    --full-mask-probability "$probability" --stop-after-steps 20013 \
    --generation-every 1000 --final-generation
  "$python_bin" -m reasoning.tfw_clue_audit --checkpoint "$run/checkpoints/last.pt" \
    --data-dir "$data" --output "$run/clue_audit.json"
  report
}
case "$action" in
  plan)
    printf '%s\n' 'GPU 2: control, original uniform-timestep Bernoulli corruption.' \
      'GPU 3: mixture, 50% forced full-answer mask + 50% original corruption.' \
      'Both: same step-5000 model/optimizer/RNG/data cursor -> step 20013 (three total data epochs).' \
      'No new model parameters, solver, data leak, shift, memory, NP or decoding change.'
    ;;
  smoke)
    mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
    for gpu in 2 3; do
      probability=0; [[ "$gpu" == 3 ]] && probability=0.5
      export CUDA_VISIBLE_DEVICES="$gpu"
      smoke_run=$(mktemp -d "$runtime_root/smoke-gpu${gpu}.XXXXXXXX")
      for stop in 5001 5002; do
        "${base[@]}" --run-dir "$smoke_run" --fork-from "$parent" \
          --full-mask-probability "$probability" --stop-after-steps "$stop" \
          --validation-examples 4 --generation-every 0 --log-every 1
      done
    done
    ;;
  control) launch control 2 0 ;;
  hardstart) launch fullmask50 3 0.5 ;;
  report) report ;;
  *) echo "Usage: bash $0 {plan|smoke|control|hardstart|report}" >&2; exit 2 ;;
esac
