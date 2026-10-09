#!/usr/bin/env bash
# Same GPT-2 recipe as GPU 2; only output alignment differs. Fresh run directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export DCACHE_TFW_GPU=3 DCACHE_TFW_LOGIT_SHIFT=0
export DCACHE_TFW_RUNTIME_ROOT="$PWD/.cache/runtime/reasoning-tfw-no-shift"
export DCACHE_TFW_RUN_DIR="${DCACHE_TFW_NO_SHIFT_RUN_DIR:-$PWD/outputs/reasoning/zebra-tfw-gpt2-no-shift-19m-gpu3}"
exec bash scripts/reasoning/run_zebra_tfw_gpu2.sh "${1:-plan}"
