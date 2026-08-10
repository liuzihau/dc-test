#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export DCACHE_DEVICES="${DCACHE_DEVICES:-2}"
export DCACHE_MAX_STEPS="${DCACHE_MAX_STEPS:-5000}"
export DCACHE_MICRO_BATCH="${DCACHE_MICRO_BATCH:-4}"
export DCACHE_GLOBAL_BATCH="${DCACHE_GLOBAL_BATCH:-512}"
export DCACHE_VAL_INTERVAL="${DCACHE_VAL_INTERVAL:-500}"
export DCACHE_VAL_BATCHES="${DCACHE_VAL_BATCHES:-100}"
export DCACHE_SANITY_VAL_STEPS="${DCACHE_SANITY_VAL_STEPS:-0}"
export DCACHE_RUN_DIR="${DCACHE_RUN_DIR:-${REPO_DIR}/outputs/owt-mdlm-pretrain-5k-2x4090}"

exec bash "${SCRIPT_DIR}/train_owt_mdlm_pretrain_100k.sh" "$@"
