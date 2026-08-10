#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${DCACHE_RUN_DIR:-}" ]]; then
  echo "DCACHE_RUN_DIR must be unset for the sequential comparison so the two runs use separate output directories." >&2
  exit 2
fi

echo "Starting the 5k-step vanilla MDLM/BD3 baseline on two GPUs."
bash "${SCRIPT_DIR}/train_owt_mdlm_pretrain_5k_2x4090.sh" "$@"

echo "Starting the 5k-step shifted-DCache three-pass run on two GPUs."
bash "${SCRIPT_DIR}/train_owt_dcache_pretrain_5k_2x4090.sh" "$@"
