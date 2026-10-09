#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting the vanilla MDLM/BD3 pretraining baseline."
bash "${SCRIPT_DIR}/train_owt_mdlm_pretrain_100k.sh" "$@"

echo "Starting five-forward DCache-v2 pretraining."
bash "${SCRIPT_DIR}/train_owt_dcache_pretrain_100k.sh" "$@"
