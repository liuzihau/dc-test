#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Stage 1/2: continue DCache-v2 from global step 5000 to 6000."
bash "${SCRIPT_DIR}/continue_owt_5k_to_6k_4x3090.sh" dcache "$@"

echo "Stage 2/2: continue vanilla MDLM from global step 5000 to 6000."
bash "${SCRIPT_DIR}/continue_owt_5k_to_6k_4x3090.sh" baseline "$@"

echo "Both 5k-to-6k continuations completed successfully."
