#!/usr/bin/env bash
# Single-GPU continuation; source checkpoints require explicit audited migration.
set -euo pipefail
export DCACHE_CUDA_VISIBLE_DEVICES=3
export DCACHE_DEVICES=1
export DCACHE_MICRO_BATCH=32
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_zebra_mdm_bernoulli_2x3090.sh" "${1:-plan}"
