#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Physical GPUs 2 and 3 become logical CUDA devices 0 and 1 inside Lightning.
export CUDA_VISIBLE_DEVICES="${DCACHE_CUDA_VISIBLE_DEVICES:-2,3}"
export DCACHE_DEVICES="${DCACHE_DEVICES:-2}"
export DCACHE_MAX_STEPS="${DCACHE_MAX_STEPS:-5000}"
export DCACHE_MICRO_BATCH="${DCACHE_MICRO_BATCH:-2}"
export DCACHE_GLOBAL_BATCH="${DCACHE_GLOBAL_BATCH:-512}"
export DCACHE_VAL_INTERVAL="${DCACHE_VAL_INTERVAL:-500}"
export DCACHE_VAL_BATCHES="${DCACHE_VAL_BATCHES:-100}"
export DCACHE_SANITY_VAL_STEPS="${DCACHE_SANITY_VAL_STEPS:-0}"
export DCACHE_CHECKPOINT_SAVE_TOP_K="${DCACHE_CHECKPOINT_SAVE_TOP_K:--1}"
export DCACHE_DATA_DIR="${DCACHE_DATA_DIR:-${REPO_DIR}/.cache/huggingface}"
export DCACHE_RUN_DIR="${DCACHE_RUN_DIR:-${REPO_DIR}/outputs/owt-dcachehooping-pretrain-5k-2x3090-exclusive}"

exec bash "${SCRIPT_DIR}/train_owt_dcache_pretrain_100k.sh" \
  dcachehooping.enabled=true \
  dcachehooping.exclusive_auxiliary_routes=true \
  dcachehooping.latent_dropout_probability=0.10 \
  dcachehooping.latent_mask_probability=0.10 \
  dcachehooping.latent_mask_loss_weight=0.10 \
  dcachehooping.tentative.enabled=true \
  dcachehooping.tentative.batch_probability=0.25 \
  dcachehooping.tentative.loss_weight=0.10 \
  dcachehooping.confidence.enabled=true \
  dcachehooping.confidence.loss_weight=0.30 \
  dcachehooping.identity_final_probability=0.50 \
  "$@"
