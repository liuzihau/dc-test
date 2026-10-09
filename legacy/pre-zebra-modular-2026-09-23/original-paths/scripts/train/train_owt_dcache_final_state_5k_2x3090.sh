#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Minimal dual-memory experiment: DCache plus the detached final state from
# the previous denoising step. Tentative correction, status embeddings,
# confidence supervision, and the latent-mask auxiliary pass are disabled.
export CUDA_VISIBLE_DEVICES="${DCACHE_CUDA_VISIBLE_DEVICES:-2,3}"
export DCACHE_DEVICES="${DCACHE_DEVICES:-2}"
export DCACHE_MAX_STEPS="${DCACHE_MAX_STEPS:-5000}"
export DCACHE_MICRO_BATCH="${DCACHE_MICRO_BATCH:-2}"
export DCACHE_GLOBAL_BATCH="${DCACHE_GLOBAL_BATCH:-512}"
export DCACHE_VAL_INTERVAL="${DCACHE_VAL_INTERVAL:-500}"
export DCACHE_VAL_BATCHES="${DCACHE_VAL_BATCHES:-100}"
export DCACHE_SANITY_VAL_STEPS="${DCACHE_SANITY_VAL_STEPS:-0}"
export DCACHE_CHECKPOINT_SAVE_TOP_K="${DCACHE_CHECKPOINT_SAVE_TOP_K:-3}"
export DCACHE_DATA_DIR="${DCACHE_DATA_DIR:-${REPO_DIR}/.cache/huggingface}"
export DCACHE_RUN_DIR="${DCACHE_RUN_DIR:-${REPO_DIR}/outputs/owt-dcache-final-state-pretrain-5k-2x3090}"

exec bash "${SCRIPT_DIR}/train_owt_dcache_pretrain_100k.sh" \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=500 \
  dcachehooping.enabled=true \
  dcachehooping.status_embedding.enabled=false \
  dcachehooping.latent_dropout_probability=0.10 \
  dcachehooping.latent_mask_probability=0.0 \
  dcachehooping.latent_mask_loss_weight=0.0 \
  dcachehooping.tentative.enabled=false \
  dcachehooping.tentative.batch_probability=0.0 \
  dcachehooping.tentative.loss_weight=0.0 \
  dcachehooping.confidence.enabled=false \
  dcachehooping.confidence.loss_weight=0.0 \
  dcachehooping.identity_final_probability=0.50 \
  "$@"
