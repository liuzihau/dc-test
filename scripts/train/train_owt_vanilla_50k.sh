#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DCACHE_DATA_DIR:-${REPO_DIR}/.cache/huggingface}"
RUN_DIR="${DCACHE_RUN_DIR:-${REPO_DIR}/outputs/owt-vanilla-50k}"
DEVICES="${DCACHE_DEVICES:-4}"
MICRO_BATCH="${DCACHE_MICRO_BATCH:-2}"
GLOBAL_BATCH="${DCACHE_GLOBAL_BATCH:-512}"

mkdir -p "$DATA_DIR" "$RUN_DIR"
cd "$REPO_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python -u main.py \
  mode=train \
  model=small \
  algo=bd3lm \
  data=openwebtext-split \
  data.cache_dir="$DATA_DIR" \
  data.insert_train_special=false \
  data.insert_valid_special=false \
  data.insert_valid_eos=false \
  model.length=1024 \
  model.attn_backend=flex \
  block_size=16 \
  trainer.devices="$DEVICES" \
  trainer.precision=bf16-mixed \
  trainer.max_steps=50000 \
  loader.global_batch_size="$GLOBAL_BATCH" \
  loader.eval_global_batch_size="$GLOBAL_BATCH" \
  loader.batch_size="$MICRO_BATCH" \
  loader.eval_batch_size="$MICRO_BATCH" \
  checkpointing.save_dir="$RUN_DIR" \
  training.resample=true \
  training.from_pretrained=null \
  step_memory.enabled=false \
  step_memory.rollout.enabled=false \
  wandb=null \
  "$@"
