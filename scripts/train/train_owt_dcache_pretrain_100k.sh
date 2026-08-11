#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DCACHE_DATA_DIR:-${REPO_DIR}/.cache/huggingface}"
RUN_DIR="${DCACHE_RUN_DIR:-${REPO_DIR}/outputs/owt-dcache-pretrain-100k}"
DEVICES="${DCACHE_DEVICES:-4}"
MICRO_BATCH="${DCACHE_MICRO_BATCH:-4}"
GLOBAL_BATCH="${DCACHE_GLOBAL_BATCH:-512}"
MAX_STEPS="${DCACHE_MAX_STEPS:-100000}"
VAL_OPTIMIZER_INTERVAL="${DCACHE_VAL_INTERVAL:-10000}"
VAL_BATCHES="${DCACHE_VAL_BATCHES:-1.0}"
SANITY_VAL_STEPS="${DCACHE_SANITY_VAL_STEPS:-2}"
CHECKPOINT_SAVE_TOP_K="${DCACHE_CHECKPOINT_SAVE_TOP_K:--1}"
PYTHON_BIN="${DCACHE_PYTHON:-python}"

PER_UPDATE_BATCHES=$(( (GLOBAL_BATCH + DEVICES * MICRO_BATCH - 1) / (DEVICES * MICRO_BATCH) ))
VAL_TRAIN_BATCH_INTERVAL=$(( VAL_OPTIMIZER_INTERVAL * PER_UPDATE_BATCHES ))

mkdir -p "$DATA_DIR" "$RUN_DIR"
cd "$REPO_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
echo "Validation every ${VAL_OPTIMIZER_INTERVAL} optimizer steps (${VAL_TRAIN_BATCH_INTERVAL} training batches with accumulation ${PER_UPDATE_BATCHES})."

"$PYTHON_BIN" -u main.py \
  mode=train \
  model=small \
  algo=mdlm \
  data=openwebtext-split \
  data.cache_dir="$DATA_DIR" \
  data.insert_train_special=false \
  data.insert_valid_special=false \
  data.insert_valid_eos=false \
  model.length=1024 \
  model.attn_backend=sdpa \
  block_size=1024 \
  trainer.devices="$DEVICES" \
  trainer.precision=bf16-mixed \
  trainer.max_steps="$MAX_STEPS" \
  trainer.log_every_n_steps=10 \
  trainer.val_check_interval="$VAL_TRAIN_BATCH_INTERVAL" \
  trainer.limit_val_batches="$VAL_BATCHES" \
  trainer.num_sanity_val_steps="$SANITY_VAL_STEPS" \
  callbacks.checkpoint_every_n_steps.save_top_k="$CHECKPOINT_SAVE_TOP_K" \
  loader.global_batch_size="$GLOBAL_BATCH" \
  loader.eval_global_batch_size="$GLOBAL_BATCH" \
  loader.batch_size="$MICRO_BATCH" \
  loader.eval_batch_size="$MICRO_BATCH" \
  checkpointing.save_dir="$RUN_DIR" \
  hydra.run.dir="$RUN_DIR/run" \
  training.resample=false \
  training.from_pretrained=null \
  step_memory.enabled=true \
  step_memory.use_previous_kv=true \
  step_memory.detach_between_steps=true \
  step_memory.pretrain.enabled=true \
  step_memory.pretrain.full_loss_weight=0.1 \
  step_memory.pretrain.s_loss_weight=1.0 \
  step_memory.pretrain.t_loss_weight=1.0 \
  step_memory.pretrain.min_mask_ratio=0.001 \
  step_memory.pretrain.teacher_token_probability=1.0 \
  step_memory.rollout.enabled=false \
  wandb=null \
  "$@"
