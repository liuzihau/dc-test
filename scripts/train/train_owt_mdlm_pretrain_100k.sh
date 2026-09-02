#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DCACHE_DATA_DIR:-${REPO_DIR}/.cache/huggingface}"
RUN_DIR="${DCACHE_RUN_DIR:-${REPO_DIR}/outputs/owt-mdlm-pretrain-100k}"
DEVICES="${DCACHE_DEVICES:-4}"
MICRO_BATCH="${DCACHE_MICRO_BATCH:-4}"
GLOBAL_BATCH="${DCACHE_GLOBAL_BATCH:-512}"
MAX_STEPS="${DCACHE_MAX_STEPS:-100000}"
VAL_OPTIMIZER_INTERVAL="${DCACHE_VAL_INTERVAL:-10000}"
VAL_BATCHES="${DCACHE_VAL_BATCHES:-1.0}"
SANITY_VAL_STEPS="${DCACHE_SANITY_VAL_STEPS:-2}"
CHECKPOINT_SAVE_TOP_K="${DCACHE_CHECKPOINT_SAVE_TOP_K:-3}"
PYTHON_BIN="${DCACHE_PYTHON:-python}"

LOCAL_UPDATE_BATCH=$(( DEVICES * MICRO_BATCH ))
if (( GLOBAL_BATCH % LOCAL_UPDATE_BATCH != 0 )); then
  echo "Global batch ${GLOBAL_BATCH} must be divisible by devices × microbatch (${LOCAL_UPDATE_BATCH})." >&2
  exit 2
fi
PER_UPDATE_BATCHES=$(( GLOBAL_BATCH / LOCAL_UPDATE_BATCH ))
VAL_TRAIN_BATCH_INTERVAL=$(( VAL_OPTIMIZER_INTERVAL * PER_UPDATE_BATCHES ))

mkdir -p "$DATA_DIR" "$RUN_DIR"
cd "$REPO_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_DIR}/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
VISIBLE_DEVICES="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$VISIBLE_DEVICES" != "$DEVICES" ]]; then
  echo "Configured trainer.devices=${DEVICES}, but PyTorch sees ${VISIBLE_DEVICES} CUDA devices." >&2
  echo "Set CUDA_VISIBLE_DEVICES to exactly ${DEVICES} GPUs before launching." >&2
  exit 2
fi
echo "PyTorch sees ${VISIBLE_DEVICES} CUDA devices (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all})."
echo "Validation every ${VAL_OPTIMIZER_INTERVAL} optimizer steps (${VAL_TRAIN_BATCH_INTERVAL} training batches with accumulation ${PER_UPDATE_BATCHES})."
if [[ "${DCACHE_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Vanilla launcher preflight passed; exiting before dataset/model startup."
  exit 0
fi

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
  step_memory.enabled=false \
  step_memory.pretrain.enabled=false \
  step_memory.rollout.enabled=false \
  wandb=null \
  "$@"
