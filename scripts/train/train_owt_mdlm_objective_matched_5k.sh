#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${DCACHE_DATA_DIR:-${REPO_DIR}/.cache/huggingface}"
RUN_DIR="${DCACHE_RUN_DIR:-${REPO_DIR}/outputs/owt-mdlm-objective-matched-5k}"
DEVICES="${DCACHE_DEVICES:-2}"
MICRO_BATCH="${DCACHE_MICRO_BATCH:-2}"
GLOBAL_BATCH="${DCACHE_GLOBAL_BATCH:-512}"
NUM_WORKERS="${DCACHE_NUM_WORKERS:-8}"
MAX_STEPS="${DCACHE_MAX_STEPS:-5000}"
VAL_OPTIMIZER_INTERVAL="${DCACHE_VAL_INTERVAL:-500}"
VAL_BATCHES="${DCACHE_VAL_BATCHES:-100}"
SANITY_VAL_STEPS="${DCACHE_SANITY_VAL_STEPS:-0}"
CHECKPOINT_SAVE_TOP_K="${DCACHE_CHECKPOINT_SAVE_TOP_K:-3}"
PYTHON_BIN="${DCACHE_PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${DCACHE_CUDA_VISIBLE_DEVICES:-2,3}"

LOCAL_UPDATE_BATCH=$(( DEVICES * MICRO_BATCH ))
if (( GLOBAL_BATCH % LOCAL_UPDATE_BATCH != 0 )); then
  echo "Global batch ${GLOBAL_BATCH} must be divisible by devices × microbatch (${LOCAL_UPDATE_BATCH})." >&2
  exit 2
fi
ACCUMULATION=$(( GLOBAL_BATCH / LOCAL_UPDATE_BATCH ))
VAL_TRAIN_BATCH_INTERVAL=$(( VAL_OPTIMIZER_INTERVAL * ACCUMULATION ))

TRAIN_DATA="${DATA_DIR}/openwebtext-train_train_bs1024_wrapped_specialFalse.dat"
VALID_DATA="${DATA_DIR}/openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat"
if [[ ! -d "$TRAIN_DATA" ]]; then
  echo "Missing prepared OpenWebText training dataset: ${TRAIN_DATA}" >&2
  exit 1
fi
if [[ ! -d "$VALID_DATA" ]]; then
  echo "Missing prepared OpenWebText validation dataset: ${VALID_DATA}" >&2
  exit 1
fi

mkdir -p "$DATA_DIR" "$RUN_DIR"
cd "$REPO_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_DIR}/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

VISIBLE_DEVICES="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$VISIBLE_DEVICES" != "$DEVICES" ]]; then
  echo "Configured trainer.devices=${DEVICES}, but PyTorch sees ${VISIBLE_DEVICES} CUDA devices." >&2
  echo "Set DCACHE_CUDA_VISIBLE_DEVICES and DCACHE_DEVICES consistently." >&2
  exit 2
fi

LAST_CHECKPOINT="${RUN_DIR}/checkpoints/last.ckpt"
if [[ -f "$LAST_CHECKPOINT" ]]; then
  CHECKPOINT_PATH="$LAST_CHECKPOINT" "$PYTHON_BIN" -c '
import os
import torch
from omegaconf import OmegaConf

path = os.environ["CHECKPOINT_PATH"]
checkpoint = torch.load(
    path, map_location="cpu", weights_only=False, mmap=True)
config = checkpoint["hyper_parameters"]["config"]
enabled = bool(OmegaConf.select(
    config, "training.objective_matched_multistate.enabled", default=False))
step_memory = bool(OmegaConf.select(
    config, "step_memory.enabled", default=False))
if not enabled or step_memory:
  raise SystemExit(
      f"Refusing to resume non-B checkpoint {path}: "
      f"objective_matched={enabled}, step_memory={step_memory}")
global_step = checkpoint.get("global_step")
print(
    f"Verified resumable B checkpoint {path}: "
    f"global_step={global_step}")
'
fi

echo "Objective-matched vanilla control B"
echo "Run directory:       ${RUN_DIR}"
echo "Visible GPUs:        ${CUDA_VISIBLE_DEVICES:-all} (${VISIBLE_DEVICES} devices)"
echo "Optimizer steps:     ${MAX_STEPS}"
echo "Global batch:        ${GLOBAL_BATCH}"
echo "Microbatch/GPU:      ${MICRO_BATCH}"
echo "Dataloader workers/rank: ${NUM_WORKERS}"
echo "Gradient accumulation: ${ACCUMULATION}"
echo "Validation interval: ${VAL_OPTIMIZER_INTERVAL} optimizer steps"
echo "Loss: (0.05 full + 0.10 t0 + 0.20 t1 + 1.00 t2 + 0.70 t3) / 2.05"
echo "Cache: disabled in architecture, training, and validation"

if [[ "${DCACHE_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Objective-matched launcher preflight passed; training was not started."
  exit 0
fi

"$PYTHON_BIN" -u main.py \
  mode=train \
  seed=1 \
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
  loader.num_workers="$NUM_WORKERS" \
  checkpointing.save_dir="$RUN_DIR" \
  hydra.run.dir="$RUN_DIR/run" \
  training.resample=false \
  training.from_pretrained=null \
  training.objective_matched_multistate.enabled=true \
  step_memory.enabled=false \
  step_memory.use_previous_kv=false \
  step_memory.gate.enabled=false \
  step_memory.pretrain.enabled=false \
  step_memory.pretrain.step_size_min=0.025 \
  step_memory.pretrain.step_size_max=0.10 \
  step_memory.pretrain.max_t0_mask_ratio=0.9975 \
  step_memory.pretrain.full_loss_weight=0.05 \
  step_memory.pretrain.t0_loss_weight=0.10 \
  step_memory.pretrain.t1_loss_weight=0.20 \
  step_memory.pretrain.t2_loss_weight=1.00 \
  step_memory.pretrain.t3_loss_weight=0.70 \
  step_memory.pretrain.teacher_token_probability=1.0 \
  step_memory.pretrain.source_dropout.enabled=false \
  step_memory.pretrain.identity.enabled=false \
  step_memory.rollout.enabled=false \
  wandb=null \
  "$@"
