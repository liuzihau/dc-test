#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "dcache" && "$MODE" != "baseline" ]]; then
  echo "usage: $0 {dcache|baseline} [Hydra overrides ...]" >&2
  exit 2
fi
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${DCACHE_PYTHON:-python}"
START_GLOBAL_STEP=5000
FINAL_GLOBAL_STEP="${DCACHE_MAX_STEPS:-6000}"

if [[ "$FINAL_GLOBAL_STEP" != "6000" ]]; then
  echo "This launcher records the 5k -> 5.5k -> 6k experiment and requires DCACHE_MAX_STEPS=6000." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${DCACHE_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export DCACHE_DEVICES=4
export DCACHE_GLOBAL_BATCH=512
export DCACHE_MAX_STEPS="$FINAL_GLOBAL_STEP"
export DCACHE_VAL_INTERVAL=500
export DCACHE_VAL_BATCHES="${DCACHE_VAL_BATCHES:-100}"
export DCACHE_SANITY_VAL_STEPS=0
# -1 retains every checkpoint emitted by the 500-step callback. Starting from
# global step 5000, the new numbered files are 0-5500.ckpt and 0-6000.ckpt.
export DCACHE_CHECKPOINT_SAVE_TOP_K=-1
export DCACHE_DATA_DIR="${DCACHE_DATA_DIR:-${REPO_DIR}/.cache/huggingface}"

if [[ "$MODE" == "dcache" ]]; then
  export DCACHE_MICRO_BATCH=2
  export DCACHE_RUN_DIR="${DCACHE_RUN_DIR:-${REPO_DIR}/outputs/owt-dcache-v2-pretrain-5k-2x3090}"
  BASE_LAUNCHER="${SCRIPT_DIR}/train_owt_dcache_pretrain_100k.sh"
else
  export DCACHE_MICRO_BATCH=4
  export DCACHE_RUN_DIR="${DCACHE_RUN_DIR:-${REPO_DIR}/outputs/owt-mdlm-pretrain-5k-2x3090}"
  BASE_LAUNCHER="${SCRIPT_DIR}/train_owt_mdlm_pretrain_100k.sh"
fi

CHECKPOINT_DIR="${DCACHE_RUN_DIR}/checkpoints"
LAST_CHECKPOINT="${CHECKPOINT_DIR}/last.ckpt"
START_CHECKPOINT="${CHECKPOINT_DIR}/0-5000.ckpt"
RESUME_CHECKPOINT="$START_CHECKPOINT"

verify_checkpoint_step() {
  local checkpoint_path="$1"
  local expected_step="$2"
  CHECKPOINT_PATH="$checkpoint_path" EXPECTED_STEP="$expected_step" \
    "$PYTHON_BIN" -c '
import os
import pathlib
import torch

path = pathlib.Path(os.environ["CHECKPOINT_PATH"])
expected = int(os.environ["EXPECTED_STEP"])
checkpoint = torch.load(
    path, map_location="cpu", weights_only=False, mmap=True)
actual = int(checkpoint.get("global_step", -1))
if actual != expected:
    raise SystemExit(
        f"Checkpoint {path} has global_step={actual}, expected {expected}")
print(f"Verified {path}: global_step={actual}")
'
}

if [[ ! -d "$DCACHE_DATA_DIR/openwebtext-train_train_bs1024_wrapped_specialFalse.dat" ]]; then
  echo "Missing prepared OpenWebText training data under ${DCACHE_DATA_DIR}." >&2
  exit 1
fi
if [[ ! -d "$DCACHE_DATA_DIR/openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat" ]]; then
  echo "Missing prepared OpenWebText validation data under ${DCACHE_DATA_DIR}." >&2
  exit 1
fi

if [[ -f "$START_CHECKPOINT" ]]; then
  verify_checkpoint_step "$START_CHECKPOINT" "$START_GLOBAL_STEP"
  if [[ -f "$LAST_CHECKPOINT" ]]; then
    # This is a restart of an interrupted continuation. Prefer last.ckpt so
    # already completed post-5k updates are not repeated.
    LAST_GLOBAL_STEP="$(CHECKPOINT_PATH="$LAST_CHECKPOINT" "$PYTHON_BIN" -c '
import os
import torch
checkpoint = torch.load(
    os.environ["CHECKPOINT_PATH"], map_location="cpu",
    weights_only=False, mmap=True)
print(int(checkpoint.get("global_step", -1)))
')"
    if (( LAST_GLOBAL_STEP < START_GLOBAL_STEP || LAST_GLOBAL_STEP > FINAL_GLOBAL_STEP )); then
      echo "Checkpoint ${LAST_CHECKPOINT} has unexpected global_step=${LAST_GLOBAL_STEP}." >&2
      exit 1
    fi
    RESUME_CHECKPOINT="$LAST_CHECKPOINT"
    echo "Found resumable last.ckpt at global step ${LAST_GLOBAL_STEP}."
  fi
elif [[ -f "$LAST_CHECKPOINT" ]]; then
  verify_checkpoint_step "$LAST_CHECKPOINT" "$START_GLOBAL_STEP"
else
  echo "Missing step-5000 checkpoint: ${LAST_CHECKPOINT}" >&2
  exit 1
fi

VISIBLE_DEVICES="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$VISIBLE_DEVICES" != "4" ]]; then
  echo "Expected four visible CUDA devices, but PyTorch sees ${VISIBLE_DEVICES}." >&2
  echo "Current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
  exit 2
fi

echo "Continuation mode: ${MODE}"
echo "Run directory:     ${DCACHE_RUN_DIR}"
echo "Visible GPUs:      ${CUDA_VISIBLE_DEVICES}"
echo "Global batch:      512"
echo "Microbatch/GPU:    ${DCACHE_MICRO_BATCH}"
echo "Total max steps:   6000 (1000 new optimizer updates)"

if [[ "${DCACHE_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Preflight passed. No checkpoint was renamed and training was not started."
  exit 0
fi

# Preserve the completed 5k weights without consuming another checkpoint-sized
# copy. Training resumes from the renamed file and creates a fresh last.ckpt.
if [[ ! -f "$START_CHECKPOINT" ]]; then
  mv "$LAST_CHECKPOINT" "$START_CHECKPOINT"
  RESUME_CHECKPOINT="$START_CHECKPOINT"
  echo "Preserved step-index 4999 as ${START_CHECKPOINT}."
fi

bash "$BASE_LAUNCHER" \
  checkpointing.resume_ckpt_path="$RESUME_CHECKPOINT" \
  "$@"

for checkpoint_step in 5500 6000; do
  checkpoint_path="${CHECKPOINT_DIR}/0-${checkpoint_step}.ckpt"
  if [[ ! -f "$checkpoint_path" ]]; then
    echo "Training returned without expected checkpoint ${checkpoint_path}." >&2
    exit 1
  fi
  verify_checkpoint_step "$checkpoint_path" "$checkpoint_step"
done

verify_checkpoint_step "$LAST_CHECKPOINT" 6000
echo "${MODE} continuation completed with checkpoints for metric indices 4999, 5499, and 5999."
