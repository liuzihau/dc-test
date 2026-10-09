#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"

# Five-state DCache + detached final-state feedback. Adjacent-gradient mode
# bridges each loss to its immediate predecessor only; simply removing detach
# would instead connect the whole recurrent trajectory.
export DCACHE_CUDA_VISIBLE_DEVICES="${DCACHE_CUDA_VISIBLE_DEVICES:-2,3}"
export DCACHE_DEVICES="${DCACHE_DEVICES:-2}"
export DCACHE_MAX_STEPS="${DCACHE_MAX_STEPS:-5000}"
export DCACHE_MICRO_BATCH="${DCACHE_MICRO_BATCH:-2}"
export DCACHE_GLOBAL_BATCH="${DCACHE_GLOBAL_BATCH:-512}"
export DCACHE_NUM_WORKERS="${DCACHE_NUM_WORKERS:-8}"
export DCACHE_VAL_INTERVAL="${DCACHE_VAL_INTERVAL:-500}"
export DCACHE_VAL_BATCHES="${DCACHE_VAL_BATCHES:-100}"
export DCACHE_SANITY_VAL_STEPS="${DCACHE_SANITY_VAL_STEPS:-0}"
export DCACHE_CHECKPOINT_SAVE_TOP_K="${DCACHE_CHECKPOINT_SAVE_TOP_K:-3}"
export DCACHE_DATA_DIR="${DCACHE_DATA_DIR:-${REPO_DIR}/.cache/huggingface}"
export DCACHE_RUN_DIR="${DCACHE_RUN_DIR:-${REPO_DIR}/outputs/owt-dcache-final-state-adjacent-pretrain-5k-2x3090}"
export DCACHE_PYTHON="${DCACHE_PYTHON:-/home/tliu0205/miniconda3/envs/dcache/bin/python}"

# Keep managed temporary files and compiled-library caches off root /tmp.
export TMPDIR="${REPO_DIR}/.cache/runtime/adjacent-grad/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export MPLCONFIGDIR="${REPO_DIR}/.cache/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="${REPO_DIR}/.cache/torchinductor"
export TRITON_CACHE_DIR="${REPO_DIR}/.cache/triton"
export HF_HOME="${REPO_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export XDG_CACHE_HOME="${REPO_DIR}/.cache"
export CUDA_CACHE_PATH="${REPO_DIR}/.cache/cuda"
export TORCH_HOME="${REPO_DIR}/.cache/torch"
export NUMBA_CACHE_DIR="${REPO_DIR}/.cache/numba"
export OMP_NUM_THREADS="${DCACHE_CPU_THREADS:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TORCH_HOME" "$NUMBA_CACHE_DIR"
ulimit -c 0
cd "$REPO_DIR"

# A new output directory starts from scratch. Reusing this trial's directory
# can resume it, but never silently resume an old detached/two-forward trial.
LAST_CHECKPOINT="${DCACHE_RUN_DIR}/checkpoints/last.ckpt"
if [[ -f "$LAST_CHECKPOINT" ]]; then
  ADJACENT_CHECKPOINT_PATH="$LAST_CHECKPOINT" "$DCACHE_PYTHON" -c '
import os
import torch
from omegaconf import OmegaConf

path = os.environ["ADJACENT_CHECKPOINT_PATH"]
checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
config = checkpoint["hyper_parameters"]["config"]
expected = {
    "dcachehooping.enabled": True,
    "dcachehooping.adjacent_grad.enabled": True,
    "dcachehooping.two_forward.enabled": False,
    "step_memory.detach_between_steps": False,
}
for key, value in expected.items():
    actual = OmegaConf.select(config, key, default=None)
    if actual != value:
        raise SystemExit(
            f"Refusing to resume non-adjacent checkpoint {path}: "
            f"{key}={actual!r}, expected {value!r}")
global_step = checkpoint.get("global_step")
print(f"Verified adjacent-gradient checkpoint {path}: global_step={global_step}")
'
fi

echo "Five-state adjacent-gradient DCache + detached final-state trial"
echo "Run directory: ${DCACHE_RUN_DIR}"
echo "Temporary files: ${TMPDIR}"
echo "Dataloader workers per rank: ${DCACHE_NUM_WORKERS}"
echo "Loss: (0.05 full + 0.10 t0 + 0.20 t1 + 1.00 t2 + 0.70 t3) / 2.05"
echo "DCache gradient horizon: one transition per loss; final feedback detached"

# Inherit the original five-forward recipe, including its source dropout and
# identity reference pass. These overrides change only DCache credit assignment
# plus the explicitly documented runtime/output defaults above.
exec bash "${SCRIPT_DIR}/train_owt_dcache_final_state_5k_2x3090.sh" \
  loader.num_workers="${DCACHE_NUM_WORKERS}" \
  step_memory.detach_between_steps=false \
  dcachehooping.two_forward.enabled=false \
  dcachehooping.adjacent_grad.enabled=true \
  "$@"
