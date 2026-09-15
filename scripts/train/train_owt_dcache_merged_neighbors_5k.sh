#!/usr/bin/env bash
# Full-data merged-attention trial with optional target-masked neighbor prediction.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
export DCACHE_GRADIENT_MODE="${DCACHE_GRADIENT_MODE:-adjacent}"
case "$DCACHE_GRADIENT_MODE" in
  adjacent) ADJACENT_ENABLED=true; DETACH_STEPS=false ;;
  detached) ADJACENT_ENABLED=false; DETACH_STEPS=true ;;
  *) echo 'DCACHE_GRADIENT_MODE must be adjacent or detached.' >&2; exit 2 ;;
esac
export DCACHE_NEIGHBOR_ENABLED="${DCACHE_NEIGHBOR_ENABLED:-true}"
case "$DCACHE_NEIGHBOR_ENABLED" in
  true) DEFAULT_TRIAL="owt-dcache-merged-neighbors-final-state-${DCACHE_GRADIENT_MODE}-5k" ;;
  false) DEFAULT_TRIAL="owt-dcache-merged-no-neighbors-final-state-${DCACHE_GRADIENT_MODE}-5k" ;;
  *) echo 'DCACHE_NEIGHBOR_ENABLED must be true or false.' >&2; exit 2 ;;
esac
export DCACHE_MERGED_POLICY="${DCACHE_MERGED_POLICY:-legacy}"
case "$DCACHE_MERGED_POLICY" in
  legacy) MERGED_GATE_ENABLED=true; CACHE_ONLY_PROBABILITY=0.20 ;;
  current_preserving)
    MERGED_GATE_ENABLED=false; CACHE_ONLY_PROBABILITY=0.0
    DEFAULT_TRIAL="${DEFAULT_TRIAL}-current-preserving" ;;
  *) echo 'DCACHE_MERGED_POLICY must be legacy or current_preserving.' >&2; exit 2 ;;
esac

export DCACHE_CUDA_VISIBLE_DEVICES="${DCACHE_CUDA_VISIBLE_DEVICES:-2,3}"
export DCACHE_DEVICES="${DCACHE_DEVICES:-2}"
export DCACHE_MAX_STEPS="${DCACHE_MAX_STEPS:-5000}"
export DCACHE_MICRO_BATCH="${DCACHE_MICRO_BATCH:-2}"
export DCACHE_EVAL_MICRO_BATCH="${DCACHE_EVAL_MICRO_BATCH:-$DCACHE_MICRO_BATCH}"
export DCACHE_GLOBAL_BATCH="${DCACHE_GLOBAL_BATCH:-512}"
export DCACHE_NUM_WORKERS="${DCACHE_NUM_WORKERS:-8}"
export DCACHE_VAL_INTERVAL="${DCACHE_VAL_INTERVAL:-500}"
export DCACHE_VAL_BATCHES="${DCACHE_EVAL_BATCHES:-${DCACHE_VAL_BATCHES:-100}}"
export DCACHE_SANITY_VAL_STEPS="${DCACHE_SANITY_VAL_STEPS:-0}"
export DCACHE_CHECKPOINT_SAVE_TOP_K="${DCACHE_CHECKPOINT_SAVE_TOP_K:-3}"
export DCACHE_DATA_DIR="${DCACHE_DATA_DIR:-$REPO_DIR/.cache/huggingface}"
export DCACHE_RUN_DIR="${DCACHE_RUN_DIR:-$REPO_DIR/outputs/$DEFAULT_TRIAL}"
export DCACHE_PYTHON="${DCACHE_PYTHON:-python}"
export DCACHE_RESTORE_DATA_CURSOR="${DCACHE_RESTORE_DATA_CURSOR:-false}"
case "$DCACHE_RESTORE_DATA_CURSOR" in
  true|false) ;;
  *) echo 'DCACHE_RESTORE_DATA_CURSOR must be true or false.' >&2; exit 2 ;;
esac

for setting in DCACHE_DEVICES DCACHE_MAX_STEPS DCACHE_MICRO_BATCH DCACHE_EVAL_MICRO_BATCH DCACHE_GLOBAL_BATCH DCACHE_VAL_INTERVAL; do
  if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "$setting must be a positive integer, got ${!setting}." >&2
    exit 2
  fi
done
if (( DCACHE_GLOBAL_BATCH % (DCACHE_DEVICES * DCACHE_MICRO_BATCH) != 0 || DCACHE_GLOBAL_BATCH % (DCACHE_DEVICES * DCACHE_EVAL_MICRO_BATCH) != 0 )); then
  echo 'Global batch must be divisible by devices times both training and evaluation microbatch.' >&2
  exit 2
fi

# Keep architecture, objective and checkpoint identity fixed for this named
# recipe. Runtime overrides (e.g. trainer.limit_train_batches) remain available.
for argument in "$@"; do
  key="${argument%%=*}"
  key="${key#++}"; key="${key#+}"; key="${key#\~}"
  case "$key" in
    step_memory|step_memory.*|dcachehooping|dcachehooping.*|neighbor_prediction|neighbor_prediction.*|training|training.*|model|model.*|algo|algo.*|data|data.*|loader|loader.*|block_size|checkpointing|checkpointing.save_dir|checkpointing.resume_from_ckpt|checkpointing.resume_ckpt_path|checkpointing.allow_batch_geometry_change)
      echo "Cannot override recipe field $key here. Use DCACHE_GRADIENT_MODE and DCACHE_* runtime settings; choose a separate recipe for scientific changes." >&2
      exit 2 ;;
  esac
done

if [[ -n "${DCACHE_RESUME_CKPT:-}" || -f "$DCACHE_DATA_DIR/compact_train.json" ]]; then
  echo 'Neighbor prediction is a NEW architecture. Unset DCACHE_RESUME_CKPT and use the full prepared dataset, not the 1500-5000 compact continuation bundle.' >&2
  exit 2
fi
for dataset in openwebtext-train_train_bs1024_wrapped_specialFalse.dat openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat; do
  if [[ ! -f "$DCACHE_DATA_DIR/$dataset/state.json" || ! -f "$DCACHE_DATA_DIR/$dataset/dataset_info.json" ]]; then
    echo "Prepared dataset metadata missing in $DCACHE_DATA_DIR/$dataset; complete the full train/validation transfer first." >&2
    exit 2
  fi
done

# Runtime caches belong on the project disk, not root /tmp.
export TMPDIR="$REPO_DIR/.cache/runtime/merged-neighbors/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$REPO_DIR/.cache"
export MPLCONFIGDIR="$REPO_DIR/.cache/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$REPO_DIR/.cache/torchinductor"
export TRITON_CACHE_DIR="$REPO_DIR/.cache/triton"
export HF_HOME="$REPO_DIR/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export CUDA_CACHE_PATH="$REPO_DIR/.cache/cuda"
export TORCH_HOME="$REPO_DIR/.cache/torch"
export NUMBA_CACHE_DIR="$REPO_DIR/.cache/numba"
export OMP_NUM_THREADS="${DCACHE_CPU_THREADS:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TORCH_HOME" "$NUMBA_CACHE_DIR"
ulimit -c 0
cd "$REPO_DIR"

LAST_CHECKPOINT="$DCACHE_RUN_DIR/checkpoints/last.ckpt"
if [[ -L "$LAST_CHECKPOINT" && ! -e "$LAST_CHECKPOINT" ]]; then
  echo 'Broken last.ckpt pointer; refusing to restart from scratch.' >&2
  exit 2
fi
if [[ ! -f "$LAST_CHECKPOINT" ]]; then
  shopt -s nullglob
  existing_checkpoints=("$DCACHE_RUN_DIR"/checkpoints/*.ckpt)
  if (( ${#existing_checkpoints[@]} )); then
    echo 'Checkpoints exist but last.ckpt is missing; choose the correct run directory and restore its last.ckpt pointer before resuming.' >&2
    exit 2
  fi
fi
if [[ -f "$LAST_CHECKPOINT" ]]; then
  NEIGHBOR_CHECKPOINT="$LAST_CHECKPOINT" "$DCACHE_PYTHON" -c '
import os, torch
from omegaconf import OmegaConf
path = os.environ["NEIGHBOR_CHECKPOINT"]
checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
config = checkpoint["hyper_parameters"]["config"]
adjacent = os.environ["DCACHE_GRADIENT_MODE"] == "adjacent"
policy = os.environ["DCACHE_MERGED_POLICY"]
current_preserving = policy == "current_preserving"
expected = {
    "step_memory.enabled": True,
    "step_memory.attention_mode": "merged",
    "step_memory.merged_policy": policy,
    "step_memory.use_previous_kv": True,
    "step_memory.detach_between_steps": not adjacent,
    "step_memory.gate.enabled": not current_preserving,
    "step_memory.gate.init": 0.1,
    "step_memory.pretrain.enabled": True,
    "step_memory.pretrain.step_size_min": 0.025,
    "step_memory.pretrain.step_size_max": 0.10,
    "step_memory.pretrain.max_t0_mask_ratio": 0.9975,
    "step_memory.pretrain.teacher_token_probability": 1.0,
    "step_memory.pretrain.full_loss_weight": 0.05,
    "step_memory.pretrain.t0_loss_weight": 0.10,
    "step_memory.pretrain.t1_loss_weight": 0.20,
    "step_memory.pretrain.t2_loss_weight": 1.00,
    "step_memory.pretrain.t3_loss_weight": 0.70,
    "step_memory.pretrain.source_dropout.enabled": True,
    "step_memory.pretrain.source_dropout.cache_only_probability": 0.0 if current_preserving else 0.20,
    "step_memory.pretrain.source_dropout.current_only_probability": 0.05,
    "step_memory.pretrain.source_dropout.warmup_steps": 1000,
    "step_memory.pretrain.identity.enabled": True,
    "step_memory.pretrain.identity.batch_probability": 0.25,
    "step_memory.pretrain.identity.margin": 0.05,
    "step_memory.pretrain.identity.weight": 0.10,
    "step_memory.rollout.enabled": False,
    "dcachehooping.enabled": True,
    "dcachehooping.two_forward.enabled": False,
    "dcachehooping.adjacent_grad.enabled": adjacent,
    "dcachehooping.status_embedding.enabled": False,
    "dcachehooping.tentative.enabled": False,
    "dcachehooping.confidence.enabled": False,
    "dcachehooping.latent_mask_probability": 0.0,
    "dcachehooping.latent_dropout_probability": 0.10,
    "dcachehooping.identity_final_probability": 0.50,
    "neighbor_prediction.enabled": os.environ["DCACHE_NEIGHBOR_ENABLED"] == "true",
    "neighbor_prediction.weight": 0.5,
    "loader.global_batch_size": int(os.environ["DCACHE_GLOBAL_BATCH"]),
    "loader.batch_size": int(os.environ["DCACHE_MICRO_BATCH"]),
    "loader.eval_batch_size": int(os.environ["DCACHE_EVAL_MICRO_BATCH"]),
    "trainer.devices": int(os.environ["DCACHE_DEVICES"]),
    "checkpointing.restore_data_cursor": os.environ["DCACHE_RESTORE_DATA_CURSOR"] == "true",
}
for key, value in expected.items():
    default = {"checkpointing.restore_data_cursor": False,
               "step_memory.merged_policy": "legacy"}.get(key)
    actual = OmegaConf.select(config, key, default=default)
    if actual != value:
        raise SystemExit(f"Refusing incompatible neighbor checkpoint {path}: {key}={actual!r}; expected {value!r}")
if not checkpoint.get("optimizer_states") or not checkpoint.get("state_dict"):
    raise SystemExit("A full training checkpoint with model and optimizer state is required")
step = checkpoint.get("global_step")
if not isinstance(step, int) or isinstance(step, bool) or step < 0:
    raise SystemExit("Checkpoint global_step must be a nonnegative integer")
if step > int(os.environ["DCACHE_MAX_STEPS"]):
    raise SystemExit("Checkpoint is beyond DCACHE_MAX_STEPS; set the intended total optimizer-step target explicitly")
print("Verified merged-neighbor", os.environ["DCACHE_GRADIENT_MODE"],
      policy, "checkpoint: step", checkpoint.get("global_step"))
'
fi

echo "Merged 2D-RoPE neighbor trial: $DCACHE_GRADIENT_MODE DCache gradients; previous final state detached."
echo "Merged policy: $DCACHE_MERGED_POLICY; previous-V gate=$MERGED_GATE_ENABLED; cache-only dropout=$CACHE_ONLY_PROBABILITY; current-only dropout=0.05."
if [[ "$DCACHE_MERGED_POLICY" == current_preserving ]]; then
  echo 'Current K/V always remain available. Previous V is unattenuated; the joint softmax still learns the source balance.'
fi
if [[ "$DCACHE_NEIGHBOR_ENABLED" == true ]]; then
  echo 'Neighbor loss: 0.5 * (previous-target mean CE + next-target mean CE) / 2; target must be masked, source need not be.'
else
  echo 'Neighbor prediction OFF: no auxiliary LM heads or neighbor loss.'
fi
echo "Run directory: $DCACHE_RUN_DIR"
echo "Temporary files: $TMPDIR"
exec bash "$SCRIPT_DIR/train_owt_dcache_final_state_5k_2x3090.sh" \
  "$@" \
  loader.num_workers="$DCACHE_NUM_WORKERS" \
  loader.eval_batch_size="$DCACHE_EVAL_MICRO_BATCH" \
  checkpointing.restore_data_cursor="$DCACHE_RESTORE_DATA_CURSOR" \
  step_memory.attention_mode=merged \
  step_memory.merged_policy="$DCACHE_MERGED_POLICY" \
  step_memory.gate.enabled="$MERGED_GATE_ENABLED" \
  step_memory.pretrain.source_dropout.enabled=true \
  step_memory.pretrain.source_dropout.cache_only_probability="$CACHE_ONLY_PROBABILITY" \
  step_memory.pretrain.source_dropout.current_only_probability=0.05 \
  step_memory.detach_between_steps="$DETACH_STEPS" \
  dcachehooping.two_forward.enabled=false \
  dcachehooping.adjacent_grad.enabled="$ADJACENT_ENABLED" \
  neighbor_prediction.enabled="$DCACHE_NEIGHBOR_ENABLED" \
  neighbor_prediction.weight=0.5 \
  neighbor_prediction.chunk_size=128 \
  neighbor_prediction.checkpoint_chunks=true
