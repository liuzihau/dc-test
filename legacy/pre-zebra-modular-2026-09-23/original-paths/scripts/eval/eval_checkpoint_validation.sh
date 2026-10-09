#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: $0 VARIANT CHECKPOINT OUTPUT_DIR [extra Hydra overrides...]" >&2
  echo "VARIANT: vanilla | objective | dcache-v2 | final-state | two-forward | final-state-adjacent" >&2
  exit 2
fi

VARIANT="$1"
CHECKPOINT="$2"
OUTPUT_DIR="$3"
shift 3
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DCACHE_DATA_DIR:-${REPO_DIR}/.cache/huggingface}"
PYTHON_BIN="${DCACHE_PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${DCACHE_EVAL_CUDA_VISIBLE_DEVICES:-2}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_DIR}/.cache/matplotlib}"

if [[ "$CHECKPOINT" != /* ]]; then
  CHECKPOINT="${REPO_DIR}/${CHECKPOINT}"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="${REPO_DIR}/${OUTPUT_DIR}"
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR"
cd "$REPO_DIR"

VARIANT_OVERRIDES=()
case "$VARIANT" in
  vanilla)
    VARIANT_OVERRIDES+=(
      training.objective_matched_multistate.enabled=false
      step_memory.enabled=false
      step_memory.pretrain.enabled=false)
    ;;
  objective)
    VARIANT_OVERRIDES+=(
      training.objective_matched_multistate.enabled=true
      step_memory.enabled=false
      step_memory.use_previous_kv=false
      step_memory.pretrain.enabled=false)
    ;;
  dcache-v2)
    VARIANT_OVERRIDES+=(
      step_memory.enabled=true
      step_memory.use_previous_kv=true
      step_memory.detach_between_steps=true
      step_memory.gate.enabled=true
      step_memory.pretrain.enabled=true)
    ;;
  final-state|final-state-adjacent)
    VARIANT_OVERRIDES+=(
      step_memory.enabled=true
      step_memory.use_previous_kv=true
      step_memory.detach_between_steps=true
      step_memory.gate.enabled=true
      step_memory.pretrain.enabled=true
      dcachehooping.enabled=true
      dcachehooping.status_embedding.enabled=false
      dcachehooping.latent_dropout_probability=0.10
      dcachehooping.latent_mask_probability=0.0
      dcachehooping.latent_mask_loss_weight=0.0
      dcachehooping.tentative.enabled=false
      dcachehooping.tentative.batch_probability=0.0
      dcachehooping.tentative.loss_weight=0.0
      dcachehooping.confidence.enabled=false
      dcachehooping.confidence.loss_weight=0.0)
    if [[ "$VARIANT" == "final-state-adjacent" ]]; then
      # Identical five-state validation inputs/losses; only the checkpoint's
      # training-gradient mode differs (validation does not backpropagate).
      VARIANT_OVERRIDES+=(
        step_memory.detach_between_steps=false
        dcachehooping.two_forward.enabled=false
        dcachehooping.adjacent_grad.enabled=true)
    fi
    ;;
  two-forward)
    VARIANT_OVERRIDES+=(
      step_memory.enabled=true
      step_memory.use_previous_kv=true
      step_memory.detach_between_steps=false
      step_memory.gate.enabled=true
      step_memory.pretrain.enabled=true
      step_memory.pretrain.identity.enabled=false
      dcachehooping.enabled=true
      dcachehooping.two_forward.enabled=true
      dcachehooping.status_embedding.enabled=false
      dcachehooping.latent_dropout_probability=0.10
      dcachehooping.latent_mask_probability=0.0
      dcachehooping.latent_mask_loss_weight=0.0
      dcachehooping.tentative.enabled=false
      dcachehooping.tentative.batch_probability=0.0
      dcachehooping.tentative.loss_weight=0.0
      dcachehooping.confidence.enabled=false
      dcachehooping.confidence.loss_weight=0.0)
    ;;
  *)
    echo "Unknown variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

"$PYTHON_BIN" -u main.py \
  mode=ppl_eval \
  model=small \
  algo=mdlm \
  data=openwebtext-split \
  data.cache_dir="$DATA_DIR" \
  data.insert_valid_special=false \
  data.insert_valid_eos=false \
  model.length=1024 \
  model.attn_backend=sdpa \
  block_size=1024 \
  trainer.devices=1 \
  trainer.precision=bf16-mixed \
  trainer.limit_val_batches="${DCACHE_VAL_BATCHES:-100}" \
  trainer.num_sanity_val_steps=0 \
  loader.eval_global_batch_size="${DCACHE_EVAL_GLOBAL_BATCH:-4}" \
  loader.eval_batch_size="${DCACHE_EVAL_BATCH:-4}" \
  eval.checkpoint_path="$CHECKPOINT" \
  eval.shuffle_valid=false \
  checkpointing.save_dir="$OUTPUT_DIR" \
  hydra.run.dir="$OUTPUT_DIR/run" \
  wandb=null \
  "${VARIANT_OVERRIDES[@]}" \
  "$@"
