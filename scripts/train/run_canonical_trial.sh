#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  cat >&2 <<'EOF'
Usage: run_canonical_trial.sh VARIANT RUN_DIR [extra Hydra overrides...]

VARIANT: vanilla | objective | dcache-v2 | final-state

Example:
  bash scripts/train/run_canonical_trial.sh final-state outputs/my-final-state
EOF
  exit 2
fi

VARIANT="$1"
RUN_DIR="$2"
shift 2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ "$RUN_DIR" != /* ]]; then
  RUN_DIR="${REPO_DIR}/${RUN_DIR}"
fi
if [[ "$RUN_DIR" == "$REPO_DIR" || "$RUN_DIR" == "/" ]]; then
  echo "Refusing unsafe run directory: ${RUN_DIR}" >&2
  exit 2
fi

export DCACHE_RUN_DIR="$RUN_DIR"
export DCACHE_CUDA_VISIBLE_DEVICES="${DCACHE_CUDA_VISIBLE_DEVICES:-2,3}"
export CUDA_VISIBLE_DEVICES="$DCACHE_CUDA_VISIBLE_DEVICES"
export DCACHE_DEVICES="${DCACHE_DEVICES:-2}"
export DCACHE_CHECKPOINT_SAVE_TOP_K="${DCACHE_CHECKPOINT_SAVE_TOP_K:-3}"

case "$VARIANT" in
  vanilla)
    LAUNCHER="${SCRIPT_DIR}/train_owt_mdlm_pretrain_5k_2x3090.sh"
    ;;
  objective)
    LAUNCHER="${SCRIPT_DIR}/train_owt_mdlm_objective_matched_5k.sh"
    ;;
  dcache-v2)
    LAUNCHER="${SCRIPT_DIR}/train_owt_dcache_v2_pretrain_5k_2x3090.sh"
    ;;
  final-state)
    LAUNCHER="${SCRIPT_DIR}/train_owt_dcache_final_state_5k_2x3090.sh"
    ;;
  *)
    echo "Unknown variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

echo "Canonical variant: ${VARIANT}"
echo "Run directory:     ${RUN_DIR}"
echo "Physical GPUs:     ${DCACHE_CUDA_VISIBLE_DEVICES}"
echo "Checkpoint policy: latest ${DCACHE_CHECKPOINT_SAVE_TOP_K} every 500 steps; last.ckpt is a symlink"
exec bash "$LAUNCHER" "$@"
