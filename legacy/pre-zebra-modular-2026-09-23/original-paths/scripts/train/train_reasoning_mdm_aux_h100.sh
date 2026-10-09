#!/usr/bin/env bash
# First reasoning baseline: objective-matched MDM + neighbor heads, no memory.
# Reuse the shared trainer; do not change OWT or select a future memory design.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
usage() {
  echo 'Usage: bash scripts/train/train_reasoning_mdm_aux_h100.sh sudoku|zebra|countdown prepare|smoke|train|tmux|evaluate|plot'
  echo 'One H100 (GPU 0), mdm_aux only: five independent states, neighbor weight 0.5, no memory.'
  echo 'Defaults: 20,000 / 1,000 / 1,000 pilot examples, seed 1, 5,000 optimizer updates.'
  echo 'Prepare ONCE; the data generator refuses to overwrite existing data.'
  echo 'Uses REASONING_DATA_DIR/RUN_DIR, GPU_IDS, MICRO_BATCH, GLOBAL_BATCH, MAX_STEPS, SIZE, SEED.'
  echo 'Pilot counts: REASONING_TRAIN_EXAMPLES, REASONING_VALID_EXAMPLES, REASONING_TEST_EXAMPLES.'
  echo 'smoke uses an isolated DEBUG model; train/tmux use the task-sized model.'
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if (( $# != 2 )); then usage >&2; exit 2; fi
TASK="$1" ACTION="$2"
case "$TASK" in sudoku|zebra|countdown) ;; *) usage >&2; exit 2 ;; esac
case "$ACTION" in prepare|smoke|train|tmux|evaluate|plot) ;; *) usage >&2; exit 2 ;; esac

export REASONING_TRAIN_EXAMPLES="${REASONING_TRAIN_EXAMPLES:-20000}"
export REASONING_VALID_EXAMPLES="${REASONING_VALID_EXAMPLES:-1000}"
export REASONING_TEST_EXAMPLES="${REASONING_TEST_EXAMPLES:-1000}"
export REASONING_SEED="${REASONING_SEED:-1}"
export REASONING_MICRO_BATCH="${REASONING_MICRO_BATCH:-8}"
export REASONING_GLOBAL_BATCH="${REASONING_GLOBAL_BATCH:-128}"
export REASONING_MAX_STEPS="${REASONING_MAX_STEPS:-5000}"
export REASONING_GPU_IDS="${REASONING_GPU_IDS:-0}"
for NAME in REASONING_TRAIN_EXAMPLES REASONING_VALID_EXAMPLES REASONING_TEST_EXAMPLES \
            REASONING_MICRO_BATCH REASONING_GLOBAL_BATCH REASONING_MAX_STEPS; do
  if [[ ! "${!NAME}" =~ ^[1-9][0-9]*$ ]]; then
    echo "$NAME must be a positive integer." >&2
    exit 2
  fi
done
if [[ ! "$REASONING_SEED" =~ ^[0-9]+$ ]]; then
  echo 'REASONING_SEED must be a nonnegative integer.' >&2
  exit 2
fi

# Label all three counts to avoid silently using an old 1,000-example demo.
# The path is variant-independent so later controls can use these exact splits.
DATA_LABEL="pilot-v1-n${REASONING_TRAIN_EXAMPLES}-v${REASONING_VALID_EXAMPLES}-t${REASONING_TEST_EXAMPLES}"
export REASONING_DATA_DIR="${REASONING_DATA_DIR:-$REPO_DIR/.cache/reasoning/${TASK}-${DATA_LABEL}}"
export REASONING_RUN_DIR="${REASONING_RUN_DIR:-$REPO_DIR/outputs/reasoning/${TASK}/mdm_aux-h100-${DATA_LABEL}-seed${REASONING_SEED}}"

# No memory-specific randomness/losses in this baseline. These mechanisms were
# already runtime-inactive for mdm_aux; make that explicit in saved contracts.
export REASONING_NO_ROBUSTNESS=1
export REASONING_GRADIENT_MODE=detached
echo 'Baseline: current-only 2D RoPE; five teacher-forced states; prev/next auxiliary weight 0.5.'
echo 'No DCache, final feedback, memory dropout, identity loss, or cross-step gradient edges.'
echo "Data: $REASONING_DATA_DIR"
echo "Run:  $REASONING_RUN_DIR"
exec /bin/bash "$SCRIPT_DIR/train_reasoning.sh" h100 "$TASK" mdm_aux "$ACTION"
