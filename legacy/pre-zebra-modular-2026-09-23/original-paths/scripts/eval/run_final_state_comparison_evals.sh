#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-parallel}"
FIVE_CUDA="${DCACHE_FIVE_FORWARD_EVAL_CUDA:-2}"
TWO_CUDA="${DCACHE_TWO_FORWARD_EVAL_CUDA:-3}"
FIVE_CHECKPOINT="${DCACHE_FIVE_FORWARD_CHECKPOINT:-${REPO_DIR}/outputs/owt-dcache-final-state-pretrain-5k-2x3090/checkpoints/last.ckpt}"
TWO_CHECKPOINT="${DCACHE_TWO_FORWARD_CHECKPOINT:-${REPO_DIR}/outputs/owt-dcache-two-forward-pretrain-5k-2x3090/checkpoints/last.ckpt}"
FIVE_OUTPUT="${DCACHE_FIVE_FORWARD_EVAL_OUTPUT:-${REPO_DIR}/outputs/eval-final-state-five-forward-memory-interventions}"
TWO_OUTPUT="${DCACHE_TWO_FORWARD_EVAL_OUTPUT:-${REPO_DIR}/outputs/eval-final-state-two-forward-memory-interventions}"
COMPARISON_OUTPUT="${DCACHE_FINAL_STATE_COMPARISON_OUTPUT:-${REPO_DIR}/outputs/eval-final-state-comparison-5k}"
LOG_DIR="${COMPARISON_OUTPUT}/logs"

if [[ "$MODE" != "parallel" && "$MODE" != "sequential" ]]; then
  echo "Usage: $0 [parallel|sequential]" >&2
  exit 2
fi
if [[ ! -f "$FIVE_CHECKPOINT" ]]; then
  echo "Five-forward checkpoint not found: $FIVE_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$TWO_CHECKPOINT" ]]; then
  echo "Two-forward checkpoint not found: $TWO_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "${REPO_DIR}/.cache/matplotlib"
export MPLCONFIGDIR="${REPO_DIR}/.cache/matplotlib"
cd "$REPO_DIR"

run_five_forward() {
  DCACHE_EVAL_CUDA_VISIBLE_DEVICES="$FIVE_CUDA" \
    bash scripts/eval/eval_final_state_interventions.sh \
      "$FIVE_CHECKPOINT" "$FIVE_OUTPUT"
}

run_two_forward() {
  DCACHE_EVAL_CUDA_VISIBLE_DEVICES="$TWO_CUDA" \
    bash scripts/eval/eval_final_state_interventions.sh \
      "$TWO_CHECKPOINT" "$TWO_OUTPUT" --two-forward
}

if [[ "$MODE" == "parallel" ]]; then
  echo "Five-forward interventions: physical CUDA $FIVE_CUDA"
  run_five_forward >"$LOG_DIR/five-forward.log" 2>&1 &
  five_pid=$!
  echo "Two-forward interventions: physical CUDA $TWO_CUDA"
  run_two_forward >"$LOG_DIR/two-forward.log" 2>&1 &
  two_pid=$!
  failure=0
  wait "$five_pid" || failure=1
  wait "$two_pid" || failure=1
  if (( failure )); then
    echo "An evaluation failed. Inspect $LOG_DIR before rerunning." >&2
    exit 1
  fi
else
  echo "Running both evaluations sequentially on physical CUDA $FIVE_CUDA"
  run_five_forward 2>&1 | tee "$LOG_DIR/five-forward.log"
  TWO_CUDA="$FIVE_CUDA" run_two_forward \
    2>&1 | tee "$LOG_DIR/two-forward.log"
fi

python scripts/eval/plot_final_state_comparison.py \
  --five-forward-eval "$FIVE_OUTPUT" \
  --two-forward-eval "$TWO_OUTPUT" \
  --output-dir "$COMPARISON_OUTPUT"

echo "Evaluation and plotting complete."
echo "Quality:       $COMPARISON_OUTPUT/five_way_transition_quality.png"
echo "Interventions: $COMPARISON_OUTPUT/dual_memory_intervention_comparison.png"
echo "Training:      $COMPARISON_OUTPUT/five_way_training_health.png"
