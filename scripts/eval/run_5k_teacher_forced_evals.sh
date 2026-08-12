#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-parallel}"
if [[ "$MODE" != "parallel" && "$MODE" != "sequential" ]]; then
  echo "usage: $0 [parallel|sequential]" >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DCACHE_EVAL_PYTHON:-python}"
DATA_DIR="${DCACHE_DATA_DIR:-${REPO_DIR}/.cache/huggingface}"
BASELINE_CKPT="${BASELINE_CKPT:-${REPO_DIR}/outputs/owt-mdlm-pretrain-5k-2x3090/checkpoints/last.ckpt}"
DCACHE_CKPT="${DCACHE_CKPT:-${REPO_DIR}/outputs/owt-dcache-pretrain-5k-2x4090/checkpoints/last.ckpt}"
OUTPUT_ROOT="${DCACHE_EVAL_OUTPUT:-${REPO_DIR}/outputs/eval-5k-teacher-forced}"
FIXED_CUDA="${DCACHE_FIXED_CUDA:-2}"
TRANSITION_CUDA="${DCACHE_TRANSITION_CUDA:-3}"
EXAMPLES="${DCACHE_EVAL_EXAMPLES:-800}"
BATCH_SIZE="${DCACHE_EVAL_BATCH:-4}"
NUM_WORKERS="${DCACHE_EVAL_WORKERS:-2}"
BOOTSTRAP_SAMPLES="${DCACHE_EVAL_BOOTSTRAP:-10000}"
SEED="${DCACHE_EVAL_SEED:-20260812}"
GATE_ENABLED="${DCACHE_EVAL_GATE_ENABLED:-0}"

if [[ "$MODE" == "parallel" && "$FIXED_CUDA" == "$TRANSITION_CUDA" ]]; then
  echo "Parallel mode requires two different GPU IDs." >&2
  exit 2
fi

for required_file in "$BASELINE_CKPT" "$DCACHE_CKPT"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing checkpoint: $required_file" >&2
    echo "Set BASELINE_CKPT/DCACHE_CKPT or copy the checkpoint to this path." >&2
    exit 1
  fi
done

VALID_DATA="${DATA_DIR}/openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat"
if [[ ! -d "$VALID_DATA" ]]; then
  echo "Missing prepared validation data: $VALID_DATA" >&2
  echo "Set DCACHE_DATA_DIR to the cache used for the completed training runs." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/.matplotlib"
cd "$REPO_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_ROOT}/.matplotlib}"

COMMON_ARGS=(
  --baseline-checkpoint "$BASELINE_CKPT"
  --dcache-checkpoint "$DCACHE_CKPT"
  --data-dir "$DATA_DIR"
  --examples "$EXAMPLES"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --bootstrap-samples "$BOOTSTRAP_SAMPLES"
  --seed "$SEED"
  --device cuda:0
)

if [[ "${DCACHE_EVAL_FORCE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--force)
fi
if [[ "$GATE_ENABLED" == "1" ]]; then
  COMMON_ARGS+=(--dcache-gate-enabled)
fi

run_fixed() {
  echo "Fixed-corruption evaluation: physical CUDA ${FIXED_CUDA}"
  CUDA_VISIBLE_DEVICES="$FIXED_CUDA" "$PYTHON_BIN" -u \
    scripts/eval/eval_fixed_corruption.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "$OUTPUT_ROOT/fixed-corruption" \
    2>&1 | tee "$OUTPUT_ROOT/logs/fixed-corruption.log"
}

run_transition() {
  echo "Teacher-forced transition evaluation: physical CUDA ${TRANSITION_CUDA}"
  CUDA_VISIBLE_DEVICES="$TRANSITION_CUDA" "$PYTHON_BIN" -u \
    scripts/eval/eval_teacher_forced_transitions.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "$OUTPUT_ROOT/transitions" \
    2>&1 | tee "$OUTPUT_ROOT/logs/transitions.log"
}

echo "Mode: $MODE"
echo "Baseline: $BASELINE_CKPT"
echo "DCache:   $DCACHE_CKPT"
echo "DCache gate enabled: $GATE_ENABLED"
echo "Data:     $VALID_DATA"
echo "Output:   $OUTPUT_ROOT"

if [[ "$MODE" == "parallel" ]]; then
  run_fixed &
  fixed_pid=$!
  run_transition &
  transition_pid=$!
  set +e
  wait "$fixed_pid"
  fixed_status=$?
  wait "$transition_pid"
  transition_status=$?
  set -e
  if [[ "$fixed_status" -ne 0 || "$transition_status" -ne 0 ]]; then
    echo "Evaluation failed: fixed=$fixed_status transition=$transition_status" >&2
    exit 1
  fi
else
  run_fixed
  run_transition
fi

echo "Both evaluations completed successfully."
echo "Fixed plot:     $OUTPUT_ROOT/fixed-corruption/fixed_corruption.png"
echo "Transition plot: $OUTPUT_ROOT/transitions/teacher_forced_transitions.png"
