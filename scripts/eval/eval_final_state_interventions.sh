#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 CHECKPOINT OUTPUT_DIR [extra evaluator arguments...]" >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT="$1"
OUTPUT_DIR="$2"
shift 2

export CUDA_VISIBLE_DEVICES="${DCACHE_EVAL_CUDA_VISIBLE_DEVICES:-2}"
PYTHON_BIN="${DCACHE_PYTHON:-python}"
DATA_DIR="${DCACHE_DATA_DIR:-${REPO_DIR}/.cache/huggingface}"

cd "$REPO_DIR"
"$PYTHON_BIN" -u scripts/eval/eval_final_state_interventions.py \
  --checkpoint "$CHECKPOINT" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  "$@"
