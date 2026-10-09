#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CHECKPOINT [ppl_eval|sample_eval] [extra Hydra overrides...]" >&2
  exit 2
fi

CHECKPOINT="$1"
MODE="${2:-ppl_eval}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DCACHE_DATA_DIR:-${REPO_DIR}/.cache/huggingface}"
mkdir -p "$DATA_DIR" "${REPO_DIR}/sample_logs"
cd "$REPO_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

COMMON=(
  "mode=${MODE}"
  model=small
  algo=bd3lm
  data=openwebtext-split
  "data.cache_dir=${DATA_DIR}"
  data.insert_valid_special=false
  data.insert_valid_eos=false
  model.length=1024
  block_size=16
  loader.eval_global_batch_size=1
  loader.eval_batch_size=1
  trainer.devices=1
  trainer.precision=bf16-mixed
  "eval.checkpoint_path=${CHECKPOINT}"
  step_memory.enabled=true
  step_memory.use_previous_kv=true
  wandb=null
)

if [[ "$MODE" == "sample_eval" ]]; then
  python -u main.py "${COMMON[@]}" \
    algo.T=5000 \
    model.attn_backend=sdpa \
    sampling.kv_cache=true \
    sampling.nucleus_p=0.9 \
    "sampling.logdir=${REPO_DIR}/sample_logs/dcache" \
    "$@"
elif [[ "$MODE" == "ppl_eval" ]]; then
  python -u main.py "${COMMON[@]}" \
    model.attn_backend=flex \
    "$@"
else
  echo "mode must be ppl_eval or sample_eval" >&2
  exit 2
fi
