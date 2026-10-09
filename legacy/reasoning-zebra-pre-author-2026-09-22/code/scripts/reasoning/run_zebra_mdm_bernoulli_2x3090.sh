#!/usr/bin/env bash
# Isolated corruption-only MDM ablation. Does not modify or join active queues.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
action="${1:-plan}"
if [[ $# -gt 1 ]]; then
  echo "Usage: bash $0 {plan|smoke|run|evaluate}; configure via DCACHE_* variables" >&2
  exit 2
fi
case "$action" in plan|smoke|run|evaluate) ;; *) echo "Unknown action: $action" >&2; exit 2 ;; esac

export CUDA_VISIBLE_DEVICES="${DCACHE_CUDA_VISIBLE_DEVICES:-2,3}"
export TMPDIR="$PWD/.cache/runtime/reasoning-mdm-bernoulli/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export MPLCONFIGDIR="$PWD/.cache/runtime/reasoning-mdm-bernoulli/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/runtime/reasoning-mdm-bernoulli/inductor"
export TRITON_CACHE_DIR="$PWD/.cache/runtime/reasoning-mdm-bernoulli/triton"
export XDG_CACHE_HOME="$PWD/.cache"
export CUDA_CACHE_PATH="$PWD/.cache/cuda"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
ulimit -c 0
python_bin="${DCACHE_PYTHON:-python}"
devices="${DCACHE_DEVICES:-2}"
micro_batch="${DCACHE_MICRO_BATCH:-32}"
eval_batch="${DCACHE_EVAL_BATCH:-32}"
timesteps="${DCACHE_CORRUPTION_TIMESTEPS:-64}"
epochs="${DCACHE_EPOCHS:-3}"
for variable in devices micro_batch eval_batch timesteps epochs; do
  value="${!variable}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$variable must be a positive integer" >&2; exit 2
  fi
done
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#gpu_ids[@]} -ne $devices ]] || (( 128 % (devices * micro_batch) )); then
  echo "List exactly DCACHE_DEVICES GPU IDs; devices * microbatch must divide 128" >&2; exit 2
fi
data="${DCACHE_DATA_DIR:-$PWD/.cache/reasoning/zebra-benchmark-full-v1}"
run="${DCACHE_RUN_DIR:-$PWD/outputs/reasoning/zebra-mdm-bernoulli-t${timesteps}-${epochs}ep-${devices}x3090-mb${micro_batch}}"
cli="$PWD/scripts/reasoning/run_reasoning.py"
launch=("$python_bin" -u)
if (( devices > 1 )); then
  launch+=(-m torch.distributed.run --standalone --nproc_per_node "$devices")
fi
train=("${launch[@]}" "$cli" train --suite split --task zebra-benchmark --variant mdm
  --data-dir "$data" --size mini --epochs "$epochs"
  --global-batch 128 --micro-batch "$micro_batch" --precision bf16
  --corruption-mode bernoulli --corruption-timesteps "$timesteps"
  --lr 0.0003 --warmup-steps 1000 --weight-decay 0 --grad-clip 1
  --seed 1 --eval-seed 2026 --val-every 500 --save-every 500 --save-seconds 1200
  --validation-protocol both --validation-examples 1000 --eval-batch-size "$eval_batch")
evaluate=("$python_bin" -u "$cli" evaluate --checkpoint "$run/checkpoints/last.pt"
  --data-dir "$data" --output "$run/generation.json" --split test
  --examples 1000 --batch-size "$eval_batch" --seed 2026 --policy top_prob)
if [[ "$action" == plan ]]; then
  echo "Fresh MDM sampler ablation; T=$timesteps is an explicit choice, not an exact paper reproduction."
  echo "Output: $run; runtime cache: $TMPDIR; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  printf '%q ' "${train[@]}" --run-dir "$run"; printf '\n'
  printf '%q ' "${evaluate[@]}"; printf '\n'
  exit 0
fi
if [[ ! -f "$data/manifest.json" ]]; then
  echo "Prepared full Zebra data missing: $data/manifest.json; no data will be downloaded." >&2; exit 2
fi
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
case "$action" in
  smoke)
    smoke_run=$(mktemp -d "$PWD/.cache/runtime/reasoning-mdm-bernoulli/smoke.XXXXXXXX")
    "${train[@]}" --run-dir "$smoke_run" --stop-after-steps 1 \
      --validation-examples 4 --eval-batch-size 4 --log-every 1
    ;;
  run)
    # Runner resumes ONLY an identical contract from this isolated directory.
    "${train[@]}" --run-dir "$run"
    "${evaluate[@]}"
    ;;
  evaluate) "${evaluate[@]}" ;;
esac
