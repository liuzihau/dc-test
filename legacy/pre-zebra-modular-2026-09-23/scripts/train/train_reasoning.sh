#!/usr/bin/env bash
# Isolated reasoning tasks. No changes to OpenWebText data, checkpoints or jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
usage() {
  echo 'Usage: bash scripts/train/train_reasoning.sh cpu|3090|h100 sudoku|zebra|countdown vanilla|mdm|mdm_aux|final|dcache|both|both_aux prepare|smoke|train|tmux|evaluate|plot'
  echo 'Default: isolated pilot data; matched five-state controls (vanilla is one-state).'
  echo 'CUDA defaults: 3090 uses 2,3; H100 uses 0. No job starts without an explicit action.'
  echo 'REASONING_DATA_DIR, REASONING_RUN_DIR, REASONING_GPU_IDS, REASONING_MICRO_BATCH,'
  echo 'REASONING_GLOBAL_BATCH, REASONING_MAX_STEPS, REASONING_SIZE, REASONING_SEED,'
  echo 'REASONING_NO_ROBUSTNESS=1, REASONING_GRADIENT_MODE=detached, DCACHE_PYTHON.'
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if (( $# != 4 )); then usage >&2; exit 2; fi
PROFILE="$1" TASK="$2" VARIANT="$3" ACTION="$4"
case "$PROFILE" in
  cpu) DEVICES=1; DEVICE=cpu; DEFAULT_GPU=''; DEFAULT_SIZE=debug; PRECISION=fp32 ;;
  3090) DEVICES=2; DEVICE=cuda; DEFAULT_GPU='2,3'; DEFAULT_SIZE=''; PRECISION=bf16 ;;
  h100) DEVICES=1; DEVICE=cuda; DEFAULT_GPU='0'; DEFAULT_SIZE=''; PRECISION=bf16 ;;
  *) usage >&2; exit 2 ;;
esac
case "$TASK" in sudoku|zebra|countdown) ;; *) usage >&2; exit 2 ;; esac
case "$VARIANT" in vanilla|mdm|mdm_aux|final|dcache|both|both_aux) ;; *) usage >&2; exit 2 ;; esac
case "$ACTION" in prepare|smoke|train|tmux|evaluate|plot) ;; *) usage >&2; exit 2 ;; esac
PYTHON_BIN="$(command -v "${DCACHE_PYTHON:-python}")"
export CUDA_VISIBLE_DEVICES="${REASONING_GPU_IDS:-$DEFAULT_GPU}"
if [[ "$DEVICE" == cuda ]]; then
  IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
  if (( ${#GPU_LIST[@]} != DEVICES )); then
    echo "Profile $PROFILE needs $DEVICES visible GPU IDs, got $CUDA_VISIBLE_DEVICES" >&2
    exit 2
  fi
  for ((i=0; i<DEVICES; i++)); do
    if [[ -z "${GPU_LIST[i]}" ]]; then echo 'Empty GPU ID' >&2; exit 2; fi
    for ((j=0; j<i; j++)); do
      if [[ "${GPU_LIST[i]}" == "${GPU_LIST[j]}" ]]; then echo 'Repeated GPU ID' >&2; exit 2; fi
    done
  done
fi
export TMPDIR="$REPO_DIR/.cache/runtime/reasoning/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export MPLCONFIGDIR="$REPO_DIR/.cache/runtime/reasoning/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$REPO_DIR/.cache/runtime/reasoning/inductor"
export TRITON_CACHE_DIR="$REPO_DIR/.cache/runtime/reasoning/triton"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
ulimit -c 0
cd "$REPO_DIR"
CLI="$REPO_DIR/scripts/reasoning/run_reasoning.py"
SEED="${REASONING_SEED:-1}"
DATA_DIR="${REASONING_DATA_DIR:-$REPO_DIR/.cache/reasoning/${TASK}-pilot-v1}"
RUN_DIR="${REASONING_RUN_DIR:-$REPO_DIR/outputs/reasoning/${TASK}/${VARIANT}-${PROFILE}-seed${SEED}}"
SIZE="${REASONING_SIZE:-$DEFAULT_SIZE}"
GLOBAL_BATCH="${REASONING_GLOBAL_BATCH:-128}"
MICRO_BATCH="${REASONING_MICRO_BATCH:-8}"
MAX_STEPS="${REASONING_MAX_STEPS:-5000}"

if [[ "$ACTION" == prepare ]]; then
  exec "$PYTHON_BIN" "$CLI" prepare --task "$TASK" --output "$DATA_DIR" \
    --train-size "${REASONING_TRAIN_EXAMPLES:-1000}" \
    --valid-size "${REASONING_VALID_EXAMPLES:-100}" \
    --test-size "${REASONING_TEST_EXAMPLES:-100}" --seed 17
fi
if [[ "$ACTION" == tmux ]]; then
  SOCKET="$REPO_DIR/.cache/tmux/reasoning.sock"
  SESSION="reasoning-${TASK}-${VARIANT}-${PROFILE}-s${SEED}"
  mkdir -p "$REPO_DIR/.cache/tmux"
  if (( ${#SOCKET} > 100 )); then echo 'tmux socket path too long; use a shorter checkout path.' >&2; exit 2; fi
  if ! tmux -S "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
    COMMAND=env
    for SETTING in "DCACHE_PYTHON=$PYTHON_BIN" "REASONING_DATA_DIR=$DATA_DIR" \
      "REASONING_RUN_DIR=$RUN_DIR" "REASONING_GPU_IDS=$CUDA_VISIBLE_DEVICES" \
      "REASONING_MICRO_BATCH=$MICRO_BATCH" "REASONING_GLOBAL_BATCH=$GLOBAL_BATCH" \
      "REASONING_MAX_STEPS=$MAX_STEPS" "REASONING_SIZE=$SIZE" "REASONING_SEED=$SEED" \
      "REASONING_GRADIENT_MODE=${REASONING_GRADIENT_MODE:-adjacent}" \
      "REASONING_NO_ROBUSTNESS=${REASONING_NO_ROBUSTNESS:-0}" "PATH=$PATH"; do
      printf -v PART ' %q' "$SETTING"; COMMAND+="$PART"
    done
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
      printf -v PART ' %q' "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"; COMMAND+="$PART"
    fi
    printf -v PART ' %q %q %q %q %q %q' /bin/bash "$SCRIPT_DIR/train_reasoning.sh" "$PROFILE" "$TASK" "$VARIANT" train
    COMMAND+="$PART"
    tmux -S "$SOCKET" new-session -d -s "$SESSION" -c "$REPO_DIR" "$COMMAND"
  fi
  printf 'Attach: tmux -S %q attach -t %q\n' "$SOCKET" "$SESSION"
  exit 0
fi
if [[ "$ACTION" == plot ]]; then
  exec "$PYTHON_BIN" "$CLI" plot --runs "$RUN_DIR" \
    --output "$RUN_DIR/training.png"
fi
if [[ "$ACTION" == evaluate ]]; then
  exec "$PYTHON_BIN" "$CLI" evaluate --checkpoint "$RUN_DIR/checkpoints/last.pt" \
    --data-dir "$DATA_DIR" --device "$DEVICE" \
    --output "$RUN_DIR/evaluation-$(date -u +%Y%m%dT%H%M%SZ).json"
fi
EXTRA=()
if [[ "$ACTION" == smoke ]]; then
  # Completely separate data and checkpoints; never advances the real trial.
  SMOKE_DIR="$(mktemp -d "$REPO_DIR/.cache/runtime/reasoning/smoke.${TASK}.${VARIANT}.XXXXXXXX")"
  DATA_DIR="$SMOKE_DIR/data"; RUN_DIR="$SMOKE_DIR/run"
  SIZE=debug; MICRO_BATCH=2; GLOBAL_BATCH=$((DEVICES * 2)); MAX_STEPS=2
  "$PYTHON_BIN" "$CLI" prepare --task "$TASK" --output "$DATA_DIR" \
    --train-size 8 --valid-size 4 --test-size 4 --seed 71
  EXTRA+=(--val-every 1 --save-every 1 --log-every 1 --validation-examples 4 --eval-batch-size 2)
  echo "SMOKE ONLY: $SMOKE_DIR"
fi
if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
  echo "Missing prepared task data: $DATA_DIR. Run the prepare action first." >&2
  exit 2
fi
if [[ -n "$SIZE" ]]; then EXTRA+=(--size "$SIZE"); fi
if [[ "${REASONING_NO_ROBUSTNESS:-0}" == 1 ]]; then EXTRA+=(--no-robustness); fi
LAUNCH=("$PYTHON_BIN")
if (( DEVICES > 1 )); then
  LAUNCH+=(-m torch.distributed.run --standalone --nproc-per-node="$DEVICES")
fi
mkdir -p "$RUN_DIR/console"
echo "Task=$TASK variant=$VARIANT device=$DEVICE GPUs=$CUDA_VISIBLE_DEVICES"
echo 'This is a pilot/reproduction, not the unreleased authors benchmark implementation.'
"${LAUNCH[@]}" "$CLI" train --task "$TASK" --variant "$VARIANT" \
  --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" --device "$DEVICE" --precision "$PRECISION" \
  --global-batch "$GLOBAL_BATCH" --micro-batch "$MICRO_BATCH" --max-steps "$MAX_STEPS" \
  --seed "$SEED" --gradient-mode "${REASONING_GRADIENT_MODE:-adjacent}" \
  "${EXTRA[@]}" 2>&1 | tee "$RUN_DIR/console/$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
