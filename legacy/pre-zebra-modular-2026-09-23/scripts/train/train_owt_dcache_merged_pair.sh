#!/usr/bin/env bash
# Matched five-forward, adjacent-gradient, merged-attention auxiliary off/on pair.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
SELF="$SCRIPT_DIR/train_owt_dcache_merged_pair.sh"

usage() {
  echo 'Usage: bash scripts/train/train_owt_dcache_merged_pair.sh 3090|h100 off|on [train|smoke|tmux] [--attention-policy legacy|current-preserving]'
  echo 'Fixed recipe: merged 2D RoPE, adjacent DCache gradients, detached final state, five states, global batch 512.'
  echo 'Default: microbatch 2, 5000 optimizer updates, validation every 500 updates on 400 examples, latest 3 periodic checkpoints.'
  echo 'Overrides: DCACHE_PYTHON, DCACHE_DATA_DIR, DCACHE_MAX_STEPS, DCACHE_NUM_WORKERS, DCACHE_CPU_THREADS;'
  echo '           DCACHE_PAIR_RUN_DIR, DCACHE_PAIR_MICRO_BATCH, DCACHE_PAIR_CUDA_VISIBLE_DEVICES.'
  echo 'Smoke uses a new isolated directory: one update from scratch and one validation batch; never changes the main run.'
  echo 'Rerun train/tmux to resume this variant from its own last.ckpt. No wall-clock recovery supervisor/onstart integration.'
  echo 'Attention policy defaults to legacy. Current-preserving disables previous-V attenuation and cache-only dropout; remaining settings are unchanged.'
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if (( $# < 2 )); then usage >&2; exit 2; fi
PROFILE="$1"
AUXILIARY="$2"
shift 2
ACTION=train
ATTENTION_POLICY=legacy
ACTION_SEEN=false
POLICY_SEEN=false
while (( $# )); do
  case "$1" in
    train|smoke|tmux)
      if [[ "$ACTION_SEEN" == true ]]; then usage >&2; exit 2; fi
      ACTION="$1"; ACTION_SEEN=true; shift ;;
    --attention-policy)
      if (( $# < 2 )) || [[ "$POLICY_SEEN" == true ]]; then usage >&2; exit 2; fi
      ATTENTION_POLICY="$2"; POLICY_SEEN=true; shift 2 ;;
    *) usage >&2; exit 2 ;;
  esac
done
case "$ATTENTION_POLICY" in
  legacy) export DCACHE_MERGED_POLICY=legacy; POLICY_SUFFIX='' ;;
  current-preserving) export DCACHE_MERGED_POLICY=current_preserving; POLICY_SUFFIX='-current-preserving' ;;
  *) echo '--attention-policy must be legacy or current-preserving.' >&2; exit 2 ;;
esac
case "$PROFILE" in
  3090) PROFILE_DEVICES=2; PROFILE_CUDA='2,3' ;;
  h100) PROFILE_DEVICES=1; PROFILE_CUDA='0' ;;
  *) usage >&2; exit 2 ;;
esac
case "$AUXILIARY" in
  on) export DCACHE_NEIGHBOR_ENABLED=true ;;
  off) export DCACHE_NEIGHBOR_ENABLED=false ;;
  *) usage >&2; exit 2 ;;
esac
case "$ACTION" in train|smoke|tmux) ;; *) usage >&2; exit 2 ;; esac

# Do not inherit another experiment's GPU count, microbatch, trial folder,
# two-forward/disconnected mode, or validation sample budget from an old shell.
export DCACHE_GRADIENT_MODE=adjacent
export DCACHE_RESTORE_DATA_CURSOR=true
export DCACHE_PREFLIGHT_ONLY=0
export DCACHE_RECOVERY_ENABLED=0
export DCACHE_DEVICES="$PROFILE_DEVICES"
export DCACHE_CUDA_VISIBLE_DEVICES="${DCACHE_PAIR_CUDA_VISIBLE_DEVICES:-$PROFILE_CUDA}"
export DCACHE_MICRO_BATCH="${DCACHE_PAIR_MICRO_BATCH:-2}"
export DCACHE_EVAL_MICRO_BATCH=2
export DCACHE_GLOBAL_BATCH=512
export DCACHE_VAL_INTERVAL=500
export DCACHE_VAL_BATCHES="$((400 / (PROFILE_DEVICES * 2)))"
unset DCACHE_EVAL_BATCHES
export DCACHE_SANITY_VAL_STEPS=0
export DCACHE_CHECKPOINT_SAVE_TOP_K=3
export DCACHE_MAX_STEPS="${DCACHE_MAX_STEPS:-5000}"
export DCACHE_NUM_WORKERS="${DCACHE_NUM_WORKERS:-8}"
export DCACHE_CPU_THREADS="${DCACHE_CPU_THREADS:-4}"
export DCACHE_DATA_DIR="${DCACHE_DATA_DIR:-$REPO_DIR/.cache/huggingface}"
export DCACHE_PAIR_RUN_DIR="${DCACHE_PAIR_RUN_DIR:-$REPO_DIR/outputs/owt-dcache-merged-final-state-adjacent-neighbors-${AUXILIARY}-${PROFILE}-5k${POLICY_SUFFIX}}"
if [[ -n "${DCACHE_RUN_DIR:-}" && "$DCACHE_RUN_DIR" != "$DCACHE_PAIR_RUN_DIR" ]]; then
  echo "Ignoring old DCACHE_RUN_DIR=$DCACHE_RUN_DIR; use DCACHE_PAIR_RUN_DIR for this off/on pair." >&2
fi
export DCACHE_RUN_DIR="$DCACHE_PAIR_RUN_DIR"
if [[ ! "$DCACHE_MICRO_BATCH" =~ ^[1-9][0-9]*$ ]] || \
   ((512 % (PROFILE_DEVICES * DCACHE_MICRO_BATCH) != 0)); then
  echo 'DCACHE_PAIR_MICRO_BATCH must be positive and divide 512 with the selected GPU count.' >&2
  exit 2
fi
if [[ ! "$DCACHE_MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo 'DCACHE_MAX_STEPS must be a positive total optimizer-step target.' >&2
  exit 2
fi
if (( DCACHE_MICRO_BATCH != 2 )); then
  echo "Nondefault microbatch $DCACHE_MICRO_BATCH changes identity-reference group size. Use the same value for off and on." >&2
fi

# Resolve the active interpreter before entering tmux, whose server environment
# can be older than this shell. Do not create or choose another conda environment.
if ! PYTHON_PATH="$(command -v "${DCACHE_PYTHON:-python}")" || [[ ! -x "$PYTHON_PATH" ]]; then
  echo 'No executable training Python found. Activate your environment or set DCACHE_PYTHON.' >&2
  exit 2
fi
export DCACHE_PYTHON="$PYTHON_PATH"
if [[ -n "${DCACHE_RESUME_CKPT:-}" || -f "$DCACHE_DATA_DIR/compact_train.json" ]]; then
  echo 'This pair starts a NEW architecture with full prepared data; unset DCACHE_RESUME_CKPT and do not use the compact continuation bundle.' >&2
  exit 2
fi
for dataset in openwebtext-train_train_bs1024_wrapped_specialFalse.dat openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat; do
  if [[ ! -f "$DCACHE_DATA_DIR/$dataset/state.json" || ! -f "$DCACHE_DATA_DIR/$dataset/dataset_info.json" ]]; then
    echo "Prepared dataset metadata missing: $DCACHE_DATA_DIR/$dataset; finish transferring the full dataset first." >&2
    exit 2
  fi
done

export TMPDIR="$REPO_DIR/.cache/runtime/merged-pair/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$TMPDIR"
ulimit -c 0
cd "$REPO_DIR"

echo "Matched merged pair: profile=$PROFILE, neighbor auxiliary=$AUXILIARY, adjacent DCache + detached final state."
echo "Attention policy: $ATTENTION_POLICY (legacy and current-preserving checkpoints must not be interchanged)."
echo "Global batch 512 = $DCACHE_DEVICES GPUs x $DCACHE_MICRO_BATCH examples x $((512 / (DCACHE_DEVICES * DCACHE_MICRO_BATCH))) accumulation."
echo 'Validation: 400 examples, microbatch 2, every 500 optimizer updates (rank-seeded masks still differ across GPU counts).'
echo "Training directory: $DCACHE_RUN_DIR"
echo 'Resume: own compatible last.ckpt only; periodic 500-update saves, latest 3. This is not the old cloud recovery/onstart launcher.'

if [[ "$ACTION" == tmux ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo 'tmux is unavailable. Install it in your environment or use the train action in an existing terminal session.' >&2
    exit 2
  fi
  SOCKET="$REPO_DIR/.cache/tmux/merged-pair.sock"
  if (( ${#SOCKET} > 100 )); then
    echo 'Repository path is too long for the tmux socket; use a shorter persistent checkout path.' >&2
    exit 2
  fi
  mkdir -p "$REPO_DIR/.cache/tmux"
  SESSION="merged-${PROFILE}-${AUXILIARY}${POLICY_SUFFIX}"
  if tmux -S "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION already exists; not replacing it or launching another job."
  else
    # Only intentional runtime settings enter the newly launched shell; fixed
    # recipe controls are re-derived from the explicit profile and off/on args.
    TMUX_COMMAND='env -u DCACHE_RESUME_CKPT -u DCACHE_TRANSFER_MANIFEST -u DCACHE_RUN_DIR -u DCACHE_EVAL_BATCHES'
    for setting in DCACHE_PYTHON DCACHE_DATA_DIR DCACHE_PAIR_RUN_DIR \
      DCACHE_MAX_STEPS DCACHE_NUM_WORKERS DCACHE_CPU_THREADS PATH; do
      printf -v PART ' %q' "$setting=${!setting}"
      TMUX_COMMAND+="$PART"
    done
    printf -v PART ' %q %q' "DCACHE_PAIR_MICRO_BATCH=$DCACHE_MICRO_BATCH" \
      "DCACHE_PAIR_CUDA_VISIBLE_DEVICES=$DCACHE_CUDA_VISIBLE_DEVICES"
    TMUX_COMMAND+="$PART"
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
      printf -v PART ' %q' "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
      TMUX_COMMAND+="$PART"
    fi
    printf -v PART ' %q %q %q %q %q %q %q' /bin/bash "$SELF" "$PROFILE" "$AUXILIARY" train --attention-policy "$ATTENTION_POLICY"
    TMUX_COMMAND+="$PART"
    tmux -S "$SOCKET" new-session -d -s "$SESSION" -c "$REPO_DIR" "$TMUX_COMMAND"
  fi
  printf 'Attach: tmux -S %q attach -t %q\n' "$SOCKET" "$SESSION"
  echo "Logs: $DCACHE_RUN_DIR/launch_logs/"
  exit 0
fi

EXTRA_OVERRIDES=(strategy=ddp trainer.num_nodes=1)
if [[ "$ACTION" == smoke ]]; then
  export DCACHE_RUN_DIR="$(mktemp -d "$REPO_DIR/.cache/runtime/merged-pair/smoke.${PROFILE}.${AUXILIARY}.${ATTENTION_POLICY}.XXXXXXXX")"
  export DCACHE_MAX_STEPS=1 DCACHE_VAL_INTERVAL=1 DCACHE_VAL_BATCHES=1
  export DCACHE_CHECKPOINT_SAVE_TOP_K=1
  EXTRA_OVERRIDES+=(callbacks.checkpoint_every_n_steps.every_n_train_steps=1)
  echo "SMOKE ONLY: fresh 0 -> 1 optimizer update with one validation batch; output $DCACHE_RUN_DIR"
fi
mkdir -p "$DCACHE_RUN_DIR/launch_logs"
if ! command -v flock >/dev/null 2>&1; then
  echo 'flock is required to prevent duplicate writers to this run directory.' >&2
  exit 2
fi
exec 9> "$DCACHE_RUN_DIR/.training.lock"
if ! flock -n 9; then
  echo "Another launch already holds $DCACHE_RUN_DIR/.training.lock; no duplicate training started." >&2
  exit 2
fi
LOG="$DCACHE_RUN_DIR/launch_logs/$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
write_status() {
  printf 'status=%s\nexit_code=%s\npid=%s\nprofile=%s\nauxiliary=%s\nattention_policy=%s\nupdated_utc=%s\nlog=%s\n' \
    "$1" "$2" "$$" "$PROFILE" "$AUXILIARY" "$ATTENTION_POLICY" "$(date -u +%FT%TZ)" "$LOG" \
    > "$DCACHE_RUN_DIR/.launch_status.$$.tmp"
  mv "$DCACHE_RUN_DIR/.launch_status.$$.tmp" "$DCACHE_RUN_DIR/launch_status.txt"
}
trap 'RESULT=$?; write_status EXITED "$RESULT"' EXIT
write_status RUNNING ''
echo "Training log: $LOG"
set +e
bash "$SCRIPT_DIR/train_owt_dcache_merged_neighbors_5k.sh" "${EXTRA_OVERRIDES[@]}" 2>&1 | tee "$LOG"
PIPE_RESULTS=("${PIPESTATUS[@]}")
set -e
RESULT="${PIPE_RESULTS[0]}"
if (( RESULT == 0 && PIPE_RESULTS[1] != 0 )); then RESULT="${PIPE_RESULTS[1]}"; fi
exit "$RESULT"
