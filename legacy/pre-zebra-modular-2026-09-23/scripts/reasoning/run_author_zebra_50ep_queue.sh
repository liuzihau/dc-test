#!/usr/bin/env bash
set -euo pipefail

action="${1:-check}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
author_root="$repo_root/third_party/reasoning_with_latent_tokens"
run_root="${DCACHE_AUTHOR_ZEBRA_RUN:-$repo_root/outputs/reasoning/author-zebra/diffu-full-mini-zebra-tat-3ep-2x3090}"
checkpoint_dir="$run_root/checkpoints"
generation_root="$run_root/generation"
python_bin="${DCACHE_PYTHON:-/home/tliu0205/miniconda3/envs/dcache/bin/python}"
train_launcher="$repo_root/scripts/reasoning/run_author_zebra_3ep_2x3090.sh"
generation_launcher="$repo_root/scripts/reasoning/run_author_zebra_generation.sh"
selector="$repo_root/scripts/reasoning/select_author_zebra_checkpoint.py"
history_script="$repo_root/scripts/reasoning/summarize_author_zebra_generation_history.py"
queue_log="${DCACHE_AUTHOR_ZEBRA_50EP_LOG:-$repo_root/logs/author-zebra-50ep-queue.log}"
tmux_tmpdir="${DCACHE_AUTHOR_ZEBRA_TMUX_TMPDIR:-$repo_root/.tmux}"
tmux_session="${DCACHE_AUTHOR_ZEBRA_50EP_SESSION:-author-zebra-mdlm-50ep}"
steps_per_epoch=2930
final_epoch=50
final_step=$((steps_per_epoch * final_epoch))
milestones=(6 9 12 15 18 21 24 27 30 33 36 39 42 45 48 50)

mkdir -p "$generation_root" "$tmux_tmpdir" "$(dirname "$queue_log")"
chmod 700 "$tmux_tmpdir"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

check_inputs() {
  [ -x "$python_bin" ] || die "Python executable not found: $python_bin"
  [ -x "$train_launcher" ] || die "Training launcher missing/not executable: $train_launcher"
  [ -x "$generation_launcher" ] || die "Generation launcher missing/not executable: $generation_launcher"
  [ -f "$selector" ] || die "Checkpoint selector missing: $selector"
  [ -d "$checkpoint_dir" ] || die "Checkpoint directory missing: $checkpoint_dir"
}

select_checkpoint() {
  "$python_bin" "$selector" "$checkpoint_dir" --author-root "$author_root"
}

checkpoint_step() {
  local checkpoint="$1"
  "$python_bin" "$selector" "$checkpoint_dir" --author-root "$author_root" --show-step \
    | cut -f1
}

generation_exists() {
  local step="$1"
  find "$generation_root/step-$step-author-native" -maxdepth 1 \
    -type f -name 'samples_*.json' -print -quit 2>/dev/null | grep -q .
}

refresh_history() {
  "$python_bin" "$history_script" "$generation_root"
}

print_contract() {
  local checkpoint step
  checkpoint="$(select_checkpoint)"
  step="$(checkpoint_step "$checkpoint")"
  printf '%s\n' \
    'Author Zebra 50-epoch continuation queue' \
    "  current checkpoint: $checkpoint" \
    "  current optimizer step: $step (logical epoch $(awk -v s="$step" -v n="$steps_per_epoch" 'BEGIN {printf "%.3f", s/n}'))" \
    "  final optimizer step: $final_step (epoch $final_epoch)" \
    '  training GPUs: physical 2,3' \
    '  validation: full held-out loss once per epoch' \
    '  recovery checkpoints: once per epoch; retain all (about 15 GiB total)' \
    '  generation: author-native 1,280-puzzle evaluation after epochs 3,6,...,48,50' \
    "  queue log: $queue_log" \
    "  generation history: $generation_root/author_native_history.csv" \
    "  live accuracy plot: $generation_root/author_native_accuracy_vs_epoch.png"
}

run_queue() {
  check_inputs
  refresh_history

  local epoch target checkpoint step
  for epoch in "${milestones[@]}"; do
    target=$((epoch * steps_per_epoch))
    checkpoint="$(select_checkpoint)"
    step="$(checkpoint_step "$checkpoint")"

    if [ "$step" -lt "$target" ]; then
      printf '\n[%s] TRAIN logical epoch %s: step %s -> %s\n' \
        "$(date --iso-8601=seconds)" "$epoch" "$step" "$target"
      DCACHE_AUTHOR_ZEBRA_RESUME_CKPT="$checkpoint" \
      DCACHE_AUTHOR_ZEBRA_TARGET_STEPS="$target" \
      DCACHE_AUTHOR_ZEBRA_VALIDATION_INTERVAL="$steps_per_epoch" \
      DCACHE_AUTHOR_ZEBRA_CHECKPOINT_INTERVAL="$steps_per_epoch" \
      DCACHE_AUTHOR_ZEBRA_CHECKPOINT_SAVE_TOP_K=-1 \
        bash "$train_launcher" resume
      checkpoint="$(select_checkpoint)"
      step="$(checkpoint_step "$checkpoint")"
      [ "$step" -eq "$target" ] || \
        die "Training segment ended at step $step, expected $target"
    else
      printf '[%s] SKIP training target %s; current step is %s\n' \
        "$(date --iso-8601=seconds)" "$target" "$step"
    fi

    if generation_exists "$target"; then
      printf '[%s] SKIP generation at step %s; result already exists\n' \
        "$(date --iso-8601=seconds)" "$target"
    else
      printf '[%s] GENERATE logical epoch %s at step %s\n' \
        "$(date --iso-8601=seconds)" "$epoch" "$target"
      DCACHE_AUTHOR_ZEBRA_GEN_CKPT="$checkpoint" \
      DCACHE_AUTHOR_ZEBRA_GEN_STEP="$target" \
      DCACHE_AUTHOR_ZEBRA_GEN_GPU=2 \
      DCACHE_AUTHOR_ZEBRA_GEN_BATCH_SIZE=128 \
      DCACHE_AUTHOR_ZEBRA_GEN_BATCHES=10 \
        bash "$generation_launcher" run
    fi
    refresh_history
  done

  printf '[%s] COMPLETE: reached epoch %s / optimizer step %s\n' \
    "$(date --iso-8601=seconds)" "$final_epoch" "$final_step"
}

launch() {
  check_inputs
  command -v tmux >/dev/null || die 'tmux is required for a persistent launch'
  if TMUX_TMPDIR="$tmux_tmpdir" tmux has-session -t "$tmux_session" 2>/dev/null; then
    die "tmux session already exists: $tmux_session"
  fi
  local command
  command="cd '$repo_root' && exec bash '$0' run >>'$queue_log' 2>&1"
  TMUX_TMPDIR="$tmux_tmpdir" tmux new-session -d -s "$tmux_session" "$command"
  printf 'Started 50-epoch queue in tmux %s\nLog: %s\n' "$tmux_session" "$queue_log"
}

case "$action" in
  check)
    check_inputs
    print_contract
    ;;
  run)
    print_contract
    run_queue
    ;;
  launch)
    launch
    ;;
  status)
    print_contract
    if TMUX_TMPDIR="$tmux_tmpdir" tmux has-session -t "$tmux_session" 2>/dev/null; then
      printf 'STATUS: RUNNING\n'
      printf 'ATTACH: TMUX_TMPDIR=%q tmux attach -t %q\n' "$tmux_tmpdir" "$tmux_session"
    else
      printf 'STATUS: NOT RUNNING\n'
    fi
    [ ! -f "$queue_log" ] || tail -n 60 "$queue_log"
    ;;
  *)
    die "Usage: $0 {check|run|launch|status}"
    ;;
esac
