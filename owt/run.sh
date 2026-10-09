#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
# The user explicitly reserves physical GPUs 2 and 3 for this project.
export CUDA_VISIBLE_DEVICES=2,3
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Resume only checkpoints produced locally by this repository.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$repo_root/.tmp" TMP="$repo_root/.tmp" TEMP="$repo_root/.tmp"
export TMUX_TMPDIR="$repo_root/.tmux"
export MPLCONFIGDIR="$repo_root/.cache/runtime/owt/matplotlib"
export TRITON_CACHE_DIR="$repo_root/.cache/runtime/owt/triton"
export TORCHINDUCTOR_CACHE_DIR="$repo_root/.cache/runtime/owt/inductor"
export CUDA_CACHE_PATH="$repo_root/.cache/runtime/owt/cuda"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$TMUX_TMPDIR" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" logs
python_bin="${DCACHE_PYTHON:-/home/tliu0205/miniconda3/envs/dcache/bin/python}"
action="${1:-status}"
shift || true
case "$action" in
  launch|launch-zero-init|launch-low-weight|launch-research-monitor|launch-reveal-sweep|launch-post-diagnostics)
    session=owt-mdm-np-5k
    queue_action=run
    log_name=owt-mdm-np-5k
    if [[ "$action" == launch-zero-init ]]; then
      session=owt-np-zero-init-5k
      queue_action=followup-zero
      log_name=owt-np-zero-init-5k
    elif [[ "$action" == launch-low-weight ]]; then
      session=owt-np-low-weight-5k
      queue_action=followup-low-weight
      log_name=owt-np-low-weight-5k
    elif [[ "$action" == launch-research-monitor ]]; then
      session=owt-research-monitor
      queue_action=research-monitor
      log_name=owt-research-monitor
    elif [[ "$action" == launch-reveal-sweep ]]; then
      session=owt-reveal-sweep
      queue_action=reveal-sweep
      log_name=owt-reveal-sweep
    elif [[ "$action" == launch-post-diagnostics ]]; then
      session=owt-post-diagnostics
      queue_action=post-diagnostics
      log_name=owt-post-diagnostics
    fi
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "Already running: $session" >&2
      exit 1
    fi
    printf -v command '%q ' bash "$repo_root/owt/run.sh" "$queue_action" "$@"
    tmux new-session -d -s "$session" "$command >> '$repo_root/logs/$log_name.log' 2>&1"
    # Keep an attachable live view even though each child preserves its own log.
    if [[ "$action" == launch-zero-init || "$action" == launch-low-weight ]]; then
      variant=mdm_np_zero_init
      if [[ "$action" == launch-low-weight ]]; then variant=mdm_np_zero_init_low_weight; fi
      tmux new-window -t "$session" -n monitor \
        "tail -n 15 -F '$repo_root/logs/$log_name.log' '$repo_root/outputs/owt/mdm-np-5k/$variant/train.log' 2>/dev/null"
    elif [[ "$action" == launch ]]; then
      tmux new-window -t "$session" -n monitor \
        "tail -n 15 -F '$repo_root/outputs/owt/mdm-np-5k/mdm/train.log' '$repo_root/outputs/owt/mdm-np-5k/mdm_np/train.log' 2>/dev/null"
    fi
    echo "Started $session; training restricted to physical GPUs 2 and 3"
    ;;
  run|smoke|followup-zero|followup-low-weight) exec "$python_bin" -m owt.schedule "$action" "$@" ;;
  research-monitor) exec "$python_bin" -m owt.research watch "$@" ;;
  reveal-sweep) exec "$python_bin" -m owt.reveal_schedule "$@" ;;
  post-diagnostics) exec "$python_bin" -m owt.post_schedule "$@" ;;
  research-snapshot) exec "$python_bin" -m owt.research snapshot "$@" ;;
  plot|status) exec "$python_bin" -m owt.report "$@" ;;
  check) exec "$python_bin" -m owt.check_model "$@" ;;
  *) echo 'Usage: bash owt/run.sh {check|smoke|launch|launch-zero-init|launch-low-weight|launch-research-monitor|research-snapshot|run|status|plot}' >&2; exit 2 ;;
esac
