#!/usr/bin/env bash
set -euo pipefail

action="${1:-check}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
author_root="$repo_root/third_party/reasoning_with_latent_tokens"
author_entrypoint="$repo_root/scripts/reasoning/author_main.py"
raw_root="${DCACHE_AUTHOR_ZEBRA_RAW_ROOT:-$repo_root/.cache/downloads/reasoning-puzzles/Reasoning puzzles public data}"
train_pickle="${DCACHE_AUTHOR_ZEBRA_TRAIN:-$raw_root/zebra-train-data.pkl}"
valid_pickle="${DCACHE_AUTHOR_ZEBRA_VALID:-$raw_root/zebra-test-data.pkl}"
cache_root="${DCACHE_AUTHOR_ZEBRA_CACHE:-$repo_root/.cache/author-zebra}"
train_root="${DCACHE_AUTHOR_ZEBRA_RUN:-$repo_root/outputs/reasoning/author-zebra/diffu-full-mini-zebra-tat-3ep-2x3090}"
checkpoint="${DCACHE_AUTHOR_ZEBRA_GEN_CKPT:-$train_root/checkpoints/3-8790.ckpt}"
checkpoint_step="${DCACHE_AUTHOR_ZEBRA_GEN_STEP:-8790}"
generation_root="${DCACHE_AUTHOR_ZEBRA_GEN_RUN:-$train_root/generation/step-$checkpoint_step-author-native}"
runtime_root="${DCACHE_AUTHOR_ZEBRA_GEN_RUNTIME:-$repo_root/.cache/runtime/author-zebra-generation}"
log_file="${DCACHE_AUTHOR_ZEBRA_GEN_LOG:-$repo_root/logs/author-zebra-generation-step-$checkpoint_step.log}"
tmux_tmpdir="${DCACHE_AUTHOR_ZEBRA_TMUX_TMPDIR:-$repo_root/.tmux}"
tmux_session="${DCACHE_AUTHOR_ZEBRA_GEN_TMUX_SESSION:-author-zebra-generation-$checkpoint_step}"
python_bin="${DCACHE_PYTHON:-/home/tliu0205/miniconda3/envs/dcache/bin/python}"
physical_gpu="${DCACHE_AUTHOR_ZEBRA_GEN_GPU:-2}"
batch_size="${DCACHE_AUTHOR_ZEBRA_GEN_BATCH_SIZE:-128}"
num_batches="${DCACHE_AUTHOR_ZEBRA_GEN_BATCHES:-10}"
workers="${DCACHE_AUTHOR_ZEBRA_GEN_WORKERS:-4}"
candidate_min="${DCACHE_AUTHOR_ZEBRA_GEN_CANDIDATE_MIN:-8}"
candidate_max="${DCACHE_AUTHOR_ZEBRA_GEN_CANDIDATE_MAX:-8}"

export CUDA_VISIBLE_DEVICES="$physical_gpu"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export DCACHE_AUTHOR_DISABLE_CUDAGRAPHS=1
export TMPDIR="${DCACHE_AUTHOR_ZEBRA_GEN_TMPDIR:-$repo_root/.tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export MPLCONFIGDIR="$runtime_root/matplotlib"
export TRITON_CACHE_DIR="$runtime_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$runtime_root/torchinductor"
export CUDA_CACHE_PATH="$runtime_root/cuda"
export HF_HOME="$runtime_root/huggingface"
export PYTHONPATH="$repo_root:$author_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$generation_root" "$runtime_root" "$TMPDIR" "$MPLCONFIGDIR" \
  "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" \
  "$HF_HOME" "$tmux_tmpdir" "$(dirname "$log_file")"
chmod 700 "$tmux_tmpdir"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

check_inputs() {
  [ -x "$python_bin" ] || die "Python executable not found: $python_bin"
  [ -f "$author_root/main.py" ] || die "Vendored author code missing: $author_root/main.py"
  [ -f "$author_entrypoint" ] || die "Local author entrypoint missing: $author_entrypoint"
  [ -f "$train_pickle" ] || die "Raw Zebra train pickle missing: $train_pickle"
  [ -f "$valid_pickle" ] || die "Raw Zebra test pickle missing: $valid_pickle"
  [ -f "$checkpoint" ] || die "Checkpoint missing: $checkpoint"
  [ "$batch_size" -gt 0 ] || die 'Batch size must be positive'
  [ "$num_batches" -gt 0 ] || die 'Number of batches must be positive'
  [ "$candidate_min" -ge 0 ] || die 'Candidate minimum must be non-negative'
  [ "$candidate_max" -ge 0 ] || die 'Candidate maximum must be non-negative'
}

print_contract() {
  printf '%s\n' \
    'Author-native Zebra generation contract' \
    '  method: standard full-attention MDLM (diffu-full), not SIDM/SCDM' \
    "  checkpoint: $checkpoint" \
    "  physical GPU: $physical_gpu" \
    "  held-out examples: $((batch_size * num_batches)) = $num_batches batches x $batch_size" \
    '  denoising: 384 steps' \
    "  position policy: top probability (topp), candidate min/max $candidate_min/$candidate_max (0/0 = unrestricted)" \
    '  token policy: author paper setting, Gumbel sampling (greedy_tokens=false)' \
    '  weights: EMA enabled' \
    '  metrics: exact puzzle, row, cell, and malformed-output rates' \
    "  output: $generation_root" \
    "  log: $log_file"
}

run_generation() {
  check_inputs
  cd "$author_root"
  "$python_bin" -u "$author_entrypoint" \
    --config-name=experiment_base \
    data=zebra \
    seed=1 \
    loader.batch_size="$batch_size" \
    loader.eval_batch_size="$batch_size" \
    loader.global_batch_size="$batch_size" \
    loader.eval_global_batch_size="$batch_size" \
    loader.num_workers="$workers" \
    wandb.name=diffu-full-mini-zebra-tat-generation \
    wandb.project=Diffusion-zebra \
    wandb.id=null \
    algo=difflm \
    model=mini \
    model.length=384 \
    data.cache_dir="$cache_root" \
    data.train_data_path="$train_pickle" \
    data.valid_data_path="$valid_pickle" \
    hydra.run.dir="$generation_root/run" \
    checkpointing.save_dir="$generation_root" \
    checkpointing.resume_from_ckpt=false \
    trainer.devices=1 \
    mode=completions \
    eval.generate_samples=true \
    eval.disable_ema=false \
    eval.checkpoint_path="$checkpoint" \
    eval.generated_samples_path="$generation_root/samples.json" \
    sampling=synthetic_base \
    sampling.steps=384 \
    sampling.num_sample_batches="$num_batches" \
    sampling.kv_cache=false \
    sampling.greedy_tokens=false \
    sampling.unmask_policy=topp \
    +sampling.topk_candidate_min="$candidate_min" \
    +sampling.topk_candidate_max="$candidate_max" \
    algo.ar_noise=false \
    algo.next_token_prediction=false \
    algo.loss_type=elbo \
    algo.diffusion_attn_mode=full \
    algo.diffusion_shuffle=true \
    training.train_on_all_tokens=true
}

latest_samples() {
  find "$generation_root" -maxdepth 1 -type f -name 'samples_*.json' -printf '%T@ %p\n' \
    | sort -n | tail -n 1 | cut -d' ' -f2-
}

summarize() {
  local samples
  samples="$(latest_samples)"
  [ -n "$samples" ] || die "No generated sample file under $generation_root"
  "$python_bin" - "$samples" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding='utf-8') as stream:
  payload = json.load(stream)
metrics = payload.get('eval_metrics', {})
print(f'Samples: {path}')
for key in (
    'n_total_puzzles', 'n_correct_puzzles', 'puzzle_accuracy',
    'mean_row_accuracy', 'mean_cell_accuracy', 'n_malformed',
    'malformed_rate', 'n_wrong_shape', 'wrong_shape_rate',
    'n_incorrect', 'incorrect_rate'):
  if key in metrics:
    print(f'{key}: {metrics[key]}')
print(f"time_per_batch_seconds: {payload.get('time_per_batch')}")
PY
}

launch() {
  check_inputs
  command -v tmux >/dev/null || die 'tmux is required for a persistent launch'
  if TMUX_TMPDIR="$tmux_tmpdir" tmux has-session -t "$tmux_session" 2>/dev/null; then
    die "tmux session already exists: $tmux_session"
  fi
  local launch_command
  launch_command="cd '$repo_root' && exec bash '$0' run >'$log_file' 2>&1"
  TMUX_TMPDIR="$tmux_tmpdir" tmux new-session -d -s "$tmux_session" "$launch_command"
  printf 'Started author-native Zebra generation in tmux %s\nLog: %s\n' \
    "$tmux_session" "$log_file"
}

case "$action" in
  check)
    check_inputs
    print_contract
    ;;
  run)
    print_contract
    run_generation
    summarize
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
    [ ! -f "$log_file" ] || tail -n 50 "$log_file"
    ;;
  summarize)
    summarize
    ;;
  *)
    die "Usage: $0 {check|run|launch|status|summarize}"
    ;;
esac
