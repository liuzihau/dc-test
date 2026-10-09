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
run_root="${DCACHE_AUTHOR_ZEBRA_RUN:-$repo_root/outputs/reasoning/author-zebra/diffu-full-mini-zebra-tat-3ep-2x3090}"
runtime_root="${DCACHE_AUTHOR_ZEBRA_RUNTIME:-$repo_root/.cache/runtime/author-zebra-mdm-3ep-2x3090}"
log_file="${DCACHE_AUTHOR_ZEBRA_LOG:-$repo_root/logs/author-zebra-mdm-3ep-2x3090.log}"
tmux_tmpdir="${DCACHE_AUTHOR_ZEBRA_TMUX_TMPDIR:-$repo_root/.tmux}"
tmux_session="${DCACHE_AUTHOR_ZEBRA_TMUX_SESSION:-author-zebra-mdlm-3ep}"
python_bin="${DCACHE_PYTHON:-/home/tliu0205/miniconda3/envs/dcache/bin/python}"
resume_ckpt="${DCACHE_AUTHOR_ZEBRA_RESUME_CKPT:-$run_root/checkpoints/last.ckpt}"
disable_cudagraphs="${DCACHE_AUTHOR_ZEBRA_DISABLE_CUDAGRAPHS:-1}"
physical_gpus="${DCACHE_AUTHOR_ZEBRA_GPUS:-2,3}"
devices="${DCACHE_AUTHOR_ZEBRA_DEVICES:-2}"
microbatch="${DCACHE_AUTHOR_ZEBRA_MICROBATCH:-256}"
workers="${DCACHE_AUTHOR_ZEBRA_WORKERS:-4}"
target_steps="${DCACHE_AUTHOR_ZEBRA_TARGET_STEPS:-8790}"
validation_interval="${DCACHE_AUTHOR_ZEBRA_VALIDATION_INTERVAL:-2930}"
checkpoint_interval="${DCACHE_AUTHOR_ZEBRA_CHECKPOINT_INTERVAL:-2930}"
checkpoint_save_top_k="${DCACHE_AUTHOR_ZEBRA_CHECKPOINT_SAVE_TOP_K:--1}"
global_batch=512
epochs=3

export CUDA_VISIBLE_DEVICES="$physical_gpus"
export NCCL_P2P_DISABLE=1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
# Keep Inductor-compiled flex attention but avoid a torch 2.7.1 CUDA-graph
# allocator failure observed during long validation on RTX 3090s.
unset TORCH_COMPILE_DISABLE
export DCACHE_AUTHOR_DISABLE_CUDAGRAPHS="$disable_cudagraphs"
# Python multiprocessing creates AF_UNIX sockets below TMPDIR. Keep this path
# deliberately short (the kernel limit is 108 bytes) while still avoiding the
# host's space-constrained /tmp filesystem.
export TMPDIR="${DCACHE_AUTHOR_ZEBRA_TMPDIR:-$repo_root/.tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export MPLCONFIGDIR="$runtime_root/matplotlib"
export TRITON_CACHE_DIR="$runtime_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$runtime_root/torchinductor"
export CUDA_CACHE_PATH="$runtime_root/cuda"
export HF_HOME="$runtime_root/huggingface"
export PYTHONPATH="$repo_root:$author_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$cache_root" "$run_root" "$runtime_root" "$TMPDIR" \
  "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$HF_HOME" "$tmux_tmpdir" "$(dirname "$log_file")"
chmod 700 "$tmux_tmpdir"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

check_inputs() {
  [ -x "$python_bin" ] || die "Python executable not found: $python_bin"
  [ -f "$author_root/main.py" ] || die "Vendored author code missing: $author_root/main.py"
  [ -f "$author_entrypoint" ] || die "Local author entrypoint missing: $author_entrypoint"
  [ -f "$author_root/LICENSE" ] || die "Vendored upstream license missing"
  [ -f "$train_pickle" ] || die "Raw Zebra train pickle missing: $train_pickle"
  [ -f "$valid_pickle" ] || die "Raw Zebra test pickle missing: $valid_pickle"
  [ "$devices" -eq 2 ] || die "This launcher is contracted for exactly two visible GPUs"
  [ $((global_batch % (devices * microbatch))) -eq 0 ] || \
    die "Global batch $global_batch is not divisible by devices*microbatch = $((devices * microbatch))"
}

print_contract() {
  local accumulation=$((global_batch / (devices * microbatch)))
  printf '%s\n' \
    'Author-faithful Zebra MDM training contract' \
    "  upstream run: diffu-full-mini-zebra-tat" \
    "  source tree: $author_root" \
    "  train pickle: $train_pickle" \
    "  test pickle: $valid_pickle" \
    '  serialization: [BOS] clues house-indices [SEP] flattened-solution [EOS], PAD to 384' \
    '  corruption: continuous-time Bernoulli masking with antithetic t sampling' \
    '  objective: ELBO, train_on_all_tokens=True' \
    '  attention: full bidirectional MDM attention (not SIDM/SCDM)' \
    '  ordering: author difflm legacy default (effective clean/masked shuffle=True; original positions retained)' \
    '  model: mini, hidden=512, blocks=6, heads=8, dropout=0.1, untied output' \
    '  optimizer: AdamW, lr=3e-4, betas=(0.9,0.999), wd=0, grad clip=1' \
    '  schedule: 2500-step linear warmup, then constant learning rate' \
    "  epochs: $epochs" \
    "  GPUs: physical $physical_gpus; Lightning devices=$devices" \
    "  global batch: $global_batch = $devices x $microbatch x accumulation $accumulation" \
    "  dataloader workers: $workers per rank" \
    "  validation cadence: every $validation_interval optimizer updates" \
    '  validation RNG: restored afterward so monitoring does not alter later training noise' \
    '  local metrics: per-update train loss/LR plus per-epoch validation NLL/PPL/BPD' \
    "  compiled flex attention: yes; CUDA graphs disabled: $disable_cudagraphs" \
    "  checkpoint cadence: every $checkpoint_interval updates; save_top_k=$checkpoint_save_top_k" \
    "  requested stopping step: $target_steps" \
    "  cache: $cache_root" \
    "  run: $run_root" \
    "  log: $log_file" \
    "  tmux: TMUX_TMPDIR=$tmux_tmpdir session=$tmux_session"
}

prepare_data() {
  check_inputs
  AUTHOR_ROOT="$author_root" TRAIN_PICKLE="$train_pickle" VALID_PICKLE="$valid_pickle" \
    CACHE_ROOT="$cache_root" "$python_bin" - <<'PY'
import os
import sys
from pathlib import Path

root = Path(os.environ['AUTHOR_ROOT'])
sys.path.insert(0, str(root))
from synthetic_data.zebra.data import generate_synthetic_data

cache = Path(os.environ['CACHE_ROOT'])
cache.mkdir(parents=True, exist_ok=True)
for split, env_name in [('train', 'TRAIN_PICKLE'), ('test', 'VALID_PICKLE')]:
    path = Path(os.environ[env_name])
    result = generate_synthetic_data(
        None,
        384,
        config={
            'name': 'zebra',
            'data_path': str(path),
            'cache_dir': str(cache),
            'split': split,
            'vocab_size': 23,
        },
    )
    # Exact row counts in the released pickle files (the paper rounds the
    # training count to 1.5M).
    expected = (1499933, 384) if split == 'train' else (100000, 384)
    if tuple(result['input_ids'].shape) != expected:
        raise RuntimeError(
            f'{split} shape {tuple(result["input_ids"].shape)} != author contract {expected}')
    if tuple(result['loss_mask'].shape) != expected:
        raise RuntimeError(f'{split} loss-mask shape mismatch')
    print(f'PREPARED {split}: {expected[0]} rows x {expected[1]} tokens')
PY
}

common_command() {
  local output_root="$1"
  shift
  cd "$author_root"
  exec "$python_bin" -u "$author_entrypoint" \
    --config-name=experiment_base \
    data=zebra \
    seed=1 \
    loader.batch_size="$microbatch" \
    loader.eval_batch_size="$microbatch" \
    loader.global_batch_size="$global_batch" \
    loader.eval_global_batch_size="$global_batch" \
    loader.num_workers="$workers" \
    wandb.name=diffu-full-mini-zebra-tat-3ep-2x3090 \
    wandb.project=Diffusion-zebra \
    algo=difflm \
    model=mini \
    model.length=384 \
    data.cache_dir="$cache_root" \
    hydra.run.dir="$output_root/run" \
    checkpointing.save_dir="$output_root" \
    checkpointing.resume_from_ckpt=false \
    trainer.val_check_interval="$validation_interval" \
    trainer.log_every_n_steps=1 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps="$checkpoint_interval" \
    callbacks.checkpoint_every_n_steps.save_top_k="$checkpoint_save_top_k" \
    +callbacks.local_metrics._target_=reasoning.author_local_metrics.AuthorLocalMetricsCallback \
    +callbacks.local_metrics.output_dir="$output_root/local_metrics" \
    +callbacks.local_metrics.train_every_n_steps=1 \
    +callbacks.local_metrics.preserve_rng_around_validation=true \
    wandb.id=null \
    +algo.log_position_losses=true \
    eval.run_task_evaluation=false \
    sampling=synthetic_base \
    sampling.num_sample_batches=1 \
    sampling.steps=384 \
    sampling.greedy_tokens=true \
    sampling.unmask_policy=topp \
    trainer.max_epochs="$epochs" \
    algo.ar_noise=false \
    algo.next_token_prediction=false \
    algo.loss_type=elbo \
    algo.diffusion_attn_mode=full \
    training.train_on_all_tokens=true \
    data.train_data_path="$train_pickle" \
    data.valid_data_path="$valid_pickle" \
    trainer.devices="$devices" \
    "$@"
}

run_smoke() {
  check_inputs
  prepare_data
  local smoke_root="$runtime_root/smoke-local-metrics"
  mkdir -p "$smoke_root"
  common_command "$smoke_root" \
    trainer.max_epochs=1 \
    +trainer.max_steps=1 \
    trainer.limit_train_batches=1 \
    trainer.limit_val_batches=1 \
    trainer.num_sanity_val_steps=0 \
    trainer.val_check_interval=1 \
    trainer.log_every_n_steps=1 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=1 \
    eval.run_task_evaluation=false
}

run_training() {
  check_inputs
  prepare_data
  if [ -e "$run_root/checkpoints/last.ckpt" ] && [ "${DCACHE_AUTHOR_ZEBRA_OVERWRITE:-0}" != 1 ]; then
    die "Run already has last.ckpt; set DCACHE_AUTHOR_ZEBRA_OVERWRITE=1 only if replacement is intentional"
  fi
  common_command "$run_root"
}

resume_training() {
  check_inputs
  prepare_data
  [ -f "$resume_ckpt" ] || die "Resume checkpoint does not exist: $resume_ckpt"
  printf 'Resuming complete Lightning state from: %s\n' "$resume_ckpt"
  printf 'Stopping at absolute optimizer step: %s\n' "$target_steps"
  common_command "$run_root" \
    checkpointing.resume_from_ckpt=true \
    checkpointing.resume_ckpt_path="$resume_ckpt" \
    trainer.max_epochs=100 \
    +trainer.max_steps="$target_steps"
}

validate_checkpoint() {
  check_inputs
  prepare_data
  [ -f "$resume_ckpt" ] || die "Validation checkpoint does not exist: $resume_ckpt"
  printf 'Validating checkpoint at optimizer step 2930: %s\n' "$resume_ckpt"
  common_command "$run_root" \
    mode=ppl_eval \
    eval.checkpoint_path="$resume_ckpt" \
    +callbacks.local_metrics.validation_step_override=2930
}

launch_action() {
  local child_action="$1"
  check_inputs
  command -v tmux >/dev/null || die 'tmux is required for a persistent launch'
  if TMUX_TMPDIR="$tmux_tmpdir" tmux has-session -t "$tmux_session" 2>/dev/null; then
    die "tmux session already exists: $tmux_session"
  fi
  local launch_command
  launch_command="cd '$repo_root' && exec bash '$0' '$child_action' >'$log_file' 2>&1"
  TMUX_TMPDIR="$tmux_tmpdir" tmux new-session -d -s "$tmux_session" "$launch_command"
  printf 'Started author Zebra MDLM %s in tmux session %s\nLog: %s\n' \
    "$child_action" "$tmux_session" "$log_file"
}

case "$action" in
  check)
    check_inputs
    print_contract
    "$python_bin" - <<'PY'
import datasets, hydra, lightning, numpy, torch, transformers
print('Runtime packages:')
for module in (torch, lightning, datasets, transformers, hydra, numpy):
    print(f'  {module.__name__}={getattr(module, "__version__", "unknown")}')
print('CHECK PASS')
PY
    ;;
  prepare)
    print_contract
    prepare_data
    ;;
  smoke)
    print_contract
    run_smoke
    ;;
  run)
    print_contract
    run_training
    ;;
  resume)
    print_contract
    resume_training
    ;;
  validate-checkpoint)
    print_contract
    validate_checkpoint
    ;;
  launch)
    launch_action run
    ;;
  launch-resume)
    [ -f "$resume_ckpt" ] || die "Resume checkpoint does not exist: $resume_ckpt"
    launch_action resume
    ;;
  status)
    print_contract
    if TMUX_TMPDIR="$tmux_tmpdir" tmux has-session -t "$tmux_session" 2>/dev/null; then
      printf 'STATUS: RUNNING (tmux session %s)\n' "$tmux_session"
      printf 'ATTACH: TMUX_TMPDIR=%q tmux attach -t %q\n' \
        "$tmux_tmpdir" "$tmux_session"
    else
      printf 'STATUS: NOT RUNNING\n'
    fi
    [ ! -f "$log_file" ] || tail -n 40 "$log_file"
    ;;
  *)
    die "Usage: $0 {check|prepare|smoke|run|resume|validate-checkpoint|launch|launch-resume|status}"
    ;;
esac
