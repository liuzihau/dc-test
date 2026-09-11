#!/usr/bin/env bash
# Portable setup / full-state continuation of the connected five-forward trial.
# No cloud API, SSH, upload, billing, or system-package operations are performed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
cd "$REPO_DIR"

usage() {
  echo "Usage: bash scripts/cloud/lightning_h100.sh ACTION [--cpu-only]"
  echo "Actions: setup | check | smoke | train | tmux | validate | plot"
  echo "Default: one visible H100 (GPU 0), microbatch 2, global batch 512, stop at 5000."
  echo "Copy imports/adjacent/0-1500.ckpt and both prepared OWT .dat directories first."
  echo "Setup installs into the active Python environment (CPython 3.9-3.12); no conda create."
  echo "Environment overrides: DCACHE_PYTHON, DCACHE_RESUME_CKPT, DCACHE_DATA_DIR,"
  echo "  DCACHE_RUN_DIR, DCACHE_CUDA_VISIBLE_DEVICES, DCACHE_NUM_WORKERS."
  echo "  DCACHE_MICRO_BATCH=4, 8 or 16 enables a separately labelled trial (default: 2)."
}

ACTION="${1:-help}"
if (( $# > 0 )); then shift; fi
if [[ "$ACTION" == help || "$ACTION" == --help || "$ACTION" == -h ]]; then
  usage
  exit 0
fi
if (( $# > 0 )) && [[ "$ACTION" != check || "$*" != --cpu-only ]]; then
  usage >&2
  exit 2
fi

# Set these BEFORE any package install or Python import. Clone the repository
# onto persistent Studio storage, not /tmp or an ephemeral mount.
export TMPDIR="${REPO_DIR}/.cache/runtime/h100/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export PIP_CACHE_DIR="${REPO_DIR}/.cache/pip"
export CONDA_PKGS_DIRS="${REPO_DIR}/.cache/conda-pkgs"
export XDG_CACHE_HOME="${REPO_DIR}/.cache"
export MPLCONFIGDIR="${REPO_DIR}/.cache/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="${REPO_DIR}/.cache/torchinductor"
export TRITON_CACHE_DIR="${REPO_DIR}/.cache/triton"
export CUDA_CACHE_PATH="${REPO_DIR}/.cache/cuda"
export TORCH_HOME="${REPO_DIR}/.cache/torch"
export NUMBA_CACHE_DIR="${REPO_DIR}/.cache/numba"
export HF_HOME="${REPO_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${DCACHE_CPU_THREADS:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ulimit -c 0
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$MPLCONFIGDIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" \
  "$TORCH_HOME" "$NUMBA_CACHE_DIR" "${REPO_DIR}/logs"

# A Lightning Studio already has its one supported default environment. Use
# that environment for every action, including new shells; never create or
# silently switch to a second environment. tmux receives this path explicitly.
export DCACHE_PYTHON="${DCACHE_PYTHON:-$(command -v python || true)}"
if [[ -z "$DCACHE_PYTHON" || ! -x "$DCACHE_PYTHON" ]]; then
  echo "No executable Python found: ${DCACHE_PYTHON:-python on PATH}." >&2
  echo "Activate the Studio default environment, or set DCACHE_PYTHON to its Python executable." >&2
  exit 2
fi
export DCACHE_CUDA_VISIBLE_DEVICES="${DCACHE_CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="$DCACHE_CUDA_VISIBLE_DEVICES"
export DCACHE_DEVICES=1
# Larger microbatches are explicit trials: they change the identity-reference group.
export DCACHE_MICRO_BATCH="${DCACHE_MICRO_BATCH:-2}"
if [[ "$DCACHE_MICRO_BATCH" != 2 && "$DCACHE_MICRO_BATCH" != 4 && "$DCACHE_MICRO_BATCH" != 8 && "$DCACHE_MICRO_BATCH" != 16 ]]; then
  echo "DCACHE_MICRO_BATCH must be 2 (reference), 4, 8 or 16 (explicit trials)." >&2
  exit 2
fi
export DCACHE_GLOBAL_BATCH=512
export DCACHE_MAX_STEPS="${DCACHE_MAX_STEPS:-5000}"
export DCACHE_NUM_WORKERS="${DCACHE_NUM_WORKERS:-4}"
export DCACHE_VAL_INTERVAL=500
# Original: 2 ranks * 2 examples * 100 batches = 400 examples.
# Validation stays at microbatch 2 / 200 batches for all training microbatches.
export DCACHE_VAL_BATCHES=200
export DCACHE_SANITY_VAL_STEPS=0
export DCACHE_CHECKPOINT_SAVE_TOP_K=3
export DCACHE_DATA_DIR="${DCACHE_DATA_DIR:-${REPO_DIR}/.cache/huggingface}"
DEFAULT_RUN_DIR="${REPO_DIR}/outputs/owt-dcache-final-state-adjacent-pretrain-5k-2x3090"
if [[ "$DCACHE_MICRO_BATCH" != 2 ]]; then
  DEFAULT_RUN_DIR="${REPO_DIR}/outputs/owt-dcache-final-state-adjacent-pretrain-5k-h100-mb${DCACHE_MICRO_BATCH}"
  echo "Microbatch-${DCACHE_MICRO_BATCH} trial: accumulation $((512 / DCACHE_MICRO_BATCH)); identity-reference groups now contain ${DCACHE_MICRO_BATCH} examples."
fi
export DCACHE_RUN_DIR="${DCACHE_RUN_DIR:-$DEFAULT_RUN_DIR}"
SOURCE_CHECKPOINT="${DCACHE_RESUME_CKPT:-${REPO_DIR}/imports/adjacent/0-1500.ckpt}"
MANIFEST="${DCACHE_TRANSFER_MANIFEST:-${REPO_DIR}/experiments/h100_transfer_manifest.json}"
LAUNCHER="${REPO_DIR}/scripts/train/train_owt_dcache_final_state_adjacent_5k_2x3090.sh"

if [[ "$ACTION" == setup ]]; then
  if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "This pinned CUDA environment targets Linux x86_64." >&2
    exit 2
  fi
  # Exact scientific package versions have CPython wheels for this window.
  # Python 3.13+ cannot use several of these pins. Stop before package changes;
  # let the user select a supported version in Studio's Environment panel.
  "$DCACHE_PYTHON" -c 'import sys
if sys.implementation.name != "cpython" or not (3, 9) <= sys.version_info[:2] <= (3, 12):
    sys.exit("This checkpoint recipe requires CPython 3.9-3.12. Select Python 3.10 in the Studio Environment panel, then rerun setup. No packages changed.")
print("Using existing environment:", sys.executable, sys.version.split()[0])'
  echo "Setup updates packages in this environment. Do not run while another job uses it."
  "$DCACHE_PYTHON" -m pip install 'pip==25.1.1'
  "$DCACHE_PYTHON" -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126
  # Do not install the local notebook/dev-tool pins into Studio's managed IDE.
  "$DCACHE_PYTHON" -m pip install -r requirements-h100.txt
  "$DCACHE_PYTHON" -m pip check
  CUDA_VISIBLE_DEVICES='' "$DCACHE_PYTHON" -c \
    'import dataloader, diffusion, recurrent_gradients, checkpoint_resume; import torch, lightning; print("Training imports OK:", torch.__version__, lightning.__version__)'
  echo "Setup complete. Interpreter: $DCACHE_PYTHON"
  echo "No training started. Next: copy prepared data/checkpoint, then run check and smoke."
  exit 0
fi

if [[ ! "$DCACHE_NUM_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "DCACHE_NUM_WORKERS must be a positive integer (persistent workers are enabled)." >&2
  exit 2
fi
if [[ ! "$DCACHE_MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "DCACHE_MAX_STEPS must be a positive optimizer-step target." >&2
  exit 2
fi

case "$ACTION" in
  plot)
    exec "$DCACHE_PYTHON" scripts/results/refresh_canonical_results.py \
      --available-only --training-only \
      --run-path dcache_final_state_adjacent "$DCACHE_RUN_DIR"
    ;;
  check|smoke|train|tmux|validate) ;;
  *) usage >&2; exit 2 ;;
esac

# Continue cloud progress on reruns. An incomplete/broken last.ckpt is an error,
# never a reason to silently restart from the original imported checkpoint.
LAST_CHECKPOINT="${DCACHE_RUN_DIR}/checkpoints/last.ckpt"
if [[ -e "$LAST_CHECKPOINT" || -L "$LAST_CHECKPOINT" ]]; then
  CHECKPOINT="$LAST_CHECKPOINT"
  TRUSTED_DESCENDANT=true
else
  CHECKPOINT="$SOURCE_CHECKPOINT"
  TRUSTED_DESCENDANT=false
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing resume checkpoint: $CHECKPOINT" >&2
  echo "Copy the actual 0-1500.ckpt file, not just the last.ckpt symlink." >&2
  exit 2
fi

CHECK_ARGS=(check --checkpoint "$CHECKPOINT" --data-dir "$DCACHE_DATA_DIR" \
  --manifest "$MANIFEST" --max-steps "$DCACHE_MAX_STEPS")
if [[ "$ACTION" == validate ]]; then
  # Validation is allowed after the target optimizer step has been reached.
  CHECK_ARGS[${#CHECK_ARGS[@]}-1]=2147483647
fi
if [[ "$TRUSTED_DESCENDANT" == true ]]; then
  CHECK_ARGS+=(--allow-descendant)
fi
CHECK_ARGS+=(--output-dir "$DCACHE_RUN_DIR")
if [[ "$ACTION" == check && "${1:-}" == --cpu-only ]]; then
  CHECK_ARGS+=(--cpu-only)
fi
"$DCACHE_PYTHON" scripts/cloud/check_h100_resume.py "${CHECK_ARGS[@]}"
if [[ "$ACTION" == check ]]; then exit 0; fi

case "$ACTION" in
  tmux)
    if ! command -v tmux >/dev/null 2>&1; then
      echo "tmux is unavailable; install it through your Studio environment or run train in an existing session." >&2
      exit 2
    fi
    SOCKET="${REPO_DIR}/.cache/tmux/h100.sock"
    if (( ${#SOCKET} > 100 )); then
      echo "Repository path is too long for the tmux socket; use a shorter persistent checkout path." >&2
      exit 2
    fi
    mkdir -p "${REPO_DIR}/.cache/tmux"
    if tmux -S "$SOCKET" has-session -t dcache-h100 2>/dev/null; then
      echo "Session already exists; not starting another training process."
    else
      # An existing tmux server can retain an older environment. Pass user
      # overrides explicitly instead of relying on that server's environment.
      printf -v TMUX_COMMAND 'env %q %q %q %q %q %q %q %q %q %q bash %q train' \
        "DCACHE_PYTHON=$DCACHE_PYTHON" \
        "DCACHE_RESUME_CKPT=$SOURCE_CHECKPOINT" \
        "DCACHE_TRANSFER_MANIFEST=$MANIFEST" \
        "DCACHE_DATA_DIR=$DCACHE_DATA_DIR" \
        "DCACHE_RUN_DIR=$DCACHE_RUN_DIR" \
        "DCACHE_CUDA_VISIBLE_DEVICES=$DCACHE_CUDA_VISIBLE_DEVICES" \
        "DCACHE_MAX_STEPS=$DCACHE_MAX_STEPS" \
        "DCACHE_NUM_WORKERS=$DCACHE_NUM_WORKERS" \
        "DCACHE_CPU_THREADS=$OMP_NUM_THREADS" \
        "DCACHE_MICRO_BATCH=$DCACHE_MICRO_BATCH" \
        "${SCRIPT_DIR}/lightning_h100.sh"
      tmux -S "$SOCKET" new-session -d -s dcache-h100 -c "$REPO_DIR" "$TMUX_COMMAND"
    fi
    printf 'Attach: tmux -S %q attach -t dcache-h100\n' "$SOCKET"
    exit 0
    ;;
  validate)
    export DCACHE_EVAL_CUDA_VISIBLE_DEVICES="$DCACHE_CUDA_VISIBLE_DEVICES"
    export DCACHE_EVAL_BATCH=2 DCACHE_EVAL_GLOBAL_BATCH=512
    STEP="$("$DCACHE_PYTHON" -c 'import sys, torch; c=torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True); print(c["global_step"])' "$CHECKPOINT")"
    # Keep standalone step-zero CSVs outside the recursive training CSV scan.
    mkdir -p "${REPO_DIR}/outputs"
    VALIDATE_DIR="$(mktemp -d "${REPO_DIR}/outputs/eval-adjacent-h100-step${STEP}.XXXXXXXX")"
    exec bash scripts/eval/eval_checkpoint_validation.sh final-state-adjacent \
      "$CHECKPOINT" "$VALIDATE_DIR" \
      loader.num_workers="$DCACHE_NUM_WORKERS"
    ;;
  smoke)
    # A true resumed optimizer update (512 / microbatch batches), plus validation and a
    # checkpoint save. Never change or resume from the smoke output in training.
    STEP="$("$DCACHE_PYTHON" -c 'import sys, torch; c=torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True); print(c["global_step"])' "$CHECKPOINT")"
    export DCACHE_MAX_STEPS=$(( STEP + 1 ))
    export DCACHE_VAL_INTERVAL=1 DCACHE_VAL_BATCHES=2
    export DCACHE_CHECKPOINT_SAVE_TOP_K=1
    export DCACHE_RUN_DIR="$(mktemp -d "${REPO_DIR}/.cache/runtime/h100/smoke.XXXXXXXX")"
    EXTRA_OVERRIDES=(callbacks.checkpoint_every_n_steps.every_n_train_steps=1)
    echo "SMOKE ONLY: ${STEP} -> ${DCACHE_MAX_STEPS}; output $DCACHE_RUN_DIR"
    ;;
  train)
    EXTRA_OVERRIDES=()
    ;;
esac

mkdir -p "$DCACHE_RUN_DIR"
# Preserve the imported checkpoint separately; normal cloud checkpoint
# retention manages only the latest three new periodic checkpoints.
echo "Resume source: $CHECKPOINT"
echo "Global batch 512 = 1 GPU x microbatch $DCACHE_MICRO_BATCH x accumulation $((512 / DCACHE_MICRO_BATCH))."
echo "Training stops at optimizer step $DCACHE_MAX_STEPS (not that many additional steps)."
echo "Managed runtime cache: $TMPDIR"
echo "Validation uses 400 examples every 500 updates in the full run."
LOG_FILE="${REPO_DIR}/logs/h100-${ACTION}-$(date -u +%Y%m%dT%H%M%SZ).log"
# flock prevents two cloud launches from writing to the same run directory.
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required to guard against duplicate launches." >&2
  exit 2
fi
(
  flock -n 9 || { echo "A process already owns this run directory." >&2; exit 2; }
  bash "$LAUNCHER" \
    strategy=ddp \
    checkpointing.resume_from_ckpt=true \
    checkpointing.resume_ckpt_path="$CHECKPOINT" \
    checkpointing.allow_batch_geometry_change=true \
    loader.eval_batch_size=2 \
    "${EXTRA_OVERRIDES[@]}"
) 9>"${DCACHE_RUN_DIR}/.training.lock" 2>&1 | tee "$LOG_FILE"
echo "${ACTION} completed; log: ${LOG_FILE}"
