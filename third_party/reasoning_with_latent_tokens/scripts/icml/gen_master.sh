#!/bin/bash
#SBATCH -J gen-master
#SBATCH --partition=preempt
#SBATCH --output=slurm/master/%j_%x.out
#SBATCH --error=slurm/master/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --time=48:00:00
#SBATCH --mem=100G

# Master generation script for all model variants
#
# Usage: sbatch gen_master.sh --method=<method> --data=<data> [options] [-- extra_args]
#
# Required:
#   --method, -m    Model method (ar, diffu-maskfix, diffu-causal, etc.)
#   --data, -d      Dataset (sudoku-small, sudoku-large, sudoku-puzzle, zebra, openwebtext-split)
#
# Optional:
#   --steps, -s     Sampling steps (default: 128, ignored for AR)
#   --batches, -b   Number of sample batches (default: 1)
#   --ckpt, -c      Checkpoint filename (default: auto-detect best*.ckpt)
#
# Pass-through:
#   -- <args>       Additional arguments passed directly to main.py
#
# Examples:
#   sbatch gen_master.sh -m ar -d sudoku-large -b 10
#   sbatch gen_master.sh --method=diffu-maskfix --data=sudoku-large --steps=64
#   sbatch gen_master.sh -m diffu-maskfix -d zebra -s 64 -- sampling.unmask_policy=entropy
#   sbatch gen_master.sh --method=ar --data=sudoku-large --ckpt=epoch-5.ckpt

set -e

# === Default values ===
METHOD=""
DATA=""
STEPS=""  # Will be set to MODEL_LENGTH if not provided
NUM_SAMPLE_BATCHES=1
CKPT_NAME=""
RUN_SUFFIX=""  # Optional suffix for checkpoint lookup (e.g., -seed2)
EXTRA_ARGS=""

# === Argument parsing ===
show_help() {
  echo "Usage: sbatch gen_master.sh --method=<method> --data=<data> [options] [-- extra_args]"
  echo ""
  echo "Required:"
  echo "  --method, -m    Model method"
  echo "  --data, -d      Dataset"
  echo ""
  echo "Optional:"
  echo "  --steps, -s     Sampling steps (default: MODEL_LENGTH for dataset, ignored for AR)"
  echo "  --batches, -b   Number of sample batches (default: 1)"
  echo "  --ckpt, -c      Checkpoint filename (default: auto-detect best*.ckpt)"
  echo ""
  echo "Pass-through:"
  echo "  -- <args>       Additional arguments passed directly to main.py"
  echo ""
  echo "Methods: ar, ar-ntp, ar-mtp-window-32, ar-mtp-window-128,"
  echo "         diffu-maskfix, diffu-causal, diffu-full, dp-qk-split-parallel, dp-4way, dp-2way"
  echo ""
  echo "Note: Some method variants (e.g., diffu-causal-output-sminy-l2r) use a different"
  echo "      checkpoint than their name implies. These have CKPT_METHOD set internally."
  echo ""
  echo "Data: sudoku-small, sudoku-large, sudoku-puzzle, sudoku-conditional, sudoku-conditional-uncond, zebra, game-of-24, openwebtext-split"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --method=*)
      METHOD="${1#*=}"
      shift
      ;;
    --method|-m)
      METHOD="$2"
      shift 2
      ;;
    --data=*)
      DATA="${1#*=}"
      shift
      ;;
    --data|-d)
      DATA="$2"
      shift 2
      ;;
    --steps=*)
      STEPS="${1#*=}"
      shift
      ;;
    --steps|-s)
      STEPS="$2"
      shift 2
      ;;
    --batches=*)
      NUM_SAMPLE_BATCHES="${1#*=}"
      shift
      ;;
    --batches|-b)
      NUM_SAMPLE_BATCHES="$2"
      shift 2
      ;;
    --ckpt=*)
      CKPT_NAME="${1#*=}"
      shift
      ;;
    --ckpt|-c)
      CKPT_NAME="$2"
      shift 2
      ;;
    --run-suffix=*)
      RUN_SUFFIX="${1#*=}"
      shift
      ;;
    --run-suffix)
      RUN_SUFFIX="$2"
      shift 2
      ;;
    --help|-h)
      show_help
      ;;
    --)
      shift
      EXTRA_ARGS="$*"
      break
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# === Validate required arguments ===
if [ -z "$METHOD" ]; then
  echo "Error: --method is required"
  echo "Use --help for usage information"
  exit 1
fi

if [ -z "$DATA" ]; then
  echo "Error: --data is required"
  echo "Use --help for usage information"
  exit 1
fi

# === Validate inputs ===
# case $DATA in
#   sudoku-small|sudoku-large|sudoku-puzzle|sudoku-conditional|sudoku-conditional-uncond|zebra|openwebtext-split|game-of-24)
#     ;;
#   *)
#     echo "Unknown data: $DATA"
#     echo "Valid data: sudoku-small, sudoku-large, sudoku-puzzle, sudoku-conditional, sudoku-conditional-uncond, zebra, game-of-24, openwebtext-split"
#     exit 1
#     ;;
# esac

# === Set data-specific parameters ===
MODEL_NAME=""  # Default: use model from config.yaml (small)
GEN_MODE=sample_eval  # Default: unconditional generation
case $DATA in
  sudoku-small)
    MODEL_LENGTH=128
    BATCH_SIZE=128
    USE_GCS=False
    ;;
  sudoku-small-solver)
    MODEL_LENGTH=128
    BATCH_SIZE=128
    USE_GCS=False
    DATA_CONFIG=sudoku-small
    EXTRA_ARGS="${EXTRA_ARGS} data.use_solver=True"
    ;;
  sudoku-large)
    MODEL_LENGTH=1536
    BATCH_SIZE=16
    USE_GCS=False
    ;;
  sudoku-puzzle)
    MODEL_LENGTH=256
    BATCH_SIZE=128
    USE_GCS=False
    MODEL_NAME=small
    GEN_MODE=completions
    TRAIN_DATA_PATH=${ESOLM_PUZZLE_DIR}/sudoku-train-data.npy
    VALID_DATA_PATH=${ESOLM_PUZZLE_DIR}/sudoku-test-data.npy
    ;;
  game-of-24)
    MODEL_LENGTH=64
    BATCH_SIZE=256
    USE_GCS=False
    MODEL_NAME=small
    GEN_MODE=completions
    ;;
  sudoku-conditional)
    # Procedurally generated sudoku puzzle-solution pairs
    MODEL_LENGTH=256
    BATCH_SIZE=128
    USE_GCS=False
    MODEL_NAME=small
    GEN_MODE=completions
    ;;
  sudoku-conditional-192)
    MODEL_LENGTH=192
    BATCH_SIZE=128
    USE_GCS=False
    MODEL_NAME=small
    GEN_MODE=completions
    DATA_CONFIG=sudoku-conditional
    ;;
  sudoku-conditional-192-0given)
    # Unconditional variant: target_givens=0
    MODEL_LENGTH=192
    BATCH_SIZE=128
    USE_GCS=False
    MODEL_NAME=small
    GEN_MODE=completions
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=0
    ;;
  sudoku-conditional-192-10given)
    MODEL_LENGTH=192
    BATCH_SIZE=128
    USE_GCS=False
    MODEL_NAME=small
    GEN_MODE=completions
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=10
    ;;
  sudoku-conditional-192-20given)
    MODEL_LENGTH=192
    BATCH_SIZE=128
    USE_GCS=False
    MODEL_NAME=small
    GEN_MODE=completions
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=20
    ;;
  sudoku-conditional-192-30given)
    MODEL_LENGTH=192
    BATCH_SIZE=128
    USE_GCS=False
    MODEL_NAME=small
    GEN_MODE=completions
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=30
    ;;
  sudoku-conditional-192-1given)
    MODEL_LENGTH=192
    BATCH_SIZE=128
    USE_GCS=False
    MODEL_NAME=small
    GEN_MODE=completions
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=1
    ;;
  sudoku-conditional-192-2given)
    MODEL_LENGTH=192
    BATCH_SIZE=128
    USE_GCS=False
    MODEL_NAME=small
    GEN_MODE=completions
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=2
    ;;
  sudoku-conditional-192-5given)
    MODEL_LENGTH=192
    BATCH_SIZE=128
    USE_GCS=False
    MODEL_NAME=small
    GEN_MODE=completions
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=5
    ;;
  zebra)
    MODEL_LENGTH=384
    BATCH_SIZE=128
    USE_GCS=False
    MODEL_NAME=small
    GEN_MODE=completions
    TRAIN_DATA_PATH=${ESOLM_PUZZLE_DIR}/zebra-train-data.pkl
    VALID_DATA_PATH=${ESOLM_PUZZLE_DIR}/zebra-test-data.pkl
    ;;
  openwebtext-split)
    MODEL_LENGTH=1024
    BATCH_SIZE=16
    USE_GCS=False
    ;;
esac

# Default STEPS to MODEL_LENGTH if not provided
if [ -z "$STEPS" ]; then
  STEPS=$MODEL_LENGTH
fi

# === Set method-specific parameters ===
# Defaults
ALGO=""
AR_NOISE=""
NEXT_TOKEN_PREDICTION=""
LOSS_TYPE="elbo"
DIFFUSION_ATTN_MODE=""
MTP_WINDOW_SIZE=""
DIFFUSION_SHUFFLE=""
POS_ENCODING_STRATEGY=""
SHUFFLE_WARMUP=""
CKPT_METHOD=""  # If set, use this method name for checkpoint lookup instead of METHOD

case $METHOD in
  diffu-causal-output)
    # Diffusion with random noise, causal_output attention, shuffle both clean and masked
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-causal-output-sminy)
    # Diffusion with random noise, causal_output attention, shuffle both, sminy model
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    MODEL_NAME=sminy
    ;;
  diffu-causal-output-sminy-iglm)
    # Diffusion with random noise, causal_output attention, shuffle both, sminy model
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    MODEL_NAME=sminy
    # TRAIN_ON_ALL_TOKENS=True
    ;;
  diffu-causal-output-sminy-l2r)
    # Diffusion with random noise, causal_output attention, shuffle both, sminy model
    # Uses diffu-causal-output-sminy checkpoint but with different sampling config
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True # TODO double check this is being applied to problem tokens correctly
    SHUFFLE_MASKED_TOKENS=False
    MODEL_NAME=sminy
    CKPT_METHOD=diffu-causal-output-sminy
    ;;
  diffu-causal-sminy)
    # Pure diffusion with ELBO loss, causal attention, with sminy model
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal
    MODEL_NAME=sminy
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-causal-tiny)
    # Pure diffusion with ELBO loss, causal attention, with sminy model
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal
    MODEL_NAME=tiny
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-causal-output-tiny)
    # Diffusion with causal_output attention, shuffle both, tiny model (~6M)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    MODEL_NAME=tiny
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-causal-output-tiny-deep)
    # Diffusion with causal_output attention, shuffle both, tiny-deep model (~5M, 12 blocks)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    MODEL_NAME=tiny-deep
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-causal-output-tiny-deeper)
    # Diffusion with causal_output attention, shuffle both, tiny-deeper model (~5M, 16 blocks)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    MODEL_NAME=tiny-deeper
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-causal-output-micro)
    # Diffusion with causal_output attention, shuffle both, micro model (~10M)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    MODEL_NAME=micro
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-causal-output-mini)
    # Diffusion with causal_output attention, shuffle both, mini model (~18M)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    MODEL_NAME=mini
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-causal-output-miny)
    # Diffusion with causal_output attention, shuffle both, miny model (~28M)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    MODEL_NAME=miny
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-causal-sminy-spt)
    # Pure diffusion with ELBO loss, causal attention, with sminy model
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal
    MODEL_NAME=sminy
    EXTRA_ARGS="${EXTRA_ARGS} +algo.shuffle_problem_tokens=True"
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-causal-tiny-spt)
    # Pure diffusion with ELBO loss, causal attention, with sminy model
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal
    MODEL_NAME=tiny
    EXTRA_ARGS="${EXTRA_ARGS} +algo.shuffle_problem_tokens=True"
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  esolmb)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    MODEL_NAME=small
    EXTRA_ARGS="${EXTRA_ARGS} model.absolute_pos_embed=False"
    EXTRA_ARGS="${EXTRA_ARGS} model.disable_adaln=False"
    EXTRA_ARGS="${EXTRA_ARGS} eval.compute_generative_perplexity=True"
    EXTRA_ARGS="${EXTRA_ARGS} sampling.kv_cache=True"
    EXTRA_ARGS="${EXTRA_ARGS} sampling.p_nucleus=0.9"

    ;;
  ar-causal-output)
    # AR noise, causal_output attention, shuffle masked but not clean
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=False
    SHUFFLE_MASKED_TOKENS=False # Note this is different from training. this is intentional
    ;;
  ar-causal-output-sminy)
    # AR noise, causal_output attention, shuffle masked but not clean, sminy model
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=False
    SHUFFLE_MASKED_TOKENS=False # Note this is different from training. this is intentional
    MODEL_NAME=sminy
    ;;
  ar)
    # Pure autoregressive baseline
    ALGO=ar
    ;;
  ar-sminy)
    ALGO=ar
    MODEL_NAME=sminy
    ;;
  ar-ntp)
    # AR noise + next token prediction with causal attention
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=True
    LOSS_TYPE=low_var
    DIFFUSION_ATTN_MODE=causal
    ;;
  ar-mtp-window-32)
    # AR noise with MTP window size 32
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=causal_context
    MTP_WINDOW_SIZE=32
    ;;
  ar-mtp-window-128)
    # AR noise with MTP window size 128
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=causal_context
    MTP_WINDOW_SIZE=128
    ;;
  ar-mtp-window--1)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=causal_context
    MTP_WINDOW_SIZE=-1
    ;;
  ar-mtp-window--1-sminy)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=causal_context
    MTP_WINDOW_SIZE=-1
    MODEL_NAME=sminy
    ;;
  ar-ntp-sminy)
    # AR noise + next token prediction with causal attention
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=True
    LOSS_TYPE=low_var
    DIFFUSION_ATTN_MODE=causal
    MODEL_NAME=sminy
    ;;
  ar-ntp-full-sminy)
    # AR noise + next token prediction with full attention
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=low_var
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=1
    MODEL_NAME=sminy
    ;;
  ar-mtp-full-sminy)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=-1
    MODEL_NAME=sminy
    ;;
  ar-mtp-32-full-sminy)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=32
    MODEL_NAME=sminy
    ;;
  ar-mtp-128-full-sminy)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=128
    MODEL_NAME=sminy
    ;;
  diffu)
    # Pure diffusion with ELBO loss
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_context
    ;;
  diffu-causal)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal
    ;;
  diffu-full)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=full
    ;;
  diffu-full-sminy)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=full
    MODEL_NAME=sminy
    ;;
  # this just helps point to the ckpt
  # config is same as above
  diffu-full-lr1e-4-bsz64)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=full
    ;;
  diffu-full-ignore-loss-mask)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=full
    ;;
  diffu-causal)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal
    ;;
  diffu-causal-sminy)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal
    MODEL_NAME=sminy
    ;;
  dp-qk-split-parallel)
    # Q/K split with shuffling
    ALGO=diffuparallel
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=qk_split
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    SHUFFLE_WARMUP=0
    ;;
  dp-4way)
    # 4-way head split
    ALGO=diffuparallel
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=4way_heads
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    SHUFFLE_WARMUP=0
    ;;
  dp-2way)
    # 2-way head split
    ALGO=diffuparallel
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=2way_heads
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    SHUFFLE_WARMUP=0
    ;;
  *)
    echo "Unknown method: $METHOD"
    echo "Valid methods: ar, ar-ntp, ar-mtp-window-32, ar-mtp-window-128,"
    echo "               diffu-maskfix, diffu-causal, diffu-full, dp-qk-split-parallel, dp-4way, dp-2way"
    exit 1
    ;;
esac

# Build run name (matches training script naming)
RUN_NAME=${METHOD}-${DATA}

# Build checkpoint run name (may differ for variants that share checkpoints)
if [ -n "$CKPT_METHOD" ]; then
  CKPT_RUN_NAME=${CKPT_METHOD}-${DATA}${RUN_SUFFIX}
else
  CKPT_RUN_NAME=${RUN_NAME}${RUN_SUFFIX}
fi

# === Environment setup ===
# Required environment variables (cluster-specific):
#   ESOLM_CONDA_PROFILE  - Path to conda.sh (e.g., /path/to/miniconda3/etc/profile.d/conda.sh)
#   ESOLM_DATADIR        - Base data directory for cache, runs, and checkpoints
#   ESOLM_HF_HOME        - HuggingFace home directory
#   ESOLM_PUZZLE_DIR     - Directory containing puzzle data files (required for zebra, sudoku-puzzle)

# export ESOLM_CONDA_PROFILE="${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
# export ESOLM_DATADIR="$HOME/"
# export ESOLM_HF_HOME="${HF_HOME:-$HOME/hf_home}"
# export ESOLM_PUZZLE_DIR="$HOME/puzzle"

if [ -z "$ESOLM_CONDA_PROFILE" ]; then
  echo "Error: ESOLM_CONDA_PROFILE environment variable is not set"
  echo "Set it to the path of your conda.sh (e.g., /path/to/miniconda3/etc/profile.d/conda.sh)"
  exit 1
fi

if [ -z "$ESOLM_DATADIR" ]; then
  echo "Error: ESOLM_DATADIR environment variable is not set"
  echo "Set it to your base data directory for cache, runs, and checkpoints"
  exit 1
fi

if [ -z "$ESOLM_HF_HOME" ]; then
  echo "Error: ESOLM_HF_HOME environment variable is not set"
  echo "Set it to your HuggingFace home directory"
  exit 1
fi

# Check ESOLM_PUZZLE_DIR for datasets that require puzzle data files
if [[ "$DATA" == "zebra" || "$DATA" == "sudoku-puzzle" ]]; then
  if [ -z "$ESOLM_PUZZLE_DIR" ]; then
    echo "Error: ESOLM_PUZZLE_DIR environment variable is not set"
    echo "Set it to the directory containing puzzle data files (e.g., $HOME/puzzle)"
    exit 1
  fi
fi

source "$ESOLM_CONDA_PROFILE"
conda activate esolm

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export HF_HOME="$ESOLM_HF_HOME"
export DATADIR="$ESOLM_DATADIR"
export GCS_DIR="gs://YOUR_BUCKET/"

CACHE_DIR=${DATADIR}/cache

# Set checkpoint directory based on dataset (uses CKPT_RUN_NAME for variants sharing checkpoints)
if [ "$USE_GCS" = "True" ]; then
  CHECKPOINT_DIR=${GCS_DIR}/checkpoints/${CKPT_RUN_NAME}
else
  CHECKPOINT_DIR=${DATADIR}/checkpoints/${CKPT_RUN_NAME}
fi

# === Find checkpoint ===
if [ -n "$CKPT_NAME" ]; then
  CKPT_PATH=${CHECKPOINT_DIR}/checkpoints/${CKPT_NAME}
  if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint not found: $CKPT_PATH"
    exit 1
  fi
else
  # Find the latest best checkpoint (handles best.ckpt, best-v1.ckpt, best-v2.ckpt, etc.)
  CKPT_PATH=$(ls -v ${CHECKPOINT_DIR}/checkpoints/best*.ckpt 2>/dev/null | tail -1)
  if [ -z "$CKPT_PATH" ]; then
    echo "Error: No checkpoint found in ${CHECKPOINT_DIR}/checkpoints/"
    exit 1
  fi
  CKPT_NAME=$(basename "$CKPT_PATH")
fi

# Build generation name with checkpoint stem
CKPT_STEM=${CKPT_NAME%.ckpt}
if [ "$ALGO" = "ar" ]; then
  GEN_NAME=${RUN_NAME}-gen-${CKPT_STEM}
else
  GEN_NAME=${RUN_NAME}-gen-steps-${STEPS}-${CKPT_STEM}
fi
WORKING_DIR=${DATADIR}/runs/${GEN_NAME}

# === Print configuration ===
echo "=== Generation Configuration ==="
echo "METHOD: $METHOD"
echo "DATA: $DATA"
echo "ALGO: $ALGO"
echo "GEN_MODE: $GEN_MODE"
echo "RUN_NAME: $RUN_NAME"
if [ -n "$CKPT_METHOD" ]; then
  echo "CKPT_METHOD: $CKPT_METHOD (loading checkpoint from ${CKPT_RUN_NAME})"
fi
echo "GEN_NAME: $GEN_NAME"
if [ "$ALGO" != "ar" ]; then
  echo "STEPS: $STEPS"
fi
echo "NUM_SAMPLE_BATCHES: $NUM_SAMPLE_BATCHES"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "MODEL_LENGTH: $MODEL_LENGTH"
if [ -n "$MODEL_NAME" ]; then
  echo "MODEL: $MODEL_NAME"
fi
if [ -n "$AR_NOISE" ]; then
  echo "AR_NOISE: $AR_NOISE"
fi
if [ -n "$NEXT_TOKEN_PREDICTION" ]; then
  echo "NEXT_TOKEN_PREDICTION: $NEXT_TOKEN_PREDICTION"
fi
if [ -n "$LOSS_TYPE" ]; then
  echo "LOSS_TYPE: $LOSS_TYPE"
fi
if [ -n "$DIFFUSION_ATTN_MODE" ]; then
  echo "DIFFUSION_ATTN_MODE: $DIFFUSION_ATTN_MODE"
fi
if [ -n "$MTP_WINDOW_SIZE" ]; then
  echo "MTP_WINDOW_SIZE: $MTP_WINDOW_SIZE"
fi
if [ -n "$DIFFUSION_SHUFFLE" ]; then
  echo "DIFFUSION_SHUFFLE: $DIFFUSION_SHUFFLE"
fi
if [ -n "$POS_ENCODING_STRATEGY" ]; then
  echo "POS_ENCODING_STRATEGY: $POS_ENCODING_STRATEGY"
fi
if [ -n "$SHUFFLE_CLEAN_TOKENS" ]; then
  echo "SHUFFLE_CLEAN_TOKENS: $SHUFFLE_CLEAN_TOKENS"
fi
if [ -n "$SHUFFLE_MASKED_TOKENS" ]; then
  echo "SHUFFLE_MASKED_TOKENS: $SHUFFLE_MASKED_TOKENS"
fi
if [ -n "$EXTRA_ARGS" ]; then
  echo "EXTRA_ARGS: $EXTRA_ARGS"
fi
echo "CACHE_DIR: $CACHE_DIR"
echo "WORKING_DIR: $WORKING_DIR"
echo "CKPT_PATH: $CKPT_PATH"
echo "================================="

# === Build generation command ===
# Use DATA_CONFIG if set, otherwise use DATA
if [ -n "$DATA_CONFIG" ]; then
  HYDRA_DATA_CONFIG=$DATA_CONFIG
else
  HYDRA_DATA_CONFIG=$DATA
fi

if [ "$ALGO" = "ar" ]; then
  # AR generation
  PYTHON_CMD="python main.py \
    --config-name=experiment_base \
    data=${HYDRA_DATA_CONFIG} \
    loader.batch_size=${BATCH_SIZE} \
    loader.eval_batch_size=${BATCH_SIZE} \
    wandb.name=${GEN_NAME} \
    algo=ar \
    model.length=${MODEL_LENGTH} \
    data.cache_dir=${CACHE_DIR} \
    hydra.run.dir=${WORKING_DIR} \
    mode=${GEN_MODE} \
    eval.generate_samples=True \
    eval.checkpoint_path=${CKPT_PATH} \
    sampling.kv_cache=True \
    sampling.num_sample_batches=${NUM_SAMPLE_BATCHES} \
    wandb.id=null"
else
  # Diffusion-based generation (difflm, diffuparallel)
  PYTHON_CMD="python main.py \
    --config-name=experiment_base \
    data=${HYDRA_DATA_CONFIG} \
    loader.batch_size=${BATCH_SIZE} \
    loader.eval_batch_size=${BATCH_SIZE} \
    wandb.name=${GEN_NAME} \
    algo=${ALGO} \
    model.length=${MODEL_LENGTH} \
    data.cache_dir=${CACHE_DIR} \
    hydra.run.dir=${WORKING_DIR} \
    mode=${GEN_MODE} \
    eval.generate_samples=True \
    eval.checkpoint_path=${CKPT_PATH} \
    sampling=synthetic_base \
    sampling.steps=${STEPS} \
    sampling.num_sample_batches=${NUM_SAMPLE_BATCHES} \
    sampling.kv_cache=False \
    wandb.id=null"

  # Add difflm-specific arguments
  if [ "$ALGO" = "difflm" ]; then
    PYTHON_CMD="${PYTHON_CMD} \
    algo.ar_noise=${AR_NOISE} \
    algo.next_token_prediction=${NEXT_TOKEN_PREDICTION} \
    algo.loss_type=${LOSS_TYPE} \
    algo.diffusion_attn_mode=${DIFFUSION_ATTN_MODE}"
    
    # Add MTP window size if set
    if [ -n "$MTP_WINDOW_SIZE" ]; then
      PYTHON_CMD="${PYTHON_CMD} \
    algo.mtp_window_size=${MTP_WINDOW_SIZE}"
    fi
  fi

  # Add diffuparallel-specific arguments
  if [ "$ALGO" = "diffuparallel" ]; then
    PYTHON_CMD="${PYTHON_CMD} \
    algo.loss_type=${LOSS_TYPE} \
    algo.diffusion_shuffle=${DIFFUSION_SHUFFLE} \
    algo.shuffle_warmup=${SHUFFLE_WARMUP} \
    algo.next_token_prediction=${NEXT_TOKEN_PREDICTION} \
    algo.pos_encoding_strategy=${POS_ENCODING_STRATEGY}"
  fi
fi

# Add model override if specified (e.g., for zebra dataset)
if [ -n "$MODEL_NAME" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  model=${MODEL_NAME}"
fi

# Add data paths if set (for zebra, sudoku-puzzle)
if [ -n "$TRAIN_DATA_PATH" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  data.train_data_path=${TRAIN_DATA_PATH} \
  data.valid_data_path=${VALID_DATA_PATH}"
fi

# Add target_givens override if set (for sudoku-conditional-uncond)
if [ -n "$TARGET_GIVENS" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  data.target_givens=${TARGET_GIVENS}"
fi

# Add shuffle token configs if set
if [ -n "$SHUFFLE_CLEAN_TOKENS" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
algo.shuffle_clean_tokens=${SHUFFLE_CLEAN_TOKENS}"
fi
if [ -n "$SHUFFLE_MASKED_TOKENS" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
algo.shuffle_masked_tokens=${SHUFFLE_MASKED_TOKENS}"
fi

# Add extra pass-through arguments if provided
if [ -n "$EXTRA_ARGS" ]; then
  PYTHON_CMD="${PYTHON_CMD} ${EXTRA_ARGS}"
fi

# Execute command
echo "Executing: $PYTHON_CMD"
eval $PYTHON_CMD
