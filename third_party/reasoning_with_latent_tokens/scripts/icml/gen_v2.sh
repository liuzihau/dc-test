#!/bin/bash
#SBATCH -J gen-v2
#SBATCH --partition=general
#SBATCH --output=slurm/master/%j_%x.out
#SBATCH --error=slurm/master/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --open-mode=append
#SBATCH --time=48:00:00
#SBATCH --mem=128G

# Generation script with separated method and model size
#
# Usage: sbatch gen_v2.sh --method=<method> --data=<data> [--size=<size>] [options] [-- extra_args]
#
# Required:
#   --method, -m    Model method/architecture (ar, diffu-causal-output, diffu-full, etc.)
#   --data, -d      Dataset (sudoku-small, sudoku-large, game-of-24, zebra, etc.)
#
# Optional:
#   --size, -z      Model size (tiny, tiny-deep, tiny-deeper, tiny-deepest, tiny-deepestest, sminy, micro, mini, miny, small)
#                   Default: small (or dataset-specific default)
#   --steps, -s     Sampling steps (default: MODEL_LENGTH for dataset, ignored for AR)
#   --batches, -b   Number of sample batches (default: 1)
#   --ckpt, -c      Checkpoint filename (default: auto-detect best*.ckpt)
#   --run-suffix    Suffix for checkpoint lookup (e.g., -seed2)
#   --gen-suffix    Suffix for generation output directory (e.g., -topp)
#
# Pass-through:
#   -- <args>       Additional arguments passed directly to main.py
#
# Examples:
#   sbatch gen_v2.sh -m ar -d sudoku-small
#   sbatch gen_v2.sh -m diffu-causal-output -z tiny-deep -d game-of-24 --steps=64
#   sbatch gen_v2.sh --method=diffu-causal --size=sminy --data=sudoku-conditional-192-0given
#   sbatch gen_v2.sh -m ar-ntp -d sudoku-small -- sampling.p_nucleus=0.9
#
# Methods:
#   ar                  - Pure autoregressive baseline
#   ar-ntp              - AR noise + next token prediction with causal attention
#   ar-ntp-full         - AR noise + NTP with full attention
#   ar-mtp-window-N     - AR noise with MTP window size N (32, 128, -1)
#   ar-mtp-full         - AR noise with full attention MTP
#   ar-mtp-32-full      - AR noise with full attention, MTP window 32
#   ar-mtp-128-full     - AR noise with full attention, MTP window 128
#   ar-causal-output    - AR noise with causal_output attention
#   diffu               - Pure diffusion with causal_context attention
#   diffu-full          - Diffusion with full (bidirectional) attention
#   diffu-causal        - Diffusion with causal attention
#   diffu-causal-spt    - Diffusion with causal attention + shuffle problem tokens
#   diffu-causal-output - Diffusion with causal_output attention (recommended)
#   diffu-causal-output-spt - Diffusion causal_output + shuffle problem tokens
#   diffu-causal-output-iglm - Diffusion causal_output + ignore loss mask
#   diffu-causal-output-l2r - Diffusion causal_output, left-to-right sampling
#   dp-qk-split-parallel - Diffusion parallel with Q/K split
#   dp-4way             - Diffusion parallel with 4-way heads
#   dp-2way             - Diffusion parallel with 2-way heads
#   esolmb              - Eso-LM variant B
#
# Model sizes:
#   tiny          - ~6M params, 8 blocks
#   tiny-deep     - ~5M params, 12 blocks
#   tiny-deeper   - ~5M params, 16 blocks
#   tiny-deepest  - ~3M params, 12 blocks, hidden=128
#   tiny-deepestest - ~2M params, 12 blocks, hidden=96
#   sminy         - ~3M params
#   micro         - ~10M params
#   mini          - ~18M params
#   miny          - ~28M params
#   small         - ~45M params (default)

set -e

# === Default values ===
METHOD=""
DATA=""
MODEL_SIZE=""  # Will use default if not specified
STEPS=""  # Will be set to MODEL_LENGTH if not provided
NUM_SAMPLE_BATCHES=1
CKPT_NAME=""
RUN_SUFFIX=""
GEN_SUFFIX=""
EXTRA_ARGS=""

# === Argument parsing ===
show_help() {
  echo "Usage: sbatch gen_v2.sh --method=<method> --data=<data> [--size=<size>] [options] [-- extra_args]"
  echo ""
  echo "Required:"
  echo "  --method, -m    Model method/architecture"
  echo "  --data, -d      Dataset"
  echo ""
  echo "Optional:"
  echo "  --size, -z      Model size (tiny, tiny-deep, tiny-deeper, tiny-deepest, tiny-deepestest, sminy, micro, mini, miny, small)"
  echo "  --steps, -s     Sampling steps (default: MODEL_LENGTH for dataset, ignored for AR)"
  echo "  --batches, -b   Number of sample batches (default: 1)"
  echo "  --ckpt, -c      Checkpoint filename (default: auto-detect best*.ckpt)"
  echo "  --run-suffix    Suffix for checkpoint lookup (e.g., -seed2)"
  echo "  --gen-suffix    Suffix for generation output directory (e.g., -topp)"
  echo ""
  echo "Pass-through:"
  echo "  -- <args>       Additional arguments passed directly to main.py"
  echo ""
  echo "Methods: ar, ar-ntp, ar-ntp-full, ar-mtp-window-32, ar-mtp-window-128, ar-mtp-window--1,"
  echo "         ar-mtp-causal-context, ar-mtp-full, ar-mtp-32-full, ar-mtp-128-full, ar-causal-output,"
  echo "         diffu, diffu-full, diffu-causal, diffu-causal-spt,"
  echo "         diffu-causal-output, diffu-causal-output-spt, diffu-causal-output-iglm,"
  echo "         diffu-solo-causal, diffu-solo-full, diffu-causal-output-l2r,"
  echo "         dp-qk-split-parallel, dp-4way, dp-2way, esolmb"
  echo ""
  echo "Sizes: tiny, tiny-deep, tiny-deeper, tiny-deepest, tiny-deepestest, sminy, micro, mini, miny, small"
  echo ""
  echo "Data: sudoku-small, sudoku-large, sudoku-puzzle, sudoku-conditional,"
  echo "      sudoku-conditional-192-Ngiven, zebra, game-of-24, openwebtext-split"
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
    --size=*)
      MODEL_SIZE="${1#*=}"
      shift
      ;;
    --size|-z)
      MODEL_SIZE="$2"
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
    --gen-suffix=*)
      GEN_SUFFIX="${1#*=}"
      shift
      ;;
    --gen-suffix)
      GEN_SUFFIX="$2"
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

# === Method defaults ===
ALGO=""
AR_NOISE=""
NEXT_TOKEN_PREDICTION=""
LOSS_TYPE="elbo"
DIFFUSION_ATTN_MODE=""
MTP_WINDOW_SIZE=""
DIFFUSION_SHUFFLE=""
POS_ENCODING_STRATEGY=""
SHUFFLE_WARMUP=""
SHUFFLE_CLEAN_TOKENS=""
SHUFFLE_MASKED_TOKENS=""
CKPT_METHOD=""  # If set, use this method name for checkpoint lookup instead of METHOD

# === Set method-specific parameters (architecture only, no model size) ===
case $METHOD in
  ar)
    ALGO=ar
    ;;
  ar-ntp)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=True
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=-1
    ;;
  ar-ntp-w1)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=True
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=1
    ;;
  ar-ntp-full)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=low_var
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=1
    ;;
  ar-mtp-window-32)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=causal_context
    MTP_WINDOW_SIZE=32
    ;;
  ar-mtp-window-128)
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
  ar-mtp-causal-context)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=causal_context
    MTP_WINDOW_SIZE=-1
    ;;
  ar-mtp-8-full)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=8
    ;;
  ar-mtp-16-full)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=16
    ;;
  ar-mtp-32-full)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=32
    ;;
  ar-mtp-64-full)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=64
    ;;
  ar-mtp-full)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=-1
    ;;
  ar-mtp-32-full)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=32
    ;;
  ar-mtp-128-full)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=128
    ;;
  ar-causal-output)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=False
    SHUFFLE_MASKED_TOKENS=False  # Note: different from training, intentional for L2R sampling
    ;;
  diffu-maskfix|diffu)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_context
    ;;
  diffu-full)
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
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-causal-spt)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    EXTRA_ARGS="${EXTRA_ARGS} +algo.shuffle_problem_tokens=True"
    ;;
  diffu-causal-output)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-causal-output-spt)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    EXTRA_ARGS="${EXTRA_ARGS} +algo.shuffle_problem_tokens=True"
    ;;
  diffu-causal-output-iglm)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-solo-causal)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=solo_causal
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-solo-full)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=solo_full
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-causal-output-l2r)
    # Uses diffu-causal-output checkpoint but with L2R sampling config
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=False
    CKPT_METHOD=diffu-causal-output
    ;;
  esolmb)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    EXTRA_ARGS="${EXTRA_ARGS} model.absolute_pos_embed=False"
    EXTRA_ARGS="${EXTRA_ARGS} model.disable_adaln=False"
    EXTRA_ARGS="${EXTRA_ARGS} eval.compute_generative_perplexity=True"
    EXTRA_ARGS="${EXTRA_ARGS} sampling.kv_cache=True"
    EXTRA_ARGS="${EXTRA_ARGS} sampling.p_nucleus=0.9"
    ;;
  dp-qk-split-parallel)
    ALGO=diffuparallel
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=qk_split
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    SHUFFLE_WARMUP=0
    ;;
  dp-4way)
    ALGO=diffuparallel
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=4way_heads
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    SHUFFLE_WARMUP=0
    ;;
  dp-2way)
    ALGO=diffuparallel
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=2way_heads
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    SHUFFLE_WARMUP=0
    ;;
  *)
    echo "Unknown method: $METHOD"
    echo "Valid methods: ar, ar-ntp, ar-ntp-full, ar-mtp-window-32, ar-mtp-window-128, ar-mtp-window--1,"
    echo "               ar-mtp-causal-context, ar-mtp-full, ar-mtp-32-full, ar-mtp-128-full, ar-causal-output,"
    echo "               diffu, diffu-full, diffu-causal, diffu-causal-spt,"
    echo "               diffu-causal-output, diffu-causal-output-spt, diffu-causal-output-iglm,"
    echo "               diffu-solo-causal, diffu-solo-full, diffu-causal-output-l2r,"
    echo "               dp-qk-split-parallel, dp-4way, dp-2way, esolmb"
    exit 1
    ;;
esac

# === Data-specific parameters ===
GEN_MODE=sample_eval  # Default: unconditional generation
DATA_CONFIG=""
TARGET_GIVENS=""

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
    MODEL_LENGTH=192
    BATCH_SIZE=128
    USE_GCS=False
    GEN_MODE=completions
    TRAIN_DATA_PATH=${ESOLM_PUZZLE_DIR}/sudoku-train-data.npy
    VALID_DATA_PATH=${ESOLM_PUZZLE_DIR}/sudoku-test-data.npy
    ;;
  game-of-24)
    MODEL_LENGTH=64
    BATCH_SIZE=256
    USE_GCS=False
    GEN_MODE=completions
    ;;
  sudoku-conditional)
    MODEL_LENGTH=256
    BATCH_SIZE=128
    USE_GCS=False
    GEN_MODE=completions
    ;;
  sudoku-conditional-192)
    MODEL_LENGTH=192
    BATCH_SIZE=128
    USE_GCS=False
    GEN_MODE=completions
    DATA_CONFIG=sudoku-conditional
    ;;
  sudoku-conditional-192-0given)
    MODEL_LENGTH=192
    BATCH_SIZE=128
    USE_GCS=False
    GEN_MODE=completions
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=0
    ;;
  sudoku-conditional-192-1given)
    MODEL_LENGTH=192
    BATCH_SIZE=128
    USE_GCS=False
    GEN_MODE=completions
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=1
    ;;
  sudoku-conditional-192-2given)
    MODEL_LENGTH=192
    BATCH_SIZE=128
    USE_GCS=False
    GEN_MODE=completions
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=2
    ;;
  sudoku-conditional-192-5given)
    MODEL_LENGTH=192
    BATCH_SIZE=128
    USE_GCS=False
    GEN_MODE=completions
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=5
    ;;
  sudoku-conditional-192-10given)
    MODEL_LENGTH=192
    BATCH_SIZE=128
    USE_GCS=False
    GEN_MODE=completions
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=10
    ;;
  sudoku-conditional-192-20given)
    MODEL_LENGTH=192
    BATCH_SIZE=128
    USE_GCS=False
    GEN_MODE=completions
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=20
    ;;
  sudoku-conditional-192-30given)
    MODEL_LENGTH=192
    BATCH_SIZE=128
    USE_GCS=False
    GEN_MODE=completions
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=30
    ;;
  zebra)
    MODEL_LENGTH=384
    BATCH_SIZE=128
    USE_GCS=False
    GEN_MODE=completions
    TRAIN_DATA_PATH=${ESOLM_PUZZLE_DIR}/zebra-train-data.pkl
    VALID_DATA_PATH=${ESOLM_PUZZLE_DIR}/zebra-test-data.pkl
    ;;
  openwebtext-split)
    MODEL_LENGTH=1024
    BATCH_SIZE=8
    USE_GCS=False
    ;;
  *)
    echo "Unknown data: $DATA"
    echo "Valid data: sudoku-small, sudoku-large, sudoku-puzzle, sudoku-conditional,"
    echo "            sudoku-conditional-192-Ngiven, zebra, game-of-24, openwebtext-split"
    exit 1
    ;;
esac

# Default STEPS to MODEL_LENGTH if not provided
if [ -z "$STEPS" ]; then
  STEPS=$MODEL_LENGTH
fi

# === Set model size (use default 'small' if not specified) ===
if [ -z "$MODEL_SIZE" ]; then
  MODEL_NAME=small
else
  MODEL_NAME=$MODEL_SIZE
fi

# Validate model size
case $MODEL_NAME in
  tiny|tiny-deep|tiny-deeper|tiny-deepest|tiny-deepestest|sminy|micro|mini|miny|small)
    ;;
  *)
    echo "Unknown model size: $MODEL_SIZE"
    echo "Valid sizes: tiny, tiny-deep, tiny-deeper, tiny-deepest, tiny-deepestest, sminy, micro, mini, miny, small"
    exit 1
    ;;
esac

# === Build run name ===
# Format: {method}-{size}-{data}[-suffix] (matches train_v2.sh naming)
if [ "$MODEL_NAME" = "small" ]; then
  # Don't include size in name if it's the default
  METHOD_SIZE_PART=${METHOD}
else
  METHOD_SIZE_PART=${METHOD}-${MODEL_NAME}
fi

RUN_NAME=${METHOD_SIZE_PART}-${DATA}

# Build checkpoint run name (may differ for variants that share checkpoints)
if [ -n "$CKPT_METHOD" ]; then
  if [ "$MODEL_NAME" = "small" ]; then
    CKPT_RUN_NAME=${CKPT_METHOD}-${DATA}${RUN_SUFFIX}
  else
    CKPT_RUN_NAME=${CKPT_METHOD}-${MODEL_NAME}-${DATA}${RUN_SUFFIX}
  fi
else
  CKPT_RUN_NAME=${RUN_NAME}${RUN_SUFFIX}
fi

# === Environment setup ===
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

if [[ "$DATA" == "zebra" || "$DATA" == "sudoku-puzzle" ]]; then
  if [ -z "$ESOLM_PUZZLE_DIR" ]; then
    echo "Error: ESOLM_PUZZLE_DIR environment variable is not set"
    echo "Set it to the directory containing puzzle data files"
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

# Set checkpoint directory based on dataset
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

# Build generation name with checkpoint stem (include run-suffix and gen-suffix if set)
CKPT_STEM=${CKPT_NAME%.ckpt}
if [ "$ALGO" = "ar" ]; then
  GEN_NAME=${RUN_NAME}${RUN_SUFFIX}-gen-${CKPT_STEM}${GEN_SUFFIX}
else
  GEN_NAME=${RUN_NAME}${RUN_SUFFIX}-gen-steps-${STEPS}-${CKPT_STEM}${GEN_SUFFIX}
fi
WORKING_DIR=${DATADIR}/runs/${GEN_NAME}

# === Print configuration ===
echo "=== Generation Configuration ==="
echo "METHOD: $METHOD"
echo "MODEL_SIZE: $MODEL_NAME"
echo "DATA: $DATA"
echo "ALGO: $ALGO"
echo "GEN_MODE: $GEN_MODE"
echo "RUN_NAME: $RUN_NAME"
if [ -n "$CKPT_METHOD" ]; then
  echo "CKPT_METHOD: $CKPT_METHOD (loading checkpoint from ${CKPT_RUN_NAME})"
fi
if [ -n "$GEN_SUFFIX" ]; then
  echo "GEN_SUFFIX: $GEN_SUFFIX"
fi
echo "GEN_NAME: $GEN_NAME"
if [ "$ALGO" != "ar" ]; then
  echo "STEPS: $STEPS"
fi
echo "NUM_SAMPLE_BATCHES: $NUM_SAMPLE_BATCHES"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "MODEL_LENGTH: $MODEL_LENGTH"
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
    model=${MODEL_NAME} \
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
    model=${MODEL_NAME} \
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

# Add data paths if set (for zebra, sudoku-puzzle)
if [ -n "$TRAIN_DATA_PATH" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  data.train_data_path=${TRAIN_DATA_PATH} \
  data.valid_data_path=${VALID_DATA_PATH}"
fi

# Add target_givens override if set
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
