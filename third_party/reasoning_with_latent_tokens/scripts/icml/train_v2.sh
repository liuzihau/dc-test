#!/bin/bash
#SBATCH -J train-v2
#SBATCH --partition=general
#SBATCH --output=slurm/master/%j_%x.out
#SBATCH --error=slurm/master/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --open-mode=append
#SBATCH --time=48:00:00
#SBATCH --mem=128G

# Training script with separated method and model size
#
# Usage: sbatch train_v2.sh --method=<method> --data=<data> [--size=<size>] [options] [-- extra_args]
#
# Required:
#   --method, -m    Model method/architecture (ar, diffu-causal-output, diffu-full, etc.)
#   --data, -d      Dataset (sudoku-small, sudoku-large, game-of-24, zebra, etc.)
#
# Optional:
#   --size, -z      Model size (tiny, tiny-deep, tiny-deeper, sminy, micro, mini, miny, small)
#                   Default: small (or dataset-specific default)
#   --resume, -r    Resume from checkpoint (default: true)
#   --no-resume     Start fresh run (equivalent to --resume=false)
#   --seed, -s      Random seed (default: 1, included in run name if not 1)
#   --run-suffix    Suffix to add to run name (e.g., -exp1)
#
# Pass-through:
#   -- <args>       Additional arguments passed directly to main.py
#
# Examples:
#   sbatch train_v2.sh -m ar -d sudoku-small
#   sbatch train_v2.sh -m diffu-causal-output -z tiny-deep -d game-of-24
#   sbatch train_v2.sh --method=diffu-causal --size=sminy --data=sudoku-conditional-192-0given
#   sbatch train_v2.sh -m ar-ntp -d sudoku-small -- optim.lr=1e-4
#
# Methods:
#   ar                  - Pure autoregressive baseline
#   ar-ntp              - AR noise + next token prediction with causal attention
#   ar-mtp-window-N     - AR noise with MTP window size N (32, 128, -1)
#   ar-mtp-causal-context - AR noise with causal_context attention, full window (alias for ar-mtp-window--1)
#   ar-mtp-full         - AR noise with full attention MTP
#   ar-causal-output    - AR noise with causal_output attention
#   diffu               - Pure diffusion with causal_context attention
#   diffu-full          - Diffusion with full (bidirectional) attention
#   diffu-causal        - Diffusion with causal attention
#   diffu-causal-output - Diffusion with causal_output attention (recommended)
#   diffu-solo-causal  - Diffusion with solo_causal attention (masked tokens independent)
#   diffu-solo-full    - Diffusion with solo_full attention (masked tokens independent, clean bidirectional)
#   dp-qk-split-parallel - Diffusion parallel with Q/K split
#   dp-4way             - Diffusion parallel with 4-way heads
#   dp-2way             - Diffusion parallel with 2-way heads
#
# Model sizes:
#   tiny          - ~6M params, 8 blocks
#   tiny-deep     - ~5M params, 12 blocks
#   tiny-deeper   - ~5M params, 16 blocks
#   tiny-deep-narrow  - ~3M params, 12 blocks, hidden=128
#   tiny-deep-narrower - ~2M params, 12 blocks, hidden=96
#   small         - ~45M params (default)

set -e
export NCCL_P2P_DISABLE=1

# === Default values ===
METHOD=""
DATA=""
MODEL_SIZE=""  # Will use dataset default if not specified
RESUME=true
SEED=1
RUN_SUFFIX=""
EXTRA_ARGS=""
NUM_GPUS=1
TRAIN_ON_ALL_TOKENS=""

# === Argument parsing ===
show_help() {
  echo "Usage: sbatch train_v2.sh --method=<method> --data=<data> [--size=<size>] [options] [-- extra_args]"
  echo ""
  echo "Required:"
  echo "  --method, -m    Model method/architecture"
  echo "  --data, -d      Dataset"
  echo ""
  echo "Optional:"
  echo "  --size, -z      Model size (tiny, tiny-deep, tiny-deeper, tiny-deep-narrow, tiny-deep-narrower, sminy, micro, mini, miny, small)"
  echo "  --resume, -r    Resume from checkpoint (default: true)"
  echo "  --no-resume     Start fresh run (equivalent to --resume=false)"
  echo "  --seed, -s      Random seed (default: 1, included in run name if not 1)"
  echo "  --run-suffix    Suffix to add to run name (e.g., -exp1)"
  echo "  --train-on-all-tokens  Train on all tokens (ignore problem/solution distinction)"
  echo "  --gpus, -g      Number of GPUs (default: 1). Must match sbatch --gres"
  echo ""
  echo "Pass-through:"
  echo "  -- <args>       Additional arguments passed directly to main.py"
  echo ""
  echo "Methods: ar, ar-ntp, ar-mtp-window-32, ar-mtp-window-128, ar-mtp-window--1,"
  echo "         ar-mtp-causal-context, ar-mtp-full, ar-causal-output, diffu, diffu-full, diffu-causal,"
  echo "         diffu-causal-output, diffu-solo-causal, diffu-solo-full,"
  echo "         dp-qk-split-parallel, dp-4way, dp-2way"
  echo ""
  echo "Sizes: tiny, tiny-deep, tiny-deeper, tiny-deep-narrow, tiny-deep-narrower, sminy, micro, mini, miny, small"
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
    --resume=*)
      RESUME="${1#*=}"
      shift
      ;;
    --resume|-r)
      if [[ -n "$2" && ! "$2" =~ ^- ]]; then
        RESUME="$2"
        shift 2
      else
        RESUME=true
        shift
      fi
      ;;
    --no-resume)
      RESUME=false
      shift
      ;;
    --seed=*)
      SEED="${1#*=}"
      shift
      ;;
    --seed|-s)
      SEED="$2"
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
    --train-on-all-tokens)
      TRAIN_ON_ALL_TOKENS=True
      shift
      ;;
    --gpus=*)
      NUM_GPUS="${1#*=}"
      shift
      ;;
    --gpus|-g)
      NUM_GPUS="$2"
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
MTP_REBALANCE_LOSS=""
SHUFFLE_CLEAN_TOKENS=""
SHUFFLE_MASKED_TOKENS=""

# === Set method-specific parameters (architecture only, no model size) ===
case $METHOD in
  ar)
    ALGO=ar
    ;;
  ar-ntp)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=True
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=-1
    ;;
  ar-ntp-w1)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=True
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=1
    ;;
  ar-mtp-full)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
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
  ar-mtp-128-full)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=128
    ;;
  diffu-full)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    ;;
  diffu-causal)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=causal
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-causal-output)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-solo-causal)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=solo_causal
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-solo-full)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=solo_full
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  dp-qk-split-parallel)
    ALGO=diffuparallel
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=qk_split
    NEXT_TOKEN_PREDICTION=False
    SHUFFLE_WARMUP=0
    ;;
  dp-4way)
    ALGO=diffuparallel
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=4way_heads
    NEXT_TOKEN_PREDICTION=False
    SHUFFLE_WARMUP=0
    ;;
  dp-2way)
    ALGO=diffuparallel
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=2way_heads
    NEXT_TOKEN_PREDICTION=False
    SHUFFLE_WARMUP=0
    ;;
  *)
    echo "Unknown method: $METHOD"
    echo "Valid methods: ar, ar-ntp, ar-mtp-window-32, ar-mtp-window-128, ar-mtp-window--1,"
    echo "               ar-mtp-causal-context, ar-mtp-full, ar-causal-output, diffu, diffu-full, diffu-causal,"
    echo "               diffu-causal-output, diffu-solo-causal, diffu-solo-full,"
    echo "               dp-qk-split-parallel, dp-4way, dp-2way"
    exit 1
    ;;
esac

# === Data-specific parameters ===
CHECKPOINT_EVERY_N_STEPS=2500
DATA_CONFIG=""
TARGET_GIVENS=""
MAX_EPOCHS=""

case $DATA in
  sudoku-small)
    MODEL_LENGTH=128
    GPU_BATCH_SIZE=128
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=100
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=100
    USE_GCS=False
    ;;
  sudoku-small-solver)
    MODEL_LENGTH=128
    GPU_BATCH_SIZE=128
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=100
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=100
    USE_GCS=False
    DATA_CONFIG=sudoku-small
    EXTRA_ARGS="${EXTRA_ARGS} data.use_solver=True"
    ;;
  sudoku-large)
    MODEL_LENGTH=1536
    GPU_BATCH_SIZE=64
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=50
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=100
    USE_GCS=False
    ;;
  sudoku-puzzle)
    MODEL_LENGTH=192
    GPU_BATCH_SIZE=256
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=20
    VAL_CHECK_INTERVAL=10000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=10000
    USE_GCS=False
    TRAIN_DATA_PATH=${ESOLM_PUZZLE_DIR}/sudoku-train-data.npy
    VALID_DATA_PATH=${ESOLM_PUZZLE_DIR}/sudoku-test-data.npy
    ;;
  zebra)
    MODEL_LENGTH=384
    GPU_BATCH_SIZE=512
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=50
    VAL_CHECK_INTERVAL=50000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=10000
    USE_GCS=False
    TRAIN_DATA_PATH=${ESOLM_PUZZLE_DIR}/zebra-train-data.pkl
    VALID_DATA_PATH=${ESOLM_PUZZLE_DIR}/zebra-test-data.pkl
    ;;
  game-of-24)
    MODEL_LENGTH=64
    GPU_BATCH_SIZE=256
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=50
    VAL_CHECK_INTERVAL=10000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=5000
    USE_GCS=False
    ;;
  sudoku-conditional)
    MODEL_LENGTH=256
    GPU_BATCH_SIZE=256
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=50
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=10000
    USE_GCS=False
    ;;
  sudoku-conditional-192)
    MODEL_LENGTH=192
    GPU_BATCH_SIZE=256
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=50
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=10000
    USE_GCS=False
    DATA_CONFIG=sudoku-conditional
    ;;
  sudoku-conditional-192-0given)
    MODEL_LENGTH=192
    GPU_BATCH_SIZE=256
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=100
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=2000
    USE_GCS=False
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=0
    ;;
  sudoku-conditional-192-10given)
    MODEL_LENGTH=192
    GPU_BATCH_SIZE=256
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=100
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=2000
    USE_GCS=False
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=10
    ;;
  sudoku-conditional-192-20given)
    MODEL_LENGTH=192
    GPU_BATCH_SIZE=256
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=100
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=2000
    USE_GCS=False
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=20
    ;;
  sudoku-conditional-192-30given)
    MODEL_LENGTH=192
    GPU_BATCH_SIZE=256
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=100
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=2000
    USE_GCS=False
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=30
    ;;
  openwebtext-split)
    MODEL_LENGTH=1024
    GPU_BATCH_SIZE=32
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=1
    VAL_CHECK_INTERVAL=10000
    LOG_EVERY_N_STEPS=1000
    USE_GCS=True
    ;;
  *)
    echo "Unknown data: $DATA"
    echo "Valid data: sudoku-small, sudoku-large, sudoku-puzzle, sudoku-conditional,"
    echo "            sudoku-conditional-192-Ngiven, zebra, game-of-24, openwebtext-split"
    exit 1
    ;;
esac

# === Set model size (use default 'small' if not specified) ===
if [ -z "$MODEL_SIZE" ]; then
  MODEL_NAME=small
else
  MODEL_NAME=$MODEL_SIZE
fi

# Validate model size
case $MODEL_NAME in
  tiny|tiny-deep|tiny-deeper|tiny-deep-narrow|tiny-deep-narrower|sminy|micro|mini|miny|small)
    ;;
  *)
    echo "Unknown model size: $MODEL_SIZE"
    echo "Valid sizes: tiny, tiny-deep, tiny-deeper, tiny-deep-narrow, tiny-deep-narrower, sminy, micro, mini, miny, small"
    exit 1
    ;;
esac

# === Build run name ===
# Format: {method}-{size}-{data}[-seedN][-suffix]
if [ "$MODEL_NAME" = "small" ]; then
  # Don't include size in name if it's the default
  METHOD_SIZE_PART=${METHOD}
else
  METHOD_SIZE_PART=${METHOD}-${MODEL_NAME}
fi

# Add -tat suffix if training on all tokens
TAT_SUFFIX=""
if [ -n "$TRAIN_ON_ALL_TOKENS" ]; then
  TAT_SUFFIX="-tat"
fi

if [ "$SEED" = "1" ]; then
  RUN_NAME=${METHOD_SIZE_PART}-${DATA}${TAT_SUFFIX}${RUN_SUFFIX}
else
  RUN_NAME=${METHOD_SIZE_PART}-${DATA}-seed${SEED}${TAT_SUFFIX}${RUN_SUFFIX}
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
WORKING_DIR=${DATADIR}/runs/${RUN_NAME}

if [ "$USE_GCS" = "True" ]; then
  CHECKPOINT_DIR=${GCS_DIR}/checkpoints/${RUN_NAME}
else
  CHECKPOINT_DIR=${DATADIR}/checkpoints/${RUN_NAME}
fi

# === Validate resume flag ===
case $RESUME in
  true|True|TRUE|1)
    RESUME_FROM_CKPT=True
    ;;
  false|False|FALSE|0)
    RESUME_FROM_CKPT=False
    ;;
  *)
    echo "Invalid resume flag: $RESUME"
    exit 1
    ;;
esac

# === Print configuration ===
echo "=== Run Configuration ==="
echo "METHOD: $METHOD"
echo "MODEL_SIZE: $MODEL_NAME"
echo "DATA: $DATA"
echo "ALGO: $ALGO"
echo "NUM_GPUS: $NUM_GPUS"
echo "SEED: $SEED"
if [ -n "$RUN_SUFFIX" ]; then
  echo "RUN_SUFFIX: $RUN_SUFFIX"
fi
echo "RUN_NAME: $RUN_NAME"
echo "RESUME_FROM_CKPT: $RESUME_FROM_CKPT"
echo "MODEL_LENGTH: $MODEL_LENGTH"
echo "GPU_BATCH_SIZE: $GPU_BATCH_SIZE"
echo "EFFECTIVE_BATCH_SIZE: $EFFECTIVE_BATCH_SIZE"
if [ -n "$MAX_EPOCHS" ]; then
  echo "MAX_EPOCHS: $MAX_EPOCHS"
fi
echo "VAL_CHECK_INTERVAL: $VAL_CHECK_INTERVAL"
echo "LOG_EVERY_N_STEPS: $LOG_EVERY_N_STEPS"
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
if [ -n "$SHUFFLE_WARMUP" ]; then
  echo "SHUFFLE_WARMUP: $SHUFFLE_WARMUP"
fi
if [ -n "$TRAIN_ON_ALL_TOKENS" ]; then
  echo "TRAIN_ON_ALL_TOKENS: $TRAIN_ON_ALL_TOKENS"
fi
if [ -n "$MTP_REBALANCE_LOSS" ]; then
  echo "MTP_REBALANCE_LOSS: $MTP_REBALANCE_LOSS"
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
echo "USE_GCS: $USE_GCS"
echo "CACHE_DIR: $CACHE_DIR"
echo "WORKING_DIR: $WORKING_DIR"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "========================="

# === Build Python command ===
if [ -n "$DATA_CONFIG" ]; then
  HYDRA_DATA_CONFIG=$DATA_CONFIG
else
  HYDRA_DATA_CONFIG=$DATA
fi

PYTHON_CMD="python main.py \
  --config-name=experiment_base \
  data=${HYDRA_DATA_CONFIG} \
  seed=${SEED} \
  loader.batch_size=${GPU_BATCH_SIZE} \
  loader.eval_batch_size=${GPU_BATCH_SIZE} \
  loader.global_batch_size=${EFFECTIVE_BATCH_SIZE} \
  loader.eval_global_batch_size=${EFFECTIVE_BATCH_SIZE} \
  wandb.name=${RUN_NAME} \
  wandb.project=Diffusion-${DATA} \
  algo=${ALGO} \
  model=${MODEL_NAME} \
  model.length=${MODEL_LENGTH} \
  data.cache_dir=${CACHE_DIR} \
  hydra.run.dir=${WORKING_DIR} \
  checkpointing.save_dir=${CHECKPOINT_DIR} \
  checkpointing.resume_from_ckpt=${RESUME_FROM_CKPT} \
  trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
  trainer.log_every_n_steps=${LOG_EVERY_N_STEPS} \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY_N_STEPS} \
  wandb.id=null \
  +algo.log_position_losses=True \
  eval.run_task_evaluation=True \
  sampling=synthetic_base \
  sampling.num_sample_batches=1 \
  sampling.steps=${MODEL_LENGTH} \
  sampling.greedy_tokens=True"

# Add unmask_policy=topp for pure diffusion methods (not AR or AR-noise methods)
if [ "$ALGO" = "difflm" ] && [ "$AR_NOISE" = "False" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  sampling.unmask_policy=topp"
fi

if [ -n "$MAX_EPOCHS" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  trainer.max_epochs=${MAX_EPOCHS}"
fi

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

  if [ -n "$MTP_REBALANCE_LOSS" ]; then
    PYTHON_CMD="${PYTHON_CMD} \
  algo.mtp_rebalance_loss=${MTP_REBALANCE_LOSS}"
  fi

  if [ -n "$SHUFFLE_CLEAN_TOKENS" ]; then
    PYTHON_CMD="${PYTHON_CMD} \
  algo.shuffle_clean_tokens=${SHUFFLE_CLEAN_TOKENS}"
  fi
  if [ -n "$SHUFFLE_MASKED_TOKENS" ]; then
    PYTHON_CMD="${PYTHON_CMD} \
  algo.shuffle_masked_tokens=${SHUFFLE_MASKED_TOKENS}"
  fi
fi

if [ -n "$TRAIN_ON_ALL_TOKENS" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  training.train_on_all_tokens=${TRAIN_ON_ALL_TOKENS}"
fi

if [ -n "$TRAIN_DATA_PATH" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  data.train_data_path=${TRAIN_DATA_PATH} \
  data.valid_data_path=${VALID_DATA_PATH}"
fi

if [ -n "$TARGET_GIVENS" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  data.target_givens=${TARGET_GIVENS}"
fi

if [ "$ALGO" = "diffuparallel" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  algo.loss_type=${LOSS_TYPE} \
  algo.diffusion_shuffle=${DIFFUSION_SHUFFLE} \
  algo.shuffle_warmup=${SHUFFLE_WARMUP} \
  algo.next_token_prediction=${NEXT_TOKEN_PREDICTION} \
  algo.pos_encoding_strategy=${POS_ENCODING_STRATEGY}"
fi

# Multi-GPU support
PYTHON_CMD="${PYTHON_CMD} trainer.devices=${NUM_GPUS}"

if [ -n "$EXTRA_ARGS" ]; then
  PYTHON_CMD="${PYTHON_CMD} ${EXTRA_ARGS}"
fi

# Execute
if [ "$NUM_GPUS" -gt 1 ]; then
  echo "Executing (multi-GPU with srun): srun $PYTHON_CMD"
  srun $PYTHON_CMD
else
  echo "Executing: $PYTHON_CMD"
  eval $PYTHON_CMD
fi
