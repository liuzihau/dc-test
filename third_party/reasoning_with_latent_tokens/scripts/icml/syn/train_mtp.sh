#!/bin/bash
#SBATCH -J train-mtp
#SBATCH --partition=preempt
#SBATCH --output=slurm/syn-mtp/%j_%x.out
#SBATCH --error=slurm/syn-mtp/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --time=48:00:00
#SBATCH --mem=100G

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

set -e

# === Configuration ===
# Usage: sbatch train_mtp.sh <method> [data]
#   method: ar-mtp-window-X where X is the window size
#           Valid window sizes: -1 (full), 1, 4, 16, 64, 256
#           Or ar-mtp-lossratio-X where X is the loss ratio (0.01-0.1)
#           Or ar-mtp-window-X-random for random span mode
#   data:   dataset name (default: sudoku-large)
#
# Examples:
#   sbatch train_mtp.sh ar-mtp-window--1      # full window (no restriction)
#   sbatch train_mtp.sh ar-mtp-window-16
#   sbatch train_mtp.sh ar-mtp-window-64 sudoku-small
#   sbatch train_mtp.sh ar-mtp-lossratio-0.01  # loss ratio sweep
#   sbatch train_mtp.sh ar-mtp-window-64-random  # random span mode

METHOD=${1:-ar-mtp-window--1}
DATA=${2:-sudoku-large}

# Base config for all MTP models (same as ar-mtp-fix)
AR_NOISE=True
NEXT_TOKEN_PREDICTION=False
DIFFUSION_ATTN_MODE=causal_context
MTP_LOSS_RATIO=""  # Default to empty (not set)
MTP_MODE="contiguous"  # Default to empty (uses 'contiguous' mode)
SEED=1

# Set method-specific parameters (extract window size from method name)
case $METHOD in
  ar-mtp-window--1-maskfix)
    MTP_WINDOW_SIZE=-1
    ;;
  ar-mtp-window-1)
    MTP_WINDOW_SIZE=1
    ;;
  ar-mtp-window-1-maskfix)
    MTP_WINDOW_SIZE=1
    ;;
  ar-ntp-window-1-maskfix)
    MTP_WINDOW_SIZE=1
    NEXT_TOKEN_PREDICTION=True
    ;;
  ar-mtp-window-1-maskfix-nocache)
    MTP_WINDOW_SIZE=1
    ;;
  ar-mtp-window-1-maskfix-yescache)
    MTP_WINDOW_SIZE=1
    ;;
  ar-mtp-window-8-maskfix)
    MTP_WINDOW_SIZE=8
    ;;
  ar-mtp-window-32-maskfix)
    MTP_WINDOW_SIZE=32
    ;;
  ar-mtp-window-64-maskfix)
    MTP_WINDOW_SIZE=64
    ;;
  ar-mtp-window-128-maskfix)
    MTP_WINDOW_SIZE=128
    ;;
  ar-ntp-window-8-maskfix)
    MTP_WINDOW_SIZE=8
    NEXT_TOKEN_PREDICTION=True
    ;;
  ar-ntp-window-32-maskfix)
    MTP_WINDOW_SIZE=32
    NEXT_TOKEN_PREDICTION=True
    ;;
  ar-ntp-window-64-maskfix)
    MTP_WINDOW_SIZE=64
    NEXT_TOKEN_PREDICTION=True
    ;;
  ar-ntp-window-128-maskfix)
    MTP_WINDOW_SIZE=128
    NEXT_TOKEN_PREDICTION=True
    ;;
  ar-ntp-window--1-maskfix)
    MTP_WINDOW_SIZE=-1
    NEXT_TOKEN_PREDICTION=True
    ;;
  ar-mtp-window-1-seed-2)
    MTP_WINDOW_SIZE=1
    SEED=2
    ;;
  ar-mtp-window-1-seed-42)
    MTP_WINDOW_SIZE=1
    SEED=42
    ;;
  ar-mtp-window-1-debug)
    MTP_WINDOW_SIZE=1
    DIFFUSION_ATTN_MODE=causal
    ;;
  ar-mtp-window-4)
    MTP_WINDOW_SIZE=4
    ;;
  ar-mtp-window-8)
    MTP_WINDOW_SIZE=8
    ;;
  ar-mtp-window-32)
    MTP_WINDOW_SIZE=32
    ;;
  ar-mtp-window-64)
    MTP_WINDOW_SIZE=64
    ;;
  ar-mtp-window-256)
    MTP_WINDOW_SIZE=256
    ;;
  ar-mtp-window-1024)
    MTP_WINDOW_SIZE=1024
    ;;
  ar-mtp-lossratio-0.01)
    MTP_WINDOW_SIZE=-1
    MTP_LOSS_RATIO=0.01
    ;;
  ar-mtp-lossratio-0.02)
    MTP_WINDOW_SIZE=-1
    MTP_LOSS_RATIO=0.02
    ;;
  ar-mtp-lossratio-0.03)
    MTP_WINDOW_SIZE=-1
    MTP_LOSS_RATIO=0.03
    ;;
  ar-mtp-lossratio-0.04)
    MTP_WINDOW_SIZE=-1
    MTP_LOSS_RATIO=0.04
    ;;
  ar-mtp-lossratio-0.05)
    MTP_WINDOW_SIZE=-1
    MTP_LOSS_RATIO=0.05
    ;;
  ar-mtp-lossratio-0.06)
    MTP_WINDOW_SIZE=-1
    MTP_LOSS_RATIO=0.06
    ;;
  ar-mtp-lossratio-0.07)
    MTP_WINDOW_SIZE=-1
    MTP_LOSS_RATIO=0.07
    ;;
  ar-mtp-lossratio-0.08)
    MTP_WINDOW_SIZE=-1
    MTP_LOSS_RATIO=0.08
    ;;
  ar-mtp-lossratio-0.09)
    MTP_WINDOW_SIZE=-1
    MTP_LOSS_RATIO=0.09
    ;;
  ar-mtp-lossratio-0.1)
    MTP_WINDOW_SIZE=-1
    MTP_LOSS_RATIO=0.1
    ;;
  ar-mtp-lossratio-0.2)
    MTP_WINDOW_SIZE=-1
    MTP_LOSS_RATIO=0.2
    ;;
  ar-mtp-window--1-random)
    MTP_WINDOW_SIZE=-1
    MTP_MODE=random
    ;;
  ar-mtp-window-1-random)
    MTP_WINDOW_SIZE=1
    MTP_MODE=random
    ;;
  ar-mtp-window-4-random)
    MTP_WINDOW_SIZE=4
    MTP_MODE=random
    ;;
  ar-mtp-window-8-random)
    MTP_WINDOW_SIZE=8
    MTP_MODE=random
    ;;
  ar-mtp-window-32-random)
    MTP_WINDOW_SIZE=32
    MTP_MODE=random
    ;;
  ar-mtp-window-64-random)
    MTP_WINDOW_SIZE=64
    MTP_MODE=random
    ;;
  ar-mtp-window-128-random)
    MTP_WINDOW_SIZE=128
    MTP_MODE=random
    ;;
  ar-mtp-window-256-random)
    MTP_WINDOW_SIZE=256
    MTP_MODE=random
    ;;
  ar-mtp-window-1024-random)
    MTP_WINDOW_SIZE=1024
    MTP_MODE=random
    ;;
  *)
    echo "Unknown method: $METHOD"
    echo "Valid methods: ar-mtp-window-{-1,1,4,8,32,64,256,1024} or ar-mtp-lossratio-{0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1} or ar-mtp-window-{-1,1,4,8,32,64,128,256,1024}-random"
    exit 1
    ;;
esac

# Set data-specific parameters
case $DATA in
  sudoku-small)
    MODEL_LENGTH=128
    BATCH_SIZE=128
    MAX_EPOCHS=100
    ;;
  sudoku-large)
    MODEL_LENGTH=1536
    BATCH_SIZE=64
    MAX_EPOCHS=50
    ;;
  *)
    echo "Unknown data: $DATA"
    echo "Valid data: sudoku-small, sudoku-large"
    exit 1
    ;;
esac

# Build run name
RUN_NAME=${METHOD}-${DATA}

# === Environment setup ===
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate esolm

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/hf_home}"
export DATADIR="$HOME/"

CACHE_DIR=${DATADIR}/cache
WORKING_DIR=${DATADIR}/runs/${RUN_NAME}
CHECKPOINT_DIR=${DATADIR}/checkpoints/${RUN_NAME}

echo "=== Run Configuration ==="
echo "METHOD: $METHOD"
echo "DATA: $DATA"
echo "RUN_NAME: $RUN_NAME"
echo "AR_NOISE: $AR_NOISE"
echo "NEXT_TOKEN_PREDICTION: $NEXT_TOKEN_PREDICTION"
echo "DIFFUSION_ATTN_MODE: $DIFFUSION_ATTN_MODE"
echo "MTP_WINDOW_SIZE: $MTP_WINDOW_SIZE"
if [ -n "$MTP_LOSS_RATIO" ]; then
  echo "MTP_LOSS_RATIO: $MTP_LOSS_RATIO"
fi
if [ -n "$MTP_MODE" ]; then
  echo "MTP_MODE: $MTP_MODE"
fi
echo "CACHE_DIR: $CACHE_DIR"
echo "WORKING_DIR: $WORKING_DIR"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "========================="

# Build python command
PYTHON_CMD="python main.py \
  --config-name=experiment_base \
  data=${DATA} \
  loader.batch_size=${BATCH_SIZE} \
  loader.eval_batch_size=${BATCH_SIZE} \
  wandb.name=${RUN_NAME} \
  algo=difflm \
  algo.diffusion_attn_mode=${DIFFUSION_ATTN_MODE} \
  model.length=${MODEL_LENGTH} \
  algo.ar_noise=${AR_NOISE} \
  algo.next_token_prediction=${NEXT_TOKEN_PREDICTION} \
  algo.mtp_window_size=${MTP_WINDOW_SIZE}"

# Add mtp_loss_ratio if set
if [ -n "$MTP_LOSS_RATIO" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  algo.mtp_loss_ratio=${MTP_LOSS_RATIO}"
fi

# Add mtp_mode if set
if [ -n "$MTP_MODE" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  algo.mtp_mode=${MTP_MODE}"
fi

# Add remaining arguments
PYTHON_CMD="${PYTHON_CMD} \
  data.cache_dir=${CACHE_DIR} \
  hydra.run.dir=${WORKING_DIR} \
  checkpointing.save_dir=${CHECKPOINT_DIR} \
  checkpointing.resume_from_ckpt=True \
  trainer.val_check_interval=1000 \
  trainer.log_every_n_steps=100 \
  trainer.max_epochs=${MAX_EPOCHS} \
  wandb.id=null \
  seed=${SEED}"

# Execute command
eval $PYTHON_CMD

