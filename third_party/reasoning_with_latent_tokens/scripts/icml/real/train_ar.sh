#!/bin/bash
#SBATCH -J train-ar
#SBATCH --partition=preempt
#SBATCH --output=slurm/real-ar/%j_%x.out
#SBATCH --error=slurm/real-ar/%j_%x.err
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
# Usage: sbatch train_ar.sh [data]
#   data: dataset name (default: sudoku-large)
#
# Examples:
#   sbatch train_ar.sh
#   sbatch train_ar.sh sudoku-small
#   sbatch train_ar.sh sudoku-large

DEBUG=${DEBUG:-false}
DATA=${1:-openwebtext-split}

# Debug mode settings
if [ "$DEBUG" = "true" ]; then
  RESUME_FROM_CKPT=False
  MAX_STEPS=10
  VALIDATE_AT_START=True
else
  RESUME_FROM_CKPT=True
  MAX_STEPS=null
  VALIDATE_AT_START=False
fi

# Set data-specific parameters
case $DATA in
  openwebtext-split)
    MODEL_LENGTH=1024
    BATCH_SIZE=32
    ;;
  *)
    echo "Unknown data: $DATA"
    echo "Valid data: openwebtext-split"
    exit 1
    ;;
esac

RUN_NAME=ar-baseline-${DATA}

# === Environment setup ===
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate esolm

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/hf_home}"
export DATADIR="$HOME/"
export GCS_DIR="gs://YOUR_BUCKET/"

CACHE_DIR=${DATADIR}/cache
WORKING_DIR=${DATADIR}/runs/${RUN_NAME}
CHECKPOINT_DIR=${GCS_DIR}/checkpoints/${RUN_NAME}

echo "=== Run Configuration ==="
echo "DEBUG: $DEBUG"
echo "DATA: $DATA"
echo "RUN_NAME: $RUN_NAME"
echo "MODEL_LENGTH: $MODEL_LENGTH"
echo "CACHE_DIR: $CACHE_DIR"
echo "WORKING_DIR: $WORKING_DIR"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "========================="

python main.py \
  --config-name=experiment_base \
  data=${DATA} \
  loader.batch_size=${BATCH_SIZE} \
  loader.eval_batch_size=${BATCH_SIZE} \
  wandb.name=${RUN_NAME} \
  algo=ar \
  model.length=${MODEL_LENGTH} \
  data.cache_dir=${CACHE_DIR} \
  hydra.run.dir=${WORKING_DIR} \
  checkpointing.save_dir=${CHECKPOINT_DIR} \
  checkpointing.resume_from_ckpt=${RESUME_FROM_CKPT} \
  trainer.val_check_interval=10000 \
  trainer.log_every_n_steps=1000 \
  +trainer.max_steps=${MAX_STEPS} \
  eval.validate_at_start=${VALIDATE_AT_START} \
  wandb.id=null
