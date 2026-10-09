#!/bin/bash
#SBATCH -J train-ar
#SBATCH --partition=preempt
#SBATCH --output=slurm/syn-ar/%j_%x.out
#SBATCH --error=slurm/syn-ar/%j_%x.err
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

DATA=${1:-sudoku-large}

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
    MAX_EPOCHS=15
    ;;
  text8)
    MODEL_LENGTH=128
    BATCH_SIZE=64
    MAX_EPOCHS=50
    ;;
  *)
    echo "Unknown data: $DATA"
    echo "Valid data: sudoku-small, sudoku-large, text8"
    exit 1
    ;;
esac

RUN_NAME=ar-parallel-${DATA}-${MAX_EPOCHS}-epochs

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
  checkpointing.resume_from_ckpt=True \
  trainer.val_check_interval=1000 \
  trainer.log_every_n_steps=100 \
  trainer.max_epochs=${MAX_EPOCHS} \
  wandb.id=null \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=500
