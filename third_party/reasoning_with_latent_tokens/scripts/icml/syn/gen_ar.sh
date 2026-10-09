#!/bin/bash
#SBATCH -J gen-ar
#SBATCH --partition=preempt
#SBATCH --output=slurm/syn-gen/%j_%x.out
#SBATCH --error=slurm/syn-gen/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --time=48:00:00
#SBATCH --mem=100G

set -e

# === Configuration ===
# Usage: sbatch gen_ar.sh [data] [num_sample_batches] [checkpoint_name]
#   data:   dataset name (default: sudoku-large)
#   num_sample_batches: number of batches to generate (default: 1)
#   checkpoint_name: checkpoint filename in the default directory (default: auto-detect best*.ckpt)
#
# Examples:
#   sbatch gen_ar.sh
#   sbatch gen_ar.sh sudoku-small
#   sbatch gen_ar.sh sudoku-large 10
#   sbatch gen_ar.sh sudoku-large 10 best.ckpt
#   sbatch gen_ar.sh sudoku-large 10 10-2000.ckpt

DATA=${1:-sudoku-large}
NUM_SAMPLE_BATCHES=${2:-1}
CKPT_NAME=${3:-}

# Set data-specific parameters
case $DATA in
  sudoku-small)
    MODEL_LENGTH=128
    BATCH_SIZE=64
    ;;
  sudoku-large)
    MODEL_LENGTH=1536
    BATCH_SIZE=16
    ;;
  text8)
    MODEL_LENGTH=128
    BATCH_SIZE=64
    ;;
  *)
    echo "Unknown data: $DATA"
    echo "Valid data: sudoku-small, sudoku-large, text8"
    exit 1
    ;;
esac

RUN_NAME=ar-baseline-${DATA}
# RUN_NAME=ar-parallel-${DATA}-15-epochs

# === Environment setup ===
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate esolm

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/hf_home}"
export DATADIR="$HOME/"

CACHE_DIR=${DATADIR}/cache
CHECKPOINT_DIR=${DATADIR}/checkpoints/${RUN_NAME}

# Find or use provided checkpoint
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

# Add checkpoint name (without .ckpt) to GEN_NAME for unique output directory
CKPT_STEM=${CKPT_NAME%.ckpt}
GEN_NAME=${RUN_NAME}-gen-${CKPT_STEM}
WORKING_DIR=${DATADIR}/runs/${GEN_NAME}

echo "=== AR Generation Configuration ==="
echo "DATA: $DATA"
echo "RUN_NAME: $RUN_NAME"
echo "GEN_NAME: $GEN_NAME"
echo "NUM_SAMPLE_BATCHES: $NUM_SAMPLE_BATCHES"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "MODEL_LENGTH: $MODEL_LENGTH"
echo "CACHE_DIR: $CACHE_DIR"
echo "WORKING_DIR: $WORKING_DIR"
echo "CKPT_PATH: $CKPT_PATH"
echo "===================================="

python main.py \
  --config-name=experiment_base \
  data=${DATA} \
  loader.batch_size=${BATCH_SIZE} \
  loader.eval_batch_size=${BATCH_SIZE} \
  wandb.name=${GEN_NAME} \
  algo=ar \
  model.length=${MODEL_LENGTH} \
  data.cache_dir=${CACHE_DIR} \
  hydra.run.dir=${WORKING_DIR} \
  mode=sample_eval \
  eval.generate_samples=True \
  eval.checkpoint_path=${CKPT_PATH} \
  sampling.kv_cache=True \
  sampling.num_sample_batches=${NUM_SAMPLE_BATCHES} \
  wandb.id=null
