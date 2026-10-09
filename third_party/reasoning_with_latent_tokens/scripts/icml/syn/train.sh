#!/bin/bash
#SBATCH -J train
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
# Usage: sbatch train.sh <method> [data] [disable_adaln]
#   method: ar-ntp | ar-diffu | diffu | diffu-full | diffu-full-lowvar
#   data:   dataset name (default: sudoku-small)
#   disable_adaln: True/False (default: False) - for debugging
#
# Examples:
#   sbatch train.sh ar-ntp
#   sbatch train.sh diffu sudoku-large
#   sbatch train.sh diffu sudoku-small True  # disable adaLN for debugging

METHOD=${1:-ar-diffu}
DATA=${2:-sudoku-large}
LOSS_TYPE=low_var

# Set method-specific parameters
case $METHOD in
  ar-ntp-maskfix)
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=True
    DIFFUSION_ATTN_MODE=causal_context
    ;;
  ar-ntp-truncate)
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=True
    DIFFUSION_ATTN_MODE=causal
    ;;
  ar-ntp-truncate-1212)
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=True
    DIFFUSION_ATTN_MODE=causal
    ;;
  ar-mtp-maskfix)
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=causal_context
    ;;
  diffu-maskfix)
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_context
    ;;
  diffu-full-maskfix)
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=full
    ;;
  diffu-full-lowvar-maskfix)
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=low_var
    DIFFUSION_ATTN_MODE=full
    ;;
  *)
    echo "Unknown method: $METHOD"
    echo "Valid methods: ar-ntp, ar-diffu, diffu, diffu-full, diffu-full-lowvar"
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
echo "LOSS_TYPE: $LOSS_TYPE"
echo "DIFFUSION_ATTN_MODE: $DIFFUSION_ATTN_MODE"
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
  algo=difflm \
  algo.diffusion_attn_mode=${DIFFUSION_ATTN_MODE} \
  model.length=${MODEL_LENGTH} \
  algo.ar_noise=${AR_NOISE} \
  algo.next_token_prediction=${NEXT_TOKEN_PREDICTION} \
  algo.loss_type=${LOSS_TYPE} \
  data.cache_dir=${CACHE_DIR} \
  hydra.run.dir=${WORKING_DIR} \
  checkpointing.save_dir=${CHECKPOINT_DIR} \
  checkpointing.resume_from_ckpt=True \
  trainer.val_check_interval=1000 \
  trainer.log_every_n_steps=100 \
  trainer.max_epochs=${MAX_EPOCHS} \
  wandb.id=null
