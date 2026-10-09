#!/bin/bash
#SBATCH -J train
#SBATCH --partition=preempt
#SBATCH --output=slurm/real/%j_%x.out
#SBATCH --error=slurm/real/%j_%x.err
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
# Usage: sbatch train.sh <method> [data]
#   method: ar-ntp | ar | diffu
#   data:   dataset name (default: sudoku-small)
#
# Examples:
#   sbatch train.sh ar-ntp
#   sbatch train.sh diffu sudoku-large

METHOD=${1:-ar}
DATA=${2:-openwebtext-split}
RESUME_FROM_CKPT=True

# Set method-specific parameters
case $METHOD in
  ar-ntp-fix)
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=True
    LOSS_TYPE=low_var
    DIFFUSION_ATTN_MODE=causal_context
    ;;
  ar-ntp-truncate)
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=True
    LOSS_TYPE=low_var
    DIFFUSION_ATTN_MODE=causal
    ;;
  ar-diffu-fix)
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=low_var
    DIFFUSION_ATTN_MODE=causal_context
    ;;
  diffu)
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_context
    ;;
  *)
    echo "Unknown method: $METHOD"
    echo "Valid methods: ar-ntp, ar-diffu, diffu"
    exit 1
    ;;
esac

RUN_NAME=${METHOD}-${DATA}
BATCH_SIZE=32

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
  model.length=1024 \
  algo.ar_noise=${AR_NOISE} \
  algo.next_token_prediction=${NEXT_TOKEN_PREDICTION} \
  algo.loss_type=${LOSS_TYPE} \
  data.cache_dir=${CACHE_DIR} \
  hydra.run.dir=${WORKING_DIR} \
  checkpointing.save_dir=${CHECKPOINT_DIR} \
  checkpointing.resume_from_ckpt=${RESUME_FROM_CKPT} \
  trainer.val_check_interval=10000 \
  trainer.log_every_n_steps=1000 \
  wandb.id=null
