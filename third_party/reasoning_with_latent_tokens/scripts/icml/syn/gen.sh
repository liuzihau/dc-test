#!/bin/bash
#SBATCH -J gen
#SBATCH --partition=preempt
#SBATCH --output=slurm/syn-gen/%j_%x.out
#SBATCH --error=slurm/syn-gen/%j_%x.err
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
# Usage: sbatch gen.sh <method> [data] [steps] [num_sample_batches]
#   method: ar-ntp | ar-diffu | diffu | diffu-full | diffu-full-lowvar
#   data:   dataset name (default: sudoku-large)
#   steps:  number of sampling steps (default: 128)
#   num_sample_batches: number of batches to generate (default: 1)
#
# Examples:
#   sbatch gen.sh diffu
#   sbatch gen.sh diffu sudoku-large 64
#   sbatch gen.sh diffu-full sudoku-small 128 10

METHOD=${1:-diffu}
DATA=${2:-sudoku-large}
STEPS=${3:-128}
NUM_SAMPLE_BATCHES=${4:-1}

MTP_WINDOW_SIZE=-1
DIFFUSION_ATTN_MODE=causal_context
LOSS_TYPE=low_var

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
  ar-mtp-window-1-maskfix)
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    MTP_WINDOW_SIZE=1
    ;;
  ar-mtp-window-8-maskfix)
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    MTP_WINDOW_SIZE=8
    ;;
  ar-mtp-window-32-maskfix)
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    MTP_WINDOW_SIZE=32
    ;;
  ar-mtp-window-64-maskfix)
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    MTP_WINDOW_SIZE=64
    ;;
  ar-mtp-window-128-maskfix)
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    MTP_WINDOW_SIZE=128
    ;;
  ar-mtp-maskfix)
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    MTP_WINDOW_SIZE=-1
    ;;
  ar-ntp-window-8-maskfix)
    AR_NOISE=True
    MTP_WINDOW_SIZE=8
    NEXT_TOKEN_PREDICTION=True
    ;;
  ar-ntp-window-32-maskfix)
    AR_NOISE=True
    MTP_WINDOW_SIZE=32
    NEXT_TOKEN_PREDICTION=True
    ;;
  ar-ntp-window-64-maskfix)
    AR_NOISE=True
    MTP_WINDOW_SIZE=64
    NEXT_TOKEN_PREDICTION=True
    ;;
  ar-ntp-window-128-maskfix)
    AR_NOISE=True
    MTP_WINDOW_SIZE=128
    NEXT_TOKEN_PREDICTION=True
    ;;
  diffu-maskfix)
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    ;;
  diffu)
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_context
    ;;
  diffu-full)
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=full
    ;;
  diffu-full-lowvar)
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
    BATCH_SIZE=64
    ;;
  sudoku-large)
    MODEL_LENGTH=1536
    BATCH_SIZE=16
    ;;
  *)
    echo "Unknown data: $DATA"
    echo "Valid data: sudoku-small, sudoku-large"
    exit 1
    ;;
esac

RUN_NAME=${METHOD}-${DATA}
GEN_NAME=${RUN_NAME}-gen-steps-${STEPS}

# === Environment setup ===
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate esolm

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/hf_home}"
export DATADIR="$HOME/"

CACHE_DIR=${DATADIR}/cache
WORKING_DIR=${DATADIR}/runs/${GEN_NAME}
CHECKPOINT_DIR=${DATADIR}/checkpoints/${RUN_NAME}

# Find the latest best checkpoint (handles best.ckpt, best-v1.ckpt, best-v2.ckpt, etc.)
CKPT_PATH=$(ls -v ${CHECKPOINT_DIR}/checkpoints/best*.ckpt 2>/dev/null | tail -1)
if [ -z "$CKPT_PATH" ]; then
  echo "Error: No checkpoint found in ${CHECKPOINT_DIR}/checkpoints/"
  exit 1
fi

echo "=== Generation Configuration ==="
echo "METHOD: $METHOD"
echo "DATA: $DATA"
echo "RUN_NAME: $RUN_NAME"
echo "GEN_NAME: $GEN_NAME"
echo "STEPS: $STEPS"
echo "NUM_SAMPLE_BATCHES: $NUM_SAMPLE_BATCHES"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "MODEL_LENGTH: $MODEL_LENGTH"
echo "AR_NOISE: $AR_NOISE"
echo "NEXT_TOKEN_PREDICTION: $NEXT_TOKEN_PREDICTION"
echo "LOSS_TYPE: $LOSS_TYPE"
echo "DIFFUSION_ATTN_MODE: $DIFFUSION_ATTN_MODE"
echo "MTP_WINDOW_SIZE: $MTP_WINDOW_SIZE"
echo "CACHE_DIR: $CACHE_DIR"
echo "WORKING_DIR: $WORKING_DIR"
echo "CKPT_PATH: $CKPT_PATH"
echo "================================"

python main.py \
  --config-name=experiment_base \
  data=${DATA} \
  loader.batch_size=${BATCH_SIZE} \
  loader.eval_batch_size=${BATCH_SIZE} \
  wandb.name=${GEN_NAME} \
  algo=difflm \
  algo.diffusion_attn_mode=${DIFFUSION_ATTN_MODE} \
  model.length=${MODEL_LENGTH} \
  algo.ar_noise=${AR_NOISE} \
  algo.next_token_prediction=${NEXT_TOKEN_PREDICTION} \
  algo.loss_type=${LOSS_TYPE} \
  algo.mtp_window_size=${MTP_WINDOW_SIZE} \
  data.cache_dir=${CACHE_DIR} \
  hydra.run.dir=${WORKING_DIR} \
  mode=sample_eval \
  eval.generate_samples=True \
  eval.checkpoint_path=${CKPT_PATH} \
  sampling=synthetic_base \
  sampling.steps=${STEPS} \
  sampling.num_sample_batches=${NUM_SAMPLE_BATCHES} \
  sampling.kv_cache=False
