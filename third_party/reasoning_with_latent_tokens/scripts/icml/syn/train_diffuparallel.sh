#!/bin/bash
#SBATCH -J train-diffuparallel
#SBATCH --partition=preempt
#SBATCH --output=slurm/syn-diffuparallel/%j_%x.out
#SBATCH --error=slurm/syn-diffuparallel/%j_%x.err
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
# Usage: sbatch train_diffuparallel.sh [method] [data] [loss_type]
#   method:    method variant (default: dp)
#   data:      dataset name (default: sudoku-large)
#   loss_type: elbo | low_var (default: elbo)
#
# Examples:
#   sbatch train_diffuparallel.sh
#   sbatch train_diffuparallel.sh dp sudoku-small
#   sbatch train_diffuparallel.sh dp-noshuffle sudoku-large low_var
#   sbatch train_diffuparallel.sh dp-warmup sudoku-small   # uses shuffle warmup

METHOD=${1:-dp}
DATA=${2:-sudoku-small}
LOSS_TYPE=elbo

# Set method-specific parameters
DIFFUSION_SHUFFLE=True
SHUFFLE_WARMUP=0  # default: no warmup
NEXT_TOKEN_PREDICTION=False  # default: on
POS_ENCODING_STRATEGY=target  # default: original behavior


case $METHOD in
  dp)
    DIFFUSION_SHUFFLE=True
    ;;
  dp-noshuffle)
    DIFFUSION_SHUFFLE=False
    ;;
  dp-noshuffle-split-heads)
    DIFFUSION_SHUFFLE=False
    POS_ENCODING_STRATEGY=split_avg_heads
    ;;
  dp-noshuffle-split-abs)
    DIFFUSION_SHUFFLE=False
    POS_ENCODING_STRATEGY=split_abs_source
    ;;
  dp-warmup)
    DIFFUSION_SHUFFLE=True
    SHUFFLE_WARMUP=5000  # warmup shuffle_alpha over 5000 steps
    ;;
  dp-warmup-100k)
    DIFFUSION_SHUFFLE=True
    SHUFFLE_WARMUP=100000  # warmup shuffle_alpha over 100000 steps
    ;;
  dp-ntp)
    DIFFUSION_SHUFFLE=True
    NEXT_TOKEN_PREDICTION=True  # next token prediction: single random position loss
    ;;
  dp-split-abs)
    # Strategy 1: abs_pos uses source position, rotary uses target position
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=split_abs_source
    ;;
  dp-split-heads)
    # Strategy 2: abs_pos uses (source+target)/2, rotary split by heads
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=split_avg_heads
    ;;
  dp-split-heads-noshuffle)
    # Strategy 2: abs_pos uses (source+target)/2, rotary split by heads
    DIFFUSION_SHUFFLE=False
    POS_ENCODING_STRATEGY=split_avg_heads
    ;;
  dp-split-abs-warmup)
    # Strategy 1 with shuffle warmup
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=split_abs_source
    SHUFFLE_WARMUP=5000
    ;;
  dp-split-heads-warmup)
    # Strategy 2 with shuffle warmup
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=split_avg_heads
    SHUFFLE_WARMUP=5000
    ;;
  dp-qk-split-parallel)
    # Q/K split: abs_pos uses source, rotary Q=target K=source
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=qk_split
    NEXT_TOKEN_PREDICTION=False
    ;;
  dp-qk-split-ntp)
    # Q/K split: abs_pos uses source, rotary Q=target K=source
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=qk_split
    NEXT_TOKEN_PREDICTION=True
    ;;
  dp-qk-split-noshuffle)
    # Q/K split without shuffling (for debugging)
    DIFFUSION_SHUFFLE=False
    POS_ENCODING_STRATEGY=qk_split
    ;;
  dp-qk-split-warmup)
    # Q/K split with shuffle warmup
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=qk_split
    SHUFFLE_WARMUP=5000
    ;;
  dp-avg-rotary)
    # Average rotary: Q=(q_src+q_tgt)/2, K=(k_src+k_tgt)/2
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=avg_rotary
    ;;
  dp-4way)
    # 4-way head split: 1/4 each of q_src×k_src, q_src×k_tgt, q_tgt×k_src, q_tgt×k_tgt
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=4way_heads
    ;;
  dp-2way)
    # 2-way head split: keys always use source, half heads q_source, half heads q_target
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=2way_heads
    ;;
  *)
    echo "Unknown method: $METHOD"
    echo "Valid methods: dp, dp-noshuffle, dp-warmup, dp-ntp, dp-split-abs, dp-split-heads, dp-qk-split-parallel, dp-qk-split-ntp, dp-avg-rotary, dp-4way, dp-2way"
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
echo "LOSS_TYPE: $LOSS_TYPE"
echo "DIFFUSION_SHUFFLE: $DIFFUSION_SHUFFLE"
echo "SHUFFLE_WARMUP: $SHUFFLE_WARMUP"
echo "NEXT_TOKEN_PREDICTION: $NEXT_TOKEN_PREDICTION"
echo "POS_ENCODING_STRATEGY: $POS_ENCODING_STRATEGY"
echo "RUN_NAME: $RUN_NAME"
echo "MODEL_LENGTH: $MODEL_LENGTH"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "MAX_EPOCHS: $MAX_EPOCHS"
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
  algo=diffuparallel \
  algo.loss_type=${LOSS_TYPE} \
  model.length=${MODEL_LENGTH} \
  data.cache_dir=${CACHE_DIR} \
  hydra.run.dir=${WORKING_DIR} \
  checkpointing.save_dir=${CHECKPOINT_DIR} \
  checkpointing.resume_from_ckpt=False \
  trainer.val_check_interval=1000 \
  trainer.log_every_n_steps=100 \
  trainer.max_epochs=${MAX_EPOCHS} \
  wandb.id=null \
  algo.diffusion_shuffle=${DIFFUSION_SHUFFLE} \
  algo.shuffle_warmup=${SHUFFLE_WARMUP} \
  algo.next_token_prediction=${NEXT_TOKEN_PREDICTION} \
  algo.pos_encoding_strategy=${POS_ENCODING_STRATEGY}

