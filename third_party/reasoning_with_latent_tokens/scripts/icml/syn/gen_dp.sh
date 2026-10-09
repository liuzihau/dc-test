#!/bin/bash
#SBATCH -J gen-dp
#SBATCH --partition=preempt
#SBATCH --output=slurm/syn-gen-dp/%j_%x.out
#SBATCH --error=slurm/syn-gen-dp/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --time=48:00:00
#SBATCH --mem=100G

# Generation script for DiffuParallel models

set -e

# === Configuration ===
# Usage: sbatch gen_dp.sh <method> [data] [num_sample_batches]
#   method: dp, dp-noshuffle, dp-qk-split-parallel, dp-qk-split-ntp, dp-avg-rotary, dp-4way, etc.
#   data:   dataset name (default: sudoku-small)
#   num_sample_batches: number of batches to generate (default: 1)
#
# Examples:
#   sbatch gen_dp.sh dp-qk-split-parallel
#   sbatch gen_dp.sh dp-qk-split-parallel sudoku-small
#   sbatch gen_dp.sh dp-4way sudoku-small 10

METHOD=${1:-dp}
DATA=${2:-sudoku-small}
NUM_SAMPLE_BATCHES=${3:-1}

# Set method-specific parameters
DIFFUSION_SHUFFLE=True
SHUFFLE_WARMUP=0
NEXT_TOKEN_PREDICTION=False
POS_ENCODING_STRATEGY=target

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
    SHUFFLE_WARMUP=5000
    ;;
  dp-warmup-100k)
    DIFFUSION_SHUFFLE=True
    SHUFFLE_WARMUP=100000
    ;;
  dp-ntp)
    DIFFUSION_SHUFFLE=True
    NEXT_TOKEN_PREDICTION=True
    ;;
  dp-split-abs)
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=split_abs_source
    ;;
  dp-split-heads)
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=split_avg_heads
    ;;
  dp-split-heads-noshuffle)
    DIFFUSION_SHUFFLE=False
    POS_ENCODING_STRATEGY=split_avg_heads
    ;;
  dp-split-abs-warmup)
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=split_abs_source
    SHUFFLE_WARMUP=5000
    ;;
  dp-split-heads-warmup)
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=split_avg_heads
    SHUFFLE_WARMUP=5000
    ;;
  dp-qk-split-parallel)
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=qk_split
    NEXT_TOKEN_PREDICTION=False
    ;;
  dp-qk-split-ntp)
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=qk_split
    NEXT_TOKEN_PREDICTION=True
    ;;
  dp-qk-split-noshuffle)
    DIFFUSION_SHUFFLE=False
    POS_ENCODING_STRATEGY=qk_split
    ;;
  dp-qk-split-warmup)
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=qk_split
    SHUFFLE_WARMUP=5000
    ;;
  dp-avg-rotary)
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=avg_rotary
    ;;
  dp-4way)
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=4way_heads
    ;;
  dp-2way)
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
GEN_NAME=${RUN_NAME}-gen

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
echo "NUM_SAMPLE_BATCHES: $NUM_SAMPLE_BATCHES"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "MODEL_LENGTH: $MODEL_LENGTH"
echo "DIFFUSION_SHUFFLE: $DIFFUSION_SHUFFLE"
echo "SHUFFLE_WARMUP: $SHUFFLE_WARMUP"
echo "NEXT_TOKEN_PREDICTION: $NEXT_TOKEN_PREDICTION"
echo "POS_ENCODING_STRATEGY: $POS_ENCODING_STRATEGY"
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
  algo=diffuparallel \
  model.length=${MODEL_LENGTH} \
  algo.diffusion_shuffle=${DIFFUSION_SHUFFLE} \
  algo.shuffle_warmup=${SHUFFLE_WARMUP} \
  algo.next_token_prediction=${NEXT_TOKEN_PREDICTION} \
  algo.pos_encoding_strategy=${POS_ENCODING_STRATEGY} \
  data.cache_dir=${CACHE_DIR} \
  hydra.run.dir=${WORKING_DIR} \
  mode=sample_eval \
  eval.generate_samples=True \
  eval.checkpoint_path=${CKPT_PATH} \
  sampling=synthetic_base \
  sampling.num_sample_batches=${NUM_SAMPLE_BATCHES} \
  sampling.kv_cache=False

