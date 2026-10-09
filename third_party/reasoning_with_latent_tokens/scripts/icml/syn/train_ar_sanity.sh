#!/bin/bash
#SBATCH -J train-ar-sanity
#SBATCH --partition=preempt
#SBATCH --output=slurm/syn-ar/%j_%x.out
#SBATCH --error=slurm/syn-ar/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --time=1:00:00
#SBATCH --mem=32G

# Sanity check: train AR on a single repeated sequence.
# The model should be able to perfectly overfit to this.
# If it can't, something is fundamentally broken.

set -e

# === Configuration ===
# Using a short sequence for fast iteration
MODEL_LENGTH=128
BATCH_SIZE=32

RUN_NAME=ar-sanity-repeat

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

echo "=== Sanity Check: AR on Repeat Dataset ==="
echo "RUN_NAME: $RUN_NAME"
echo "MODEL_LENGTH: $MODEL_LENGTH"
echo "CACHE_DIR: $CACHE_DIR"
echo "WORKING_DIR: $WORKING_DIR"
echo "============================================"
echo ""
echo "Expected behavior:"
echo "  - Loss should drop quickly (within ~100 steps)"
echo "  - Loss should approach 0 (perfect prediction)"
echo "  - If loss stays high (~2.3 = log(10)), model is broken"
echo ""

python main.py \
  --config-name=experiment_base \
  data=repeat \
  loader.batch_size=${BATCH_SIZE} \
  loader.eval_batch_size=${BATCH_SIZE} \
  wandb.name=${RUN_NAME} \
  algo=ar \
  model.length=${MODEL_LENGTH} \
  model.absolute_pos_embed=False \
  data.cache_dir=${CACHE_DIR} \
  hydra.run.dir=${WORKING_DIR} \
  checkpointing.save_dir=${CHECKPOINT_DIR} \
  checkpointing.resume_from_ckpt=False \
  trainer.val_check_interval=100 \
  trainer.log_every_n_steps=10 \
  trainer.max_epochs=10 \
  wandb.id=null \
  optim.lr=1e-3
