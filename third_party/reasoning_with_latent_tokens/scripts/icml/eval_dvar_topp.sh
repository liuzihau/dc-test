#!/bin/bash
#SBATCH -J eval-dvar-topp
#SBATCH --partition=general
#SBATCH --output=slurm/dvar/%j_%x.out
#SBATCH --error=slurm/dvar/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --open-mode=append
#SBATCH --time=48:00:00
#SBATCH --mem=128G

# Batch evaluation script for MDM-full models with topp unmask policy
#
# Usage: sbatch scripts/icml/eval_dvar_topp.sh

set -e

# === Environment setup ===
if [ -z "$ESOLM_CONDA_PROFILE" ]; then
  echo "Error: ESOLM_CONDA_PROFILE environment variable is not set"
  exit 1
fi

if [ -z "$ESOLM_DATADIR" ]; then
  echo "Error: ESOLM_DATADIR environment variable is not set"
  exit 1
fi

if [ -z "$ESOLM_HF_HOME" ]; then
  echo "Error: ESOLM_HF_HOME environment variable is not set"
  exit 1
fi

source "$ESOLM_CONDA_PROFILE"
conda activate esolm

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export HF_HOME="$ESOLM_HF_HOME"
export DATADIR="$ESOLM_DATADIR"

# Tasks and their checkpoints
declare -A CHECKPOINTS
CHECKPOINTS[sudoku]="last-v1.ckpt"
CHECKPOINTS[cd3]="last.ckpt"
CHECKPOINTS[cd4]="last.ckpt"
CHECKPOINTS[cd5]="last.ckpt"
CHECKPOINTS[3sat5]="last.ckpt"
CHECKPOINTS[3sat7]="last.ckpt"
CHECKPOINTS[3sat9]="last.ckpt"
CHECKPOINTS[path]="last.ckpt"

# Model lengths (must match what was used during training)
declare -A MODEL_LENS
MODEL_LENS[sudoku]=165
MODEL_LENS[cd3]=64
MODEL_LENS[cd4]=64
MODEL_LENS[cd5]=74
MODEL_LENS[3sat5]=325
MODEL_LENS[3sat7]=325
MODEL_LENS[3sat9]=325
MODEL_LENS[path]=75

# Experiment configs
declare -A EXP_CONFIGS
EXP_CONFIGS[sudoku]=experiment_dvar_sudoku
EXP_CONFIGS[cd3]=experiment_dvar_cd
EXP_CONFIGS[cd4]=experiment_dvar_cd
EXP_CONFIGS[cd5]=experiment_dvar_cd
EXP_CONFIGS[3sat5]=experiment_dvar_3sat
EXP_CONFIGS[3sat7]=experiment_dvar_3sat
EXP_CONFIGS[3sat9]=experiment_dvar_3sat
EXP_CONFIGS[path]=experiment_dvar_path

TASKS="sudoku cd3 cd4 cd5 3sat5 3sat7 3sat9 path"
CACHE_DIR=${DATADIR}/cache

echo "=== Running DVAR MDM-full Evaluations with topp ==="
echo ""

for task in $TASKS; do
  ckpt=${CHECKPOINTS[$task]}
  model_len=${MODEL_LENS[$task]}
  exp_config=${EXP_CONFIGS[$task]}

  RUN_NAME="dvar-mdm-full-${task}"
  EVAL_NAME="${RUN_NAME}-topp-eval-steps${model_len}-${ckpt%.ckpt}"
  CHECKPOINT_DIR=${DATADIR}/checkpoints/${RUN_NAME}
  CKPT_PATH=${CHECKPOINT_DIR}/checkpoints/${ckpt}
  WORKING_DIR=${DATADIR}/runs/${EVAL_NAME}

  echo "Evaluating: $EVAL_NAME"

  if [ ! -f "$CKPT_PATH" ]; then
    echo "  WARNING: Checkpoint not found: $CKPT_PATH, skipping"
    continue
  fi

  python main.py \
    --config-name=${exp_config} \
    data=dvar-${task} \
    algo=difflm \
    model=dvar-tiny-legacy \
    model.length=${model_len} \
    wandb.name=${EVAL_NAME} \
    wandb.project=DVAR-Replication \
    data.cache_dir=${CACHE_DIR} \
    hydra.run.dir=${WORKING_DIR} \
    mode=completions \
    eval.checkpoint_path=${CKPT_PATH} \
    sampling.num_sample_batches=-1 \
    sampling=synthetic_base \
    sampling.steps=${model_len} \
    sampling.unmask_policy=topp \
    sampling.kv_cache=False \
    algo.diffusion_attn_mode=full \
    algo.shuffle_clean_tokens=True \
    algo.shuffle_masked_tokens=True \
    wandb.id=null

  echo "  Done: $EVAL_NAME"
  echo ""
done

echo ""
echo "=== All topp evaluations complete ==="
