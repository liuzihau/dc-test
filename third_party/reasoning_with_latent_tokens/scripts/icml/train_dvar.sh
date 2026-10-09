#!/bin/bash
#SBATCH -J train-dvar
#SBATCH --partition=general
#SBATCH --output=slurm/dvar/%j_%x.out
#SBATCH --error=slurm/dvar/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --open-mode=append
#SBATCH --time=48:00:00
#SBATCH --mem=128G

# Training and evaluation script for diffusion-vs-ar replication experiments
#
# Usage: sbatch train_dvar.sh --method=<method> --data=<data> [options] [-- extra_args]
#
# Required:
#   --method, -m    Method: ar, ar-mtp-full, mdm-causal-output, mdm-full, mdm-solo-full, noshuffle-mdlm
#   --data, -d      Dataset: sudoku, 3sat5, 3sat7, 3sat9, cd3, cd4, cd5, path
#
# Optional:
#   --mode          Mode: train (default) or eval
#   --resume, -r    Resume from checkpoint (default: true, training only)
#   --no-resume     Start fresh run (training only)
#   --seed, -s      Random seed (default: 1)
#   --run-suffix    Suffix to add to run name
#   --ckpt, -c      Checkpoint filename (default: best.ckpt, eval only)
#   --steps         Sampling steps for diffusion (default: seq length, eval only)
#   --batches, -b   Number of sample batches (default: 1, eval only)
#   --model         Model config override (default: from experiment config)
#                   Use --model=dvar-tiny-legacy for old models with RoPE + no weight tying
#   --fp16          Use fp16 precision (default: bf16)
#
# Examples:
#   # Training
#   sbatch train_dvar.sh -m mdm-full -d sudoku
#   sbatch train_dvar.sh -m ar -d 3sat9 --no-resume
#   sbatch train_dvar.sh -m mdm-full -d cd4 -- optim.lr=1e-4
#
#   # Evaluation
#   sbatch train_dvar.sh -m ar -d 3sat9 --mode=eval
#   sbatch train_dvar.sh -m mdm-full -d cd4 --mode=eval --ckpt=epoch-100.ckpt
#   sbatch train_dvar.sh -m mdm-full -d sudoku --mode=eval --steps=64 --batches=10
#
#   # Evaluate legacy model (with RoPE, no weight tying)
#   sbatch train_dvar.sh -m mdm-full -d 3sat9 --mode=eval --model=dvar-tiny-legacy

set -e
export NCCL_P2P_DISABLE=1

# === Default values ===
METHOD=""
DATA=""
MODE="train"
RESUME=true
SEED=1
RUN_SUFFIX=""
CKPT_NAME="best.ckpt"
STEPS=""
NUM_SAMPLE_BATCHES=1
MODEL_OVERRIDE=""
PRECISION="bf16"
TRAIN_ON_ALL_TOKENS=""
EXTRA_ARGS=""

# === Argument parsing ===
show_help() {
  echo "Usage: sbatch train_dvar.sh --method=<method> --data=<data> [options] [-- extra_args]"
  echo ""
  echo "Required:"
  echo "  --method, -m    Method: ar, ar-ntp, ar-mtp-full, mdm-causal-output, mdm-full, mdm-solo-full, noshuffle-mdlm"
  echo "  --data, -d      Dataset: sudoku, 3sat5, 3sat7, 3sat9, cd3, cd4, cd5, path"
  echo ""
  echo "Optional:"
  echo "  --mode          Mode: train (default) or eval"
  echo "  --resume, -r    Resume from checkpoint (default: true, training only)"
  echo "  --no-resume     Start fresh run (training only)"
  echo "  --seed, -s      Random seed (default: 1)"
  echo "  --run-suffix    Suffix to add to run name"
  echo "  --ckpt, -c      Checkpoint filename (default: best.ckpt, eval only)"
  echo "  --steps         Sampling steps for diffusion (default: seq length, eval only)"
  echo "  --batches, -b   Number of sample batches (default: 1, eval only)"
  echo "  --model         Model config override (e.g., dvar-tiny-legacy for old models)"
  echo "  --fp16          Use fp16 precision (default: bf16)"
  echo "  --train-on-all-tokens  Train on all tokens (ignore problem/solution distinction)"
  echo ""
  echo "Training configs (from diffusion-vs-ar paper):"
  echo "  Dataset   | MDM Epochs | AR Epochs | LR (tiny) | Seq Len"
  echo "  ----------|------------|-----------|-----------|--------"
  echo "  sudoku    | 300        | 100       | 1e-3      | 164"
  echo "  3sat*     | 600        | 300       | 1e-3      | 325"
  echo "  cd*       | 600        | 40        | 1e-3      | 64"
  echo "  path      | 100        | 200       | 1e-3      | 75"
  echo ""
  echo "Note: LR is model-dependent (1e-3 for tiny, 3e-4 for small/medium)"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --method=*) METHOD="${1#*=}"; shift ;;
    --method|-m) METHOD="$2"; shift 2 ;;
    --data=*) DATA="${1#*=}"; shift ;;
    --data|-d) DATA="$2"; shift 2 ;;
    --mode=*) MODE="${1#*=}"; shift ;;
    --mode) MODE="$2"; shift 2 ;;
    --resume=*) RESUME="${1#*=}"; shift ;;
    --resume|-r)
      if [[ -n "$2" && ! "$2" =~ ^- ]]; then
        RESUME="$2"; shift 2
      else
        RESUME=true; shift
      fi ;;
    --no-resume) RESUME=false; shift ;;
    --seed=*) SEED="${1#*=}"; shift ;;
    --seed|-s) SEED="$2"; shift 2 ;;
    --run-suffix=*) RUN_SUFFIX="${1#*=}"; shift ;;
    --run-suffix) RUN_SUFFIX="$2"; shift 2 ;;
    --ckpt=*) CKPT_NAME="${1#*=}"; shift ;;
    --ckpt|-c) CKPT_NAME="$2"; shift 2 ;;
    --steps=*) STEPS="${1#*=}"; shift ;;
    --steps) STEPS="$2"; shift 2 ;;
    --batches=*) NUM_SAMPLE_BATCHES="${1#*=}"; shift ;;
    --batches|-b) NUM_SAMPLE_BATCHES="$2"; shift 2 ;;
    --model=*) MODEL_OVERRIDE="${1#*=}"; shift ;;
    --model) MODEL_OVERRIDE="$2"; shift 2 ;;
    --fp16) PRECISION="16-true"; shift ;;
    --train-on-all-tokens) TRAIN_ON_ALL_TOKENS=True; shift ;;
    --help|-h) show_help ;;
    --)
      shift
      EXTRA_ARGS="$*"
      break ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1 ;;
  esac
done

# === Validate required arguments ===
if [ -z "$METHOD" ]; then
  echo "Error: --method is required"
  exit 1
fi

if [ -z "$DATA" ]; then
  echo "Error: --data is required"
  exit 1
fi

# Validate mode
case $MODE in
  train|eval) ;;
  *)
    echo "Invalid mode: $MODE"
    echo "Valid modes: train, eval"
    exit 1 ;;
esac

# === Method configuration ===
ALGO=""
MAX_EPOCHS=""
MODEL_LENGTH=""
NUM_TRAINING_STEPS=""

case $METHOD in
  ar)
    ALGO=ar
    ;;
  ar-ntp)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=True
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=-1
    ;;
  ar-ntp-w1)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=True
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=1
    ;;
  ar-mtp-full)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=-1
    ;;
  mdm-causal-output)
    ALGO=difflm
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN=True
    SHUFFLE_MASKED=True
    ;;
  mdm-full)
    ALGO=difflm
    DIFFUSION_ATTN_MODE=full
    SHUFFLE_CLEAN=True
    SHUFFLE_MASKED=True
    ;;
  mdm-solo-full)
    ALGO=difflm
    DIFFUSION_ATTN_MODE=solo_full
    SHUFFLE_CLEAN=True
    SHUFFLE_MASKED=True
    ;;
  noshuffle-mdlm)
    # NoShuffleMDLM: matches diffusion-vs-ar exactly
    # - No token shuffling (tokens stay in sequence order)
    # - Shifted logits (position i-1 predicts position i)
    # - Token reweighting (focal loss)
    # - Full bidirectional attention
    ALGO=noshuffle_mdlm
    DIFFUSION_ATTN_MODE=full
    NOSHUFFLE_MDLM=True
    ;;
  *)
    echo "Unknown method: $METHOD"
    echo "Valid methods: ar, ar-ntp, ar-mtp-full, mdm-causal-output, mdm-full, mdm-solo-full, noshuffle-mdlm"
    exit 1 ;;
esac

# === Data configuration ===
EXPERIMENT_CONFIG=""
DATA_CONFIG=""
DEFAULT_SEQ_LEN=""

case $DATA in
  sudoku)
    EXPERIMENT_CONFIG=experiment_dvar_sudoku
    DATA_CONFIG=dvar-sudoku
    DEFAULT_SEQ_LEN=164
    if [ "$ALGO" = "ar" ]; then
      MAX_EPOCHS=100
    else
      MAX_EPOCHS=300
    fi
    ;;
  3sat5)
    EXPERIMENT_CONFIG=experiment_dvar_3sat
    DATA_CONFIG=dvar-3sat5
    DEFAULT_SEQ_LEN=258
    if [ "$ALGO" = "ar" ]; then
      MAX_EPOCHS=300
      NUM_TRAINING_STEPS=14648  # (50000/1024)*300
    else
      MAX_EPOCHS=600
      NUM_TRAINING_STEPS=29297  # (50000/1024)*600
    fi
    ;;
  3sat7)
    EXPERIMENT_CONFIG=experiment_dvar_3sat7
    DATA_CONFIG=dvar-3sat7
    DEFAULT_SEQ_LEN=285
    if [ "$ALGO" = "ar" ]; then
      MAX_EPOCHS=300
      NUM_TRAINING_STEPS=14648  # (50000/1024)*300
    else
      MAX_EPOCHS=600
      NUM_TRAINING_STEPS=29297  # (50000/1024)*600
    fi
    ;;
  3sat9)
    EXPERIMENT_CONFIG=experiment_dvar_3sat9
    DATA_CONFIG=dvar-3sat9
    DEFAULT_SEQ_LEN=325
    if [ "$ALGO" = "ar" ]; then
      MAX_EPOCHS=300
      NUM_TRAINING_STEPS=29297  # (100000/1024)*300
    else
      MAX_EPOCHS=600
      NUM_TRAINING_STEPS=58594  # (100000/1024)*600
    fi
    ;;
  cd3)
    EXPERIMENT_CONFIG=experiment_dvar_cd
    DATA_CONFIG=dvar-cd3
    MODEL_LENGTH=37  # Paper: cutoff_len=37
    DEFAULT_SEQ_LEN=37
    if [ "$ALGO" = "ar" ]; then
      MAX_EPOCHS=40
      NUM_TRAINING_STEPS=19531  # (500000/1024)*40
    else
      MAX_EPOCHS=600
      NUM_TRAINING_STEPS=292969  # (500000/1024)*600
    fi
    ;;
  cd4)
    EXPERIMENT_CONFIG=experiment_dvar_cd
    DATA_CONFIG=dvar-cd4
    MODEL_LENGTH=64  # Paper: cutoff_len=64
    DEFAULT_SEQ_LEN=64
    if [ "$ALGO" = "ar" ]; then
      MAX_EPOCHS=40
      NUM_TRAINING_STEPS=19531  # (500000/1024)*40
    else
      MAX_EPOCHS=600
      NUM_TRAINING_STEPS=292969  # (500000/1024)*600
    fi
    ;;
  cd5)
    EXPERIMENT_CONFIG=experiment_dvar_cd
    DATA_CONFIG=dvar-cd5
    MODEL_LENGTH=74  # Paper: cutoff_len=74
    DEFAULT_SEQ_LEN=74
    if [ "$ALGO" = "ar" ]; then
      MAX_EPOCHS=40
      NUM_TRAINING_STEPS=19531  # (500000/1024)*40
    else
      MAX_EPOCHS=600
      NUM_TRAINING_STEPS=292969  # (500000/1024)*600
    fi
    ;;
  path)
    EXPERIMENT_CONFIG=experiment_dvar_path
    DATA_CONFIG=dvar-path
    DEFAULT_SEQ_LEN=75
    if [ "$ALGO" = "ar" ]; then
      MAX_EPOCHS=200
    else
      MAX_EPOCHS=100
    fi
    ;;
  *)
    echo "Unknown data: $DATA"
    echo "Valid data: sudoku, 3sat5, 3sat7, 3sat9, cd3, cd4, cd5, path"
    exit 1 ;;
esac

# Set focal gamma based on task type (matching diffusion-vs-ar)
# - countdown (cd*): gamma=2
# - 3sat*, sudoku, path: gamma=1
case $DATA in
  cd3|cd4|cd5) FOCAL_GAMMA=2.0 ;;
  *) FOCAL_GAMMA=1.0 ;;
esac

# Default STEPS to sequence length if not provided
# For noshuffle_mdlm, default to 20 steps (matching diffusion-vs-ar)
if [ -z "$STEPS" ]; then
  if [ "$ALGO" = "noshuffle_mdlm" ]; then
    STEPS=20
  else
    STEPS=$DEFAULT_SEQ_LEN
  fi
fi

# === Build run name ===
# Add -tat suffix if training on all tokens
TAT_SUFFIX=""
if [ -n "$TRAIN_ON_ALL_TOKENS" ]; then
  TAT_SUFFIX="-tat"
fi

if [ "$SEED" = "1" ]; then
  RUN_NAME=dvar-${METHOD}-${DATA}${TAT_SUFFIX}${RUN_SUFFIX}
else
  RUN_NAME=dvar-${METHOD}-${DATA}-seed${SEED}${TAT_SUFFIX}${RUN_SUFFIX}
fi

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

CACHE_DIR=${DATADIR}/cache
CHECKPOINT_DIR=${DATADIR}/checkpoints/${RUN_NAME}

# === Mode-specific setup ===
if [ "$MODE" = "train" ]; then
  WORKING_DIR=${DATADIR}/runs/${RUN_NAME}

  # Validate resume flag
  case $RESUME in
    true|True|TRUE|1) RESUME_FROM_CKPT=True ;;
    false|False|FALSE|0) RESUME_FROM_CKPT=False ;;
    *) echo "Invalid resume flag: $RESUME"; exit 1 ;;
  esac
else
  # Eval mode: find checkpoint
  CKPT_PATH=${CHECKPOINT_DIR}/checkpoints/${CKPT_NAME}
  if [ ! -f "$CKPT_PATH" ]; then
    # Try to find best*.ckpt if specified ckpt doesn't exist
    if [ "$CKPT_NAME" = "best.ckpt" ]; then
      CKPT_PATH=$(ls -v ${CHECKPOINT_DIR}/checkpoints/best*.ckpt 2>/dev/null | tail -1)
      if [ -z "$CKPT_PATH" ]; then
        echo "Error: No checkpoint found in ${CHECKPOINT_DIR}/checkpoints/"
        echo "Available checkpoints:"
        ls -la ${CHECKPOINT_DIR}/checkpoints/ 2>/dev/null || echo "  (directory not found)"
        exit 1
      fi
      CKPT_NAME=$(basename "$CKPT_PATH")
    else
      echo "Error: Checkpoint not found: $CKPT_PATH"
      echo "Available checkpoints:"
      ls -la ${CHECKPOINT_DIR}/checkpoints/ 2>/dev/null || echo "  (directory not found)"
      exit 1
    fi
  fi

  # Build eval run name
  CKPT_STEM=${CKPT_NAME%.ckpt}
  if [ "$ALGO" = "ar" ]; then
    EVAL_NAME=${RUN_NAME}-eval-${CKPT_STEM}
  else
    EVAL_NAME=${RUN_NAME}-eval-steps${STEPS}-${CKPT_STEM}
  fi
  WORKING_DIR=${DATADIR}/runs/${EVAL_NAME}
fi

# === Print configuration ===
echo "=== DVAR Run Configuration ==="
echo "MODE: $MODE"
echo "METHOD: $METHOD"
echo "DATA: $DATA"
echo "ALGO: $ALGO"
echo "EXPERIMENT_CONFIG: $EXPERIMENT_CONFIG"
echo "DATA_CONFIG: $DATA_CONFIG"
if [ "$MODE" = "train" ]; then
  echo "MAX_EPOCHS: $MAX_EPOCHS"
  echo "RESUME_FROM_CKPT: $RESUME_FROM_CKPT"
else
  echo "CKPT_PATH: $CKPT_PATH"
  if [ "$ALGO" != "ar" ]; then
    echo "STEPS: $STEPS"
  fi
  echo "NUM_SAMPLE_BATCHES: $NUM_SAMPLE_BATCHES"
fi
if [ -n "$MODEL_LENGTH" ]; then
  echo "MODEL_LENGTH: $MODEL_LENGTH"
fi
if [ -n "$NUM_TRAINING_STEPS" ] && [ "$MODE" = "train" ]; then
  echo "NUM_TRAINING_STEPS: $NUM_TRAINING_STEPS"
fi
echo "SEED: $SEED"
echo "RUN_NAME: $RUN_NAME"
if [ -n "$DIFFUSION_ATTN_MODE" ]; then
  echo "DIFFUSION_ATTN_MODE: $DIFFUSION_ATTN_MODE"
fi
if [ -n "$AR_NOISE" ]; then
  echo "AR_NOISE: $AR_NOISE"
fi
if [ -n "$NEXT_TOKEN_PREDICTION" ]; then
  echo "NEXT_TOKEN_PREDICTION: $NEXT_TOKEN_PREDICTION"
fi
if [ -n "$MTP_WINDOW_SIZE" ]; then
  echo "MTP_WINDOW_SIZE: $MTP_WINDOW_SIZE"
fi
if [ -n "$NOSHUFFLE_MDLM" ]; then
  echo "NOSHUFFLE_MDLM: shifted_logits=True, token_reweighting=True, focal_gamma=${FOCAL_GAMMA}, time_reweighting=linear"
fi
if [ -n "$MODEL_OVERRIDE" ]; then
  echo "MODEL_OVERRIDE: $MODEL_OVERRIDE"
fi
echo "PRECISION: $PRECISION"
if [ -n "$TRAIN_ON_ALL_TOKENS" ]; then
  echo "TRAIN_ON_ALL_TOKENS: $TRAIN_ON_ALL_TOKENS"
fi
if [ -n "$EXTRA_ARGS" ]; then
  echo "EXTRA_ARGS: $EXTRA_ARGS"
fi
echo "WORKING_DIR: $WORKING_DIR"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "=============================="

# === Build Python command ===
if [ "$MODE" = "train" ]; then
  # === Training mode ===
  PYTHON_CMD="python main.py \
    --config-name=${EXPERIMENT_CONFIG} \
    data=${DATA_CONFIG} \
    algo=${ALGO} \
    seed=${SEED} \
    trainer.max_epochs=${MAX_EPOCHS} \
    wandb.name=${RUN_NAME} \
    wandb.project=DVAR-Replication \
    data.cache_dir=${CACHE_DIR} \
    hydra.run.dir=${WORKING_DIR} \
    checkpointing.save_dir=${CHECKPOINT_DIR} \
    checkpointing.resume_from_ckpt=${RESUME_FROM_CKPT} \
    wandb.id=null \
    eval.run_task_evaluation=True \
    sampling=synthetic_base \
    sampling.num_sample_batches=1 \
    sampling.steps=${STEPS} \
    sampling.greedy_tokens=True"

  # Add unmask_policy=topp only for diffusion-based methods (not ar or ar-mtp-full)
  if [ "$METHOD" != "ar" ] && [ "$METHOD" != "ar-ntp" ] && [ "$METHOD" != "ar-ntp-w1" ] && [ "$METHOD" != "ar-mtp-full" ]; then
    PYTHON_CMD="${PYTHON_CMD} \
    sampling.unmask_policy=topp"
  fi

  # Add num_training_steps for cosine LR schedule if set
  if [ -n "$NUM_TRAINING_STEPS" ]; then
    PYTHON_CMD="${PYTHON_CMD} \
    lr_scheduler.num_training_steps=${NUM_TRAINING_STEPS}"
  fi

  # Add model override if set
  if [ -n "$MODEL_OVERRIDE" ]; then
    PYTHON_CMD="${PYTHON_CMD} \
    model=${MODEL_OVERRIDE}"
  fi

  # Add model.length override if set (for cd variants with different seq lengths)
  if [ -n "$MODEL_LENGTH" ]; then
    PYTHON_CMD="${PYTHON_CMD} \
    model.length=${MODEL_LENGTH}"
  fi

  # Add difflm-specific arguments
  if [ "$ALGO" = "difflm" ]; then
    PYTHON_CMD="${PYTHON_CMD} \
    algo.diffusion_attn_mode=${DIFFUSION_ATTN_MODE} \
    +algo.log_position_losses=True"

    if [ -n "$SHUFFLE_CLEAN" ]; then
      PYTHON_CMD="${PYTHON_CMD} \
    algo.shuffle_clean_tokens=${SHUFFLE_CLEAN}"
    fi
    if [ -n "$SHUFFLE_MASKED" ]; then
      PYTHON_CMD="${PYTHON_CMD} \
    algo.shuffle_masked_tokens=${SHUFFLE_MASKED}"
    fi
    if [ -n "$AR_NOISE" ]; then
      PYTHON_CMD="${PYTHON_CMD} \
    algo.ar_noise=${AR_NOISE}"
    fi
    if [ -n "$NEXT_TOKEN_PREDICTION" ]; then
      PYTHON_CMD="${PYTHON_CMD} \
    algo.next_token_prediction=${NEXT_TOKEN_PREDICTION}"
    fi
    if [ -n "$LOSS_TYPE" ]; then
      PYTHON_CMD="${PYTHON_CMD} \
    algo.loss_type=${LOSS_TYPE}"
    fi
    if [ -n "$MTP_WINDOW_SIZE" ]; then
      PYTHON_CMD="${PYTHON_CMD} \
    algo.mtp_window_size=${MTP_WINDOW_SIZE}"
    fi
  fi

  # Add noshuffle_mdlm-specific arguments
  # keep things consistent with eval: greedy_tokens=True, steps=20
  if [ "$ALGO" = "noshuffle_mdlm" ]; then
    PYTHON_CMD="${PYTHON_CMD} \
    algo.shifted_logits=True \
    algo.token_reweighting=True \
    algo.focal_alpha=0.25 \
    algo.focal_gamma=${FOCAL_GAMMA} \
    algo.time_reweighting=linear \
    algo.num_diffusion_steps=20 \
    model.use_gpt2_arch=True \
    sampling=dvar_eval"
  fi

else
  # === Evaluation mode ===
  # dvar datasets are conditional (input -> output), so use completions mode
  PYTHON_CMD="python main.py \
    --config-name=${EXPERIMENT_CONFIG} \
    data=${DATA_CONFIG} \
    algo=${ALGO} \
    seed=${SEED} \
    wandb.name=${EVAL_NAME} \
    wandb.project=DVAR-Replication \
    data.cache_dir=${CACHE_DIR} \
    hydra.run.dir=${WORKING_DIR} \
    mode=completions \
    eval.checkpoint_path=${CKPT_PATH} \
    sampling.num_sample_batches=${NUM_SAMPLE_BATCHES} \
    wandb.id=null \
    trainer.precision=${PRECISION}"

  # Add model override if set
  if [ -n "$MODEL_OVERRIDE" ]; then
    PYTHON_CMD="${PYTHON_CMD} \
    model=${MODEL_OVERRIDE}"
  fi

  # Add model.length override if set
  if [ -n "$MODEL_LENGTH" ]; then
    PYTHON_CMD="${PYTHON_CMD} \
    model.length=${MODEL_LENGTH}"
  fi

  if [ "$ALGO" = "ar" ]; then
    # AR evaluation
    PYTHON_CMD="${PYTHON_CMD} \
    sampling.kv_cache=True"
  else
    # Diffusion evaluation
    PYTHON_CMD="${PYTHON_CMD} \
    sampling=synthetic_base \
    sampling.steps=${STEPS} \
    sampling.kv_cache=False \
    sampling.greedy_tokens=True \
    algo.diffusion_attn_mode=${DIFFUSION_ATTN_MODE}"

    if [ -n "$SHUFFLE_CLEAN" ]; then
      PYTHON_CMD="${PYTHON_CMD} \
    algo.shuffle_clean_tokens=${SHUFFLE_CLEAN}"
    fi
    if [ -n "$SHUFFLE_MASKED" ]; then
      PYTHON_CMD="${PYTHON_CMD} \
    algo.shuffle_masked_tokens=${SHUFFLE_MASKED}"
    fi
    if [ -n "$AR_NOISE" ]; then
      PYTHON_CMD="${PYTHON_CMD} \
    algo.ar_noise=${AR_NOISE}"
    fi
    if [ -n "$NEXT_TOKEN_PREDICTION" ]; then
      PYTHON_CMD="${PYTHON_CMD} \
    algo.next_token_prediction=${NEXT_TOKEN_PREDICTION}"
    fi
    if [ -n "$LOSS_TYPE" ]; then
      PYTHON_CMD="${PYTHON_CMD} \
    algo.loss_type=${LOSS_TYPE}"
    fi
    if [ -n "$MTP_WINDOW_SIZE" ]; then
      PYTHON_CMD="${PYTHON_CMD} \
    algo.mtp_window_size=${MTP_WINDOW_SIZE}"
    fi

    # Add noshuffle_mdlm-specific eval arguments
    # Uses ESOLM unmask paradigm with greedy tokens (best single change from ablation study)
    if [ "$ALGO" = "noshuffle_mdlm" ]; then
      PYTHON_CMD="${PYTHON_CMD} \
    sampling=dvar_eval \
    algo.shifted_logits=True \
    model.use_gpt2_arch=True"
    fi
  fi
fi

# Add train_on_all_tokens if set
if [ -n "$TRAIN_ON_ALL_TOKENS" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  training.train_on_all_tokens=${TRAIN_ON_ALL_TOKENS}"
fi

if [ -n "$EXTRA_ARGS" ]; then
  PYTHON_CMD="${PYTHON_CMD} ${EXTRA_ARGS}"
fi

# Execute
echo "Executing: $PYTHON_CMD"
eval $PYTHON_CMD
