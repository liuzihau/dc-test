#!/bin/bash
#SBATCH -J train-master
#SBATCH --partition=preempt
#SBATCH --output=slurm/master/%j_%x.out
#SBATCH --error=slurm/master/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --time=48:00:00
#SBATCH --mem=128G

# Master training script for all model variants
#
# Usage: sbatch train_master.sh --method=<method> --data=<data> [options] [-- extra_args]
#
# Required:
#   --method, -m    Model method (ar, diffu-maskfix, dp-4way, etc.)
#   --data, -d      Dataset (sudoku-small, sudoku-large, sudoku-puzzle, zebra, openwebtext-split)
#
# Optional:
#   --resume, -r    Resume from checkpoint (default: true)
#   --no-resume     Start fresh run (equivalent to --resume=false)
#
# Pass-through:
#   -- <args>       Additional arguments passed directly to main.py
#
# Examples:
#   sbatch train_master.sh -m ar -d sudoku-large
#   sbatch train_master.sh --method=diffu-maskfix --data=zebra --no-resume
#   sbatch train_master.sh -m dp-4way -d sudoku-small -- optim.lr=1e-4

set -e

# === Default values ===
METHOD=""
DATA=""
RESUME=true
EXTRA_ARGS=""

# === Argument parsing ===
show_help() {
  echo "Usage: sbatch train_master.sh --method=<method> --data=<data> [options] [-- extra_args]"
  echo ""
  echo "Required:"
  echo "  --method, -m    Model method"
  echo "  --data, -d      Dataset"
  echo ""
  echo "Optional:"
  echo "  --resume, -r    Resume from checkpoint (default: true)"
  echo "  --no-resume     Start fresh run (equivalent to --resume=false)"
  echo ""
  echo "Pass-through:"
  echo "  -- <args>       Additional arguments passed directly to main.py"
  echo ""
  echo "Methods: ar, ar-ntp, ar-mtp-window-32, ar-mtp-window-128,"
  echo "         diffu-maskfix, diffu-causal, diffu-full-lr1e-4-bsz64,"
  echo "         dp-qk-split-parallel, dp-4way, dp-2way"
  echo ""
  echo "Data: sudoku-small, sudoku-large, sudoku-puzzle, sudoku-conditional, sudoku-conditional-uncond, zebra, game-of-24, openwebtext-split"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --method=*)
      METHOD="${1#*=}"
      shift
      ;;
    --method|-m)
      METHOD="$2"
      shift 2
      ;;
    --data=*)
      DATA="${1#*=}"
      shift
      ;;
    --data|-d)
      DATA="$2"
      shift 2
      ;;
    --resume=*)
      RESUME="${1#*=}"
      shift
      ;;
    --resume|-r)
      # Check if next arg is a value or another flag
      if [[ -n "$2" && ! "$2" =~ ^- ]]; then
        RESUME="$2"
        shift 2
      else
        RESUME=true
        shift
      fi
      ;;
    --no-resume)
      RESUME=false
      shift
      ;;
    --help|-h)
      show_help
      ;;
    --)
      shift
      EXTRA_ARGS="$*"
      break
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# === Validate required arguments ===
if [ -z "$METHOD" ]; then
  echo "Error: --method is required"
  echo "Use --help for usage information"
  exit 1
fi

if [ -z "$DATA" ]; then
  echo "Error: --data is required"
  echo "Use --help for usage information"
  exit 1
fi

# === Validate inputs ===
# case $DATA in
#   sudoku-small|sudoku-small-solver|sudoku-large|sudoku-puzzle|sudoku-conditional|sudoku-conditional-uncond|zebra|game-of-24|openwebtext-split)
#     ;;
#   *)
#     echo "Unknown data: $DATA"
#     echo "Valid data: sudoku-small, sudoku-large, sudoku-puzzle, sudoku-conditional, sudoku-conditional-uncond, zebra, game-of-24, openwebtext-split"
#     exit 1
#     ;;
# esac

# Defaults
ALGO=""
AR_NOISE=""
NEXT_TOKEN_PREDICTION=""
LOSS_TYPE="elbo"
DIFFUSION_ATTN_MODE=""
MTP_WINDOW_SIZE=""
DIFFUSION_SHUFFLE=""
POS_ENCODING_STRATEGY=""
SHUFFLE_WARMUP=""
TRAIN_ON_ALL_TOKENS=""
LEARNING_RATE=""
CHECKPOINT_EVERY_N_STEPS=2500
MTP_REBALANCE_LOSS=""
SHUFFLE_CLEAN_TOKENS=""
SHUFFLE_MASKED_TOKENS=""
DATA_CONFIG=""  # Config file name (defaults to DATA if not set)
TARGET_GIVENS=""  # Override for data.target_givens

# === Set data-specific parameters ===
# GPU_BATCH_SIZE: per-GPU batch size (constrained by memory)
# EFFECTIVE_BATCH_SIZE: total effective batch size (gradient accumulation computed automatically)
MODEL_NAME=""  # Default: use model from config.yaml (small)
case $DATA in
  sudoku-small)
    MODEL_LENGTH=128
    GPU_BATCH_SIZE=128
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=100
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=100
    USE_GCS=False
    ;;
  sudoku-small-solver)
    MODEL_LENGTH=128
    GPU_BATCH_SIZE=128
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=100
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=100
    USE_GCS=False
    DATA_CONFIG=sudoku-small
    EXTRA_ARGS="${EXTRA_ARGS} data.use_solver=True"
    ;;
  sudoku-large)
    MODEL_LENGTH=1536
    GPU_BATCH_SIZE=64
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=50
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=100
    USE_GCS=False
    ;;
  sudoku-puzzle)
    MODEL_LENGTH=192
    GPU_BATCH_SIZE=256
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=50
    VAL_CHECK_INTERVAL=10000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=10000
    USE_GCS=False
    MODEL_NAME=small
    # LEARNING_RATE=1e-4
    TRAIN_DATA_PATH=${ESOLM_PUZZLE_DIR}/sudoku-train-data.npy
    VALID_DATA_PATH=${ESOLM_PUZZLE_DIR}/sudoku-test-data.npy
    ;;
  zebra)
    MODEL_LENGTH=384
    GPU_BATCH_SIZE=128
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=50
    VAL_CHECK_INTERVAL=10000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=50000
    USE_GCS=False
    MODEL_NAME=small
    LEARNING_RATE=1e-4
    TRAIN_DATA_PATH=${ESOLM_PUZZLE_DIR}/zebra-train-data.pkl
    VALID_DATA_PATH=${ESOLM_PUZZLE_DIR}/zebra-test-data.pkl
    ;;
  game-of-24)
    MODEL_LENGTH=64
    GPU_BATCH_SIZE=256
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=50
    VAL_CHECK_INTERVAL=10000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=50000
    USE_GCS=False
    MODEL_NAME=small
    LEARNING_RATE=1e-4
    # Data is programmatically generated and cached in ${CACHE_DIR}
    ;;
  sudoku-conditional)
    # Procedurally generated sudoku puzzle-solution pairs (allows non-unique solutions)
    MODEL_LENGTH=256
    GPU_BATCH_SIZE=256
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=50
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=10000
    USE_GCS=False
    MODEL_NAME=small
    # LEARNING_RATE=1e-4
    # Data is programmatically generated and cached in ${CACHE_DIR}
    ;;
  sudoku-conditional-192)
    # Procedurally generated sudoku puzzle-solution pairs (allows non-unique solutions)
    MODEL_LENGTH=192
    GPU_BATCH_SIZE=256
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=50
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=10000
    USE_GCS=False
    MODEL_NAME=small
    DATA_CONFIG=sudoku-conditional
    # LEARNING_RATE=1e-4
    # Data is programmatically generated and cached in ${CACHE_DIR}
    ;;
  sudoku-conditional-192-0given)
    # Unconditional variant: target_givens=0 (no puzzle, just generate valid boards)
    # Similar to sudoku-small but uses conditional format [BOS] puzzle [SEP] solution [EOS]
    MODEL_LENGTH=192
    GPU_BATCH_SIZE=256
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=50
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=10000
    USE_GCS=False
    MODEL_NAME=small
    # LEARNING_RATE=1e-4
    # Use sudoku-conditional config but override target_givens to 0
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=0
    ;;
  sudoku-conditional-192-10given)
    # Unconditional variant: target_givens=0 (no puzzle, just generate valid boards)
    # Similar to sudoku-small but uses conditional format [BOS] puzzle [SEP] solution [EOS]
    MODEL_LENGTH=192
    GPU_BATCH_SIZE=256
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=50
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=10000
    USE_GCS=False
    MODEL_NAME=small
    # LEARNING_RATE=1e-4
    # Use sudoku-conditional config but override target_givens to 0
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=10
    ;;
  sudoku-conditional-192-20given)
    # Unconditional variant: target_givens=0 (no puzzle, just generate valid boards)
    # Similar to sudoku-small but uses conditional format [BOS] puzzle [SEP] solution [EOS]
    MODEL_LENGTH=192
    GPU_BATCH_SIZE=256
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=50
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=10000
    USE_GCS=False
    MODEL_NAME=small
    # LEARNING_RATE=1e-4
    # Use sudoku-conditional config but override target_givens to 0
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=20
    ;;
  sudoku-conditional-192-30given)
    # Unconditional variant: target_givens=0 (no puzzle, just generate valid boards)
    # Similar to sudoku-small but uses conditional format [BOS] puzzle [SEP] solution [EOS]
    MODEL_LENGTH=192
    GPU_BATCH_SIZE=256
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=50
    VAL_CHECK_INTERVAL=1000
    LOG_EVERY_N_STEPS=1000
    CHECKPOINT_EVERY_N_STEPS=10000
    USE_GCS=False
    MODEL_NAME=small
    # LEARNING_RATE=1e-4
    # Use sudoku-conditional config but override target_givens to 0
    DATA_CONFIG=sudoku-conditional
    TARGET_GIVENS=30
    ;;
  openwebtext-split)
    MODEL_LENGTH=1024
    GPU_BATCH_SIZE=32
    EFFECTIVE_BATCH_SIZE=512
    MAX_EPOCHS=1
    VAL_CHECK_INTERVAL=10000
    LOG_EVERY_N_STEPS=1000
    USE_GCS=True
    ;;
esac

case $METHOD in
  ar)
    # Pure autoregressive baseline
    ALGO=ar
    ;;
  ar-sminy)
    # Pure autoregressive baseline with sminy model
    ALGO=ar
    MODEL_NAME=sminy
    ;;
  ar-ignore-loss-mask)
    # Pure autoregressive baseline
    ALGO=ar
    TRAIN_ON_ALL_TOKENS=True
    ;;
  ar-ntp)
    # From train.sh: AR noise + next token prediction with causal attention
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=True
    LOSS_TYPE=low_var
    DIFFUSION_ATTN_MODE=causal
    ;;
  ar-ntp-sminy)
    # From train.sh: AR noise + next token prediction with causal attention
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=True
    LOSS_TYPE=low_var
    DIFFUSION_ATTN_MODE=causal
    MODEL_NAME=sminy
    ;;
  ar-ntp-full-sminy)
    # From train.sh: AR noise + next token prediction with causal attention
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=low_var
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=1
    MODEL_NAME=sminy
    ;;
  ar-mtp-window-32)
    # From train_mtp.sh: AR noise with MTP window size 32
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=causal_context
    MTP_WINDOW_SIZE=32
    ;;
  ar-mtp-window-32-ignore-loss-mask)
    # From train_mtp.sh: AR noise with MTP window size 32
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=causal_context
    MTP_WINDOW_SIZE=32
    TRAIN_ON_ALL_TOKENS=True
    ;;
  ar-mtp-window-128)
    # From train_mtp.sh: AR noise with MTP window size 128
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=causal_context
    MTP_WINDOW_SIZE=128
    ;;
  ar-mtp-window-32-rebalance)
    # AR noise with MTP window size 32 and rebalanced loss (first token not downweighted)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=causal_context
    MTP_WINDOW_SIZE=32
    MTP_REBALANCE_LOSS=True
    ;;
  ar-mtp-window-128-rebalance)
    # AR noise with MTP window size 128 and rebalanced loss (first token not downweighted)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=causal_context
    MTP_WINDOW_SIZE=128
    MTP_REBALANCE_LOSS=True
    ;;
  ar-mtp-window--1)
    # From train_mtp.sh: AR noise with MTP window size -1 (full context)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=causal_context
    MTP_WINDOW_SIZE=-1
    ;;
  ar-mtp-window--1-sminy)
    # AR noise with MTP window size -1 (full context) with sminy model
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=causal_context
    MTP_WINDOW_SIZE=-1
    MODEL_NAME=sminy
    ;;
  ar-mtp-full-sminy)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=-1
    MODEL_NAME=sminy
    ;;
  ar-mtp-full-sminy-rebalance)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=-1
    MODEL_NAME=sminy
    MTP_REBALANCE_LOSS=True
    ;;
  ar-mtp-32-full-sminy-rebalance)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=32
    MODEL_NAME=sminy
    MTP_REBALANCE_LOSS=True
    ;;
  ar-mtp-128-full-sminy-rebalance)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=128
    MODEL_NAME=sminy
    MTP_REBALANCE_LOSS=True
    ;;
  ar-mtp-32-full-sminy)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=32
    MODEL_NAME=sminy
    ;;
  ar-mtp-128-full-sminy)
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    DIFFUSION_ATTN_MODE=full
    MTP_WINDOW_SIZE=128
    MODEL_NAME=sminy
    ;;  
  diffu-maskfix)
    # From train.sh: Pure diffusion with ELBO loss
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_context
    ;;
  diffu)
    # From train.sh: Pure diffusion with ELBO loss
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_context
    ;;
  diffu-full)
    # From train.sh: Pure diffusion with ELBO loss
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=full
    ;;
  diffu-full-sminy)
    # Pure diffusion with ELBO loss, full attention, with sminy model
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=full
    MODEL_NAME=sminy
    ;;
  diffu-full-ignore-loss-mask)
    # Pure diffusion with ELBO loss, full attention, ignore loss mask
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=full
    TRAIN_ON_ALL_TOKENS=True
    ;;
  diffu-causal)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal
    ;;
  diffu-causal-sminy)
    # Pure diffusion with ELBO loss, causal attention, with sminy model
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal
    MODEL_NAME=sminy
    ;;
  diffu-causal-tiny)
    # Pure diffusion with ELBO loss, causal attention, with sminy model
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal
    MODEL_NAME=tiny
    MAX_EPOCHS=300
    ;;
  diffu-causal-sminy-spt)
    # Pure diffusion with ELBO loss, causal attention, with sminy model
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal
    MODEL_NAME=sminy
    EXTRA_ARGS="${EXTRA_ARGS} +algo.shuffle_problem_tokens=True"
    ;;
  diffu-causal-tiny-spt)
    # Pure diffusion with ELBO loss, causal attention, with sminy model
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal
    MODEL_NAME=tiny
    MAX_EPOCHS=300
    EXTRA_ARGS="${EXTRA_ARGS} +algo.shuffle_problem_tokens=True"
    ;;
  diffu-causal-output)
    # Diffusion with random noise, causal_output attention, shuffle both clean and masked
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    ;;
  diffu-causal-output-sminy-spt)
    # Diffusion with random noise, causal_output attention, shuffle both clean and masked
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    MODEL_NAME=sminy
    EXTRA_ARGS="${EXTRA_ARGS} +algo.shuffle_problem_tokens=True"
    ;;
  diffu-causal-output-sminy)
    # Diffusion with random noise, causal_output attention, shuffle both, sminy model
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    MODEL_NAME=sminy
    ;;
  diffu-causal-output-tiny)
    # Diffusion with random noise, causal_output attention, shuffle both, tiny model
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    MODEL_NAME=tiny
    MAX_EPOCHS=300
    ;;
  diffu-causal-output-tiny-deep)
    # Diffusion with causal_output attention, shuffle both, tiny-deep model (~5M, 12 blocks)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    MODEL_NAME=tiny-deep
    MAX_EPOCHS=300
    ;;
  diffu-causal-output-tiny-deeper)
    # Diffusion with causal_output attention, shuffle both, tiny-deeper model (~5M, 16 blocks)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    MODEL_NAME=tiny-deeper
    MAX_EPOCHS=300
    ;;
  diffu-causal-output-micro)
    # Diffusion with random noise, causal_output attention, shuffle both, micro model (~10M)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    MODEL_NAME=micro
    MAX_EPOCHS=300
    ;;
  diffu-causal-output-mini)
    # Diffusion with random noise, causal_output attention, shuffle both, mini model (~18M)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    MODEL_NAME=mini
    MAX_EPOCHS=300
    ;;
  diffu-causal-output-miny)
    # Diffusion with random noise, causal_output attention, shuffle both, miny model (~28M)
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    MODEL_NAME=miny
    MAX_EPOCHS=300
    ;;
  diffu-causal-output-sminy-iglm)
    # Diffusion with random noise, causal_output attention, shuffle both, sminy model
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    MODEL_NAME=sminy
    TRAIN_ON_ALL_TOKENS=True
    ;;
  diffu-causal-output-small)
    # Diffusion with random noise, causal_output attention, shuffle both, small model
    ALGO=difflm
    AR_NOISE=False
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=True
    SHUFFLE_MASKED_TOKENS=True
    MODEL_NAME=small
    ;;
  ar-causal-output)
    # AR noise, causal_output attention, shuffle masked but not clean
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=False
    SHUFFLE_MASKED_TOKENS=True
    ;;
  ar-causal-output-sminy)
    # AR noise, causal_output attention, shuffle masked but not clean, sminy model
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=False
    SHUFFLE_MASKED_TOKENS=True
    MODEL_NAME=sminy
    ;;
  ar-causal-output-small)
    # AR noise, causal_output attention, shuffle masked but not clean, small model
    ALGO=difflm
    AR_NOISE=True
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    DIFFUSION_ATTN_MODE=causal_output
    SHUFFLE_CLEAN_TOKENS=False
    SHUFFLE_MASKED_TOKENS=True
    MODEL_NAME=small
    ;;
  dp-qk-split-parallel)
    # From train_diffuparallel.sh: Q/K split with shuffling
    ALGO=diffuparallel
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=qk_split
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    SHUFFLE_WARMUP=0
    ;;
  dp-4way)
    # From train_diffuparallel.sh: 4-way head split
    ALGO=diffuparallel
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=4way_heads
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    SHUFFLE_WARMUP=0
    ;;
  dp-2way)
    # From train_diffuparallel.sh: 2-way head split
    ALGO=diffuparallel
    DIFFUSION_SHUFFLE=True
    POS_ENCODING_STRATEGY=2way_heads
    NEXT_TOKEN_PREDICTION=False
    LOSS_TYPE=elbo
    SHUFFLE_WARMUP=0
    ;;
  *)
    echo "Unknown method: $METHOD"
    echo "Valid methods: ar, ar-ntp, ar-mtp-window-32, ar-mtp-window-128,"
    echo "               diffu-maskfix, diffu-causal, diffu-full-lr1e-4-bsz64,"
    echo "               dp-qk-split-parallel, dp-4way, dp-2way"
    exit 1
    ;;
esac

# Build run name
RUN_NAME=${METHOD}-${DATA}

# === Environment setup ===
# Required environment variables (cluster-specific):
#   ESOLM_CONDA_PROFILE  - Path to conda.sh (e.g., /path/to/miniconda3/etc/profile.d/conda.sh)
#   ESOLM_DATADIR        - Base data directory for cache, runs, and checkpoints
#   ESOLM_HF_HOME        - HuggingFace home directory
#   ESOLM_PUZZLE_DIR     - Directory containing puzzle data files (required for zebra, sudoku-puzzle)

# export ESOLM_CONDA_PROFILE="${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
# export ESOLM_DATADIR="$HOME/"
# export ESOLM_HF_HOME="${HF_HOME:-$HOME/hf_home}"
# export ESOLM_PUZZLE_DIR="$HOME/puzzle"

if [ -z "$ESOLM_CONDA_PROFILE" ]; then
  echo "Error: ESOLM_CONDA_PROFILE environment variable is not set"
  echo "Set it to the path of your conda.sh (e.g., /path/to/miniconda3/etc/profile.d/conda.sh)"
  exit 1
fi

if [ -z "$ESOLM_DATADIR" ]; then
  echo "Error: ESOLM_DATADIR environment variable is not set"
  echo "Set it to your base data directory for cache, runs, and checkpoints"
  exit 1
fi

if [ -z "$ESOLM_HF_HOME" ]; then
  echo "Error: ESOLM_HF_HOME environment variable is not set"
  echo "Set it to your HuggingFace home directory"
  exit 1
fi

# Check ESOLM_PUZZLE_DIR for datasets that require puzzle data files
if [[ "$DATA" == "zebra" || "$DATA" == "sudoku-puzzle" ]]; then
  if [ -z "$ESOLM_PUZZLE_DIR" ]; then
    echo "Error: ESOLM_PUZZLE_DIR environment variable is not set"
    echo "Set it to the directory containing puzzle data files (e.g., $HOME/puzzle)"
    exit 1
  fi
fi

source "$ESOLM_CONDA_PROFILE"
conda activate esolm

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export HF_HOME="$ESOLM_HF_HOME"
export DATADIR="$ESOLM_DATADIR"
export GCS_DIR="gs://YOUR_BUCKET/"

CACHE_DIR=${DATADIR}/cache
WORKING_DIR=${DATADIR}/runs/${RUN_NAME}

# Set checkpoint directory based on dataset
if [ "$USE_GCS" = "True" ]; then
  CHECKPOINT_DIR=${GCS_DIR}/checkpoints/${RUN_NAME}
else
  CHECKPOINT_DIR=${DATADIR}/checkpoints/${RUN_NAME}
fi

# === Validate resume flag ===
case $RESUME in
  true|True|TRUE|1)
    RESUME_FROM_CKPT=True
    ;;
  false|False|FALSE|0)
    RESUME_FROM_CKPT=False
    ;;
  *)
    echo "Invalid resume flag: $RESUME"
    echo "Valid values: true, false, True, False, TRUE, FALSE, 1, 0"
    exit 1
    ;;
esac

# === Print configuration ===
echo "=== Run Configuration ==="
echo "METHOD: $METHOD"
echo "DATA: $DATA"
echo "ALGO: $ALGO"
echo "RUN_NAME: $RUN_NAME"
echo "RESUME_FROM_CKPT: $RESUME_FROM_CKPT"
if [ -n "$MODEL_NAME" ]; then
  echo "MODEL: $MODEL_NAME"
fi
echo "MODEL_LENGTH: $MODEL_LENGTH"
echo "GPU_BATCH_SIZE: $GPU_BATCH_SIZE"
echo "EFFECTIVE_BATCH_SIZE: $EFFECTIVE_BATCH_SIZE"
if [ -n "$MAX_EPOCHS" ]; then
  echo "MAX_EPOCHS: $MAX_EPOCHS"
fi
echo "VAL_CHECK_INTERVAL: $VAL_CHECK_INTERVAL"
echo "LOG_EVERY_N_STEPS: $LOG_EVERY_N_STEPS"
if [ -n "$AR_NOISE" ]; then
  echo "AR_NOISE: $AR_NOISE"
fi
if [ -n "$NEXT_TOKEN_PREDICTION" ]; then
  echo "NEXT_TOKEN_PREDICTION: $NEXT_TOKEN_PREDICTION"
fi
if [ -n "$LOSS_TYPE" ]; then
  echo "LOSS_TYPE: $LOSS_TYPE"
fi
if [ -n "$DIFFUSION_ATTN_MODE" ]; then
  echo "DIFFUSION_ATTN_MODE: $DIFFUSION_ATTN_MODE"
fi
if [ -n "$MTP_WINDOW_SIZE" ]; then
  echo "MTP_WINDOW_SIZE: $MTP_WINDOW_SIZE"
fi
if [ -n "$DIFFUSION_SHUFFLE" ]; then
  echo "DIFFUSION_SHUFFLE: $DIFFUSION_SHUFFLE"
fi
if [ -n "$POS_ENCODING_STRATEGY" ]; then
  echo "POS_ENCODING_STRATEGY: $POS_ENCODING_STRATEGY"
fi
if [ -n "$SHUFFLE_WARMUP" ]; then
  echo "SHUFFLE_WARMUP: $SHUFFLE_WARMUP"
fi
if [ -n "$TRAIN_ON_ALL_TOKENS" ]; then
  echo "TRAIN_ON_ALL_TOKENS: $TRAIN_ON_ALL_TOKENS"
fi
if [ -n "$MTP_REBALANCE_LOSS" ]; then
  echo "MTP_REBALANCE_LOSS: $MTP_REBALANCE_LOSS"
fi
if [ -n "$SHUFFLE_CLEAN_TOKENS" ]; then
  echo "SHUFFLE_CLEAN_TOKENS: $SHUFFLE_CLEAN_TOKENS"
fi
if [ -n "$SHUFFLE_MASKED_TOKENS" ]; then
  echo "SHUFFLE_MASKED_TOKENS: $SHUFFLE_MASKED_TOKENS"
fi
if [ -n "$EXTRA_ARGS" ]; then
  echo "EXTRA_ARGS: $EXTRA_ARGS"
fi
echo "USE_GCS: $USE_GCS"
echo "CACHE_DIR: $CACHE_DIR"
echo "WORKING_DIR: $WORKING_DIR"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "========================="

# === Build and execute Python command ===
# Use DATA_CONFIG if set, otherwise use DATA
if [ -n "$DATA_CONFIG" ]; then
  HYDRA_DATA_CONFIG=$DATA_CONFIG
else
  HYDRA_DATA_CONFIG=$DATA
fi

PYTHON_CMD="python main.py \
  --config-name=experiment_base \
  data=${HYDRA_DATA_CONFIG} \
  loader.batch_size=${GPU_BATCH_SIZE} \
  loader.eval_batch_size=${GPU_BATCH_SIZE} \
  loader.global_batch_size=${EFFECTIVE_BATCH_SIZE} \
  loader.eval_global_batch_size=${EFFECTIVE_BATCH_SIZE} \
  wandb.name=${RUN_NAME} \
  wandb.project=Diffusion-${DATA} \
  algo=${ALGO} \
  model.length=${MODEL_LENGTH} \
  data.cache_dir=${CACHE_DIR} \
  hydra.run.dir=${WORKING_DIR} \
  checkpointing.save_dir=${CHECKPOINT_DIR} \
  checkpointing.resume_from_ckpt=${RESUME_FROM_CKPT} \
  trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
  trainer.log_every_n_steps=${LOG_EVERY_N_STEPS} \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY_N_STEPS} \
  wandb.id=null"

# Add model override if specified (e.g., for zebra dataset)
if [ -n "$MODEL_NAME" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  model=${MODEL_NAME}"
fi

# Add max_epochs for sudoku datasets
if [ -n "$MAX_EPOCHS" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  trainer.max_epochs=${MAX_EPOCHS}"
fi

# Add difflm-specific arguments
if [ "$ALGO" = "difflm" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  algo.ar_noise=${AR_NOISE} \
  algo.next_token_prediction=${NEXT_TOKEN_PREDICTION} \
  algo.loss_type=${LOSS_TYPE} \
  algo.diffusion_attn_mode=${DIFFUSION_ATTN_MODE}"
  
  # Add MTP window size if set
  if [ -n "$MTP_WINDOW_SIZE" ]; then
    PYTHON_CMD="${PYTHON_CMD} \
  algo.mtp_window_size=${MTP_WINDOW_SIZE}"
  fi
  
  # Add MTP rebalance loss if set
  if [ -n "$MTP_REBALANCE_LOSS" ]; then
    PYTHON_CMD="${PYTHON_CMD} \
  algo.mtp_rebalance_loss=${MTP_REBALANCE_LOSS}"
  fi
  
  # Add shuffle token configs if set
  if [ -n "$SHUFFLE_CLEAN_TOKENS" ]; then
    PYTHON_CMD="${PYTHON_CMD} \
  algo.shuffle_clean_tokens=${SHUFFLE_CLEAN_TOKENS}"
  fi
  if [ -n "$SHUFFLE_MASKED_TOKENS" ]; then
    PYTHON_CMD="${PYTHON_CMD} \
  algo.shuffle_masked_tokens=${SHUFFLE_MASKED_TOKENS}"
  fi
fi

# Add train_on_all_tokens if set
if [ -n "$TRAIN_ON_ALL_TOKENS" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  training.train_on_all_tokens=${TRAIN_ON_ALL_TOKENS}"
fi

# Add learning rate if set
if [ -n "$LEARNING_RATE" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  optim.lr=${LEARNING_RATE}"
fi

# Add data paths if set (for zebra, sudoku-puzzle)
if [ -n "$TRAIN_DATA_PATH" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  data.train_data_path=${TRAIN_DATA_PATH} \
  data.valid_data_path=${VALID_DATA_PATH}"
fi

# Add target_givens override if set (for sudoku-conditional-uncond)
if [ -n "$TARGET_GIVENS" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  data.target_givens=${TARGET_GIVENS}"
fi

# Add diffuparallel-specific arguments
if [ "$ALGO" = "diffuparallel" ]; then
  PYTHON_CMD="${PYTHON_CMD} \
  algo.loss_type=${LOSS_TYPE} \
  algo.diffusion_shuffle=${DIFFUSION_SHUFFLE} \
  algo.shuffle_warmup=${SHUFFLE_WARMUP} \
  algo.next_token_prediction=${NEXT_TOKEN_PREDICTION} \
  algo.pos_encoding_strategy=${POS_ENCODING_STRATEGY}"
fi

# Add extra pass-through arguments if provided
if [ -n "$EXTRA_ARGS" ]; then
  PYTHON_CMD="${PYTHON_CMD} ${EXTRA_ARGS}"
fi

# Execute command
echo "Executing: $PYTHON_CMD"
eval $PYTHON_CMD
