#!/bin/bash
#SBATCH -J sweep-latent-steps
#SBATCH --partition=preempt
#SBATCH --output=slurm/master/%j_%x.out
#SBATCH --error=slurm/master/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --time=48:00:00
#SBATCH --mem=100G

# Sweep script for steps and latent tokens
#
# Usage: sbatch sweep_latent_steps.sh --method=<method> --data=<data> [options]
#
# Required:
#   --method, -m    Model method (diffu-causal, etc.)
#   --data, -d      Dataset (sudoku-small, etc.)
#
# Optional:
#   --steps-list    Comma-separated list of steps (default: 32,64,128,256)
#   --latent-list   Comma-separated list of latent tokens (default: 4,8,16,32)
#   --batches, -b   Number of sample batches (default: 1)
#   --ckpt, -c      Checkpoint filename (default: auto-detect best*.ckpt)
#   --output-dir    Directory to save outputs (default: sweep_results)
#
# Examples:
#   sbatch sweep_latent_steps.sh -m diffu-causal -d sudoku-small
#   sbatch sweep_latent_steps.sh -m diffu-causal -d sudoku-small --steps-list=64,128 --latent-list=8,16

set -e

# === Default values ===
METHOD=""
DATA=""
STEPS_LIST="32,64,128,256"
LATENT_LIST="4,8,16,32"
NUM_BATCHES=1
CKPT_NAME=""
RUN_SUFFIX=""
OUTPUT_DIR="sweep_results_tb"
EXTRA_ARGS=""

# === Argument parsing ===
show_help() {
  echo "Usage: sbatch sweep_latent_steps.sh --method=<method> --data=<data> [options]"
  echo ""
  echo "Required:"
  echo "  --method, -m    Model method"
  echo "  --data, -d      Dataset"
  echo ""
  echo "Optional:"
  echo "  --steps-list    Comma-separated list of steps (default: 32,64,128,256)"
  echo "  --latent-list   Comma-separated list of latent tokens (default: 4,8,16,32)"
  echo "  --batches, -b   Number of sample batches (default: 1)"
  echo "  --ckpt, -c      Checkpoint filename (default: auto-detect best*.ckpt)"
  echo "  --output-dir    Directory to save outputs (default: sweep_results)"
  echo ""
  echo "Pass-through:"
  echo "  -- <args>       Additional arguments passed directly to gen_master.sh"
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
    --steps-list=*)
      STEPS_LIST="${1#*=}"
      shift
      ;;
    --steps-list)
      STEPS_LIST="$2"
      shift 2
      ;;
    --latent-list=*)
      LATENT_LIST="${1#*=}"
      shift
      ;;
    --latent-list)
      LATENT_LIST="$2"
      shift 2
      ;;
    --batches=*)
      NUM_BATCHES="${1#*=}"
      shift
      ;;
    --batches|-b)
      NUM_BATCHES="$2"
      shift 2
      ;;
    --ckpt=*)
      CKPT_NAME="${1#*=}"
      shift
      ;;
    --ckpt|-c)
      CKPT_NAME="$2"
      shift 2
      ;;
    --run-suffix=*)
      RUN_SUFFIX="${1#*=}"
      shift
      ;;
    --run-suffix)
      RUN_SUFFIX="$2"
      shift 2
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
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
      show_help
      ;;
  esac
done

# === Validation ===
if [[ -z "$METHOD" ]]; then
  echo "Error: --method is required"
  show_help
fi

if [[ -z "$DATA" ]]; then
  echo "Error: --data is required"
  show_help
fi

# === Setup ===
SCRIPT_DIR="scripts/icml"
mkdir -p "$OUTPUT_DIR/$DATA"

# Convert comma-separated lists to arrays
IFS=',' read -ra STEPS_ARRAY <<< "$STEPS_LIST"
IFS=',' read -ra LATENT_ARRAY <<< "$LATENT_LIST"

echo "=== Sweep Configuration ==="
echo "Method: $METHOD"
echo "Data: $DATA"
echo "Steps: ${STEPS_ARRAY[*]}"
echo "Latent tokens: ${LATENT_ARRAY[*]}"
echo "Batches: $NUM_BATCHES"
echo "Checkpoint: ${CKPT_NAME:-auto-detect}"
echo "Run suffix: ${RUN_SUFFIX:-none}"
echo "Output directory: $OUTPUT_DIR"
echo "Extra args: $EXTRA_ARGS"
echo ""

# === Run sweep ===
for steps in "${STEPS_ARRAY[@]}"; do
  for latent in "${LATENT_ARRAY[@]}"; do
    # Build output filename (include ckpt and suffix if specified)
    base_name="${METHOD}${RUN_SUFFIX}"
    if [[ -n "$CKPT_NAME" ]]; then
      # Remove .ckpt extension for cleaner filename
      ckpt_stem="${CKPT_NAME%.ckpt}"
      output_file="${OUTPUT_DIR}/${DATA}/${base_name}_${ckpt_stem}_steps${steps}_latent${latent}.txt"
    else
      output_file="${OUTPUT_DIR}/${DATA}/${base_name}_steps${steps}_latent${latent}.txt"
    fi
    
    echo "=== Running: steps=$steps, latent=$latent ==="
    echo "Output: $output_file"
    
    # Build the command
    cmd="bash ${SCRIPT_DIR}/gen_master.sh -m $METHOD -d $DATA --steps $steps --batches $NUM_BATCHES"
    if [[ -n "$CKPT_NAME" ]]; then
      cmd="$cmd --ckpt $CKPT_NAME"
    fi
    if [[ -n "$RUN_SUFFIX" ]]; then
      cmd="$cmd --run-suffix $RUN_SUFFIX"
    fi
    
    if [[ -n "$EXTRA_ARGS" ]]; then
      cmd="$cmd -- +sampling.n_latent_tokens=$latent $EXTRA_ARGS"
    else
      cmd="$cmd -- +sampling.n_latent_tokens=$latent"
    fi
    
    echo "Command: $cmd"
    
    # Run and save output
    $cmd 2>&1 | tee "$output_file"
    
    echo ""
    echo "=== Completed: steps=$steps, latent=$latent ==="
    echo ""
  done
done

echo "=== All sweeps completed ==="
echo "Results saved in: $OUTPUT_DIR"
