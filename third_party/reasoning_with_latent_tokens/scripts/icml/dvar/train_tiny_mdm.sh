#!/bin/bash
#SBATCH -J dvar-tiny
#SBATCH --partition=general
#SBATCH --output=slurm/dvar/%j_%x.out
#SBATCH --error=slurm/dvar/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G

# Train tiny MDM models on diffusion-vs-ar tasks
# Usage: sbatch train_tiny_mdm.sh --task=<task> [--no-token-reweight]
#   task: 3sat7, 3sat9, cd4, cd5

set -e

# Parse arguments
TASK=""
TOKEN_REWEIGHT="True"
while [[ $# -gt 0 ]]; do
    case $1 in
        --task=*) TASK="${1#*=}"; shift ;;
        -t) TASK="$2"; shift 2 ;;
        --no-token-reweight) TOKEN_REWEIGHT="False"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$TASK" ]; then
    echo "Error: --task is required"
    echo "Usage: sbatch train_tiny_mdm.sh --task=<3sat7|3sat9|cd4|cd5> [--no-token-reweight]"
    exit 1
fi

# Task-specific parameters
case $TASK in
    3sat7)
        DATASET="3sat7_train"
        TEST_DATASET="3sat7_test"
        CUTOFF_LEN=285
        GAMMA=1
        ;;
    3sat9)
        DATASET="3sat9_train"
        TEST_DATASET="3sat9_test"
        CUTOFF_LEN=325
        GAMMA=1
        ;;
    cd4)
        DATASET="cd4_train"
        TEST_DATASET="cd4_test cd4_tot24"
        CUTOFF_LEN=64
        GAMMA=2
        ;;
    cd5)
        DATASET="cd5_train"
        TEST_DATASET="cd5_test"
        CUTOFF_LEN=74
        GAMMA=2
        ;;
    *)
        echo "Unknown task: $TASK"
        exit 1
        ;;
esac

# Common parameters for tiny model
MODEL="model_config_tiny"
LR="1e-3"
BATCH_SIZE=128  # per device, total = batch_size * num_gpus
EPOCHS=600
DIFFUSION_STEPS=20
ALPHA=0.25

# Setup environment
source ~/.bashrc
conda activate dvar

# Add src to Python path for llmtuner imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)/diffusion-vs-ar/src"

# Create output directory
TIMESTAMP=$(date "+%Y%m%d-%H%M%S")
if [ "$TOKEN_REWEIGHT" = "True" ]; then
    EXP_DIR="diffusion-vs-ar/output/${TASK}/tiny-mdm-a${ALPHA}-g${GAMMA}-lr${LR}-ep${EPOCHS}-T${DIFFUSION_STEPS}-${TIMESTAMP}"
    RUN_NAME="${TASK}_tiny_mdm"
else
    EXP_DIR="diffusion-vs-ar/output/${TASK}/tiny-mdm-noTR-a${ALPHA}-g${GAMMA}-lr${LR}-ep${EPOCHS}-T${DIFFUSION_STEPS}-${TIMESTAMP}"
    RUN_NAME="${TASK}_tiny_mdm_noTR"
fi
mkdir -p "$EXP_DIR"

echo "=== Training Configuration ==="
echo "Task: $TASK"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Cutoff len: $CUTOFF_LEN"
echo "Learning rate: $LR"
echo "Epochs: $EPOCHS"
echo "Alpha: $ALPHA"
echo "Gamma: $GAMMA"
echo "Token reweighting: $TOKEN_REWEIGHT"
echo "Output: $EXP_DIR"
echo "=============================="

cd diffusion-vs-ar

export NCCL_P2P_DISABLE=1

# Training (using torchrun for proper distributed setup)
torchrun --standalone --nproc_per_node=1 src/train_bash.py \
    --stage mdm \
    --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path "$MODEL" \
    --do_train \
    --dataset "$DATASET" \
    --finetuning_type full \
    --cutoff_len "$CUTOFF_LEN" \
    --output_dir "../$EXP_DIR" \
    --overwrite_cache \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --val_size 448 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_steps 1000 \
    --learning_rate "$LR" \
    --num_train_epochs "$EPOCHS" \
    --plot_loss \
    --run_name "$RUN_NAME" \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 3 \
    --remove_unused_columns False \
    --diffusion_steps "$DIFFUSION_STEPS" \
    --save_safetensors False \
    --token_reweighting "$TOKEN_REWEIGHT" \
    --time_reweighting linear \
    --topk_decoding True \
    --alpha "$ALPHA" \
    --gamma "$GAMMA" \
    --report_to wandb \
    2>&1 | tee "../$EXP_DIR/train.log"

# Evaluation
for eval_dataset in $TEST_DATASET; do
    echo "=== Evaluating on $eval_dataset ==="
    mkdir -p "../$EXP_DIR/$eval_dataset"

    torchrun --standalone --nproc_per_node=1 src/train_bash.py \
        --stage mdm \
        --overwrite_output_dir \
        --cache_dir ./cache \
        --model_name_or_path "$MODEL" \
        --do_predict \
        --cutoff_len "$CUTOFF_LEN" \
        --dataset "$eval_dataset" \
        --finetuning_type full \
        --diffusion_steps "$DIFFUSION_STEPS" \
        --output_dir "../$EXP_DIR/${eval_dataset}" \
        --checkpoint_dir "../$EXP_DIR" \
        --remove_unused_columns False \
        --decoding_strategy stochastic0.5-linear \
        --topk_decoding True \
        --report_to wandb \
        2>&1 | tee "../$EXP_DIR/${eval_dataset}/eval.log"
done

echo "=== Training complete ==="
echo "Output: $EXP_DIR"
