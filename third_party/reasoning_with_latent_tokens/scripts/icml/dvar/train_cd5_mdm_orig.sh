#!/bin/bash
#SBATCH -J cd5-mdm
#SBATCH --partition=general
#SBATCH --output=slurm/dvar/%j_%x.out
#SBATCH --error=slurm/dvar/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G

# Train MDM on cd5 using the original diffusion-vs-ar script structure

set -e

source ~/.bashrc
conda activate dvar

cd diffusion-vs-ar

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export NCCL_P2P_DISABLE=1

# Create output directory (matching their naming convention)
exp=output/cd5/mdm-alpha0.25-gamma2-bs1024-lr1e-3-ep600-T20-$(date "+%Y%m%d-%H%M%S")
mkdir -p $exp

echo "=== Training Configuration ==="
echo "Task: cd5"
echo "Output: $exp"
echo "Batch size: 1024 (512 x 2 grad_accum)"
echo "==============================="

# Training
torchrun --standalone --nproc_per_node=1 src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_tiny \
    --do_train \
    --dataset cd5_train \
    --finetuning_type full \
    --cutoff_len 74 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 512 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 448 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 500 \
    --learning_rate 1e-3 \
    --num_train_epochs 600.0 \
    --plot_loss \
    --run_name cd5_mdm \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting True \
    --time_reweighting linear \
    --topk_decoding True \
    --alpha 0.25 \
    --gamma 2 \
    --report_to wandb \
    2>&1 | tee $exp/train.log

# Evaluation
for dataset in cd5_test; do
    echo "=== Evaluating on $dataset ==="
    topk_decoding=True
    mkdir -p $exp/$dataset
    python3 -u src/train_bash.py \
        --stage mdm --overwrite_output_dir \
        --cache_dir ./cache \
        --model_name_or_path model_config_tiny \
        --do_predict \
        --cutoff_len 74 \
        --dataset $dataset \
        --finetuning_type full \
        --diffusion_steps 20 \
        --output_dir $exp/${dataset} \
        --checkpoint_dir $exp \
        --remove_unused_columns False \
        --decoding_strategy stochastic0.5-linear \
        --topk_decoding $topk_decoding \
        --report_to wandb \
        2>&1 | tee $exp/${dataset}/eval-TopK$topk_decoding.log
done

echo "=== Training complete ==="
echo "Output: $exp"
