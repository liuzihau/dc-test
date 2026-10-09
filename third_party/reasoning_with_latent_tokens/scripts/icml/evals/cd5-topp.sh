#!/bin/bash
#SBATCH -J cd5-topp
#SBATCH --partition=general
#SBATCH --output=slurm/master/%j_%x.out
#SBATCH --error=slurm/master/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --open-mode=append
#SBATCH --time=48:00:00
#SBATCH --mem=128G

source "$ESOLM_CONDA_PROFILE"
conda activate esolm

mkdir -p paper/countdown-topp
# for n_latent_tokens in 0 4 8 16 20 24 28 32 36; do
# for n_latent_tokens in 40 44 48 52; do
for n_latent_tokens in 56 60 64 68 72; do
    ./scripts/icml/train_dvar.sh \
        -m mdm-causal-output \
        -d cd5 \
        --model dvar-tiny-legacy \
        --run-suffix=-tat-121gm \
        --mode=eval \
        -- \
        sampling.unmask_policy=topp \
        sampling.greedy_tokens=True \
        +sampling.n_latent_tokens=${n_latent_tokens} \
        +sampling.topk_candidate_max=8 \
        +sampling.topk_candidate_min=8 \
        > paper/countdown-topp/cd5-tat-latent${n_latent_tokens}.out 2>&1
done