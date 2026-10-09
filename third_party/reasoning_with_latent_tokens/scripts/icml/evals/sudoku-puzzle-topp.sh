#!/bin/bash
#SBATCH -J sudoku-puzzle-topp
#SBATCH --partition=general
#SBATCH --output=slurm/master/%j_%x.out
#SBATCH --error=slurm/master/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --open-mode=append
#SBATCH --time=48:00:00
#SBATCH --mem=128G

topk=$1

source "$ESOLM_CONDA_PROFILE"
conda activate esolm

mkdir paper/sudoku-puzzle-topp-$topk
SDKS_EVAL_CONFIGS="sampling.greedy_tokens=false sampling.unmask_policy=topp +sampling.topk_candidate_min=$topk +sampling.topk_candidate_max=$topk"

for n_latent_tokens in 0 8 16 32 64 128 256; do
    ./scripts/icml/gen_v2.sh -m diffu-causal-output -d sudoku-puzzle -z mini --run-suffix -tat -b 10 -- $SDKS_EVAL_CONFIGS +sampling.n_latent_tokens=${n_latent_tokens} > paper/sudoku-puzzle-topp-$topk/diffu-causal-sudoku-puzzle-mini-tat-latent${n_latent_tokens}.out 2>&1
done