#!/bin/bash
#SBATCH -J game-of-24
#SBATCH --partition=general
#SBATCH --output=slurm/game_of_24/%j_%x.out
#SBATCH --error=slurm/game_of_24/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --time=48:00:00
#SBATCH --mem=100G

source ~/.bashrc
conda activate esolm

python -m synthetic_data.game_of_24.data