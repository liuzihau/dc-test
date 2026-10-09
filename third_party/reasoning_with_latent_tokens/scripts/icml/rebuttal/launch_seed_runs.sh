#!/bin/bash
# NeurIPS rebuttal: multi-seed re-runs of core results (Fig 4 / Table 1 / Table 2
# on Sudoku-Gen and Zebra) plus LR-robustness sweep on Sudoku-Gen.
# Seed 1 = original paper runs; we add seeds 2,3.
#
# Usage: ./scripts/icml/rebuttal/launch_seed_runs.sh [--dry-run]
set -e
cd "$(dirname "$0")/../../.."

DRY=""
[ "$1" = "--dry-run" ] && DRY="echo"

submit() {
  $DRY sbatch scripts/icml/train_v2.sh "$@"
}

for SEED in 2 3; do
  # --- Sudoku-Gen (sudoku-small-solver, sminy, suffix -sms as in paper) ---
  for M in diffu-full diffu-causal-output ar ar-mtp-full; do
    submit -m $M -d sudoku-small-solver -z sminy --run-suffix=-sms -s $SEED --no-resume
  done
  for M in ar-ntp ar-ntp-w1; do
    submit -m $M -d sudoku-small-solver -z sminy --run-suffix=-sms -s $SEED --train-on-all-tokens --no-resume
  done

  # --- Zebra (mini; tat flags per original runs) ---
  for M in diffu-full diffu-causal-output ar-ntp ar-ntp-w1; do
    submit -m $M -d zebra -z mini --train-on-all-tokens -s $SEED --no-resume
  done
  for M in ar ar-mtp-full; do
    submit -m $M -d zebra -z mini -s $SEED --no-resume
  done
done

# --- LR robustness sweep (Sudoku-Gen, seed 1; original used optim.lr=3e-4) ---
for LR in 1e-4 1e-3; do
  for M in diffu-full diffu-causal-output; do
    submit -m $M -d sudoku-small-solver -z sminy --run-suffix=-sms-lr${LR} --no-resume -- optim.lr=${LR}
  done
done
