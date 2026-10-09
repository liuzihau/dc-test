#!/bin/bash
# Emit gen_v2.sh arg lines for all rebuttal seed models -> eval_queue.txt
# Mirrors paper eval configs:
#  - diffusion on reasoning tasks: top-prob decoding with candidate set k=8
#  - SCDM additionally swept over sampling.n_latent_tokens
#  - AR-family: uniform (l2r) with greedy_tokens=false
# Paper used 10 batches x 128 = 1280 samples.
set -eu
cd "$(dirname "$0")"
TOPP="sampling.greedy_tokens=false sampling.unmask_policy=topp +sampling.topk_candidate_min=8 +sampling.topk_candidate_max=8"
OUT=eval_queue.txt
: > $OUT

for S in 2 3; do
  # --- Sudoku-Gen ---
  echo "-m diffu-full -d sudoku-small-solver -z sminy --run-suffix=-seed${S}-sms -b 10 -- $TOPP" >> $OUT
  for N in 0 8 16 32 64 128; do
    echo "-m diffu-causal-output -d sudoku-small-solver -z sminy --run-suffix=-seed${S}-sms --gen-suffix=-latent${N} -b 10 -- $TOPP +sampling.n_latent_tokens=${N}" >> $OUT
  done
  echo "-m ar -d sudoku-small-solver -z sminy --run-suffix=-seed${S}-sms -b 10 -- +sampling.greedy_tokens=false" >> $OUT
  echo "-m ar-mtp-full -d sudoku-small-solver -z sminy --run-suffix=-seed${S}-sms -b 10 -- sampling.greedy_tokens=false" >> $OUT
  echo "-m ar-ntp -d sudoku-small-solver -z sminy --run-suffix=-seed${S}-tat-sms -b 10 -- sampling.greedy_tokens=false" >> $OUT
  echo "-m ar-ntp-w1 -d sudoku-small-solver -z sminy --run-suffix=-seed${S}-tat-sms -b 10 -- sampling.greedy_tokens=false" >> $OUT

  # --- Zebra ---
  echo "-m diffu-full -d zebra -z mini --run-suffix=-seed${S}-tat -b 10 -- $TOPP" >> $OUT
  for N in 0 8 16 32 64 128 256 384; do
    echo "-m diffu-causal-output -d zebra -z mini --run-suffix=-seed${S}-tat --gen-suffix=-latent${N} -b 10 -- $TOPP +sampling.n_latent_tokens=${N}" >> $OUT
  done
  echo "-m ar -d zebra -z mini --run-suffix=-seed${S} -b 10 -- +sampling.greedy_tokens=false" >> $OUT
  echo "-m ar-mtp-full -d zebra -z mini --run-suffix=-seed${S} -b 10 -- sampling.greedy_tokens=false" >> $OUT
  echo "-m ar-ntp -d zebra -z mini --run-suffix=-seed${S}-tat -b 10 -- sampling.greedy_tokens=false" >> $OUT
  echo "-m ar-ntp-w1 -d zebra -z mini --run-suffix=-seed${S}-tat -b 10 -- sampling.greedy_tokens=false" >> $OUT
done

# --- LR robustness models (seed 1) ---
for LR in 1e-4 1e-3; do
  echo "-m diffu-full -d sudoku-small-solver -z sminy --run-suffix=-sms-lr${LR} -b 10 -- $TOPP" >> $OUT
  for N in 0 8 32 128; do
    echo "-m diffu-causal-output -d sudoku-small-solver -z sminy --run-suffix=-sms-lr${LR} --gen-suffix=-latent${N} -b 10 -- $TOPP +sampling.n_latent_tokens=${N}" >> $OUT
  done
done

wc -l $OUT
