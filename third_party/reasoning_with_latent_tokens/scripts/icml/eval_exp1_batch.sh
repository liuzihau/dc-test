#!/bin/bash
# Batch evaluation script for exp1 experiments
# Submits 24 evaluation jobs (4 methods × 4 datasets + 2 diffusion methods × 4 datasets with topp)
#
# Usage:
#   bash scripts/icml/eval_exp1_batch.sh
#
# All runs use sminy model size and -exp1 suffix

set -e

# Methods and datasets
METHODS=(ar ar-mtp-full diffu-full diffu-solo-full)
DIFFUSION_METHODS=(diffu-full diffu-solo-full)  # Methods that also get topp sampling
DATASETS=(sudoku-puzzle sudoku-small-solver zebra game-of-24)

# Common parameters
SIZE=sminy
RUN_SUFFIX=-exp1
NUM_BATCHES=10

echo "=== Submitting exp1 evaluation jobs ==="
echo "Methods: ${METHODS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Size: $SIZE"
echo "Run suffix: $RUN_SUFFIX"
echo "Batches: $NUM_BATCHES"
echo ""

# Track submitted jobs
JOBS=()

for METHOD in "${METHODS[@]}"; do
    for DATA in "${DATASETS[@]}"; do
        echo "Submitting: $METHOD on $DATA"

        JOB_OUTPUT=$(sbatch scripts/icml/gen_v2.sh \
            --method="$METHOD" \
            --data="$DATA" \
            --size="$SIZE" \
            --run-suffix="$RUN_SUFFIX" \
            --batches="$NUM_BATCHES" 2>&1)

        JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'Submitted batch job \K\d+' || echo "FAILED")
        echo "  -> Job ID: $JOB_ID"
        JOBS+=("$JOB_ID:$METHOD:$DATA")
    done
done

# Submit topp sampling jobs for diffusion methods
echo ""
echo "=== Submitting topp sampling jobs for diffusion methods ==="
for METHOD in "${DIFFUSION_METHODS[@]}"; do
    for DATA in "${DATASETS[@]}"; do
        echo "Submitting: ${METHOD}-topp on $DATA"

        JOB_OUTPUT=$(sbatch scripts/icml/gen_v2.sh \
            --method="$METHOD" \
            --data="$DATA" \
            --size="$SIZE" \
            --run-suffix="$RUN_SUFFIX" \
            --gen-suffix="-topp" \
            --batches="$NUM_BATCHES" \
            -- sampling.unmask_policy=topp 2>&1)

        JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'Submitted batch job \K\d+' || echo "FAILED")
        echo "  -> Job ID: $JOB_ID"
        JOBS+=("$JOB_ID:${METHOD}-topp:$DATA")
    done
done

echo ""
echo "=== Submitted ${#JOBS[@]} jobs ==="
for JOB in "${JOBS[@]}"; do
    echo "  $JOB"
done

echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "After completion, run: python scripts/icml/aggregate_exp1_results.py"
