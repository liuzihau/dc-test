#!/bin/bash
# Launch all 4 tiny model experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Launching 4 tiny model MDM experiments..."

for task in 3sat7 3sat9 cd4 cd5; do
    echo "Submitting $task..."
    sbatch --partition=general --gres=gpu:L40S:1 "$SCRIPT_DIR/train_tiny_mdm.sh" --task="$task"
done

echo "All jobs submitted. Check status with: squeue -u $USER"
