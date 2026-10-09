#!/bin/bash
# Drip-feed train->eval pairs: submit each training job when a slot frees,
# and immediately submit its eval jobs with --dependency=afterok:<train_jid>
# so results arrive incrementally as each model finishes.
# Consumed blocks are appended to paired_queue.done.txt (restart-safe).
set -u
cd "$(dirname "$0")/../../.."
QDIR=scripts/icml/rebuttal
MAX_INFLIGHT=${MAX_INFLIGHT:-6}

while true; do
  if [ ! -s "$QDIR/paired_queue.txt" ]; then
    echo "QUEUE_EMPTY: all train+eval pairs submitted"
    exit 0
  fi
  inflight=$(squeue -u $USER -h -n train-v2 | wc -l)
  if [ "$inflight" -lt "$MAX_INFLIGHT" ]; then
    # Read the first block (up to ---)
    block=$(sed -n '1,/^---$/p' "$QDIR/paired_queue.txt")
    train_args=$(echo "$block" | sed -n 's/^TRAIN //p')
    out=$(sbatch scripts/icml/train_v2.sh $train_args 2>&1)
    if echo "$out" | grep -q "Submitted"; then
      jid=$(echo "$out" | grep -o '[0-9]*$')
      echo "SUBMITTED train $jid: $train_args"
      echo "$jid TRAIN $train_args" >> "$QDIR/submitted.log"
      echo "$block" | sed -n 's/^EVAL //p' | while read -r eval_args; do
        [ -z "$eval_args" ] && continue
        eout=$(sbatch --dependency=afterok:$jid scripts/icml/gen_v2.sh $eval_args 2>&1)
        ejid=$(echo "$eout" | grep -o '[0-9]*$')
        echo "  chained eval $ejid (afterok:$jid)"
        echo "$ejid EVAL(dep:$jid) $eval_args" >> "$QDIR/submitted.log"
      done
      # pop the block
      n=$(echo "$block" | wc -l)
      echo "$block" >> "$QDIR/paired_queue.done.txt"
      sed -i "1,${n}d" "$QDIR/paired_queue.txt"
    else
      echo "SBATCH_FAILED: $out"
      sleep 600
    fi
  fi
  sleep 120
done
