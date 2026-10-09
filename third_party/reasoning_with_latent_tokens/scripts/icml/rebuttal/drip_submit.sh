#!/bin/bash
# Drip-feed rebuttal jobs: keep at most $MAX_INFLIGHT of our train-v2 jobs
# in the queue (pending+running); submit the next line of job_queue.txt when
# a slot frees. Submitted job IDs are appended to submitted.log; lines already
# consumed are moved to job_queue.done.txt so the script is restart-safe.
set -u
cd "$(dirname "$0")/../../.."
QDIR=scripts/icml/rebuttal
MAX_INFLIGHT=${MAX_INFLIGHT:-6}

while true; do
  next=$(head -1 "$QDIR/job_queue.txt" 2>/dev/null)
  if [ -z "$next" ]; then
    echo "QUEUE_EMPTY: all jobs submitted"
    exit 0
  fi
  inflight=$(squeue -u $USER -h -n train-v2 | wc -l)
  if [ "$inflight" -lt "$MAX_INFLIGHT" ]; then
    out=$(sbatch scripts/icml/train_v2.sh $next 2>&1)
    if echo "$out" | grep -q "Submitted"; then
      jid=$(echo "$out" | grep -o '[0-9]*$')
      echo "SUBMITTED $jid: $next"
      echo "$jid $next" >> "$QDIR/submitted.log"
      sed -i '1d' "$QDIR/job_queue.txt"
      echo "$next" >> "$QDIR/job_queue.done.txt"
    else
      echo "SBATCH_FAILED: $out"
      sleep 600
    fi
  fi
  sleep 120
done
