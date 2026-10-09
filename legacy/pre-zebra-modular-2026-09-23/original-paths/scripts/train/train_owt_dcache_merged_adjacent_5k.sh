#!/usr/bin/env bash
# New architecture: fresh full-data training, never the old compact continuation.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
if [[ "${DCACHE_MERGED_POLICY:-legacy}" != legacy ]]; then
  echo 'This historical launcher is legacy-only. Use train_owt_dcache_merged_pair.sh --attention-policy current-preserving for the new policy.' >&2
  exit 2
fi
for argument in "$@"; do
  key="${argument%%=*}"
  key="${key#++}"; key="${key#+}"; key="${key#\~}"
  if [[ "$key" == step_memory || "$key" == step_memory.merged_policy ]]; then
    echo 'The historical merged launcher fixes step_memory.merged_policy=legacy; use the explicit pair policy selector.' >&2
    exit 2
  fi
done
export DCACHE_RUN_DIR="${DCACHE_RUN_DIR:-$REPO_DIR/outputs/owt-dcache-merged-final-state-adjacent-5k}"
export DCACHE_PYTHON="${DCACHE_PYTHON:-/home/tliu0205/miniconda3/envs/dcache/bin/python}"
export DCACHE_DATA_DIR="${DCACHE_DATA_DIR:-$REPO_DIR/.cache/huggingface}"
if [[ -n "${DCACHE_RESUME_CKPT:-}" || -f "$DCACHE_DATA_DIR/compact_train.json" ]]; then
  echo 'Merged attention is a NEW architecture. Unset DCACHE_RESUME_CKPT and use the full prepared training cache, not the 1500-5000 compact bundle.' >&2
  exit 2
fi
if [[ -L "$DCACHE_RUN_DIR/checkpoints/last.ckpt" && ! -e "$DCACHE_RUN_DIR/checkpoints/last.ckpt" ]]; then
  echo 'Broken last.ckpt pointer; refusing to start from scratch.' >&2
  exit 2
fi
if [[ -f "$DCACHE_RUN_DIR/checkpoints/last.ckpt" ]]; then
  MERGED_CHECKPOINT="$DCACHE_RUN_DIR/checkpoints/last.ckpt" "$DCACHE_PYTHON" -c '
import os, torch
from omegaconf import OmegaConf
c = torch.load(os.environ["MERGED_CHECKPOINT"], map_location="cpu", weights_only=False, mmap=True)
if OmegaConf.select(c["hyper_parameters"]["config"], "step_memory.attention_mode", default="separate") != "merged":
    raise SystemExit("Refusing to load a separate-attention checkpoint into the merged trial")
if OmegaConf.select(c["hyper_parameters"]["config"], "step_memory.merged_policy", default="legacy") != "legacy":
    raise SystemExit("Refusing to load a current-preserving checkpoint into the historical legacy merged trial")
'
fi
echo 'Merged trial: shared QKV, one joint softmax, spatial/iteration-age RoPE.'
echo 'The optional gate scales previous V; cache-only dropout now removes the sole current-KV route.'
exec bash "$SCRIPT_DIR/train_owt_dcache_final_state_adjacent_5k_2x3090.sh" \
  step_memory.attention_mode=merged step_memory.merged_policy=legacy "$@"
