#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Project policy: evaluation/training jobs default to physical GPU 2 (or an
# explicitly supplied subset of physical GPUs 2 and 3).  Inside that visible
# set the process addresses the first device as cuda:0.
export CUDA_VISIBLE_DEVICES="${DCACHE_LAYER_COSINE_GPU:-2}"
export PYTHONPATH="$repo_root:$repo_root/third_party/reasoning_with_latent_tokens${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$repo_root/.tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export MPLCONFIGDIR="$repo_root/.cache/runtime/layer-cosine/matplotlib"
export TRITON_CACHE_DIR="$repo_root/.cache/runtime/layer-cosine/triton"
export TORCHINDUCTOR_CACHE_DIR="$repo_root/.cache/runtime/layer-cosine/inductor"
export CUDA_CACHE_PATH="$repo_root/.cache/runtime/layer-cosine/cuda"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" logs

python_bin="${DCACHE_PYTHON:-/home/tliu0205/miniconda3/envs/dcache/bin/python}"
exec "$python_bin" -u -m representation.layer_cosine \
  --device cuda:0 \
  --samples 100 \
  --batch-size "${DCACHE_LAYER_COSINE_BATCH:-25}" \
  "$@"
