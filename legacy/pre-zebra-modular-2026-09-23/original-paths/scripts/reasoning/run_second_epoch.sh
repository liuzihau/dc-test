#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export TMPDIR="$PWD/.cache/runtime/reasoning-second-epoch/tmp"
export MPLCONFIGDIR="$PWD/.cache/runtime/reasoning-second-epoch/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/runtime/reasoning-second-epoch/inductor"
export TRITON_CACHE_DIR="$PWD/.cache/runtime/reasoning-second-epoch/triton"
export CUDA_CACHE_PATH="$PWD/.cache/runtime/reasoning-second-epoch/cuda"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
exec "${DCACHE_PYTHON:-python}" -u -m reasoning.second_epoch_queue "$@"
