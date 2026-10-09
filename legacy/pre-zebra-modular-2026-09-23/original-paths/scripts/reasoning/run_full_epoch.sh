#!/usr/bin/env bash
# Ten full-data runs; one epoch each, no old subset checkpoint initialization.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export TMPDIR="$PWD/.cache/runtime/reasoning-full-epoch/tmp"
export MPLCONFIGDIR="$PWD/.cache/runtime/reasoning-full-epoch/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/runtime/reasoning-full-epoch/inductor"
export TRITON_CACHE_DIR="$PWD/.cache/runtime/reasoning-full-epoch/triton"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
exec "${DCACHE_PYTHON:-python}" -u -m reasoning.full_epoch_queue "$@"
