#!/usr/bin/env bash
# Paper-informed reconstruction. Read manifest benchmark_protocol for caveats.
# All five methods use identical immutable data, examples, seeds and decoding.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export TMPDIR="$PWD/.cache/runtime/reasoning-benchmark/tmp"
export MPLCONFIGDIR="$PWD/.cache/runtime/reasoning-benchmark/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/runtime/reasoning-benchmark/inductor"
export TRITON_CACHE_DIR="$PWD/.cache/runtime/reasoning-benchmark/triton"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
exec "${DCACHE_PYTHON:-python}" -u -m reasoning.benchmark_queue "$@"
