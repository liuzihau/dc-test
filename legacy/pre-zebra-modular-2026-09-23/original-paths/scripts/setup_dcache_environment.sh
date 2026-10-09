#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available. Install Miniconda, then rerun this script." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx dcache; then
  conda env update --name dcache --file environment.yml --prune
else
  conda env create --file environment.yml
fi
conda activate dcache
python -m pip check
python scripts/check_training_environment.py --expected-gpus 2 \
  --data-dir "${DCACHE_DATA_DIR:-${REPO_DIR}/.cache/huggingface}"
