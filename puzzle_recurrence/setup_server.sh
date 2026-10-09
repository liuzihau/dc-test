#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [ "${CONDA_DEFAULT_ENV:-}" != dcache ]; then
  echo "First: conda create -n dcache python=3.9 -y; conda activate dcache" >&2
  exit 2
fi
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-h100.txt
# Only tokenizer files are fetched; model weights are not needed for puzzle runs.
export HF_HOME="${HF_HOME:-$repo_root/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
python -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("gpt2-large")'
python -c 'import torch, lightning; print("Torch:",torch.__version__,"Lightning:",lightning.__version__); print("Visible GPUs:",torch.cuda.device_count()); assert torch.cuda.is_available(); print("GPU0:",torch.cuda.get_device_name(0),"BF16:",torch.cuda.is_bf16_supported()); assert torch.cuda.is_bf16_supported(), "This author backbone requires BF16 support"'
