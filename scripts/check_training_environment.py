#!/usr/bin/env python3
"""Fail-fast environment check before launching the two-GPU experiments."""

import argparse
import shutil
import sys
from pathlib import Path

import datasets
import lightning
import torch
import transformers


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--expected-gpus', type=int, default=2)
  parser.add_argument('--data-dir', default='.cache/huggingface')
  args = parser.parse_args()

  print(f'Python: {sys.version.split()[0]}')
  print(f'PyTorch: {torch.__version__}')
  print(f'Lightning: {lightning.__version__}')
  print(f'Transformers: {transformers.__version__}')
  print(f'Datasets: {datasets.__version__}')
  print(f'CUDA build: {torch.version.cuda}')
  if not torch.cuda.is_available():
    raise SystemExit('ERROR: PyTorch cannot access CUDA.')
  count = torch.cuda.device_count()
  if count < args.expected_gpus:
    raise SystemExit(
      f'ERROR: expected at least {args.expected_gpus} GPUs, found {count}.')
  for index in range(count):
    props = torch.cuda.get_device_properties(index)
    print(f'GPU {index}: {props.name}, {props.total_memory / 2**30:.1f} GiB, '
          f'compute capability {props.major}.{props.minor}')
  if not torch.cuda.is_bf16_supported():
    raise SystemExit('ERROR: the selected CUDA device does not support BF16.')

  data_dir = Path(args.data_dir).expanduser().resolve()
  free_gib = shutil.disk_usage(data_dir.parent if data_dir.parent.exists()
                               else Path.cwd()).free / 2**30
  print(f'Data cache: {data_dir}')
  print(f'Free disk near data cache: {free_gib:.1f} GiB')
  train_cache = list(data_dir.glob('openwebtext-train_train_bs1024*'))
  valid_cache = list(data_dir.glob('openwebtext-valid_validation_bs1024*'))
  if train_cache and valid_cache:
    print('Prepared OpenWebText train and validation caches: found')
  else:
    print('WARNING: prepared OpenWebText cache is absent; the first run will '
          'download and preprocess a very large dataset.')
  print('Environment check passed.')


if __name__ == '__main__':
  main()
