#!/usr/bin/env python3
"""Profile one real-shape vanilla or three-pass DCache pretraining update."""

import argparse
import json
import pathlib
import sys
import time

import hydra
import lightning as L
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import dataloader  # noqa: E402
import diffusion  # noqa: E402


class SyntheticOpenWebTextBatch(Dataset):
  def __init__(self, tokenizer, size, length=1024, seed=23):
    generator = torch.Generator().manual_seed(seed)
    self.input_ids = torch.randint(
      0, tokenizer.vocab_size, (size, length), generator=generator)
    self.input_ids[:, 0] = tokenizer.bos_token_id

  def __len__(self):
    return self.input_ids.shape[0]

  def __getitem__(self, index):
    tokens = self.input_ids[index]
    return {
      'input_ids': tokens,
      'attention_mask': torch.ones_like(tokens),
    }


def compose_config(micro_batch, recurrent):
  for name, resolver in {
      'cwd': lambda: str(REPO_ROOT),
      'device_count': torch.cuda.device_count,
      'eval': eval,
      'div_up': lambda x, y: (x + y - 1) // y,
  }.items():
    if not OmegaConf.has_resolver(name):
      OmegaConf.register_new_resolver(name, resolver)
  with hydra.initialize_config_dir(
      version_base=None, config_dir=str(REPO_ROOT / 'configs')):
    return hydra.compose(
      config_name='config',
      overrides=[
        'algo=mdlm',
        'model=small',
        'model.length=1024',
        'model.attn_backend=sdpa',
        'block_size=1024',
        f'loader.global_batch_size={micro_batch}',
        f'loader.eval_global_batch_size={micro_batch}',
        f'loader.batch_size={micro_batch}',
        f'loader.eval_batch_size={micro_batch}',
        'loader.num_workers=1',
        'trainer.devices=1',
        'trainer.accumulate_grad_batches=1',
        'training.from_pretrained=null',
        f'step_memory.enabled={str(recurrent).lower()}',
        f'step_memory.pretrain.enabled={str(recurrent).lower()}',
        'step_memory.use_previous_kv=true',
        'step_memory.detach_between_steps=true',
        'step_memory.pretrain.teacher_token_probability=1.0',
        'step_memory.rollout.enabled=false',
        'wandb=null',
      ])


def gib(value):
  return round(value / 1024 ** 3, 3)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--micro-batch', type=int, default=1)
  parser.add_argument('--steps', type=int, default=1)
  parser.add_argument('--vanilla', action='store_true')
  parser.add_argument(
    '--work-dir', type=pathlib.Path,
    default=REPO_ROOT / 'outputs' / 'dcache_pretrain_profile')
  args = parser.parse_args()
  if not torch.cuda.is_available():
    raise RuntimeError('This profile requires a CUDA GPU.')

  recurrent = not args.vanilla
  torch.set_float32_matmul_precision('high')
  L.seed_everything(23, workers=True)
  config = compose_config(args.micro_batch, recurrent)
  tokenizer = dataloader.get_tokenizer(config)
  loader = DataLoader(
    SyntheticOpenWebTextBatch(tokenizer, args.micro_batch * args.steps),
    batch_size=args.micro_batch, num_workers=1)
  args.work_dir.mkdir(parents=True, exist_ok=True)
  model = diffusion.Diffusion(config, tokenizer=tokenizer)

  torch.cuda.reset_peak_memory_stats()
  trainer = L.Trainer(
    accelerator='gpu', devices=1, strategy='auto', precision='bf16-mixed',
    max_steps=args.steps, limit_train_batches=args.steps, limit_val_batches=0,
    num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
    enable_progress_bar=False, default_root_dir=str(args.work_dir))
  started = time.perf_counter()
  trainer.fit(model, loader)
  elapsed = time.perf_counter() - started

  print('DCACHE_PRETRAIN_PROFILE_OK')
  print(json.dumps({
    'mode': 'shifted_dcache' if recurrent else 'vanilla_mdlm',
    'micro_batch': args.micro_batch,
    'steps': args.steps,
    'gpu': torch.cuda.get_device_name(torch.cuda.current_device()),
    'parameters': sum(parameter.numel() for parameter in model.parameters()),
    'total_seconds': round(elapsed, 3),
    'average_update_seconds': round(elapsed / args.steps, 3),
    'peak_allocated_gib': gib(torch.cuda.max_memory_allocated()),
    'peak_reserved_gib': gib(torch.cuda.max_memory_reserved()),
  }, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
