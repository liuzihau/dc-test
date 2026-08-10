#!/usr/bin/env python3
"""Profile one full BD3-small recurrent train step and validation batch."""

import argparse
import json
import pathlib
import sys

import hydra
import lightning as L
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import dataloader  # noqa: E402
import diffusion  # noqa: E402


class OneBatch(Dataset):
  def __init__(self, tokenizer, length, seed, size=1):
    generator = torch.Generator().manual_seed(seed)
    self.input_ids = torch.randint(
      0, tokenizer.vocab_size, (size, length), generator=generator)
    self.input_ids[:, 0] = tokenizer.bos_token_id

  def __len__(self):
    return self.input_ids.shape[0]

  def __getitem__(self, index):
    return {
      'input_ids': self.input_ids[index],
      'attention_mask': torch.ones_like(self.input_ids[index]),
    }


def compose_config(
    attn_backend, rollout_forwards, final_mask_ratio, micro_batch):
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
        'model=small',
        'model.length=1024',
        f'model.attn_backend={attn_backend}',
        'block_size=16',
        f'loader.global_batch_size={micro_batch}',
        'loader.eval_global_batch_size=1',
        f'loader.batch_size={micro_batch}',
        'loader.eval_batch_size=1',
        'loader.num_workers=1',
        'trainer.devices=1',
        'trainer.accumulate_grad_batches=1',
        'training.resample=true',
        'training.from_pretrained=null',
        'algo.var_min=false',
        'step_memory.enabled=true',
        'step_memory.use_previous_kv=true',
        'step_memory.detach_between_steps=true',
        'step_memory.rollout.enabled=true',
        f'step_memory.rollout.forwards_start={rollout_forwards}',
        f'step_memory.rollout.forwards_end={rollout_forwards}',
        f'step_memory.rollout.final_mask_ratio_start={final_mask_ratio}',
        f'step_memory.rollout.final_mask_ratio_end={final_mask_ratio}',
        'eval.gen_ppl_eval_model_name_or_path=sshleifer/tiny-gpt2',
        'wandb=null',
      ])


def make_trainer(work_dir, max_steps, limit_val_batches):
  return L.Trainer(
    accelerator='gpu', devices=1, strategy='auto',
    precision='bf16-mixed', max_steps=max_steps,
    limit_train_batches=1, limit_val_batches=limit_val_batches,
    num_sanity_val_steps=0, logger=False,
    enable_checkpointing=False, enable_progress_bar=False,
    default_root_dir=str(work_dir))


def gib(value):
  return round(value / 1024 ** 3, 3)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--attn-backend', choices=['flex', 'sdpa'], default='flex')
  parser.add_argument('--rollout-forwards', type=int, default=5)
  parser.add_argument('--final-mask-ratio', type=float, default=0.0625)
  parser.add_argument('--micro-batch', type=int, default=1)
  parser.add_argument(
    '--work-dir', type=pathlib.Path,
    default=REPO_ROOT / 'outputs' / 'dcache_full_profile')
  args = parser.parse_args()
  if not torch.cuda.is_available():
    raise RuntimeError('This profile requires a CUDA GPU.')

  torch.set_float32_matmul_precision('high')
  L.seed_everything(17, workers=True)
  config = compose_config(
    args.attn_backend, args.rollout_forwards,
    args.final_mask_ratio, args.micro_batch)
  tokenizer = dataloader.get_tokenizer(config)
  train_loader = DataLoader(
    OneBatch(
      tokenizer, config.model.length, 17, size=args.micro_batch),
    batch_size=args.micro_batch, num_workers=1)
  valid_loader = DataLoader(
    OneBatch(tokenizer, config.model.length, 19),
    batch_size=1, num_workers=1)
  args.work_dir.mkdir(parents=True, exist_ok=True)

  model = diffusion.Diffusion(config, tokenizer=tokenizer)
  parameter_count = sum(p.numel() for p in model.parameters())
  torch.cuda.reset_peak_memory_stats()
  trainer = make_trainer(args.work_dir, max_steps=1, limit_val_batches=0)
  trainer.fit(model, train_loader)
  train_peak_allocated = torch.cuda.max_memory_allocated()
  train_peak_reserved = torch.cuda.max_memory_reserved()

  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
  validator = make_trainer(args.work_dir, max_steps=1, limit_val_batches=1)
  validation = validator.validate(
    model, dataloaders=valid_loader, verbose=False)[0]
  validation_peak_allocated = torch.cuda.max_memory_allocated()
  validation_peak_reserved = torch.cuda.max_memory_reserved()

  print('DCACHE_FULL_PROFILE_OK')
  print(json.dumps({
    'attention_backend': args.attn_backend,
    'rollout_forwards': args.rollout_forwards,
    'final_mask_ratio': args.final_mask_ratio,
    'micro_batch': args.micro_batch,
    'gpu': torch.cuda.get_device_name(torch.cuda.current_device()),
    'parameters': parameter_count,
    'train_peak_allocated_gib': gib(train_peak_allocated),
    'train_peak_reserved_gib': gib(train_peak_reserved),
    'validation_peak_allocated_gib': gib(validation_peak_allocated),
    'validation_peak_reserved_gib': gib(validation_peak_reserved),
    'validation': {key: float(value) for key, value in validation.items()},
  }, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
