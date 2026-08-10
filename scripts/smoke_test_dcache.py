#!/usr/bin/env python3
"""One-batch CUDA smoke test for training, validation, and sample evaluation."""

import argparse
import json
import pathlib
import shutil
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


class SyntheticTokens(Dataset):
  def __init__(self, tokenizer, size=4, length=8, seed=7):
    generator = torch.Generator().manual_seed(seed)
    self.tokens = torch.randint(
      8, tokenizer.vocab_size, (size, length), generator=generator)
    self.tokens[:, 0] = tokenizer.bos_token_id

  def __len__(self):
    return self.tokens.shape[0]

  def __getitem__(self, index):
    input_ids = self.tokens[index]
    return {
      'input_ids': input_ids,
      'attention_mask': torch.ones_like(input_ids),
    }


def compose_smoke_config(eval_model):
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
        'model=tiny',
        'model.length=8',
        'model.hidden_size=32',
        'model.cond_dim=16',
        'model.n_blocks=2',
        'model.n_heads=4',
        'model.dropout=0.0',
        'model.attn_backend=sdpa',
        'block_size=4',
        'loader.global_batch_size=2',
        'loader.eval_global_batch_size=1',
        'loader.batch_size=2',
        'loader.eval_batch_size=1',
        'loader.num_workers=1',
        'trainer.devices=1',
        'trainer.accumulate_grad_batches=1',
        'training.ema=0',
        'training.resample=false',
        'algo.var_min=false',
        'step_memory.enabled=true',
        'step_memory.use_previous_kv=true',
        'step_memory.detach_between_steps=true',
        'step_memory.rollout.enabled=true',
        'step_memory.rollout.forwards_start=2',
        'step_memory.rollout.forwards_end=2',
        'step_memory.rollout.final_mask_ratio_start=0.75',
        'step_memory.rollout.final_mask_ratio_end=0.75',
        'step_memory.rollout.curriculum_steps=1',
        'sampling.first_hitting=true',
        # Training rollout and completed-prefix inference caching have separate
        # lifetimes, so the prefix cache is enabled only after training.
        'sampling.kv_cache=false',
        f'eval.gen_ppl_eval_model_name_or_path={eval_model}',
        'wandb=null',
      ])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--work-dir', type=pathlib.Path,
    default=REPO_ROOT / 'outputs' / 'dcache_smoke')
  parser.add_argument(
    '--eval-model', default='sshleifer/tiny-gpt2',
    help='Small causal LM used to exercise generative-PPL evaluation.')
  parser.add_argument(
    '--skip-generative-ppl', action='store_true',
    help='Skip the Hugging Face download while still testing generation.')
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError('This smoke test requires a CUDA GPU.')

  L.seed_everything(7, workers=True)
  config = compose_smoke_config(args.eval_model)
  tokenizer = dataloader.Text8Tokenizer()
  train_loader = DataLoader(
    SyntheticTokens(tokenizer), batch_size=2, num_workers=1)
  valid_loader = DataLoader(
    SyntheticTokens(tokenizer, size=2, seed=11),
    batch_size=1, num_workers=1)

  if args.work_dir.exists():
    shutil.rmtree(args.work_dir)
  args.work_dir.mkdir(parents=True)

  model = diffusion.Diffusion(config, tokenizer=tokenizer)
  trainer = L.Trainer(
    accelerator='gpu', devices=1, strategy='auto',
    precision='bf16-mixed', max_steps=1,
    limit_train_batches=1, limit_val_batches=1,
    num_sanity_val_steps=0, logger=False,
    enable_checkpointing=False, enable_progress_bar=False,
    default_root_dir=str(args.work_dir))
  trainer.fit(model, train_loader, valid_loader)

  checkpoint_path = args.work_dir / 'smoke.ckpt'
  trainer.save_checkpoint(checkpoint_path)
  loaded = diffusion.Diffusion.load_from_checkpoint(
    checkpoint_path, config=config, tokenizer=tokenizer,
    strict=True, weights_only=False)
  validation = trainer.validate(
    loaded, dataloaders=valid_loader, verbose=False)[0]

  # Use a fresh checkpoint load for generation, matching main.py's separate
  # ppl_eval and sample_eval processes. Lightning validation uses inference
  # tensors internally, which should not leak into generative metric state.
  sample_model = diffusion.Diffusion.load_from_checkpoint(
    checkpoint_path, config=config, tokenizer=tokenizer,
    strict=True, weights_only=False).to('cuda').eval()
  sample_model.config.sampling.kv_cache = True
  samples, sampling_steps = sample_model._semi_ar_sampler(
    n_samples=1, num_steps=8, num_strides=2,
    seqlen=8, context_size=8)
  decoded = tokenizer.batch_decode(
    samples, skip_special_tokens=True)

  generative_ppl = None
  entropy = None
  if not args.skip_generative_ppl:
    sample_model.metrics.record_generative_perplexity(
      decoded, max_length=32, batch_size=1,
      device=sample_model.device)
    generative_ppl = float(sample_model.metrics.gen_ppl.compute().cpu())
    entropy = float(sample_model.metrics.gen_entropy.compute().cpu())

  result = {
    'checkpoint': str(checkpoint_path),
    'train_steps': int(trainer.global_step),
    'validation': {key: float(value) for key, value in validation.items()},
    'sample': decoded[0],
    'sampling_steps': int(sampling_steps),
    'generative_ppl': generative_ppl,
    'entropy': entropy,
    'gpu': torch.cuda.get_device_name(torch.cuda.current_device()),
  }
  print('DCACHE_SMOKE_OK')
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
