#!/usr/bin/env python3
"""Shared utilities for controlled, teacher-forced DCache evaluations."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from typing import Any

import hydra
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MASK_SEED_MODULUS = 2 ** 63 - 1
MASK_SEED_STRIDE = 1_000_003


def register_resolvers() -> None:
  for name, resolver in {
      'cwd': lambda: str(REPO_ROOT),
      'device_count': torch.cuda.device_count,
      'eval': eval,
      'div_up': lambda x, y: (x + y - 1) // y,
  }.items():
    if not OmegaConf.has_resolver(name):
      OmegaConf.register_new_resolver(name, resolver)


def compose_eval_config(
    batch_size: int,
    data_dir: pathlib.Path,
    recurrent: bool,
    num_workers: int = 2,
    gate_enabled: bool = False,
    final_state_enabled: bool = False,
):
  """Compose the exact small/MDLM/1024 configuration used by both runs."""
  register_resolvers()
  recurrent_text = str(recurrent).lower()
  with hydra.initialize_config_dir(
      version_base=None, config_dir=str(REPO_ROOT / 'configs')):
    overrides = [
        'algo=mdlm',
        'model=small',
        'model.length=1024',
        'model.attn_backend=sdpa',
        'block_size=1024',
        'data=openwebtext-split',
        f'data.cache_dir={data_dir}',
        'data.insert_valid_special=false',
        'data.insert_valid_eos=false',
        f'loader.global_batch_size={batch_size}',
        f'loader.eval_global_batch_size={batch_size}',
        f'loader.batch_size={batch_size}',
        f'loader.eval_batch_size={batch_size}',
        f'loader.num_workers={num_workers}',
        'trainer.devices=1',
        'trainer.accumulate_grad_batches=1',
        'training.from_pretrained=null',
        f'step_memory.enabled={recurrent_text}',
        f'step_memory.pretrain.enabled={recurrent_text}',
        f'step_memory.gate.enabled={str(gate_enabled).lower()}',
        'step_memory.gate.init=0.1',
        'step_memory.use_previous_kv=true',
        'step_memory.detach_between_steps=true',
        'step_memory.pretrain.teacher_token_probability=1.0',
        'step_memory.rollout.enabled=false',
        'wandb=null',
    ]
    if final_state_enabled:
      if not recurrent:
        raise ValueError('Final-state recurrence requires recurrent=true')
      overrides.extend([
        'dcachehooping.enabled=true',
        'dcachehooping.status_embedding.enabled=false',
        'dcachehooping.latent_dropout_probability=0.10',
        'dcachehooping.latent_mask_probability=0.0',
        'dcachehooping.latent_mask_loss_weight=0.0',
        'dcachehooping.tentative.enabled=false',
        'dcachehooping.tentative.batch_probability=0.0',
        'dcachehooping.tentative.loss_weight=0.0',
        'dcachehooping.confidence.enabled=false',
        'dcachehooping.confidence.loss_weight=0.0',
        'dcachehooping.identity_final_probability=0.50',
      ])
    return hydra.compose(
      config_name='config',
      overrides=overrides)


def load_validation_data(config, tokenizer, examples: int, batch_size: int,
                         num_workers: int) -> DataLoader:
  import dataloader

  dataset = dataloader.get_dataset(
    config.data.valid,
    tokenizer,
    wrap=bool(config.data.wrap),
    mode='validation',
    cache_dir=str(config.data.cache_dir),
    block_size=int(config.model.length),
    streaming=bool(config.data.streaming),
    insert_eos=bool(config.data.insert_valid_eos),
    insert_special_tokens=bool(config.data.insert_valid_special))
  if examples > len(dataset):
    raise ValueError(
      f'Requested {examples} examples but validation set has {len(dataset)}')
  subset = Subset(dataset, range(examples))
  return DataLoader(
    subset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=num_workers > 0,
    drop_last=False)


def load_tokenizer(config):
  import dataloader
  return dataloader.get_tokenizer(config)


def load_ema_model(checkpoint: pathlib.Path, config, tokenizer,
                   device: torch.device):
  """Strictly load a training checkpoint and install its EMA parameters."""
  import diffusion

  if not checkpoint.is_file():
    raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
  model = diffusion.Diffusion.load_from_checkpoint(
    str(checkpoint),
    tokenizer=tokenizer,
    config=config,
    strict=True,
    weights_only=False,
    map_location='cpu')
  model.to(device)
  if model.ema is None:
    raise RuntimeError(f'Checkpoint has no EMA weights: {checkpoint}')
  model.ema.move_shadow_params_to_device(device)
  model.ema.copy_to(model._get_parameters())
  model.ema = None
  model.eval()
  model.backbone.eval()
  model.noise.eval()
  return model


def unload_model(model) -> None:
  del model
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()


def eligible_positions(attention_mask: torch.Tensor,
                       ignore_first: bool = True) -> torch.Tensor:
  eligible = attention_mask.bool().clone()
  if ignore_first:
    eligible[:, 0] = False
  return eligible


def deterministic_nested_masks(
    attention_mask: torch.Tensor,
    example_ids: Sequence[int],
    ratios: Sequence[float],
    seed: int,
    ignore_first: bool = True,
) -> dict[float, torch.Tensor]:
  """Return exact-count, nested masks shared by every compared condition."""
  if len(example_ids) != attention_mask.shape[0]:
    raise ValueError('example_ids must match the batch dimension')
  clean_ratios = sorted(set(float(ratio) for ratio in ratios))
  if not clean_ratios or clean_ratios[0] <= 0 or clean_ratios[-1] > 1:
    raise ValueError('Mask ratios must be in (0, 1]')
  eligible = eligible_positions(attention_mask, ignore_first=ignore_first)
  output = {
    ratio: torch.zeros_like(eligible, dtype=torch.bool)
    for ratio in clean_ratios
  }
  for row, example_id in enumerate(example_ids):
    positions = eligible[row].nonzero(as_tuple=False).flatten().cpu()
    if positions.numel() == 0:
      raise ValueError(f'Example {example_id} has no maskable positions')
    generator = torch.Generator(device='cpu')
    row_seed = (int(seed) + int(example_id) * MASK_SEED_STRIDE)
    generator.manual_seed(row_seed % MASK_SEED_MODULUS)
    permutation = positions[
      torch.randperm(positions.numel(), generator=generator)]
    for ratio in clean_ratios:
      count = max(1, int(round(ratio * positions.numel())))
      count = min(count, positions.numel())
      output[ratio][row, permutation[:count]] = True
  return output


def masked_state(tokens: torch.Tensor, mask: torch.Tensor,
                 mask_index: int) -> torch.Tensor:
  return torch.where(mask.to(tokens.device), mask_index, tokens)


def sigma_for_ratio(model, ratio: float, batch_size: int,
                    device: torch.device) -> torch.Tensor:
  probability = torch.full(
    (batch_size, 1), float(ratio), dtype=torch.float32, device=device)
  return model._sigma_from_p(probability)


def score_masked_tokens(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    example_ids: Sequence[int],
    **fields: Any,
) -> list[dict[str, Any]]:
  """Compute raw conditional metrics per document on masked positions only."""
  mask = mask.to(log_probs.device).bool()
  targets = targets.to(log_probs.device)
  rows: list[dict[str, Any]] = []
  for row, example_id in enumerate(example_ids):
    selected = mask[row]
    count = int(selected.sum().item())
    if count == 0:
      raise ValueError(f'Example {example_id} has no scored tokens')
    row_scores = log_probs[row, selected].float()
    row_targets = targets[row, selected]
    target_log_probs = row_scores.gather(
      -1, row_targets.unsqueeze(-1)).squeeze(-1)
    top_k = min(5, row_scores.shape[-1])
    top_predictions = row_scores.topk(top_k, dim=-1).indices
    top1 = top_predictions[:, 0].eq(row_targets).float().mean().item()
    top5 = top_predictions.eq(row_targets.unsqueeze(-1)).any(
      dim=-1).float().mean().item()
    result = {
      'example_id': int(example_id),
      'masked_tokens': count,
      'nll': float(-target_log_probs.mean().item()),
      'top1_accuracy': float(top1),
      'top5_accuracy': float(top5),
    }
    result.update(fields)
    rows.append(result)
  return rows


def roll_cache_batch(cache: Sequence[torch.Tensor], shift: int = 1):
  return [entry.roll(shifts=shift, dims=0) for entry in cache]


def zero_cache(cache: Sequence[torch.Tensor]):
  return [torch.zeros_like(entry) for entry in cache]


def checkpoint_fingerprint(path: pathlib.Path) -> dict[str, Any]:
  path = path.expanduser().resolve()
  stat = path.stat()
  return {
    'path': str(path),
    'size_bytes': stat.st_size,
    'mtime_ns': stat.st_mtime_ns,
  }


def git_revision() -> str:
  try:
    return subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
  except (OSError, subprocess.CalledProcessError):
    return 'unknown'


def _canonical_json(value: Any) -> str:
  return json.dumps(value, sort_keys=True, separators=(',', ':'))


def prepare_output(output_dir: pathlib.Path, metadata: dict[str, Any],
                   force: bool) -> pathlib.Path:
  output_dir = output_dir.expanduser().resolve()
  if force and output_dir.exists():
    if output_dir == pathlib.Path('/') or len(output_dir.parts) < 3:
      raise ValueError(f'Refusing to clear unsafe output path: {output_dir}')
    shutil.rmtree(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  manifest = output_dir / 'manifest.json'
  metadata = dict(metadata)
  metadata['schema_version'] = 1
  metadata['git_revision'] = git_revision()
  metadata['fingerprint'] = hashlib.sha256(
    _canonical_json(metadata).encode()).hexdigest()
  if manifest.exists():
    existing = json.loads(manifest.read_text())
    if existing != metadata:
      raise RuntimeError(
        f'Existing evaluation metadata differs at {manifest}. '
        'Use --force to start this output directory again.')
  else:
    atomic_write_json(manifest, metadata)
  (output_dir / 'parts').mkdir(exist_ok=True)
  return output_dir


def atomic_write_json(path: pathlib.Path, value: Any) -> None:
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  os.replace(temporary, path)


def atomic_write_csv(path: pathlib.Path, frame: pd.DataFrame) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  frame.to_csv(temporary, index=False)
  os.replace(temporary, path)


def part_path(output_dir: pathlib.Path, phase: str,
              batch_index: int) -> pathlib.Path:
  return output_dir / 'parts' / phase / f'batch_{batch_index:05d}.csv'


def write_part(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
  frame = pd.DataFrame(list(rows))
  if frame.empty:
    raise ValueError(f'Refusing to write empty result part: {path}')
  atomic_write_csv(path, frame)


def collect_parts(output_dir: pathlib.Path) -> pd.DataFrame:
  paths = sorted((output_dir / 'parts').glob('*/*.csv'))
  if not paths:
    raise RuntimeError(f'No completed result parts under {output_dir}')
  return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def aggregate_metrics(frame: pd.DataFrame,
                      group_columns: Sequence[str]) -> pd.DataFrame:
  rows = []
  for keys, group in frame.groupby(list(group_columns), sort=True):
    if not isinstance(keys, tuple):
      keys = (keys,)
    weights = group['masked_tokens'].to_numpy(dtype=np.float64)
    total = weights.sum()
    nll = float(np.dot(group['nll'], weights) / total)
    row = dict(zip(group_columns, keys))
    row.update({
      'examples': int(len(group)),
      'masked_tokens': int(total),
      'conditional_nll': nll,
      'conditional_ppl': float(math.exp(nll)),
      'top1_accuracy': float(np.dot(group['top1_accuracy'], weights) / total),
      'top5_accuracy': float(np.dot(group['top5_accuracy'], weights) / total),
    })
    rows.append(row)
  return pd.DataFrame(rows)


def paired_bootstrap_mean(
    values: np.ndarray,
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float, float]:
  values = np.asarray(values, dtype=np.float64)
  if values.ndim != 1 or values.size == 0:
    raise ValueError('Bootstrap values must be a nonempty vector')
  if values.size == 1:
    value = float(values[0])
    return value, value, value
  rng = np.random.default_rng(seed)
  means = np.empty(samples, dtype=np.float64)
  chunk = max(1, min(samples, 1_000_000 // values.size))
  for start in range(0, samples, chunk):
    stop = min(samples, start + chunk)
    indices = rng.integers(0, values.size, size=(stop - start, values.size))
    means[start:stop] = values[indices].mean(axis=1)
  return (
    float(values.mean()),
    float(np.quantile(means, 0.025)),
    float(np.quantile(means, 0.975)),
  )


def paired_condition_differences(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    comparisons: Sequence[tuple[str, str]],
    seed: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
  """Return condition-minus-reference paired NLL differences."""
  rows = []
  for group_keys, group in frame.groupby(list(group_columns), sort=True):
    if not isinstance(group_keys, tuple):
      group_keys = (group_keys,)
    group_values = dict(zip(group_columns, group_keys))
    for comparison_index, (condition, reference) in enumerate(comparisons):
      left = group[group.condition == condition][['example_id', 'nll']]
      right = group[group.condition == reference][['example_id', 'nll']]
      paired = left.merge(
        right, on='example_id', suffixes=('_condition', '_reference'),
        validate='one_to_one')
      if paired.empty:
        continue
      delta = (
        paired['nll_condition'].to_numpy()
        - paired['nll_reference'].to_numpy())
      mean, low, high = paired_bootstrap_mean(
        delta,
        seed=seed + comparison_index + sum(int(float(v) * 10_000)
                                           for v in group_keys),
        samples=bootstrap_samples)
      rows.append({
        **group_values,
        'condition': condition,
        'reference': reference,
        'n_examples': int(len(paired)),
        'mean_delta_nll': mean,
        'ci95_low': low,
        'ci95_high': high,
      })
  return pd.DataFrame(rows)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
  parser.add_argument('--baseline-checkpoint', type=pathlib.Path, required=True)
  parser.add_argument('--dcache-checkpoint', type=pathlib.Path, required=True)
  parser.add_argument('--data-dir', type=pathlib.Path, required=True)
  parser.add_argument('--output-dir', type=pathlib.Path, required=True)
  parser.add_argument('--examples', type=int, default=800)
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--num-workers', type=int, default=2)
  parser.add_argument('--seed', type=int, default=20260812)
  parser.add_argument('--bootstrap-samples', type=int, default=10_000)
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument(
    '--dcache-gate-enabled', action='store_true',
    help='Compose the gated DCache-v2 architecture for the DCache checkpoint.')
  parser.add_argument('--force', action='store_true')


def validate_common_arguments(args) -> None:
  if args.examples <= 0:
    raise ValueError('--examples must be positive')
  if args.batch_size <= 0:
    raise ValueError('--batch-size must be positive')
  if args.num_workers < 0:
    raise ValueError('--num-workers cannot be negative')
  if args.bootstrap_samples <= 0:
    raise ValueError('--bootstrap-samples must be positive')
  if str(args.device).startswith('cuda') and not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested but torch.cuda.is_available() is false')
  args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
  args.dcache_checkpoint = args.dcache_checkpoint.expanduser().resolve()
  args.data_dir = args.data_dir.expanduser().resolve()
  for checkpoint in (args.baseline_checkpoint, args.dcache_checkpoint):
    if not checkpoint.is_file():
      raise FileNotFoundError(checkpoint)
  expected_data = (
    args.data_dir
    / 'openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat')
  if not expected_data.is_dir():
    raise FileNotFoundError(
      f'Prepared validation dataset not found: {expected_data}')


def batch_ids(batch_index: int, batch_size: int,
              actual_batch_size: int) -> list[int]:
  start = batch_index * batch_size
  return list(range(start, start + actual_batch_size))


def print_device_banner(device: torch.device) -> None:
  if device.type == 'cuda':
    print(f'Using logical {device}; physical visibility={os.getenv("CUDA_VISIBLE_DEVICES")}; '
          f'GPU={torch.cuda.get_device_name(device)}', flush=True)
  else:
    print(f'Using device={device}', flush=True)


def save_fixed_plot(summary: pd.DataFrame, paired: pd.DataFrame,
                    path: pathlib.Path) -> None:
  fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
  for condition, group in summary.groupby('condition'):
    group = group.sort_values('mask_ratio')
    label = condition.replace('_', ' ')
    axes[0].plot(group.mask_ratio * 100, group.conditional_nll,
                 marker='o', label=label)
    axes[1].plot(group.mask_ratio * 100, group.top1_accuracy * 100,
                 marker='o', label=label)
  gain = paired[
    (paired.condition == 'dcache_no_cache')
    & (paired.reference == 'dcache_correct')].sort_values('mask_ratio')
  if not gain.empty:
    x = gain.mask_ratio.to_numpy() * 100
    y = gain.mean_delta_nll.to_numpy()
    axes[2].plot(x, y, marker='o', label='no-cache NLL - correct-cache NLL')
    axes[2].fill_between(x, gain.ci95_low, gain.ci95_high, alpha=0.2)
  axes[0].set_ylabel('Conditional masked-token NLL')
  axes[1].set_ylabel('Masked-token top-1 accuracy (%)')
  axes[2].set_ylabel('Paired NLL gain (positive = cache helps)')
  for axis in axes:
    axis.set_xlabel('Mask ratio (%)')
    axis.grid(alpha=0.25)
  axes[0].legend(fontsize=8)
  axes[2].axhline(0, color='black', linewidth=0.8)
  fig.suptitle('Teacher-forced fixed-corruption evaluation')
  fig.tight_layout()
  fig.savefig(path, dpi=180, bbox_inches='tight')
  plt.close(fig)


def save_transition_plot(summary: pd.DataFrame, paired: pd.DataFrame,
                         path: pathlib.Path) -> None:
  fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
  for condition, group in summary.groupby('condition'):
    group = group.sort_values(['t_mask_ratio', 's_mask_ratio'])
    label = condition.replace('_', ' ')
    axes[0].plot(group.t_mask_ratio * 100, group.conditional_nll,
                 marker='o', label=label)
    axes[1].plot(group.t_mask_ratio * 100, group.top1_accuracy * 100,
                 marker='o', label=label)
  gain = paired[
    (paired.condition == 'dcache_no_cache')
    & (paired.reference == 'dcache_correct')].sort_values('t_mask_ratio')
  if not gain.empty:
    x = gain.t_mask_ratio.to_numpy() * 100
    y = gain.mean_delta_nll.to_numpy()
    axes[2].plot(x, y, marker='o', label='no-cache NLL - correct-cache NLL')
    axes[2].fill_between(x, gain.ci95_low, gain.ci95_high, alpha=0.2)
  axes[0].set_ylabel('Conditional masked-token NLL')
  axes[1].set_ylabel('Masked-token top-1 accuracy (%)')
  axes[2].set_ylabel('Paired NLL gain (positive = cache helps)')
  for axis in axes:
    axis.set_xlabel('Final t mask ratio (%)')
    axis.grid(alpha=0.25)
  axes[0].legend(fontsize=7)
  axes[2].axhline(0, color='black', linewidth=0.8)
  fig.suptitle('Teacher-forced s→t transition evaluation')
  fig.tight_layout()
  fig.savefig(path, dpi=180, bbox_inches='tight')
  plt.close(fig)
