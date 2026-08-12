#!/usr/bin/env python3
"""Controlled fixed-corruption evaluation for the 5k baseline and DCache."""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import dcache_eval_common as common  # noqa: E402


DEFAULT_RATIOS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00)


def parse_args():
  parser = argparse.ArgumentParser(
    description=(
      'Evaluate identical teacher-forced corruptions with the vanilla model, '
      'DCache with its full-mask cache, and DCache without a previous cache.'))
  common.add_common_arguments(parser)
  parser.add_argument(
    '--ratios', type=float, nargs='+', default=DEFAULT_RATIOS,
    help='Mask ratios; exact rounded token counts are used per document.')
  return parser.parse_args()


def run_baseline(args, loader, tokenizer, output_dir, device):
  phase = 'baseline'
  missing = [
    index for index in range(len(loader))
    if not common.part_path(output_dir, phase, index).is_file()
  ]
  if not missing:
    print('Baseline phase is already complete; skipping model load.', flush=True)
    return
  config = common.compose_eval_config(
    args.batch_size, args.data_dir, recurrent=False,
    num_workers=args.num_workers)
  model = common.load_ema_model(
    args.baseline_checkpoint, config, tokenizer, device)
  print(f'Loaded baseline EMA; {len(missing)}/{len(loader)} batches remain.',
        flush=True)
  with torch.inference_mode():
    for batch_index, batch in enumerate(loader):
      path = common.part_path(output_dir, phase, batch_index)
      if path.is_file():
        continue
      tokens_cpu = batch['input_ids'].long()
      attention = batch['attention_mask']
      ids = common.batch_ids(batch_index, args.batch_size, tokens_cpu.shape[0])
      masks = common.deterministic_nested_masks(
        attention, ids, args.ratios, args.seed, ignore_first=True)
      tokens = tokens_cpu.to(device, non_blocking=True)
      rows = []
      for ratio in sorted(masks):
        mask = masks[ratio].to(device, non_blocking=True)
        state = common.masked_state(tokens, mask, model.mask_index)
        sigma = common.sigma_for_ratio(model, ratio, len(ids), device)
        log_probs = model.forward(state, sigma=sigma, sample_mode=True)
        rows.extend(common.score_masked_tokens(
          log_probs, tokens, mask, ids,
          condition='baseline', mask_ratio=ratio))
        del log_probs, state, sigma, mask
      common.write_part(path, rows)
      if (batch_index + 1) % 10 == 0 or batch_index + 1 == len(loader):
        print(f'Baseline: completed batch {batch_index + 1}/{len(loader)}',
              flush=True)
  common.unload_model(model)


def run_dcache(args, loader, tokenizer, output_dir, device):
  phase = 'dcache'
  missing = [
    index for index in range(len(loader))
    if not common.part_path(output_dir, phase, index).is_file()
  ]
  if not missing:
    print('DCache phase is already complete; skipping model load.', flush=True)
    return
  config = common.compose_eval_config(
    args.batch_size, args.data_dir, recurrent=True,
    num_workers=args.num_workers)
  model = common.load_ema_model(
    args.dcache_checkpoint, config, tokenizer, device)
  print(f'Loaded DCache EMA; {len(missing)}/{len(loader)} batches remain.',
        flush=True)
  with torch.inference_mode():
    for batch_index, batch in enumerate(loader):
      path = common.part_path(output_dir, phase, batch_index)
      if path.is_file():
        continue
      tokens_cpu = batch['input_ids'].long()
      attention = batch['attention_mask']
      ids = common.batch_ids(batch_index, args.batch_size, tokens_cpu.shape[0])
      masks = common.deterministic_nested_masks(
        attention, ids, args.ratios, args.seed, ignore_first=True)
      tokens = tokens_cpu.to(device, non_blocking=True)
      eligible = common.eligible_positions(attention, ignore_first=True).to(
        device, non_blocking=True)
      full_state = common.masked_state(tokens, eligible, model.mask_index)
      full_sigma = common.sigma_for_ratio(model, 1.0, len(ids), device)
      full_log_probs, full_cache = model.forward(
        full_state,
        sigma=full_sigma,
        sample_mode=True,
        previous_step_kv=None,
        return_step_kv=True)
      del full_log_probs, full_state, full_sigma

      rows = []
      for ratio in sorted(masks):
        mask = masks[ratio].to(device, non_blocking=True)
        state = common.masked_state(tokens, mask, model.mask_index)
        sigma = common.sigma_for_ratio(model, ratio, len(ids), device)
        correct_log_probs = model.forward(
          state,
          sigma=sigma,
          sample_mode=True,
          previous_step_kv=full_cache)
        rows.extend(common.score_masked_tokens(
          correct_log_probs, tokens, mask, ids,
          condition='dcache_correct', mask_ratio=ratio))
        del correct_log_probs

        no_cache_log_probs = model.forward(
          state,
          sigma=sigma,
          sample_mode=True,
          previous_step_kv=None)
        rows.extend(common.score_masked_tokens(
          no_cache_log_probs, tokens, mask, ids,
          condition='dcache_no_cache', mask_ratio=ratio))
        del no_cache_log_probs, state, sigma, mask
      del full_cache
      common.write_part(path, rows)
      if (batch_index + 1) % 10 == 0 or batch_index + 1 == len(loader):
        print(f'DCache: completed batch {batch_index + 1}/{len(loader)}',
              flush=True)
  common.unload_model(model)


def write_reports(args, output_dir):
  raw = common.collect_parts(output_dir)
  expected = args.examples * len(args.ratios) * 3
  if len(raw) != expected:
    raise RuntimeError(f'Expected {expected} raw rows, found {len(raw)}')
  raw = raw.sort_values(['mask_ratio', 'condition', 'example_id'])
  common.atomic_write_csv(output_dir / 'per_document_metrics.csv', raw)
  summary = common.aggregate_metrics(raw, ['mask_ratio', 'condition'])
  common.atomic_write_csv(output_dir / 'summary.csv', summary)
  paired = common.paired_condition_differences(
    raw,
    group_columns=['mask_ratio'],
    comparisons=[
      ('dcache_correct', 'baseline'),
      ('dcache_no_cache', 'baseline'),
      ('dcache_no_cache', 'dcache_correct'),
    ],
    seed=args.seed,
    bootstrap_samples=args.bootstrap_samples)
  common.atomic_write_csv(output_dir / 'paired_nll_differences.csv', paired)
  common.save_fixed_plot(summary, paired, output_dir / 'fixed_corruption.png')
  print('\nFixed-corruption summary:', flush=True)
  print(summary.to_string(index=False), flush=True)
  print(
    '\nPaired deltas use condition NLL - reference NLL. For '
    'dcache_no_cache - dcache_correct, positive means the cache helped.',
    flush=True)


def main():
  args = parse_args()
  common.validate_common_arguments(args)
  args.ratios = sorted(set(float(value) for value in args.ratios))
  if args.ratios[0] <= 0 or args.ratios[-1] > 1:
    raise ValueError('--ratios must be in (0, 1]')
  torch.set_float32_matmul_precision('high')
  device = torch.device(args.device)
  common.print_device_banner(device)
  metadata = {
    'evaluation': 'teacher_forced_fixed_corruption',
    'protocol': (
      'Exact-count nested corruptions; first sequence position stays visible; '
      'all other visible tokens are ground truth; score masked positions only.'),
    'examples': args.examples,
    'batch_size': args.batch_size,
    'seed': args.seed,
    'ratios': args.ratios,
    'bootstrap_samples': args.bootstrap_samples,
    'baseline_checkpoint': common.checkpoint_fingerprint(
      args.baseline_checkpoint),
    'dcache_checkpoint': common.checkpoint_fingerprint(args.dcache_checkpoint),
    'data_dir': str(args.data_dir),
    'weights': 'EMA',
  }
  output_dir = common.prepare_output(args.output_dir, metadata, args.force)
  base_config = common.compose_eval_config(
    args.batch_size, args.data_dir, recurrent=False,
    num_workers=args.num_workers)
  tokenizer = common.load_tokenizer(base_config)
  loader = common.load_validation_data(
    base_config, tokenizer, args.examples, args.batch_size, args.num_workers)
  run_baseline(args, loader, tokenizer, output_dir, device)
  run_dcache(args, loader, tokenizer, output_dir, device)
  write_reports(args, output_dir)
  print(f'Wrote fixed-corruption evaluation to {output_dir}', flush=True)


if __name__ == '__main__':
  main()
