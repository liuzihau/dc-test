#!/usr/bin/env python3
"""Teacher-forced s→t transition evaluation with DCache ablations."""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import dcache_eval_common as common  # noqa: E402


DEFAULT_TRANSITIONS = (
  (0.25, 0.05),
  (0.30, 0.10),
  (0.40, 0.20),
  (0.50, 0.30),
  (0.70, 0.50),
  (0.90, 0.70),
)


def parse_transition(value: str) -> tuple[float, float]:
  try:
    s_text, t_text = value.split(':', maxsplit=1)
    s_ratio, t_ratio = float(s_text), float(t_text)
  except (ValueError, TypeError) as error:
    raise argparse.ArgumentTypeError(
      f'Expected S:T, for example 0.50:0.30; got {value!r}') from error
  if not 0 < t_ratio < s_ratio <= 1:
    raise argparse.ArgumentTypeError(
      f'Transition must satisfy 0 < T < S <= 1; got {value!r}')
  return s_ratio, t_ratio


def parse_args():
  parser = argparse.ArgumentParser(
    description=(
      'Compare predictions at the identical teacher-forced x_t using the '
      'baseline and DCache with correct, absent, shuffled, or zero M_s.'))
  common.add_common_arguments(parser)
  parser.add_argument(
    '--transitions', type=parse_transition, nargs='+',
    default=DEFAULT_TRANSITIONS,
    metavar='S:T')
  return parser.parse_args()


def masks_for_batch(attention, ids, transitions, seed):
  ratios = sorted({ratio for transition in transitions for ratio in transition})
  return common.deterministic_nested_masks(
    attention, ids, ratios, seed, ignore_first=True)


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
      masks = masks_for_batch(attention, ids, args.transitions, args.seed)
      tokens = tokens_cpu.to(device, non_blocking=True)
      rows = []
      for s_ratio, t_ratio in args.transitions:
        t_mask = masks[t_ratio].to(device, non_blocking=True)
        t_state = common.masked_state(tokens, t_mask, model.mask_index)
        t_sigma = common.sigma_for_ratio(model, t_ratio, len(ids), device)
        log_probs = model.forward(t_state, sigma=t_sigma, sample_mode=True)
        rows.extend(common.score_masked_tokens(
          log_probs, tokens, t_mask, ids,
          condition='baseline', s_mask_ratio=s_ratio,
          t_mask_ratio=t_ratio))
        del log_probs, t_state, t_sigma, t_mask
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
    num_workers=args.num_workers, gate_enabled=args.dcache_gate_enabled)
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
      masks = masks_for_batch(attention, ids, args.transitions, args.seed)
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
      for s_ratio, t_ratio in args.transitions:
        s_mask = masks[s_ratio].to(device, non_blocking=True)
        t_mask = masks[t_ratio].to(device, non_blocking=True)
        if not torch.all(t_mask <= s_mask):
          raise RuntimeError('Internal error: t mask is not nested inside s mask')
        s_state = common.masked_state(tokens, s_mask, model.mask_index)
        t_state = common.masked_state(tokens, t_mask, model.mask_index)
        s_sigma = common.sigma_for_ratio(model, s_ratio, len(ids), device)
        t_sigma = common.sigma_for_ratio(model, t_ratio, len(ids), device)
        s_log_probs, s_cache = model.forward(
          s_state,
          sigma=s_sigma,
          sample_mode=True,
          previous_step_kv=full_cache,
          return_step_kv=True)
        del s_log_probs

        conditions = [
          ('dcache_correct', s_cache),
          ('dcache_no_cache', None),
          ('dcache_shuffled_cache', common.roll_cache_batch(s_cache)),
          ('dcache_zero_cache', common.zero_cache(s_cache)),
        ]
        for condition, previous_cache in conditions:
          log_probs = model.forward(
            t_state,
            sigma=t_sigma,
            sample_mode=True,
            previous_step_kv=previous_cache)
          rows.extend(common.score_masked_tokens(
            log_probs, tokens, t_mask, ids,
            condition=condition, s_mask_ratio=s_ratio,
            t_mask_ratio=t_ratio))
          del log_probs
        del conditions, s_cache, s_state, t_state, s_sigma, t_sigma
        del s_mask, t_mask
      del full_cache
      common.write_part(path, rows)
      if (batch_index + 1) % 10 == 0 or batch_index + 1 == len(loader):
        print(f'DCache: completed batch {batch_index + 1}/{len(loader)}',
              flush=True)
  common.unload_model(model)


def write_reports(args, output_dir):
  raw = common.collect_parts(output_dir)
  expected = args.examples * len(args.transitions) * 5
  if len(raw) != expected:
    raise RuntimeError(f'Expected {expected} raw rows, found {len(raw)}')
  raw = raw.sort_values(
    ['t_mask_ratio', 's_mask_ratio', 'condition', 'example_id'])
  common.atomic_write_csv(output_dir / 'per_document_metrics.csv', raw)
  summary = common.aggregate_metrics(
    raw, ['s_mask_ratio', 't_mask_ratio', 'condition'])
  common.atomic_write_csv(output_dir / 'summary.csv', summary)
  paired = common.paired_condition_differences(
    raw,
    group_columns=['s_mask_ratio', 't_mask_ratio'],
    comparisons=[
      ('dcache_correct', 'baseline'),
      ('dcache_no_cache', 'baseline'),
      ('dcache_no_cache', 'dcache_correct'),
      ('dcache_shuffled_cache', 'dcache_correct'),
      ('dcache_zero_cache', 'dcache_correct'),
    ],
    seed=args.seed,
    bootstrap_samples=args.bootstrap_samples)
  common.atomic_write_csv(output_dir / 'paired_nll_differences.csv', paired)
  common.save_transition_plot(
    summary, paired, output_dir / 'teacher_forced_transitions.png')
  print('\nTeacher-forced transition summary:', flush=True)
  print(summary.to_string(index=False), flush=True)
  print(
    '\nPaired deltas use condition NLL - reference NLL. For '
    'dcache_no_cache - dcache_correct, positive means the correct M_s helped.',
    flush=True)


def main():
  args = parse_args()
  common.validate_common_arguments(args)
  args.transitions = sorted(set(tuple(pair) for pair in args.transitions))
  if args.batch_size < 2:
    raise ValueError('--batch-size must be at least 2 for shuffled-cache control')
  if args.examples % args.batch_size != 0:
    raise ValueError(
      '--examples must be divisible by --batch-size so every shuffled-cache '
      'batch contains another document')
  torch.set_float32_matmul_precision('high')
  device = torch.device(args.device)
  common.print_device_banner(device)
  metadata = {
    'evaluation': 'teacher_forced_s_to_t_transitions',
    'protocol': (
      'Full mask -> M_full; ground-truth nested x_s -> M_s; score identical '
      'ground-truth x_t masked targets using correct/absent/shuffled/zero M_s.'),
    'examples': args.examples,
    'batch_size': args.batch_size,
    'seed': args.seed,
    'transitions': [list(pair) for pair in args.transitions],
    'bootstrap_samples': args.bootstrap_samples,
    'baseline_checkpoint': common.checkpoint_fingerprint(
      args.baseline_checkpoint),
    'dcache_checkpoint': common.checkpoint_fingerprint(args.dcache_checkpoint),
    'data_dir': str(args.data_dir),
    'weights': 'EMA',
    'dcache_gate_enabled': args.dcache_gate_enabled,
    'zero_cache_caveat': (
      'Zero K/V entries remain present in the denoising-attention softmax; '
      'no-cache is the primary removal ablation.'),
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
  print(f'Wrote teacher-forced transition evaluation to {output_dir}',
        flush=True)


if __name__ == '__main__':
  main()
