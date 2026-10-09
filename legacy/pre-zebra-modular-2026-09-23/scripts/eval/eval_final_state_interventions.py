#!/usr/bin/env python3
"""Teacher-forced causal interventions on DCache and final-state memory."""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import dcache_eval_common as common  # noqa: E402
from scripts.eval.eval_teacher_forced_transitions import (  # noqa: E402
  DEFAULT_TRANSITIONS,
  masks_for_batch,
  parse_transition,
)


CONDITIONS = (
  'correct_dcache_correct_final',
  'shuffled_dcache_correct_final',
  'correct_dcache_shuffled_final',
  'shuffled_dcache_shuffled_final',
  'absent_dcache_absent_final',
)


def parse_args():
  parser = argparse.ArgumentParser(
    description=(
      'At identical teacher-forced x_t, independently contaminate the '
      'previous DCache and previous final-layer representation.'))
  parser.add_argument('--checkpoint', type=pathlib.Path, required=True)
  parser.add_argument('--data-dir', type=pathlib.Path, required=True)
  parser.add_argument('--output-dir', type=pathlib.Path, required=True)
  parser.add_argument('--examples', type=int, default=800)
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--num-workers', type=int, default=2)
  parser.add_argument('--seed', type=int, default=20260812)
  parser.add_argument('--bootstrap-samples', type=int, default=10_000)
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--force', action='store_true')
  parser.add_argument(
    '--two-forward', action='store_true',
    help='Compose the strict two-forward checkpoint architecture.')
  parser.add_argument(
    '--transitions', type=parse_transition, nargs='+',
    default=DEFAULT_TRANSITIONS, metavar='S:T')
  return parser.parse_args()


def validate_args(args):
  if args.examples <= 0 or args.batch_size < 2:
    raise ValueError('Use positive examples and batch-size >= 2')
  if args.examples % args.batch_size:
    raise ValueError('--examples must be divisible by --batch-size')
  if args.num_workers < 0 or args.bootstrap_samples <= 0:
    raise ValueError('Invalid worker or bootstrap count')
  if str(args.device).startswith('cuda') and not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested but is unavailable')
  args.checkpoint = args.checkpoint.expanduser().resolve()
  args.data_dir = args.data_dir.expanduser().resolve()
  if not args.checkpoint.is_file():
    raise FileNotFoundError(args.checkpoint)
  expected = (
    args.data_dir
    / 'openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat')
  if not expected.is_dir():
    raise FileNotFoundError(expected)


def run(args, loader, tokenizer, output_dir, device):
  phase = 'final_state_interventions'
  missing = [
    index for index in range(len(loader))
    if not common.part_path(output_dir, phase, index).is_file()
  ]
  if not missing:
    print('All intervention batches already exist; skipping model load.',
          flush=True)
    return
  config = common.compose_eval_config(
    args.batch_size,
    args.data_dir,
    recurrent=True,
    num_workers=args.num_workers,
    gate_enabled=True,
    final_state_enabled=True,
    two_forward_enabled=args.two_forward)
  model = common.load_ema_model(args.checkpoint, config, tokenizer, device)
  print(f'Loaded final-state EMA; {len(missing)}/{len(loader)} batches remain.',
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
      full_output = model.forward(
        full_state,
        sigma=full_sigma,
        sample_mode=True,
        return_step_kv=True,
        return_dcachehooping=True)

      rows = []
      for s_ratio, t_ratio in args.transitions:
        s_mask = masks[s_ratio].to(device, non_blocking=True)
        t_mask = masks[t_ratio].to(device, non_blocking=True)
        if not torch.all(t_mask <= s_mask):
          raise RuntimeError('Internal error: x_t is not nested inside x_s')
        s_state = common.masked_state(tokens, s_mask, model.mask_index)
        t_state = common.masked_state(tokens, t_mask, model.mask_index)
        s_sigma = common.sigma_for_ratio(model, s_ratio, len(ids), device)
        t_sigma = common.sigma_for_ratio(model, t_ratio, len(ids), device)
        s_output = model.forward(
          s_state,
          sigma=s_sigma,
          sample_mode=True,
          previous_step_kv=full_output.step_kv,
          previous_final_hidden=full_output.final_hidden,
          return_step_kv=True,
          return_dcachehooping=True)

        shuffled_cache = common.roll_cache_batch(s_output.step_kv)
        shuffled_final = s_output.final_hidden.roll(shifts=1, dims=0)
        sources = (
          (CONDITIONS[0], s_output.step_kv, s_output.final_hidden),
          (CONDITIONS[1], shuffled_cache, s_output.final_hidden),
          (CONDITIONS[2], s_output.step_kv, shuffled_final),
          (CONDITIONS[3], shuffled_cache, shuffled_final),
          (CONDITIONS[4], None, None),
        )
        for condition, previous_cache, previous_final in sources:
          log_probs = model.forward(
            t_state,
            sigma=t_sigma,
            sample_mode=True,
            previous_step_kv=previous_cache,
            previous_final_hidden=previous_final)
          rows.extend(common.score_masked_tokens(
            log_probs,
            tokens,
            t_mask,
            ids,
            condition=condition,
            s_mask_ratio=s_ratio,
            t_mask_ratio=t_ratio))
        del s_output, shuffled_cache, shuffled_final
      common.write_part(path, rows)
      if (batch_index + 1) % 10 == 0 or batch_index + 1 == len(loader):
        print(f'Interventions: batch {batch_index + 1}/{len(loader)}',
              flush=True)
  common.unload_model(model)


def save_plot(summary: pd.DataFrame, paired: pd.DataFrame,
              path: pathlib.Path):
  fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))
  for condition, group in summary.groupby('condition'):
    group = group.sort_values('t_mask_ratio')
    label = condition.replace('_', ' ')
    x = group.t_mask_ratio * 100
    axes[0].plot(x, group.conditional_nll, marker='o', label=label)
    axes[1].plot(x, group.top1_accuracy * 100, marker='o', label=label)
  correct = CONDITIONS[0]
  for condition in CONDITIONS[1:]:
    group = paired[
      (paired.condition == condition) & (paired.reference == correct)
    ].sort_values('t_mask_ratio')
    axes[2].plot(
      group.t_mask_ratio * 100,
      group.mean_delta_nll,
      marker='o',
      label=condition.replace('_', ' '))
  axes[0].set_ylabel('Conditional masked-token NLL')
  axes[1].set_ylabel('Masked-token top-1 accuracy (%)')
  axes[2].set_ylabel('NLL minus correct/correct (lower is better)')
  for axis in axes:
    axis.set_xlabel('Final t mask ratio (%)')
    axis.grid(alpha=0.25)
  axes[0].legend(fontsize=6.5)
  axes[2].legend(fontsize=6.5)
  axes[2].axhline(0, color='black', linewidth=0.8)
  fig.suptitle('Independent DCache × final-state causal interventions')
  fig.tight_layout()
  fig.savefig(path, dpi=180, bbox_inches='tight')
  plt.close(fig)


def write_reports(args, output_dir):
  raw = common.collect_parts(output_dir)
  expected = args.examples * len(args.transitions) * len(CONDITIONS)
  if len(raw) != expected:
    raise RuntimeError(f'Expected {expected} rows, found {len(raw)}')
  raw = raw.sort_values(
    ['t_mask_ratio', 's_mask_ratio', 'condition', 'example_id'])
  common.atomic_write_csv(output_dir / 'per_document_metrics.csv', raw)
  summary = common.aggregate_metrics(
    raw, ['s_mask_ratio', 't_mask_ratio', 'condition'])
  common.atomic_write_csv(output_dir / 'summary.csv', summary)
  correct = CONDITIONS[0]
  paired = common.paired_condition_differences(
    raw,
    group_columns=['s_mask_ratio', 't_mask_ratio'],
    comparisons=[(condition, correct) for condition in CONDITIONS[1:]],
    seed=args.seed,
    bootstrap_samples=args.bootstrap_samples)
  common.atomic_write_csv(output_dir / 'paired_nll_differences.csv', paired)
  save_plot(summary, paired, output_dir / 'memory_interventions.png')
  print('\nFinal-state intervention summary:', flush=True)
  print(summary.to_string(index=False), flush=True)
  print(
    '\nPaired NLL is condition minus correct-DCache/correct-final-state; '
    'positive means the intervention hurt.', flush=True)


def main():
  args = parse_args()
  validate_args(args)
  args.transitions = sorted(set(tuple(pair) for pair in args.transitions))
  torch.set_float32_matmul_precision('high')
  device = torch.device(args.device)
  common.print_device_banner(device)
  metadata = {
    'evaluation': 'final_state_memory_interventions',
    'protocol': (
      'Full-mask forward -> teacher-forced x_s with both correct memories -> '
      'identical x_t scored under independent batch-roll interventions.'),
    'conditions': list(CONDITIONS),
    'examples': args.examples,
    'batch_size': args.batch_size,
    'seed': args.seed,
    'transitions': [list(pair) for pair in args.transitions],
    'bootstrap_samples': args.bootstrap_samples,
    'checkpoint': common.checkpoint_fingerprint(args.checkpoint),
    'data_dir': str(args.data_dir),
    'weights': 'EMA',
    'shuffle': 'cyclic batch roll by one document, independently per source',
  }
  output_dir = common.prepare_output(args.output_dir, metadata, args.force)
  config = common.compose_eval_config(
    args.batch_size,
    args.data_dir,
    recurrent=True,
    num_workers=args.num_workers,
    gate_enabled=True,
    final_state_enabled=True,
    two_forward_enabled=args.two_forward)
  tokenizer = common.load_tokenizer(config)
  loader = common.load_validation_data(
    config, tokenizer, args.examples, args.batch_size, args.num_workers)
  run(args, loader, tokenizer, output_dir, device)
  write_reports(args, output_dir)
  print(f'Wrote final-state interventions to {output_dir}', flush=True)


if __name__ == '__main__':
  main()
