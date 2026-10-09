#!/usr/bin/env python3
"""Compare correct-vs-shuffled DCache identity across checkpoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
  'MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'dcache-matplotlib'))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_result(value: str) -> tuple[int, Path]:
  try:
    step_text, path_text = value.split(':', maxsplit=1)
    step = int(step_text)
  except (ValueError, TypeError) as error:
    raise argparse.ArgumentTypeError(
      f'Expected GLOBAL_STEP:SUMMARY_CSV, got {value!r}') from error
  return step, Path(path_text)


def load_checkpoint_result(global_step: int, path: Path) -> pd.DataFrame:
  frame = pd.read_csv(path)
  required = {
    's_mask_ratio', 't_mask_ratio', 'condition', 'masked_tokens',
    'conditional_nll', 'top1_accuracy',
  }
  missing = required.difference(frame.columns)
  if missing:
    raise ValueError(f'{path} is missing columns: {sorted(missing)}')

  rows = []
  groups = frame.groupby(['s_mask_ratio', 't_mask_ratio'], sort=True)
  for (s_ratio, t_ratio), group in groups:
    indexed = group.set_index('condition')
    for condition in ('dcache_correct', 'dcache_shuffled_cache'):
      if condition not in indexed.index:
        raise ValueError(f'{path} has no {condition} row at {s_ratio}->{t_ratio}')
    correct = indexed.loc['dcache_correct']
    shuffled = indexed.loc['dcache_shuffled_cache']
    rows.append({
      'global_step': global_step,
      'metric_index': global_step - 1,
      's_mask_ratio': float(s_ratio),
      't_mask_ratio': float(t_ratio),
      'masked_tokens': int(correct.masked_tokens),
      'correct_top1_accuracy': float(correct.top1_accuracy),
      'shuffled_top1_accuracy': float(shuffled.top1_accuracy),
      'correct_minus_shuffled_top1_pp': 100.0 * (
        float(correct.top1_accuracy) - float(shuffled.top1_accuracy)),
      'relative_top1_gain_percent': 100.0 * (
        float(correct.top1_accuracy) / float(shuffled.top1_accuracy) - 1.0),
      'correct_nll': float(correct.conditional_nll),
      'shuffled_nll': float(shuffled.conditional_nll),
      'shuffled_minus_correct_nll': (
        float(shuffled.conditional_nll) - float(correct.conditional_nll)),
    })
  return pd.DataFrame(rows)


def checkpoint_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
  rows = []
  for global_step, group in frame.groupby('global_step', sort=True):
    weights = group.masked_tokens.to_numpy(dtype=np.float64)
    correct = group.correct_top1_accuracy.to_numpy(dtype=np.float64)
    shuffled = group.shuffled_top1_accuracy.to_numpy(dtype=np.float64)
    rows.append({
      'global_step': int(global_step),
      'metric_index': int(global_step) - 1,
      'macro_correct_top1_accuracy': float(
        group.correct_top1_accuracy.mean()),
      'token_weighted_correct_top1_accuracy': float(
        np.average(correct, weights=weights)),
      'macro_correct_nll': float(group.correct_nll.mean()),
      'token_weighted_correct_nll': float(np.average(
        group.correct_nll, weights=weights)),
      'macro_accuracy_gap_pp': float(
        group.correct_minus_shuffled_top1_pp.mean()),
      'token_weighted_accuracy_gap_pp': float(
        100.0 * np.average(correct - shuffled, weights=weights)),
      'macro_nll_gap': float(group.shuffled_minus_correct_nll.mean()),
      'token_weighted_nll_gap': float(np.average(
        group.shuffled_minus_correct_nll, weights=weights)),
    })
  return pd.DataFrame(rows)


def save_plot(frame: pd.DataFrame, aggregate: pd.DataFrame,
              output: Path) -> None:
  figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
  colors = plt.cm.viridis(np.linspace(0.05, 0.9, frame.t_mask_ratio.nunique()))
  for color, (t_ratio, group) in zip(
      colors, frame.groupby('t_mask_ratio', sort=True)):
    group = group.sort_values('metric_index')
    transition = f'{group.s_mask_ratio.iloc[0]:.2f}->{t_ratio:.2f}'
    axes[0].plot(
      group.metric_index, group.correct_minus_shuffled_top1_pp,
      marker='o', linewidth=1.8, color=color, label=transition)
    axes[1].plot(
      group.metric_index, group.shuffled_minus_correct_nll,
      marker='o', linewidth=1.8, color=color, label=transition)

  axes[0].plot(
    aggregate.metric_index, aggregate.macro_accuracy_gap_pp,
    marker='s', linewidth=3.0, linestyle='--', color='black',
    label='macro mean')
  axes[1].plot(
    aggregate.metric_index, aggregate.macro_nll_gap,
    marker='s', linewidth=3.0, linestyle='--', color='black',
    label='macro mean')

  axes[0].axhline(0, color='gray', linewidth=1)
  axes[1].axhline(0, color='gray', linewidth=1)
  axes[0].set_title('Correct - shuffled top-1 accuracy')
  axes[1].set_title('Shuffled - correct conditional NLL')
  axes[0].set_ylabel('Accuracy gap (percentage points)')
  axes[1].set_ylabel('NLL gap (positive = correct cache helps)')
  for axis in axes:
    axis.set_xlabel('Zero-based optimizer-step index')
    axis.set_xticks(sorted(frame.metric_index.unique()))
    axis.grid(alpha=0.25)
  axes[1].legend(title='s -> t mask ratio', loc='best')
  figure.suptitle('DCache identity signal across checkpoints')
  figure.tight_layout()
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output, dpi=160)
  plt.close(figure)


def save_quality_plot(frame: pd.DataFrame, aggregate: pd.DataFrame,
                      output: Path) -> None:
  figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
  colors = plt.cm.viridis(np.linspace(0.05, 0.9, frame.t_mask_ratio.nunique()))
  for color, (t_ratio, group) in zip(
      colors, frame.groupby('t_mask_ratio', sort=True)):
    group = group.sort_values('metric_index')
    transition = f'{group.s_mask_ratio.iloc[0]:.2f}->{t_ratio:.2f}'
    axes[0].plot(
      group.metric_index, 100.0 * group.correct_top1_accuracy,
      marker='o', linewidth=1.8, color=color, label=transition)
    axes[1].plot(
      group.metric_index, group.correct_nll,
      marker='o', linewidth=1.8, color=color, label=transition)

  axes[0].plot(
    aggregate.metric_index,
    100.0 * aggregate.macro_correct_top1_accuracy,
    marker='s', linewidth=3.0, linestyle='--', color='black',
    label='macro mean')
  axes[1].plot(
    aggregate.metric_index, aggregate.macro_correct_nll,
    marker='s', linewidth=3.0, linestyle='--', color='black',
    label='macro mean')

  axes[0].set_title('Correct-cache top-1 accuracy')
  axes[1].set_title('Correct-cache conditional NLL')
  axes[0].set_ylabel('Top-1 accuracy (%)')
  axes[1].set_ylabel('Conditional NLL')
  for axis in axes:
    axis.set_xlabel('Zero-based optimizer-step index')
    axis.set_xticks(sorted(frame.metric_index.unique()))
    axis.grid(alpha=0.25)
  axes[1].legend(title='s -> t mask ratio', loc='best')
  figure.suptitle('DCache-v2 correct-cache quality across checkpoints')
  figure.tight_layout()
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output, dpi=160)
  plt.close(figure)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--result', action='append', type=parse_result, required=True,
    metavar='GLOBAL_STEP:SUMMARY_CSV',
    help='Repeat once for every transition-evaluation summary')
  parser.add_argument('--output-dir', type=Path, required=True)
  args = parser.parse_args()

  frames = [load_checkpoint_result(step, path) for step, path in args.result]
  combined = pd.concat(frames, ignore_index=True).sort_values(
    ['global_step', 't_mask_ratio'])
  if combined.global_step.nunique() != len(args.result):
    raise ValueError('Every --result must use a unique global step')
  aggregate = checkpoint_aggregate(combined)

  args.output_dir.mkdir(parents=True, exist_ok=True)
  combined.to_csv(args.output_dir / 'identity_by_transition.csv', index=False)
  aggregate.to_csv(args.output_dir / 'identity_by_checkpoint.csv', index=False)
  save_plot(combined, aggregate, args.output_dir / 'identity_trend.png')
  save_quality_plot(
    combined, aggregate, args.output_dir / 'correct_cache_quality_trend.png')

  print('\nCorrect-vs-shuffled accuracy gaps by transition (percentage points):')
  print(combined.pivot(
    index='metric_index', columns='t_mask_ratio',
    values='correct_minus_shuffled_top1_pp').to_string())
  print('\nCheckpoint aggregates:')
  print(aggregate.to_string(index=False))
  print(f'Wrote {args.output_dir.resolve()}')


if __name__ == '__main__':
  main()
