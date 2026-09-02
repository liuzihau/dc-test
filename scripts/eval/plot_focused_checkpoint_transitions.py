#!/usr/bin/env python3
"""Focused transition plot for several DCache checkpoints and one baseline."""

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


def parse_checkpoint(value: str) -> tuple[int, Path]:
  try:
    step_text, directory_text = value.split(':', maxsplit=1)
    return int(step_text), Path(directory_text)
  except (ValueError, TypeError) as error:
    raise argparse.ArgumentTypeError(
      f'Expected GLOBAL_STEP:TRANSITION_DIR, got {value!r}') from error


def bootstrap_mean(values: np.ndarray, seed: int,
                   samples: int) -> tuple[float, float, float]:
  values = np.asarray(values, dtype=np.float64)
  rng = np.random.default_rng(seed)
  means = np.empty(samples, dtype=np.float64)
  chunk = max(1, min(samples, 1_000_000 // len(values)))
  for start in range(0, samples, chunk):
    stop = min(samples, start + chunk)
    indices = rng.integers(0, len(values), size=(stop - start, len(values)))
    means[start:stop] = values[indices].mean(axis=1)
  return (
    float(values.mean()),
    float(np.quantile(means, 0.025)),
    float(np.quantile(means, 0.975)),
  )


def load_condition(directory: Path, condition: str,
                   max_t_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
  summary = pd.read_csv(directory / 'summary.csv')
  documents = pd.read_csv(directory / 'per_document_metrics.csv')
  summary = summary[
    (summary.condition == condition)
    & (summary.t_mask_ratio < max_t_ratio)].copy()
  documents = documents[
    (documents.condition == condition)
    & (documents.t_mask_ratio < max_t_ratio)].copy()
  if summary.empty or documents.empty:
    raise ValueError(
      f'No {condition} results below t={max_t_ratio} in {directory}')
  return summary, documents


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--baseline-dir', type=Path, required=True,
    help='Transition directory providing the single baseline curve')
  parser.add_argument('--baseline-step', type=int, required=True)
  parser.add_argument(
    '--dcache', action='append', type=parse_checkpoint, required=True,
    metavar='GLOBAL_STEP:TRANSITION_DIR')
  parser.add_argument('--max-t-ratio', type=float, default=0.30)
  parser.add_argument('--accuracy-ymin', type=float, default=49.0)
  parser.add_argument('--accuracy-ymax', type=float, default=70.0)
  parser.add_argument('--bootstrap-samples', type=int, default=10_000)
  parser.add_argument('--seed', type=int, default=20260819)
  parser.add_argument('--output-dir', type=Path, required=True)
  args = parser.parse_args()

  baseline_summary, baseline_documents = load_condition(
    args.baseline_dir, 'baseline', args.max_t_ratio)
  baseline_summary['series'] = f'Vanilla step {args.baseline_step - 1}'
  baseline_summary['global_step'] = args.baseline_step

  summaries = [baseline_summary]
  paired_rows = []
  for global_step, directory in sorted(args.dcache):
    summary, documents = load_condition(
      directory, 'dcache_correct', args.max_t_ratio)
    summary['series'] = f'DCache correct step {global_step - 1}'
    summary['global_step'] = global_step
    summaries.append(summary)

    keys = ['example_id', 's_mask_ratio', 't_mask_ratio']
    paired = baseline_documents[keys + ['nll']].merge(
      documents[keys + ['nll']], on=keys,
      suffixes=('_baseline', '_dcache'), validate='one_to_one')
    paired['gain'] = paired.nll_baseline - paired.nll_dcache
    for comparison_index, ((s_ratio, t_ratio), group) in enumerate(
        paired.groupby(['s_mask_ratio', 't_mask_ratio'], sort=True)):
      mean, low, high = bootstrap_mean(
        group.gain.to_numpy(),
        seed=args.seed + global_step + comparison_index,
        samples=args.bootstrap_samples)
      paired_rows.append({
        'global_step': global_step,
        'metric_index': global_step - 1,
        's_mask_ratio': s_ratio,
        't_mask_ratio': t_ratio,
        'n_examples': len(group),
        'baseline_minus_dcache_nll': mean,
        'ci95_low': low,
        'ci95_high': high,
      })

  combined = pd.concat(summaries, ignore_index=True)
  paired_frame = pd.DataFrame(paired_rows)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  combined.to_csv(args.output_dir / 'focused_summary.csv', index=False)
  paired_frame.to_csv(args.output_dir / 'paired_baseline_gain.csv', index=False)

  figure, axes = plt.subplots(1, 3, figsize=(17, 5.0))
  series_order = [f'Vanilla step {args.baseline_step - 1}'] + [
    f'DCache correct step {step - 1}' for step, _ in sorted(args.dcache)]
  colors = ['#1f77b4', '#ff9f1c', '#f05d23', '#2ca02c']
  color_map = dict(zip(series_order, colors))
  for series in series_order:
    group = combined[combined.series == series].sort_values('t_mask_ratio')
    axes[0].plot(
      group.t_mask_ratio * 100, group.conditional_nll,
      marker='o', linewidth=2.2, color=color_map[series], label=series)
    axes[1].plot(
      group.t_mask_ratio * 100, group.top1_accuracy * 100,
      marker='o', linewidth=2.2, color=color_map[series], label=series)

  for global_step, group in paired_frame.groupby('global_step', sort=True):
    group = group.sort_values('t_mask_ratio')
    label = f'DCache step {int(global_step) - 1}'
    x = group.t_mask_ratio.to_numpy() * 100
    y = group.baseline_minus_dcache_nll.to_numpy()
    axes[2].plot(
      x, y, marker='o', linewidth=2.2,
      color=color_map[f'DCache correct step {int(global_step) - 1}'],
      label=label)
    axes[2].fill_between(
      x, group.ci95_low.to_numpy(), group.ci95_high.to_numpy(),
      color=color_map[f'DCache correct step {int(global_step) - 1}'],
      alpha=0.12)

  axes[0].set_ylabel('Conditional masked-token NLL')
  axes[1].set_ylabel('Masked-token top-1 accuracy (%)')
  axes[2].set_ylabel('Paired NLL gain (baseline - DCache)')
  axes[1].set_ylim(args.accuracy_ymin, args.accuracy_ymax)
  axes[2].axhline(0, color='black', linewidth=0.8)
  for axis in axes:
    axis.set_xlabel('Final t mask ratio (%)')
    axis.set_xticks(sorted(combined.t_mask_ratio.unique() * 100))
    axis.grid(alpha=0.25)
  axes[0].legend(fontsize=8)
  axes[2].legend(fontsize=8)
  figure.suptitle(
    f'Teacher-forced transitions with final mask ratio < '
    f'{args.max_t_ratio * 100:g}%')
  figure.tight_layout()
  output = args.output_dir / 'focused_teacher_forced_transitions.png'
  figure.savefig(output, dpi=180, bbox_inches='tight')
  plt.close(figure)

  print(combined[[
    'series', 's_mask_ratio', 't_mask_ratio', 'conditional_nll',
    'top1_accuracy']].to_string(index=False))
  print('\nPaired baseline-minus-DCache NLL gains:')
  print(paired_frame.to_string(index=False))
  print(f'Wrote {output.resolve()}')


if __name__ == '__main__':
  main()
