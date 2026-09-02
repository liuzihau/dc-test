#!/usr/bin/env python3
"""Compare step-matched vanilla, objective-control, and DCache evaluations."""

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


SERIES = ('Vanilla', 'Objective-matched no-DCache', 'DCache-v2 correct cache')
COLORS = dict(zip(SERIES, ('#1f77b4', '#2ca02c', '#ff7f0e')))


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


def load_condition(directory: Path, condition: str, series: str,
                   groups: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
  summary = pd.read_csv(directory / 'summary.csv')
  documents = pd.read_csv(directory / 'per_document_metrics.csv')
  summary = summary[summary.condition == condition].copy()
  documents = documents[documents.condition == condition].copy()
  if summary.empty or documents.empty:
    raise ValueError(f'No {condition!r} rows in {directory}')
  summary['series'] = series
  documents['series'] = series
  summary = summary.sort_values(groups)
  documents = documents.sort_values(groups + ['example_id'])
  return summary, documents


def compare_protocol(
    vanilla_dir: Path,
    objective_dir: Path,
    output_dir: Path,
    groups: list[str],
    x_column: str,
    title: str,
    seed: int,
    bootstrap_samples: int,
) -> None:
  vanilla_summary, vanilla_docs = load_condition(
    vanilla_dir, 'baseline', SERIES[0], groups)
  objective_summary, objective_docs = load_condition(
    objective_dir, 'baseline', SERIES[1], groups)
  dcache_summary, dcache_docs = load_condition(
    objective_dir, 'dcache_correct', SERIES[2], groups)

  # Both evaluations used the same DCache checkpoint and corruption seed. Catch
  # accidental protocol drift before combining their baseline-side results.
  old_dcache_summary, _ = load_condition(
    vanilla_dir, 'dcache_correct', SERIES[2], groups)
  metric_columns = ['conditional_nll', 'top1_accuracy', 'top5_accuracy']
  np.testing.assert_allclose(
    old_dcache_summary[metric_columns].to_numpy(),
    dcache_summary[metric_columns].to_numpy(), rtol=0, atol=1e-10)

  summaries = pd.concat(
    [vanilla_summary, objective_summary, dcache_summary], ignore_index=True)
  docs = pd.concat([vanilla_docs, objective_docs, dcache_docs],
                   ignore_index=True)
  output_dir.mkdir(parents=True, exist_ok=True)
  summaries.to_csv(output_dir / 'three_way_summary.csv', index=False)

  keys = groups + ['example_id']
  wide = None
  for series in SERIES:
    current = docs[docs.series == series][keys + ['nll']].rename(
      columns={'nll': series})
    wide = current if wide is None else wide.merge(
      current, on=keys, validate='one_to_one')

  comparisons = (
    (SERIES[1], SERIES[0]),
    (SERIES[2], SERIES[0]),
    (SERIES[2], SERIES[1]),
  )
  paired_rows = []
  for group_index, (group_keys, group) in enumerate(
      wide.groupby(groups, sort=True)):
    if not isinstance(group_keys, tuple):
      group_keys = (group_keys,)
    group_values = dict(zip(groups, group_keys))
    for comparison_index, (condition, reference) in enumerate(comparisons):
      # Positive gain means the condition has lower NLL than its reference.
      gains = group[reference].to_numpy() - group[condition].to_numpy()
      mean, low, high = bootstrap_mean(
        gains, seed + 10 * group_index + comparison_index,
        bootstrap_samples)
      paired_rows.append({
        **group_values,
        'condition': condition,
        'reference': reference,
        'n_examples': len(group),
        'mean_nll_gain': mean,
        'ci95_low': low,
        'ci95_high': high,
      })
  paired = pd.DataFrame(paired_rows)
  paired.to_csv(output_dir / 'paired_nll_gains.csv', index=False)

  figure, axes = plt.subplots(2, 2, figsize=(14, 9))
  for series in SERIES:
    frame = summaries[summaries.series == series].sort_values(x_column)
    x = frame[x_column].to_numpy() * 100
    axes[0, 0].plot(x, frame.conditional_nll, marker='o', linewidth=2.2,
                    color=COLORS[series], label=series)
    axes[0, 1].plot(x, frame.top1_accuracy * 100, marker='o', linewidth=2.2,
                    color=COLORS[series], label=series)

  gain_specs = (
    (SERIES[1], SERIES[0], COLORS[SERIES[1]],
     'Objective-matched over vanilla'),
    (SERIES[2], SERIES[0], COLORS[SERIES[2]], 'DCache-v2 over vanilla'),
  )
  for condition, reference, color, label in gain_specs:
    frame = paired[
      (paired.condition == condition) & (paired.reference == reference)
    ].sort_values(x_column)
    x = frame[x_column].to_numpy() * 100
    axes[1, 0].plot(x, frame.mean_nll_gain, marker='o', linewidth=2.2,
                    color=color, label=label)
    axes[1, 0].fill_between(x, frame.ci95_low, frame.ci95_high,
                            color=color, alpha=0.14)

  direct = paired[
    (paired.condition == SERIES[2]) & (paired.reference == SERIES[1])
  ].sort_values(x_column)
  x = direct[x_column].to_numpy() * 100
  axes[1, 1].plot(x, direct.mean_nll_gain, marker='o', linewidth=2.2,
                  color=COLORS[SERIES[2]], label='DCache-v2 over objective control')
  axes[1, 1].fill_between(x, direct.ci95_low, direct.ci95_high,
                          color=COLORS[SERIES[2]], alpha=0.14)

  axes[0, 0].set_ylabel('Conditional masked-token NLL')
  axes[0, 1].set_ylabel('Masked-token top-1 accuracy (%)')
  axes[1, 0].set_ylabel('Paired NLL gain over vanilla')
  axes[1, 1].set_ylabel('Paired NLL gain over objective control')
  axes[0, 0].legend(fontsize=9)
  axes[1, 0].legend(fontsize=9)
  axes[1, 1].legend(fontsize=9)
  for axis in axes.flat:
    axis.set_xlabel('Mask ratio (%)' if x_column == 'mask_ratio'
                    else 'Final t mask ratio (%)')
    axis.set_xticks(sorted(summaries[x_column].unique() * 100))
    axis.grid(alpha=0.25)
  axes[1, 0].axhline(0, color='black', linewidth=0.8)
  axes[1, 1].axhline(0, color='black', linewidth=0.8)
  figure.suptitle(title)
  figure.tight_layout()
  output = output_dir / 'three_way_comparison.png'
  figure.savefig(output, dpi=180, bbox_inches='tight')
  plt.close(figure)
  print(f'Wrote {output.resolve()}')


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--vanilla-eval-root', type=Path, required=True)
  parser.add_argument('--objective-eval-root', type=Path, required=True)
  parser.add_argument('--output-root', type=Path, required=True)
  parser.add_argument('--seed', type=int, default=20260823)
  parser.add_argument('--bootstrap-samples', type=int, default=10_000)
  args = parser.parse_args()

  compare_protocol(
    args.vanilla_eval_root / 'fixed-corruption',
    args.objective_eval_root / 'fixed-corruption',
    args.output_root / 'fixed-corruption', ['mask_ratio'], 'mask_ratio',
    'Step-5000 fixed-corruption evaluation', args.seed,
    args.bootstrap_samples)
  compare_protocol(
    args.vanilla_eval_root / 'transitions',
    args.objective_eval_root / 'transitions',
    args.output_root / 'transitions', ['s_mask_ratio', 't_mask_ratio'],
    't_mask_ratio', 'Step-5000 teacher-forced transitions', args.seed + 1000,
    args.bootstrap_samples)


if __name__ == '__main__':
  main()
