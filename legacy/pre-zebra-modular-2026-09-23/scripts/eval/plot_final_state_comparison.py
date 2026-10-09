#!/usr/bin/env python3
"""Compare the 5k two- and five-forward dual-memory candidates."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_SUMMARY = (
  REPO_ROOT / 'outputs/eval-three-way-5k-teacher-forced/transitions'
  / 'three_way_summary.csv')
FIVE_FORWARD_EVAL = (
  REPO_ROOT / 'outputs/eval-final-state-five-forward-memory-interventions')
TWO_FORWARD_EVAL = (
  REPO_ROOT / 'outputs/eval-final-state-two-forward-memory-interventions')
OUTPUT_DIR = REPO_ROOT / 'outputs/eval-final-state-comparison-5k'

QUALITY_SERIES = (
  'Vanilla',
  'Objective-matched no-DCache',
  'DCache-v2 correct cache',
  'DCache + final state (5-forward)',
  'DCache + final state (2-forward)',
)
QUALITY_COLORS = dict(zip(
  QUALITY_SERIES,
  ('#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728')))
CORRECT = 'correct_dcache_correct_final'
CONDITION_LABELS = {
  CORRECT: 'correct DCache + correct final',
  'shuffled_dcache_correct_final': 'shuffled DCache',
  'correct_dcache_shuffled_final': 'shuffled final state',
  'shuffled_dcache_shuffled_final': 'both shuffled',
  'absent_dcache_absent_final': 'both absent',
}
CONDITION_COLORS = {
  CORRECT: '#2ca02c',
  'shuffled_dcache_correct_final': '#ff7f0e',
  'correct_dcache_shuffled_final': '#9467bd',
  'shuffled_dcache_shuffled_final': '#d62728',
  'absent_dcache_absent_final': '#7f7f7f',
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--previous-summary', type=Path,
                      default=PREVIOUS_SUMMARY)
  parser.add_argument('--five-forward-eval', type=Path,
                      default=FIVE_FORWARD_EVAL)
  parser.add_argument('--two-forward-eval', type=Path,
                      default=TWO_FORWARD_EVAL)
  parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
  return parser.parse_args()


def require_columns(frame, columns, source):
  missing = set(columns) - set(frame.columns)
  if missing:
    raise ValueError(f'{source} is missing columns: {sorted(missing)}')


def transition_keys(frame):
  return set(zip(frame.s_mask_ratio.astype(float),
                 frame.t_mask_ratio.astype(float)))


def load_previous(path):
  frame = pd.read_csv(path)
  require_columns(
    frame,
    ['s_mask_ratio', 't_mask_ratio', 'series', 'conditional_nll',
     'top1_accuracy', 'top5_accuracy'],
    path)
  frame = frame[frame.series.isin(QUALITY_SERIES[:3])].copy()
  found = set(frame.series)
  expected = set(QUALITY_SERIES[:3])
  if found != expected:
    raise ValueError(
      f'{path} has series {sorted(found)}, expected {sorted(expected)}')
  return frame


def load_intervention(directory, series):
  summary_path = directory / 'summary.csv'
  paired_path = directory / 'paired_nll_differences.csv'
  summary = pd.read_csv(summary_path)
  paired = pd.read_csv(paired_path)
  require_columns(
    summary,
    ['s_mask_ratio', 't_mask_ratio', 'condition', 'conditional_nll',
     'top1_accuracy', 'top5_accuracy'],
    summary_path)
  require_columns(
    paired,
    ['s_mask_ratio', 't_mask_ratio', 'condition', 'reference',
     'mean_delta_nll', 'ci95_low', 'ci95_high'],
    paired_path)
  correct = summary[summary.condition == CORRECT].copy()
  if correct.empty:
    raise ValueError(f'{summary_path} has no {CORRECT!r} rows')
  correct['series'] = series
  summary['training_style'] = series
  paired['training_style'] = series
  return correct, summary, paired


def save_quality_plot(frame, path):
  fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
  panels = (
    ('conditional_nll', 'Conditional masked-token NLL'),
    ('top1_accuracy', 'Masked-token top-1 accuracy (%)'),
    ('top5_accuracy', 'Masked-token top-5 accuracy (%)'),
  )
  for series in QUALITY_SERIES:
    group = frame[frame.series == series].sort_values('t_mask_ratio')
    x = group.t_mask_ratio * 100
    for axis, (metric, _) in zip(axes, panels):
      y = group[metric] if metric == 'conditional_nll' else group[metric] * 100
      axis.plot(x, y, marker='o', linewidth=2, color=QUALITY_COLORS[series],
                label=series)
  for axis, (_, ylabel) in zip(axes, panels):
    axis.set_xlabel('Final t mask ratio (%)')
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
  axes[0].legend(fontsize=7.5)
  fig.suptitle('Step-5000 teacher-forced transition quality')
  fig.tight_layout()
  fig.savefig(path, dpi=180, bbox_inches='tight')
  plt.close(fig)


def save_intervention_plot(summaries, paired_frames, path):
  fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex='col')
  rows = (
    ('DCache + final state (5-forward)', summaries[0], paired_frames[0]),
    ('DCache + final state (2-forward)', summaries[1], paired_frames[1]),
  )
  for row_index, (title, summary, paired) in enumerate(rows):
    for condition, label in CONDITION_LABELS.items():
      group = summary[summary.condition == condition].sort_values(
        't_mask_ratio')
      x = group.t_mask_ratio * 100
      axes[row_index, 0].plot(
        x, group.conditional_nll, marker='o', linewidth=1.8,
        color=CONDITION_COLORS[condition], label=label)
      axes[row_index, 1].plot(
        x, group.top1_accuracy * 100, marker='o', linewidth=1.8,
        color=CONDITION_COLORS[condition], label=label)
      if condition == CORRECT:
        continue
      delta = paired[
        (paired.condition == condition) & (paired.reference == CORRECT)
      ].sort_values('t_mask_ratio')
      delta_x = (delta.t_mask_ratio * 100).to_numpy()
      axes[row_index, 2].plot(
        delta_x, delta.mean_delta_nll.to_numpy(), marker='o',
        linewidth=1.8, color=CONDITION_COLORS[condition], label=label)
      axes[row_index, 2].fill_between(
        delta_x, delta.ci95_low.to_numpy(), delta.ci95_high.to_numpy(),
        color=CONDITION_COLORS[condition], alpha=0.12)
    axes[row_index, 0].set_ylabel(f'{title}\nConditional NLL')
    axes[row_index, 1].set_ylabel('Top-1 accuracy (%)')
    axes[row_index, 2].set_ylabel('NLL minus correct/correct')
    axes[row_index, 2].axhline(0, color='black', linewidth=0.8)
    for axis in axes[row_index]:
      axis.grid(alpha=0.25)
  for axis in axes[-1]:
    axis.set_xlabel('Final t mask ratio (%)')
  axes[0, 0].legend(fontsize=7)
  axes[0, 2].legend(fontsize=7)
  fig.suptitle('Independent DCache and final-state interventions')
  fig.tight_layout()
  fig.savefig(path, dpi=180, bbox_inches='tight')
  plt.close(fig)


def metric_series(frame, metric):
  if metric not in frame:
    return pd.DataFrame(columns=['step', metric])
  selected = frame[['step', metric, '_file_index', '_row_index']].copy()
  selected['step'] = pd.to_numeric(selected.step, errors='coerce')
  selected[metric] = pd.to_numeric(selected[metric], errors='coerce')
  return (selected.dropna(subset=['step', metric])
          .sort_values(['step', '_file_index', '_row_index'])
          .drop_duplicates('step', keep='last')[['step', metric]])


def load_metrics(directory):
  paths = sorted(directory.rglob('metrics.csv'), key=lambda p: p.stat().st_mtime)
  if not paths:
    raise FileNotFoundError(f'No metrics.csv under {directory}')
  frames = []
  for index, path in enumerate(paths):
    frame = pd.read_csv(path)
    frame['_file_index'] = index
    frame['_row_index'] = range(len(frame))
    frames.append(frame)
  return pd.concat(frames, ignore_index=True, sort=False)


def save_training_plot(path, table_path):
  runs = (
    ('Vanilla', REPO_ROOT / 'outputs/owt-mdlm-pretrain-5k-2x3090',
     'trainer/loss', 'val/nll', None),
    ('Objective-matched no-DCache',
     REPO_ROOT / 'outputs/owt-mdlm-objective-matched-5k',
     'trainer/loss_t2', 'val/loss_t2', None),
    ('DCache-v2 correct cache',
     REPO_ROOT / 'outputs/owt-dcache-v2-pretrain-5k-2x3090',
     'trainer/loss_t2', 'val/loss_t2', None),
    ('DCache + final state (5-forward)',
     REPO_ROOT / 'outputs/owt-dcache-final-state-pretrain-5k-2x3090',
     'trainer/loss_t2', 'val/loss_t2',
     REPO_ROOT / 'outputs/eval-final-state-step5000-validation/summary.csv'),
    ('DCache + final state (2-forward)',
     REPO_ROOT / 'outputs/owt-dcache-two-forward-pretrain-5k-2x3090',
     'trainer/loss_t', 'val/loss_t', None),
  )
  fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
  latest_rows = []
  for label, directory, train_metric, val_metric, supplement in runs:
    frame = load_metrics(directory)
    train = metric_series(frame, train_metric)
    validation = metric_series(frame, val_metric)
    if supplement is not None and supplement.is_file():
      extra = pd.read_csv(supplement)
      require_columns(extra, ['checkpoint_step', 'val_loss_t2'], supplement)
      extra = extra[['checkpoint_step', 'val_loss_t2']].rename(columns={
        'checkpoint_step': 'step', 'val_loss_t2': val_metric})
      validation = (pd.concat([validation, extra], ignore_index=True)
                    .sort_values('step').drop_duplicates('step', keep='last'))
    train = train[(train.step >= 400) & (train.step <= 5000)]
    validation = validation[
      (validation.step >= 400) & (validation.step <= 5000)]
    if train.empty or validation.empty:
      raise ValueError(f'Missing matched metrics for {label}')
    smoothed = train[train_metric].rolling(
      min(60, len(train)), min_periods=1).mean()
    color = QUALITY_COLORS[label]
    axes[0].plot(train.step, train[train_metric], color=color, alpha=0.12,
                 linewidth=0.7)
    axes[0].plot(train.step, smoothed, color=color, linewidth=2, label=label)
    axes[1].plot(validation.step, validation[val_metric], color=color,
                 marker='o', markersize=4, linewidth=1.8, label=label)
    latest_rows.append({
      'series': label,
      'train_metric': train_metric,
      'train_step': int(train.iloc[-1].step),
      'train_value': float(train.iloc[-1][train_metric]),
      'validation_metric': val_metric,
      'validation_step': int(validation.iloc[-1].step),
      'validation_value': float(validation.iloc[-1][val_metric]),
    })
  axes[0].set_ylabel('Training loss')
  axes[0].set_title('Training health (60 logged-sample smoothing)')
  axes[1].set_ylabel('Validation loss / NLL')
  axes[1].set_xlabel('Optimizer step')
  axes[1].set_xlim(400, 5200)
  for axis in axes:
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7.5)
  fig.tight_layout()
  fig.savefig(path, dpi=180, bbox_inches='tight')
  plt.close(fig)
  with table_path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=latest_rows[0].keys())
    writer.writeheader()
    writer.writerows(latest_rows)


def main():
  args = parse_args()
  previous = load_previous(args.previous_summary.resolve())
  five_correct, five_summary, five_paired = load_intervention(
    args.five_forward_eval.resolve(), QUALITY_SERIES[3])
  two_correct, two_summary, two_paired = load_intervention(
    args.two_forward_eval.resolve(), QUALITY_SERIES[4])
  expected_keys = transition_keys(previous)
  for name, frame in (
      ('five-forward', five_correct), ('two-forward', two_correct)):
    if transition_keys(frame) != expected_keys:
      raise ValueError(
        f'{name} transitions do not match the previous controlled evaluation')

  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  quality = pd.concat(
    [previous, five_correct, two_correct], ignore_index=True, sort=False)
  quality = quality.sort_values(['series', 't_mask_ratio'])
  quality.to_csv(output_dir / 'five_way_transition_summary.csv', index=False)
  interventions = pd.concat(
    [five_summary, two_summary], ignore_index=True, sort=False)
  interventions.to_csv(
    output_dir / 'dual_memory_intervention_summary.csv', index=False)
  paired = pd.concat([five_paired, two_paired], ignore_index=True, sort=False)
  paired.to_csv(
    output_dir / 'dual_memory_paired_nll_differences.csv', index=False)

  save_quality_plot(quality, output_dir / 'five_way_transition_quality.png')
  save_intervention_plot(
    (five_summary, two_summary), (five_paired, two_paired),
    output_dir / 'dual_memory_intervention_comparison.png')
  save_training_plot(
    output_dir / 'five_way_training_health.png',
    output_dir / 'five_way_latest_training_metrics.csv')
  print(f'Wrote final-state comparison plots to {output_dir}')


if __name__ == '__main__':
  main()
