#!/usr/bin/env python3
"""Regenerate registry-driven training plots, status table, and key evidence."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / 'experiments' / 'canonical_runs.json'
DEFAULT_OUTPUT = REPO_ROOT / 'results' / 'generated'
MPL_CACHE = REPO_ROOT / '.cache' / 'matplotlib'
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MPL_CACHE))

RUN_COLORS = {
  'bd3_vanilla': '#1f77b4',
  'dcache_v2': '#ff7f0e',
  'objective_aligned': '#2ca02c',
  'dcache_final_state': '#9467bd',
  'dcache_final_state_adjacent': '#d62728',
}


def load_manifest(path):
  with path.open() as handle:
    manifest = json.load(handle)
  runs = {run['id']: run for run in manifest['runs']}
  if len(runs) != len(manifest['runs']):
    raise ValueError('Manifest contains duplicate run IDs')
  required = {
    'bd3_vanilla', 'objective_aligned', 'dcache_v2',
    'dcache_final_state'}
  missing = required - set(runs)
  if missing:
    raise ValueError(f'Manifest is missing canonical runs: {sorted(missing)}')
  return manifest, runs


def resolve(path_string):
  path = Path(path_string)
  return path if path.is_absolute() else REPO_ROOT / path


def select_runs(manifest, available_only=False, path_overrides=()):
  """Allow a partial cloud transfer without relaxing the default strict mode."""
  runs = {run['id']: dict(run) for run in manifest['runs']}
  for run_id, path in path_overrides:
    if run_id not in runs:
      raise ValueError(f'Unknown run ID for --run-path: {run_id}')
    runs[run_id]['path'] = path
  if available_only:
    present = {}
    for run_id, run in runs.items():
      if not any(resolve(run['path']).rglob('metrics.csv')):
        print(f'Skipping {run_id}: no local metrics.csv')
        continue
      supplement = run.get('validation_supplement')
      if supplement and not resolve(supplement['path']).is_file():
        print(f'{run_id}: optional historical validation supplement not transferred')
        run.pop('validation_supplement')
      present[run_id] = run
    runs = present
    if not runs:
      raise FileNotFoundError('No registered runs have local metrics.csv yet')
  return {**manifest, 'runs': list(runs.values())}, runs


def run_plot(runs, output, min_step=None, max_step=5000, x_max=5200):
  import matplotlib

  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
  for index, run in enumerate(runs.values()):
    frame = append_validation_supplement(load_metrics(run['path']), run)
    color = run_color(run, index)
    train_metric = run['train_metric']
    train = metric_series(frame, train_metric)
    # Smooth the full history before clipping, matching the existing template.
    smoothed = train[train_metric].rolling(60, min_periods=1).mean()
    keep = train.step <= max_step
    if min_step is not None:
      keep &= train.step >= min_step
    train, smoothed = train[keep], smoothed[keep]
    if not train.empty:
      axes[0].plot(train.step, train[train_metric], color=color,
                   alpha=0.16, linewidth=0.8)
      axes[0].plot(train.step, smoothed, color=color, linewidth=2.0,
                   label=f'{run["label"]} {train_metric.removeprefix("trainer/")}')
    validation = metric_series(frame, run['validation_metric'])
    validation = validation[validation.step <= max_step]
    if min_step is not None:
      validation = validation[validation.step >= min_step]
    if not validation.empty:
      axes[1].plot(
        validation.step, validation[run['validation_metric']], marker='o',
        markersize=4, color=color, linewidth=1.8,
        label=validation_label(run))
    # Never substitute training loss for missing validation, or extrapolate.
    else:
      print(f'{run["label"]}: no validation points in this plot window')
  axes[0].set_ylabel('Training loss')
  axes[0].set_title(f'OpenWebText pretraining health ({len(runs)} trials)')
  axes[1].set_ylabel('Validation loss / NLL')
  axes[1].set_xlabel('Optimizer step (logged index)')
  axes[1].set_xlim(left=min_step, right=x_max)
  for axis in axes:
    axis.grid(alpha=0.25)
    if axis.get_legend_handles_labels()[0]:
      axis.legend(fontsize=9)
  if not axes[1].get_legend_handles_labels()[0]:
    axes[1].text(0.5, 0.5, 'No validation logged in this step range',
                 ha='center', va='center', transform=axes[1].transAxes)
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.tight_layout()
  figure.savefig(output, dpi=160)
  plt.close(figure)


def run_color(run, index):
  return run.get('color', RUN_COLORS.get(run['id'], f'C{index % 10}'))


def validation_label(run):
  metric = run['validation_metric'].removeprefix('val/')
  return f'{run["label"]} {"NLL" if metric == "nll" else metric}'


def metric_series(frame, metric):
  import pandas as pd

  if metric not in frame.columns:
    return pd.DataFrame(columns=['step', metric])
  series = frame[['step', metric, '_file_index', '_row_index']].copy()
  series['step'] = pd.to_numeric(series['step'], errors='coerce')
  series[metric] = pd.to_numeric(series[metric], errors='coerce')
  series = series.dropna(subset=['step', metric])
  series = series.sort_values(['step', '_file_index', '_row_index'])
  return series.drop_duplicates('step', keep='last')[['step', metric]]


def load_metrics(run_path):
  import pandas as pd

  files = sorted(resolve(run_path).rglob('metrics.csv'),
                 key=lambda path: path.stat().st_mtime)
  if not files:
    raise FileNotFoundError(f'No metrics.csv under {resolve(run_path)}')
  frames = []
  for file_index, path in enumerate(files):
    frame = pd.read_csv(path)
    frame['_file_index'] = file_index
    frame['_row_index'] = range(len(frame))
    frames.append(frame)
  return pd.concat(frames, ignore_index=True, sort=False)


def append_validation_supplement(frame, run):
  import pandas as pd

  supplement = run.get('validation_supplement')
  if supplement is None:
    return frame
  points = pd.read_csv(resolve(supplement['path']))
  step_column = supplement['step_column']
  metric_column = supplement['metric_column']
  missing = {step_column, metric_column} - set(points.columns)
  if missing:
    raise ValueError(
      f'{run["id"]} validation supplement is missing columns: '
      f'{sorted(missing)}')
  normalized = pd.DataFrame({
    'step': points[step_column],
    run['validation_metric']: points[metric_column],
  })
  normalized['_file_index'] = int(frame['_file_index'].max()) + 1
  normalized['_row_index'] = range(len(normalized))
  return pd.concat([frame, normalized], ignore_index=True, sort=False)


def plot_validation_zoom(manifest, output, min_step=1900, x_max=5100):
  import matplotlib

  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  figure, axis = plt.subplots(figsize=(10, 6.2))
  comparison_limit = int(manifest['comparison_step_limit'])
  for index, run in enumerate(manifest['runs']):
    frame = append_validation_supplement(load_metrics(run['path']), run)
    series = metric_series(frame, run['validation_metric'])
    series = series[
      (series.step >= min_step) & (series.step <= comparison_limit)]
    if series.empty:
      continue
    axis.plot(
      series.step, series[run['validation_metric']], marker='o',
      markersize=5, linewidth=2.1, color=run_color(run, index),
      label=validation_label(run))
  axis.set_xlim(min_step, x_max)
  axis.set_xlabel('Optimizer step (logged index)')
  axis.set_ylabel('Validation loss / NLL')
  axis.set_title(f'OpenWebText validation health from step {min_step}')
  axis.grid(alpha=0.25)
  if axis.get_legend_handles_labels()[0]:
    axis.legend(fontsize=9)
  else:
    axis.text(0.5, 0.5, 'No validation logged in this step range',
              ha='center', va='center', transform=axis.transAxes)
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.tight_layout()
  figure.savefig(output, dpi=160)
  plt.close(figure)


def write_status_table(manifest, runs, output):
  comparison_limit = int(manifest['comparison_step_limit'])
  rows = []
  for run in manifest['runs']:
    frame = load_metrics(run['path'])
    frame = append_validation_supplement(frame, run)
    train = metric_series(frame, run['train_metric'])
    validation = metric_series(frame, run['validation_metric'])
    train = train[train.step <= comparison_limit]
    validation = validation[validation.step <= comparison_limit]
    row = {
      'id': run['id'],
      'label': run['label'],
      'role': run['role'],
      'status': run['status'],
      'train_metric': run['train_metric'],
      'train_step': '',
      'train_value': '',
      'validation_metric': run['validation_metric'],
      'validation_step': '',
      'validation_value': '',
    }
    if not train.empty:
      latest = train.iloc[-1]
      row['train_step'] = int(latest.step)
      row['train_value'] = float(latest[run['train_metric']])
    if not validation.empty:
      latest = validation.iloc[-1]
      row['validation_step'] = int(latest.step)
      row['validation_value'] = float(
        latest[run['validation_metric']])
    rows.append(row)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


CURATED_FILES = {
  'outputs/eval-three-way-5k-teacher-forced/fixed-corruption/three_way_comparison.png':
    'figures/mechanism/three_way_fixed_corruption.png',
  'outputs/eval-three-way-5k-teacher-forced/transitions/three_way_comparison.png':
    'figures/mechanism/three_way_transitions.png',
  'outputs/eval-v2-checkpoint-trend/identity-summary/identity_trend.png':
    'figures/mechanism/dcache_v2_identity_trend.png',
  'outputs/eval-v2-checkpoint-trend/focused-under-30/focused_teacher_forced_transitions.png':
    'figures/mechanism/dcache_v2_late_denoising_quality.png',
  'outputs/eval-v2-same-state-recurrence/step-6000-r30/same_state_recurrence.png':
    'figures/mechanism/dcache_v2_same_state_recurrence.png',
  'outputs/eval-three-way-5k-teacher-forced/fixed-corruption/three_way_summary.csv':
    'tables/mechanism/three_way_fixed_corruption.csv',
  'outputs/eval-three-way-5k-teacher-forced/transitions/three_way_summary.csv':
    'tables/mechanism/three_way_transitions.csv',
  'outputs/eval-v2-checkpoint-trend/identity-summary/identity_by_checkpoint.csv':
    'tables/mechanism/dcache_v2_identity_by_checkpoint.csv',
  'outputs/eval-v2-same-state-recurrence/step-6000-r30/summary.csv':
    'tables/mechanism/dcache_v2_same_state_recurrence.csv',
}


def copy_curated_evidence(output_root):
  for source_string, destination_string in CURATED_FILES.items():
    source = REPO_ROOT / source_string
    if not source.exists():
      raise FileNotFoundError(f'Canonical evidence is missing: {source}')
    destination = output_root / destination_string
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument('--available-only', action='store_true',
                      help='Plot only locally available runs on a partial checkout')
  parser.add_argument('--training-only', action='store_true',
                      help='Do not copy historical mechanism-evaluation artifacts')
  parser.add_argument('--run-path', nargs=2, action='append', default=[],
                      metavar=('RUN_ID', 'PATH'),
                      help='Override a registered run directory without editing the registry')
  args = parser.parse_args()

  manifest_path = args.manifest.resolve()
  output_root = args.output_dir.resolve()
  manifest, runs = load_manifest(manifest_path)
  manifest, runs = select_runs(manifest, args.available_only, args.run_path)
  training_dir = output_root / 'figures' / 'training'
  training_dir.mkdir(parents=True, exist_ok=True)
  run_plot(
    runs, training_dir / 'canonical_5k_smooth60.png',
    min_step=400, max_step=manifest['comparison_step_limit'], x_max=5200)
  run_plot(
    runs, training_dir / 'canonical_early_smooth60.png',
    min_step=None, max_step=1000, x_max=1000)
  plot_validation_zoom(
    manifest, training_dir / 'canonical_validation_from_1900.png')
  plot_validation_zoom(
    manifest, training_dir / 'canonical_validation_nll.png',
    min_step=400, x_max=5200)
  # Keep existing bookmarks working; these legacy names now include every run.
  for canonical, legacy in (
      ('canonical_5k_smooth60.png', 'four_way_5k_smooth60.png'),
      ('canonical_early_smooth60.png', 'four_way_early_smooth60.png'),
      ('canonical_validation_from_1900.png', 'four_way_validation_from_1900.png')):
    shutil.copy2(training_dir / canonical, training_dir / legacy)
  write_status_table(
    manifest, runs,
    output_root / 'tables' / 'training' / 'canonical_status.csv')
  if not args.training_only:
    copy_curated_evidence(output_root)
  print(f'Refreshed canonical results under {output_root}')


if __name__ == '__main__':
  main()
