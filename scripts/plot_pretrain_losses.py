#!/usr/bin/env python3
"""Plot matched vanilla and DCache training/validation losses from CSV logs."""

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
  'MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'dcache-matplotlib'))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def find_metric_files(path_string):
  path = Path(path_string).expanduser()
  if path.is_file():
    return [path]
  if not path.exists():
    raise FileNotFoundError(f'Log path does not exist: {path}')
  files = sorted(path.rglob('metrics.csv'), key=lambda item: item.stat().st_mtime)
  if not files:
    raise FileNotFoundError(f'No metrics.csv found below: {path}')
  return files


def load_metrics(path_string):
  frames = []
  for file_index, path in enumerate(find_metric_files(path_string)):
    frame = pd.read_csv(path)
    frame['_file_index'] = file_index
    frame['_row_index'] = range(len(frame))
    frames.append(frame)
  return pd.concat(frames, ignore_index=True, sort=False)


def metric_series(frame, metric):
  if metric not in frame.columns:
    return pd.DataFrame(columns=['step', metric])
  selected = frame[['step', metric, '_file_index', '_row_index']].copy()
  selected['step'] = pd.to_numeric(selected['step'], errors='coerce')
  selected[metric] = pd.to_numeric(selected[metric], errors='coerce')
  selected = selected.dropna(subset=['step', metric])
  selected = selected.sort_values(['step', '_file_index', '_row_index'])
  selected = selected.drop_duplicates('step', keep='last')
  return selected[['step', metric]]


def plot_training(axis, frame, metric, label, color, smooth):
  series = metric_series(frame, metric)
  if series.empty:
    return None
  axis.plot(series.step, series[metric], color=color, alpha=0.16, linewidth=0.8)
  window = max(1, min(smooth, len(series)))
  smoothed = series[metric].rolling(window, min_periods=1).mean()
  axis.plot(series.step, smoothed, color=color, linewidth=2.0, label=label)
  return series.iloc[-1]


def plot_validation(axis, frame, metric, label, color):
  series = metric_series(frame, metric)
  if series.empty:
    return None
  axis.plot(series.step, series[metric], marker='o', markersize=4,
            color=color, linewidth=1.8, label=label)
  return series.iloc[-1]


def print_latest(label, row, metric):
  if row is None:
    print(f'{label}: not logged yet')
  else:
    print(f'{label}: step={int(row.step)} {metric}={row[metric]:.6f}')


def first_available_metric(frame, candidates):
  for metric in candidates:
    if not metric_series(frame, metric).empty:
      return metric
  return candidates[0]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--vanilla', required=True,
                      help='Vanilla run directory or metrics.csv')
  parser.add_argument('--dcache', required=True,
                      help='DCache run directory or metrics.csv')
  parser.add_argument('--output', default='outputs/loss_comparison.png')
  parser.add_argument('--smooth', type=int, default=20,
                      help='Training-point rolling-average window')
  args = parser.parse_args()

  vanilla = load_metrics(args.vanilla)
  dcache = load_metrics(args.dcache)
  dcache_train_metric = first_available_metric(
    dcache, ('trainer/loss_t2', 'trainer/loss_s'))
  figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

  vanilla_train = plot_training(
    axes[0], vanilla, 'trainer/loss', 'Vanilla train loss', '#1f77b4',
    args.smooth)
  dcache_train = plot_training(
    axes[0], dcache, dcache_train_metric,
    f'DCache train {dcache_train_metric.removeprefix("trainer/")}', '#ff7f0e',
    args.smooth)
  total = metric_series(dcache, 'trainer/loss')
  if not total.empty:
    axes[0].plot(total.step,
                 total['trainer/loss'].rolling(
                   max(1, min(args.smooth, len(total))), min_periods=1).mean(),
                 color='#d62728', linestyle=':', linewidth=1.2,
                 label='DCache total objective')
  axes[0].set_ylabel('Training loss')
  axes[0].set_title('Matched OpenWebText pretraining')
  axes[0].grid(alpha=0.25)
  axes[0].legend()

  vanilla_val = plot_validation(
    axes[1], vanilla, 'val/nll', 'Vanilla validation NLL', '#1f77b4')
  dcache_val = plot_validation(
    axes[1], dcache, 'val/nll', 'DCache validation NLL at t2', '#ff7f0e')
  axes[1].set_xlabel('Optimizer step')
  axes[1].set_ylabel('Validation NLL')
  axes[1].grid(alpha=0.25)
  if vanilla_val is not None or dcache_val is not None:
    axes[1].legend()
  else:
    axes[1].text(0.5, 0.5, 'Validation begins at step 500',
                 ha='center', va='center', transform=axes[1].transAxes)

  output = Path(args.output).expanduser()
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.tight_layout()
  figure.savefig(output, dpi=160)
  plt.close(figure)

  print_latest('Vanilla training', vanilla_train, 'trainer/loss')
  print_latest('DCache training', dcache_train, dcache_train_metric)
  print_latest('Vanilla validation', vanilla_val, 'val/nll')
  print_latest('DCache validation', dcache_val, 'val/nll')
  print(f'Wrote {output.resolve()}')


if __name__ == '__main__':
  main()
