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


def append_metric_points(frame, path_string, step_column, metric_column,
                         target_metric):
  """Append explicitly step-labeled standalone evaluation points."""
  points = pd.read_csv(Path(path_string).expanduser())
  missing = {step_column, metric_column} - set(points.columns)
  if missing:
    raise ValueError(
      f'Supplemental validation file is missing columns: {sorted(missing)}')
  normalized = pd.DataFrame({
    'step': points[step_column],
    target_metric: points[metric_column],
  })
  normalized['_file_index'] = int(frame['_file_index'].max()) + 1
  normalized['_row_index'] = range(len(normalized))
  return pd.concat([frame, normalized], ignore_index=True, sort=False)


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


def plot_training(axis, frame, metric, label, color, smooth, min_step=None,
                  max_step=None):
  series = metric_series(frame, metric)
  if series.empty:
    return None
  window = max(1, min(smooth, len(series)))
  smoothed = series[metric].rolling(window, min_periods=1).mean()
  keep = pd.Series(True, index=series.index)
  if min_step is not None:
    keep &= series.step >= min_step
  if max_step is not None:
    keep &= series.step <= max_step
  smoothed = smoothed[keep]
  series = series[keep]
  if series.empty:
    return None
  axis.plot(series.step, series[metric], color=color, alpha=0.16, linewidth=0.8)
  axis.plot(series.step, smoothed, color=color, linewidth=2.0, label=label)
  return series.iloc[-1]


def plot_validation(axis, frame, metric, label, color, min_step=None,
                    max_step=None):
  series = metric_series(frame, metric)
  if min_step is not None:
    series = series[series.step >= min_step]
  if max_step is not None:
    series = series[series.step <= max_step]
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
  parser.add_argument(
    '--ablation', default=None,
    help='Optional objective-matched no-DCache run directory or metrics.csv')
  parser.add_argument(
    '--dcachehooping', default=None,
    help='Optional Dcachehooping run directory or metrics.csv')
  parser.add_argument('--output', default='outputs/loss_comparison.png')
  parser.add_argument('--smooth', type=int, default=20,
                      help='Training-point rolling-average window')
  parser.add_argument(
    '--dcache-train-metric', default='auto',
    help=('DCache metric for the training curve, for example '
          'trainer/loss_t1. The default selects loss_t2, then loss_s.'))
  parser.add_argument(
    '--dcache-val-metric', default='val/nll',
    help=('DCache metric for the validation curve, for example '
          'val/loss_t1. The default is val/nll.'))
  parser.add_argument(
    '--ablation-train-metric', default='auto',
    help=('Ablation training metric. The default uses the selected DCache '
          'training metric when available, then loss_base or total loss.'))
  parser.add_argument(
    '--ablation-val-metric', default='auto',
    help=('Ablation validation metric. The default uses the selected DCache '
          'validation metric when available, then val/nll.'))
  parser.add_argument(
    '--dcachehooping-train-metric', default='trainer/loss_base',
    help='Comparable Dcachehooping training metric')
  parser.add_argument(
    '--dcachehooping-val-metric', default='val/loss_t2',
    help='Comparable Dcachehooping validation metric')
  parser.add_argument(
    '--dcachehooping-val-supplement', default=None,
    help='Optional CSV containing standalone final-state validation points')
  parser.add_argument(
    '--dcachehooping-val-supplement-step-column', default='checkpoint_step',
    help='Step column in --dcachehooping-val-supplement')
  parser.add_argument(
    '--dcachehooping-val-supplement-metric-column', default='val_loss_t2',
    help='Metric column in --dcachehooping-val-supplement')
  parser.add_argument(
    '--dcachehooping-label', default='Dcachehooping',
    help='Display label for the optional recurrent final-state run')
  parser.add_argument(
    '--min-step', type=float, default=None,
    help='Optional inclusive optimizer-step lower bound for all curves')
  parser.add_argument(
    '--max-step', type=float, default=None,
    help='Optional inclusive optimizer-step upper bound for all curves')
  parser.add_argument(
    '--x-max', type=float, default=None,
    help='Optional displayed x-axis maximum, independent of --max-step')
  parser.add_argument(
    '--hide-dcache-total', action='store_true',
    help='Do not overlay the DCache total objective on the training panel')
  args = parser.parse_args()

  vanilla = load_metrics(args.vanilla)
  dcache = load_metrics(args.dcache)
  ablation = load_metrics(args.ablation) if args.ablation else None
  dcachehooping = (
    load_metrics(args.dcachehooping) if args.dcachehooping else None)
  if (dcachehooping is not None
      and args.dcachehooping_val_supplement is not None):
    dcachehooping = append_metric_points(
      dcachehooping,
      args.dcachehooping_val_supplement,
      args.dcachehooping_val_supplement_step_column,
      args.dcachehooping_val_supplement_metric_column,
      args.dcachehooping_val_metric)
  if args.dcache_train_metric == 'auto':
    dcache_train_metric = first_available_metric(
      dcache, ('trainer/loss_t2', 'trainer/loss_s'))
  else:
    dcache_train_metric = args.dcache_train_metric
    if metric_series(dcache, dcache_train_metric).empty:
      raise ValueError(
        f'DCache training metric is absent from the logs: '
        f'{dcache_train_metric}')
  if metric_series(dcache, args.dcache_val_metric).empty:
    raise ValueError(
        f'DCache validation metric is absent from the logs: '
        f'{args.dcache_val_metric}')
  ablation_train_metric = None
  ablation_val_metric = None
  if ablation is not None:
    if args.ablation_train_metric == 'auto':
      ablation_train_metric = first_available_metric(
        ablation, (dcache_train_metric, 'trainer/loss_base', 'trainer/loss'))
    else:
      ablation_train_metric = args.ablation_train_metric
    if metric_series(ablation, ablation_train_metric).empty:
      raise ValueError(
        f'Ablation training metric is absent from the logs: '
        f'{ablation_train_metric}')
    if args.ablation_val_metric == 'auto':
      ablation_val_metric = first_available_metric(
        ablation, (args.dcache_val_metric, 'val/nll'))
    else:
      ablation_val_metric = args.ablation_val_metric
    if metric_series(ablation, ablation_val_metric).empty:
      raise ValueError(
        f'Ablation validation metric is absent from the logs: '
        f'{ablation_val_metric}')
  if dcachehooping is not None:
    if metric_series(
        dcachehooping, args.dcachehooping_train_metric).empty:
      raise ValueError(
        'Dcachehooping training metric is absent from the logs: '
        f'{args.dcachehooping_train_metric}')
  figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

  vanilla_train = plot_training(
    axes[0], vanilla, 'trainer/loss', 'Vanilla train loss', '#1f77b4',
    args.smooth, args.min_step, args.max_step)
  dcache_train = plot_training(
    axes[0], dcache, dcache_train_metric,
    f'DCache train {dcache_train_metric.removeprefix("trainer/")}', '#ff7f0e',
    args.smooth, args.min_step, args.max_step)
  ablation_train = None
  if ablation is not None:
    ablation_train = plot_training(
      axes[0], ablation, ablation_train_metric,
      f'Objective-matched no-DCache {ablation_train_metric.removeprefix("trainer/")}',
      '#2ca02c', args.smooth, args.min_step, args.max_step)
  dcachehooping_train = None
  if dcachehooping is not None:
    dcachehooping_train = plot_training(
      axes[0], dcachehooping, args.dcachehooping_train_metric,
      (f'{args.dcachehooping_label} train '
       f'{args.dcachehooping_train_metric.removeprefix("trainer/")}'),
      '#9467bd',
      args.smooth, args.min_step, args.max_step)
  total = (pd.DataFrame(columns=['step', 'trainer/loss'])
           if args.hide_dcache_total
           else metric_series(dcache, 'trainer/loss'))
  if args.min_step is not None:
    total = total[total.step >= args.min_step]
  if args.max_step is not None:
    total = total[total.step <= args.max_step]
  if not total.empty:
    axes[0].plot(total.step,
                 total['trainer/loss'].rolling(
                   max(1, min(args.smooth, len(total))), min_periods=1).mean(),
                 color='#d62728', linestyle=':', linewidth=1.2,
                 label='DCache total objective')
  axes[0].set_ylabel('Training loss')
  axes[0].set_title('OpenWebText pretraining health')
  axes[0].grid(alpha=0.25)
  axes[0].legend()

  vanilla_val = plot_validation(
    axes[1], vanilla, 'val/nll', 'Vanilla validation NLL', '#1f77b4',
    args.min_step, args.max_step)
  dcache_val = plot_validation(
    axes[1], dcache, args.dcache_val_metric,
    f'DCache validation {args.dcache_val_metric.removeprefix("val/")}',
    '#ff7f0e', args.min_step, args.max_step)
  ablation_val = None
  if ablation is not None:
    ablation_val = plot_validation(
      axes[1], ablation, ablation_val_metric,
      f'Objective-matched no-DCache {ablation_val_metric.removeprefix("val/")}',
      '#2ca02c', args.min_step, args.max_step)
  dcachehooping_val = None
  if dcachehooping is not None:
    dcachehooping_val = plot_validation(
      axes[1], dcachehooping, args.dcachehooping_val_metric,
      (f'{args.dcachehooping_label} validation '
       f'{args.dcachehooping_val_metric.removeprefix("val/")}'), '#9467bd',
      args.min_step, args.max_step)
  axes[1].set_xlabel('Optimizer step')
  axes[1].set_ylabel('Validation loss / NLL')
  displayed_x_max = args.x_max if args.x_max is not None else args.max_step
  if args.min_step is not None or displayed_x_max is not None:
    axes[1].set_xlim(left=args.min_step, right=displayed_x_max)
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
  if ablation is not None:
    print_latest('Objective-matched no-DCache training', ablation_train,
                 ablation_train_metric)
  print_latest('Vanilla validation', vanilla_val, 'val/nll')
  print_latest('DCache validation', dcache_val, args.dcache_val_metric)
  if ablation is not None:
    print_latest('Objective-matched no-DCache validation', ablation_val,
                 ablation_val_metric)
  if dcachehooping is not None:
    print_latest(
      f'{args.dcachehooping_label} training', dcachehooping_train,
      args.dcachehooping_train_metric)
    print_latest(
      f'{args.dcachehooping_label} validation', dcachehooping_val,
      args.dcachehooping_val_metric)
  print(f'Wrote {output.resolve()}')


if __name__ == '__main__':
  main()
