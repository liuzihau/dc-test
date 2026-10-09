#!/usr/bin/env python3
"""Plot vanilla validation NLL against DCache t0--t3 validation losses."""

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
  'MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'dcache-matplotlib'))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_pretrain_losses import load_metrics, metric_series


def after_step(frame, metric, minimum_step):
  series = metric_series(frame, metric)
  return series[series.step > minimum_step]


def latest_mask_ratio(frame, state):
  series = metric_series(frame, f'val/mask_ratio_{state}')
  if series.empty:
    return None
  return float(series.iloc[-1][f'val/mask_ratio_{state}'])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--vanilla', required=True,
                      help='Vanilla run directory or metrics.csv')
  parser.add_argument('--dcache', required=True,
                      help='DCache run directory or metrics.csv')
  parser.add_argument('--output',
                      default='outputs/dcache-validation-t0-t3.png')
  parser.add_argument('--min-step', type=float, default=900,
                      help='Plot points strictly after this optimizer step')
  args = parser.parse_args()

  vanilla = load_metrics(args.vanilla)
  dcache = load_metrics(args.dcache)

  figure, axis = plt.subplots(figsize=(10, 6.5))

  vanilla_series = after_step(vanilla, 'val/nll', args.min_step)
  if vanilla_series.empty:
    raise ValueError('No vanilla val/nll points remain after the step filter')
  axis.plot(
    vanilla_series.step, vanilla_series['val/nll'],
    color='#1f77b4', linewidth=2.3, label='Vanilla validation NLL')

  state_colors = {
    't0': '#d62728',
    't1': '#ff7f0e',
    't2': '#2ca02c',
    't3': '#9467bd',
  }
  for state, color in state_colors.items():
    metric = f'val/loss_{state}'
    series = after_step(dcache, metric, args.min_step)
    if series.empty:
      raise ValueError(f'No {metric} points remain after the step filter')
    mask_ratio = latest_mask_ratio(dcache, state)
    ratio_label = '' if mask_ratio is None else f', mask={mask_ratio:.3f}'
    axis.plot(
      series.step, series[metric], marker='o', markersize=5,
      linewidth=2.0, color=color,
      label=f'DCache {state} validation loss ({ratio_label.lstrip(", ")})')
    latest = series.iloc[-1]
    print(f'{state}: step={int(latest.step)} {metric}={latest[metric]:.6f}')

  latest_vanilla = vanilla_series.iloc[-1]
  print(
    f'vanilla: step={int(latest_vanilla.step)} '
    f'val/nll={latest_vanilla["val/nll"]:.6f}')

  axis.set_title(f'Validation losses after optimizer step {args.min_step:g}')
  axis.set_xlabel('Optimizer step')
  axis.set_ylabel('Validation loss / NLL')
  axis.grid(alpha=0.25)
  axis.legend()

  output = Path(args.output).expanduser()
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.tight_layout()
  figure.savefig(output, dpi=160)
  plt.close(figure)
  print(f'Wrote {output.resolve()}')


if __name__ == '__main__':
  main()
