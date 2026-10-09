"""Compare weighted auxiliary/main loss in existing MDM+NP training logs.

No model is loaded and no training state is changed. The smoothed statistic is
the ratio of rolling component means, not the mean of per-update ratios.
"""
import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/runtime/analysis/matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator, FuncFormatter
import numpy as np
import pandas as pd
import yaml

COLORS = {'OWT': '#2563eb', 'Sudoku': '#d97706', 'Zebra': '#0f8b76'}


def ratio_frame(frame, weights, main_weight=1.0, smooth=128):
    """Calculate ratios in an actual optimizer-update window, handling gaps."""
    if smooth < 1 or main_weight <= 0:
        raise ValueError('Smoothing and main weight must be positive')
    if set(weights) != {-1, 1}:
        raise ValueError('This report requires exactly previous/next NP heads')
    columns = ['optimizer_step', 'main_elbo', 'np_prev', 'np_next']
    data = frame.copy()
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    good = np.isfinite(data[columns]).all(axis=1) & (data.main_elbo > 0)
    good &= data.optimizer_step.ge(1) & data.optimizer_step.eq(data.optimizer_step.round())
    data = data[good].drop_duplicates('optimizer_step', keep='last').sort_values('optimizer_step')
    data = data.reset_index(drop=True)
    if data.empty:
        raise ValueError('No finite positive-main-loss rows available')
    data['optimizer_step'] = data.optimizer_step.astype('int64')
    data['weighted_main'] = main_weight * data.main_elbo
    data['weighted_auxiliary'] = weights[-1]*data.np_prev + weights[1]*data.np_next
    data['raw_aux_to_main'] = data.weighted_auxiliary/data.weighted_main
    # A synthetic time index gives rolling windows (step - smooth, step], so
    # missing optimizer updates cannot silently turn 128 updates into 128 rows.
    index = pd.to_timedelta(data.optimizer_step.to_numpy(), unit='s')
    for source, dest in [('weighted_main', 'rolling_main'),
                         ('weighted_auxiliary', 'rolling_auxiliary')]:
        series = pd.Series(data[source].to_numpy(), index=index)
        data[dest] = series.rolling(f'{smooth}s', min_periods=1).mean().to_numpy()
    data['aux_to_main'] = data.rolling_auxiliary/data.rolling_main
    data['aux_share_total'] = data.aux_to_main/(1+data.aux_to_main)
    return data


def read_run(name, root, smooth):
    root = Path(root).resolve()
    source = root/'mdm_np/local_metrics/train.csv'
    config = yaml.safe_load((root/'mdm_np/resolved_config.yaml').read_text())
    np_config = config['mechanisms']['np']
    if not np_config['enabled']:
        raise ValueError(f'{name}: NP is disabled')
    weights = dict(zip(np_config['offsets'], np_config['weights']))
    frame = pd.read_csv(source)
    data = ratio_frame(frame, weights, float(config['objective']['current_weight']), smooth)
    total = 'objective' if 'objective' in data else 'train_loss'
    error = (data[total]-data.weighted_main-data.weighted_auxiliary).abs()
    if not np.allclose(data[total], data.weighted_main+data.weighted_auxiliary,
                       rtol=1e-5, atol=1e-4):
        raise ValueError(f'{name}: logged objective disagrees with configured loss weights')
    metadata = dict(source=str(source), source_mtime_ns=source.stat().st_mtime_ns,
        rows_read=len(frame), rows_used=len(data), weights=weights,
        main_weight=float(config['objective']['current_weight']),
        missing_update_gaps=int(data.optimizer_step.diff().gt(1).sum()),
        maximum_objective_identity_error=float(error.max()))
    return data, metadata


def style_axis(ax, ymax):
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.set_xlabel('Optimizer updates')
    ax.grid(axis='both', alpha=.17)
    ax.spines[['top', 'right']].set_visible(False)
    ax.axhline(.5, color='#64748b', lw=1, ls='--', alpha=.65, zorder=0)


def save_figure(fig, output, stem):
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'{stem}.{extension}', dpi=190, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smooth', type=int, default=128)
    parser.add_argument('--owt-run', type=Path, default=ROOT/'outputs/owt/mdm-np-5k')
    parser.add_argument('--sudoku-run', type=Path, default=ROOT/'outputs/sudoku/mdm-np-20ep')
    parser.add_argument('--zebra-run', type=Path, default=ROOT/'outputs/zebra/mdm-np-40ep')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT/'results/generated/figures/training/auxiliary-loss-ratio')
    parser.add_argument('--table-dir', type=Path,
                        default=ROOT/'results/generated/tables/training/auxiliary-loss-ratio')
    args = parser.parse_args()
    if args.smooth < 1:
        parser.error('--smooth must be positive')
    runs, metadata = {}, {}
    for name, root in [('OWT', args.owt_run), ('Sudoku', args.sudoku_run), ('Zebra', args.zebra_run)]:
        runs[name], metadata[name] = read_run(name, root, args.smooth)
    output = args.output_dir.resolve()
    tables = args.table_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    common_end = min(int(d.optimizer_step.iloc[-1]) for d in runs.values())
    ymax = max(.6, math.ceil(max(d.aux_to_main.max() for d in runs.values())*10)/10)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 13, 'axes.labelsize': 11})
    formula = ('Weighted auxiliary / main loss: (w_prev L_prev + w_next L_next) / (w_main L_main)'
               f'\nRatio of trailing {args.smooth}-update means; shorter window at the start')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), sharey=True)
    for ax, (name, d) in zip(axes, runs.items()):
        ax.plot(d.optimizer_step, d.aux_to_main, color=COLORS[name], lw=1.7)
        ax.scatter(d.optimizer_step.iloc[-1], d.aux_to_main.iloc[-1], color=COLORS[name], s=24, zorder=3)
        ax.set_title(f'{name} · MDM + NP\nLatest: {d.aux_to_main.iloc[-1]:.1%} at update {d.optimizer_step.iloc[-1]:,}')
        ax.set_xlim(0, int(d.optimizer_step.iloc[-1])*1.015)
        style_axis(ax, ymax)
    axes[0].set_ylabel('Weighted auxiliary / main loss')
    fig.suptitle('Auxiliary loss relative to main loss — full training histories', fontsize=16, y=1.02)
    fig.text(.5, -.055, formula+'\nDifferent x-axis ranges. Dashed line: auxiliary equals 50% of main (not of total loss).',
             ha='center', va='top', fontsize=10, color='#475569')
    fig.tight_layout()
    save_figure(fig, output, 'auxiliary_loss_ratio_full_history')
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, d in runs.items():
        early = d[d.optimizer_step <= common_end]
        ax.plot(early.optimizer_step, early.aux_to_main, lw=2, color=COLORS[name], label=name)
    style_axis(ax, ymax)
    ax.set_xlim(0, common_end)
    ax.set_ylabel('Weighted auxiliary / main loss')
    ax.set_title(f'Matched early updates — through {common_end:,}', fontsize=15)
    ax.legend(loc='best', frameon=False)
    fig.text(.5, -.025, formula+'\nTraining-loss magnitudes only; not gradient shares or proof of task interference.',
             ha='center', va='top', fontsize=10, color='#475569')
    fig.tight_layout()
    save_figure(fig, output, 'auxiliary_loss_ratio_matched_updates')
    summaries = []
    for name, d in runs.items():
        exported = d[['optimizer_step', 'main_elbo', 'np_prev', 'np_next',
                      'weighted_main', 'weighted_auxiliary', 'raw_aux_to_main',
                      'rolling_main', 'rolling_auxiliary', 'aux_to_main', 'aux_share_total']]
        exported.to_csv(tables/f'{name.lower()}_ratio_by_update.csv', index=False)
        for label, step in [('step_2500', 2500), ('common_latest', common_end),
                            ('latest', int(d.optimizer_step.iloc[-1]))]:
            selected = d[d.optimizer_step == step]
            if not selected.empty:
                row = selected.iloc[-1]
                summaries.append(dict(benchmark=name, point=label, optimizer_step=step,
                    weighted_aux_to_main_pct=100*row.aux_to_main,
                    weighted_aux_share_total_pct=100*row.aux_share_total,
                    rolling_main=row.rolling_main, rolling_auxiliary=row.rolling_auxiliary))
    summary = pd.DataFrame(summaries)
    summary.to_csv(tables/'summary.csv', index=False)
    (tables/'provenance.json').write_text(json.dumps(dict(smoothing_updates=args.smooth,
        aggregation='ratio of rolling component means, not mean of ratios',
        common_latest_optimizer_step=common_end,
        interpretation='loss magnitudes; not gradient magnitudes or causal effects', runs=metadata), indent=2)+'\n')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('Figures:', output)
    print('Tables:', tables)


if __name__ == '__main__':
    main()
