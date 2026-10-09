"""Refresh loss plots from per-update OWT CSV files."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [('mdm', 'MDM'), ('mdm_np', 'MDM + NP (random init)'),
            ('mdm_np_zero_init', 'MDM + NP (zero init)'),
            ('mdm_np_zero_init_low_weight', 'MDM + NP (zero init, weight 0.05)')]


def refresh(root, smooth=128):
    root = Path(root)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for variant, label in VARIANTS:
        for kind, column, ax in [('train','main_elbo',axes[0]),
                                 ('train','objective',axes[1]),
                                 ('validation','val_nll',axes[2])]:
            path = root/variant/'local_metrics'/f'{kind}.csv'
            if not path.exists():
                continue
            data = pd.read_csv(path).drop_duplicates('optimizer_step', keep='last').sort_values('optimizer_step')
            y = data[column].rolling(smooth, min_periods=1).mean() if kind == 'train' else data[column]
            ax.plot(data.optimizer_step, y, label=label, marker='o' if kind != 'train' else None,
                    markersize=3)
    for ax, title in zip(axes, ['Train main ELBO', 'Train total objective (includes NP)', 'Validation main ELBO (EMA)']):
        ax.set(title=title, xlabel='Optimizer updates', ylabel='Loss (nats/token)')
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend()
    fig.suptitle(f'OWT / BD3 pretraining • training smoothing: {smooth} updates • validation: 1,024 fixed rows')
    fig.tight_layout()
    fig.savefig(root/'loss_vs_step.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT/'outputs/owt/mdm-np-5k')
    parser.add_argument('--smooth', type=int, default=128)
    args = parser.parse_args()
    state = args.root/'current.json'
    if state.exists():
        print(state.read_text())
    for name in ('zero_init_queue.json', 'low_weight_queue.json'):
        followup = args.root/name
        if followup.exists():
            print(name, followup.read_text())
    for variant, _ in VARIANTS:
        for kind in ('train', 'validation'):
            path = args.root/variant/'local_metrics'/f'{kind}.csv'
            if path.exists():
                print(variant, kind, pd.read_csv(path).tail(1).to_dict('records'))
    refresh(args.root, args.smooth)
    print('Plot:', args.root/'loss_vs_step.png')


if __name__ == '__main__':
    main()
