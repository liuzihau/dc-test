"""Plot saved main training and EMA validation losses for MDM, A and B."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/runtime/np-live-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CURRENT = 'mdm_np_zero_init_transformer_pair_count_control'
RECIPES = [
    ('mdm-np-5k', 'mdm', 'MDM', '#247bb8'),
    ('transformer-np-5k', 'mdm_np_zero_init_transformer_masked_source', 'A: both masked', '#cc4545'),
    ('transformer-np-5k', CURRENT, 'B: matched pairs', '#159da0'),
]


def read_rows(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    raw = raw[:raw.rfind(b'\n') + 1]
    rows = []
    for row in csv.DictReader(io.StringIO(raw.decode())):
        if None in row or any(value in (None, '') for value in row.values()):
            continue
        values = {key: float(value) for key, value in row.items()}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f'Nonfinite metrics: {path}')
        if not values['optimizer_step'].is_integer():
            raise ValueError(f'Invalid step: {path}')
        rows.append(values)
    if any(a['optimizer_step'] >= b['optimizer_step'] for a, b in zip(rows, rows[1:])):
        raise ValueError(f'Non-increasing steps: {path}')
    return rows


def refresh(out):
    out.mkdir(parents=True, exist_ok=True)
    snapshot = {'captured_at': datetime.now(timezone.utc).isoformat(), 'model_forwards': 0,
                'point_cutoff': 800, 'x_axis_start': 700, 'curves': {}}
    for name, filename, metric, window, title in (
        ('loss_vs_step', 'train.csv', 'main_elbo', 32, 'Main training ELBO'),
        ('validation_vs_step', 'validation.csv', 'val_nll', 1, 'Main EMA validation loss'),
    ):
        fig, ax = plt.subplots(figsize=(11, 6))
        for group, variant, label, color in RECIPES:
            path = ROOT / 'outputs/owt' / group / variant / 'local_metrics' / filename
            rows = read_rows(path)
            last = int(rows[-1]['optimizer_step']) if rows else 0
            snapshot['curves'].setdefault(variant, {})[name] = {'last_step': last, 'source': str(path.relative_to(ROOT))}
            if variant == CURRENT:
                current = last
            if not rows:
                continue
            x = np.array([row['optimizer_step'] for row in rows], dtype=int)
            y = np.array([row[metric] for row in rows])
            # Smooth using prior saved updates, then apply the plotting cutoff.
            if window > 1:
                sums = np.r_[0., np.cumsum(y)]
                ends = np.arange(1, len(y) + 1)
                starts = np.maximum(ends - window, 0)
                y = (sums[ends] - sums[starts]) / (ends - starts)
            keep = x >= 800
            if keep.any():
                ax.plot(x[keep], y[keep], label=label, color=color, lw=2,
                        marker='o' if window == 1 else None, markersize=4)
        ax.set_xlim(700, 5100)
        ax.set_xticks([700, *range(1000, 5001, 500)])
        ax.set_xlabel('Optimizer step')
        ax.set_ylabel('Nats/token (lower is better)')
        ax.set_title(f'OWT: {title} · B through step {current:,}', loc='left', fontweight='bold')
        ax.grid(alpha=.16)
        ax.spines[['top', 'right']].set_visible(False)
        if ax.lines:
            ax.legend(fontsize=10)
        note = ('Trailing 32-update mean; main loss only' if window > 1 else
                'Actual evaluations only; first point after cutoff is step 1,000; no smoothing')
        if current < 800:
            note += f' · B has no points ≥800 yet (latest {current})'
        fig.text(.08, .025, note, fontsize=9, color='#555555')
        fig.tight_layout(rect=(0, .055, 1, 1))
        partial = out / f'{name}.{os.getpid()}.partial.png'
        fig.savefig(partial, dpi=180, facecolor='white')
        partial.replace(out / f'{name}.png')
        plt.close(fig)
    partial = out / f'snapshot.{os.getpid()}.partial.json'
    partial.write_text(json.dumps(snapshot, indent=2) + '\n')
    partial.replace(out / 'snapshot.json')
    current = snapshot['curves'][CURRENT]
    print(f"Updated {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}: "
          f"B train={current['loss_vs_step']['last_step']}, "
          f"validation={current['validation_vs_step']['last_step']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true', help='Refresh until Ctrl+C')
    parser.add_argument('--interval', type=float, default=60, help='Refresh seconds, default60')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/analysis/owt-transformer-np-control-live')
    args = parser.parse_args()
    if not np.isfinite(args.interval) or args.interval < 5:
        parser.error('--interval must be finite and at least5 seconds')
    try:
        while True:
            refresh(args.output.resolve())
            if not args.watch:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
