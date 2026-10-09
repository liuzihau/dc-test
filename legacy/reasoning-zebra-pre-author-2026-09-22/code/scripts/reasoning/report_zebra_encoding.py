"""Plot per-encoding validation and generation; independent outputs avoid races."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main(gpu):
    encoding = {2: 'answer_relative', 3: 'typed_coordinates'}[gpu]
    run = ROOT / 'outputs/reasoning' / f'zebra-encoding-{encoding}-gpu{gpu}'
    output = ROOT / 'results/generated/figures/reasoning/zebra-encoding' / encoding
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for name, directory in [('Old absolute positions', ROOT/'outputs/reasoning/zebra-tfw-answer-only-no-shift-lr3e4-gpu3'),
                             (encoding, run)]:
        values = []
        for path in sorted((directory/'validation').glob('step-*.json')):
            v = json.loads(path.read_text())
            row = dict(run=name, step=int(path.stem.split('-')[-1]),
                validation_nll=v['mean_over_ratios_nll'],
                full_mask_accuracy=v['full_mask_diagnostic']['content_accuracy'])
            values.append(row); rows.append(row)
        if values:
            axes[0].plot([v['step'] for v in values], [v['validation_nll'] for v in values], label=name)
            axes[1].plot([v['step'] for v in values], [100*v['full_mask_accuracy'] for v in values], label=name)
    generation = [json.loads(p.read_text()) for p in sorted((run/'generation').glob('test-step-*.json'))]
    for policy in ('upstream_remask', 'paper_monotonic', 'matched_candidate8'):
        if generation:
            axes[2].plot([v['step'] for v in generation],
                         [100*v['reports'][policy]['metrics']['valid_solution'] for v in generation], '.-', label=policy)
    for ax, title in zip(axes, ('Common validation NLL', 'Fully masked content accuracy (%)', 'Complete TEST puzzle accuracy (%)')):
        ax.set(title=title, xlabel='Optimizer step'); ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=8)
    fig.suptitle(f'Zebra public-encoding diagnostic: {encoding} (not an exact paper reproduction)')
    fig.tight_layout()
    fig.savefig(output/'status.png', dpi=160)
    plt.close(fig)
    if rows:
        with (output/'validation.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    print(output/'status.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(2,3), required=True)
    main(parser.parse_args().gpu)
