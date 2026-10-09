#!/usr/bin/env python3
"""CPU-only figures for the synthetic clue-binding experiment."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ('direct_equality', 'direct_adjacency', 'equality_star', 'adjacency_pairs', 'equality_chain', 'adjacency_chain')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoding', choices=('answer_relative', 'typed_coordinates'))
    args = p.parse_args()
    rows = []
    for enc, gpu in (('answer_relative', 2), ('typed_coordinates', 3)):
        if args.encoding and args.encoding != enc:
            continue
        root = ROOT/f'outputs/reasoning/zebra-binding-{enc}-gpu{gpu}/binding'
        for path in sorted(root.glob('*.json')):
            result = json.loads(path.read_text())
            for family, group in result['groups'].items():
                rows.append(dict(encoding=enc, step=result['step'], split=result['split'], family=family,
                    fullmask_accuracy=group['fullmask_accuracy'], fullmask_nll=group['fullmask_nll'],
                    greedy_solve_rate=group['argmax_solve_rate'],
                    sampled_solve_rate=group.get('sample_solve_rate', ''),
                    examples=group['examples'], trained_family=group['trained_family']))
    if not rows:
        raise ValueError('No completed binding evaluations yet')
    output = ROOT/'results/generated/figures/reasoning/zebra-clue-binding'/(args.encoding or 'comparison')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for ax, family in zip(axes.flat, FAMILIES):
        for enc, color in (('answer_relative', '#718096'), ('typed_coordinates', '#007f86')):
            selected = sorted([r for r in rows if r['encoding'] == enc and r['split'] == 'validation' and r['family'] == family], key=lambda r:r['step'])
            if not selected: continue
            short = 'Fixed' if enc == 'answer_relative' else 'Typed'
            for key, style, label in (('fullmask_accuracy', '--', 'full-mask token acc'), ('greedy_solve_rate', '-', 'greedy solve rate')):
                ax.plot([r['step'] for r in selected], [100*r[key] for r in selected], style, marker='o', color=color, label=short+' '+label)
        ax.set_title(family.replace('_', ' ') + (' [UNSEEN GRAPH]' if family.endswith('chain') else ''))
        ax.set_ylim(0, 102); ax.grid(alpha=.2); ax.legend(fontsize=7)
        ax.set_xlabel('Optimizer updates'); ax.set_ylabel('Validation accuracy (%)')
    fig.suptitle('Synthetic 5x5 clue-binding diagnostic — NOT original Zebra benchmark accuracy')
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output/'progress.png', dpi=160); plt.close(fig)
    print('Wrote', output, flush=True)


if __name__ == '__main__':
    main()
