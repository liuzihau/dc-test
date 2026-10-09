#!/usr/bin/env python3
"""Compare frozen-checkpoint Zebra trace audits; no GPU or model loading."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main():
    runs = {}
    for encoding in ('answer_relative', 'typed_coordinates'):
        path = ROOT / 'outputs/reasoning' / ('zebra-trace-audit-' + encoding) / 'audit.json'
        runs[encoding] = json.loads(path.read_text())
    first, second = runs.values()
    for key in ('data_sha256', 'split', 'seed', 'examples', 'step', 'protocol', 'precision', 'batch_size'):
        if first[key] != second[key]:
            raise ValueError('Unmatched audit setting: ' + key)
    if [x['id'] for x in first['probes']] != [x['id'] for x in second['probes']]:
        raise ValueError('Audits used different validation examples')
    output = ROOT / 'results/generated/figures/reasoning/zebra-trace-audit'
    output.mkdir(parents=True, exist_ok=True)
    labels = dict(answer_relative='Fixed positions', typed_coordinates='Typed coordinates')
    colors = dict(answer_relative='#718096', typed_coordinates='#007f86')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    summary_rows, clue_rows, rank_rows = [], [], []
    for i, (encoding, result) in enumerate(runs.items()):
        summary = result['summary']
        color, label = colors[encoding], labels[encoding]
        groups = summary['fullmask_groups']
        values = [100 * groups[k]['accuracy'] for k in ('direct_equality', 'other_content')]
        bars = axes[0].bar(np.arange(2)+(i-.5)*.34, values, width=.34, color=color, label=label)
        axes[0].bar_label(bars, fmt='%.1f', fontsize=9)
        for name, style in (('sample', '-'), ('argmax', '--')):
            condition = summary['conditions'][name]
            ks = (1, 2, 4, 8)
            axes[1].plot(ks, [100 * condition['prefix_survival'][str(k)] for k in ks],
                         style, marker='o', color=color, label=label + (' / sample' if name == 'sample' else ' / greedy'))
            for kind, counts in condition['clue_satisfaction'].items():
                clue_rows.append(dict(model=label, decoder=name, kind=kind, **counts,
                    satisfied_fraction=counts['satisfied']/counts['total'],
                    invalid_reference_fraction=counts['invalid_reference']/counts['total']))
            for rank, counts in condition['by_content_rank'].items():
                rank_rows.append(dict(model=label, decoder=name, rank=int(rank), **counts))
        actual_correct = oracle_correct = tokens = 0
        for a, o in zip(result['conditions']['sample'], result['conditions']['oracle_replay']):
            assert a['id'] == o['id'] and a['decode_order'] == o['decode_order']
            had_error = False
            for ae, oe in zip(a['events'], o['events']):
                if had_error and ae['content']:
                    tokens += 1
                    actual_correct += ae['argmax_correct']
                    oracle_correct += oe['argmax_correct']
                had_error |= not ae['committed_correct']
        rates = [100 * actual_correct/tokens, 100 * oracle_correct/tokens]
        bars = axes[2].bar(np.arange(2)+(i-.5)*.34, rates, width=.34, color=color, label=label)
        axes[2].bar_label(bars, fmt='%.1f', fontsize=9)
        sample, greedy = [summary['conditions'][k] for k in ('sample', 'argmax')]
        summary_rows.append(dict(model=label, examples=result['examples'],
            fullmask_direct_accuracy=groups['direct_equality']['accuracy'],
            fullmask_other_accuracy=groups['other_content']['accuracy'],
            sample_solve_rate=sample['scores']['valid_solution'],
            greedy_solve_rate=greedy['scores']['valid_solution'],
            sample_first_content_accuracy=sample['prefix_survival']['1'],
            greedy_first_content_accuracy=greedy['prefix_survival']['1'],
            sample_all_rows_valid=sample['all_rows_valid'], greedy_all_rows_valid=greedy['all_rows_valid'],
            after_error_tokens=tokens, actual_history_argmax_accuracy=actual_correct/tokens,
            oracle_history_argmax_accuracy=oracle_correct/tokens,
            **{'oracle_' + k: v for k, v in summary['oracle_after_first_error'].items()}))
    axes[0].set_xticks([0, 1], ['Explicit equality\nfixes this cell', 'Other content cells'])
    axes[0].set_title('A. Clue use with every answer masked')
    axes[0].set_ylabel('Token argmax accuracy (%)')
    axes[0].legend(fontsize=8)
    axes[1].set_title('B. Early errors during normal decoding')
    axes[1].set_xlabel('First k content commitments (EOS excluded)')
    axes[1].set_ylabel('Puzzles with all first k tokens correct (%)')
    axes[1].set_xticks([1, 2, 4, 8])
    axes[1].legend(fontsize=7)
    axes[2].set_xticks([0, 1], ['Actual sampled\nhistory', 'Corrected (gold)\nhistory'])
    axes[2].set_title('C. After an earlier decoding error')
    axes[2].set_ylabel('Argmax accuracy at the SAME later positions (%)')
    for ax in axes:
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Zebra diagnostic audit — two frozen 40,026-update models; 1,000 validation puzzles')
    fig.text(.5, .015, 'Candidate-8 in both free decoders. Gold-history replay is diagnostic only, not a solve rate. '
             'Direct vs other cells are different difficulty subsets.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(output/'diagnosis.png', dpi=170)
    plt.close(fig)
    for name, rows in (('summary.csv', summary_rows), ('clues.csv', clue_rows), ('ranks.csv', rank_rows)):
        with (output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary_rows, indent=2))
    print('Figure:', output/'diagnosis.png')


if __name__ == '__main__':
    main()
