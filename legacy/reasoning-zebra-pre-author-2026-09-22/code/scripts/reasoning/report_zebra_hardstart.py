"""Refresh the repaired baseline / hard-start continuation diagnostic figure."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RUNS = [
    ('Repaired baseline', 'zebra-tfw-answer-only-no-shift-lr3e4-gpu3', '#64748b'),
    ('Ordinary continuation (GPU 2)', 'zebra-tfw-hardstart-control-from5000-gpu2', '#2563eb'),
    ('50% full-mask mixture (GPU 3)', 'zebra-tfw-hardstart-fullmask50-from5000-gpu3', '#ea580c')]


def main():
    output = ROOT / 'results/generated/figures/reasoning/zebra-tfw-hardstart'
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    table = []
    for name, directory, color in RUNS:
        run = ROOT / 'outputs/reasoning' / directory
        rows = []
        for path in sorted((run / 'validation').glob('step-*.json')):
            value = json.loads(path.read_text())
            step = int(path.stem.split('-')[-1])
            hard = value['full_mask_diagnostic']
            # Same common validation protocol as the historic runner.
            ratios = value['ratios']
            if isinstance(ratios, dict):
                ratios = list(ratios.values())
            nll = sum(v['conditional_nll'] for v in ratios) / len(ratios)
            row = dict(run=name, step=step, common_nll=nll,
                       full_mask_content_nll=hard['content_nll'],
                       full_mask_content_accuracy=hard['content_accuracy'])
            rows.append(row); table.append(row)
        if rows:
            axes[0].plot([r['step'] for r in rows], [r['common_nll'] for r in rows], '.-', color=color, label=name)
            axes[1].plot([r['step'] for r in rows], [100*r['full_mask_content_accuracy'] for r in rows], '.-', color=color, label=name)
    audit_path = ROOT / 'results/generated/tables/reasoning/zebra-tfw-hardstart/clue_audit_step5000.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        for offset, key, label, color in [(-.25, 'original', 'Correct clues', '#2563eb'),
                                         (0, 'shuffled', 'Shuffled clue references', '#f97316'),
                                         (.25, 'blind', 'Clue-blind permutation baseline', '#94a3b8')]:
            axes[2].bar([i+offset for i in range(3)], [100*r[key+'_accuracy'] for r in audit['rows']],
                        width=.24, color=color, label=label)
        axes[2].set_xticks(range(3), ['10%', '70%', '100%'])
        axes[1].axhline(100*audit['rows'][-1]['blind_accuracy'], color='#94a3b8', ls='--', label='Clue-blind expected accuracy')
    axes[0].set(title='Common validation NLL (lower is better)', xlabel='Optimizer step', ylabel='NLL')
    axes[1].set(title='Fully masked answer — cell accuracy', xlabel='Optimizer step', ylabel='Accuracy (%)')
    axes[2].set(title='Step 5000: clue-sensitivity diagnostic', xlabel='Answer mask ratio', ylabel='Content accuracy (%)')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle('Zebra: answer completion is not yet clue-conditioned solving')
    fig.text(.5, .015, 'Frozen held-out validation. Clue shuffling is OOD; sensitivity is not a reasoning proof. '
             'Whole-puzzle generation at step 5000: 0/1000 with all three tested decoders.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .94))
    fig.savefig(output / 'status.png', dpi=170)
    plt.close(fig)
    if table:
        with (output / 'validation.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=table[0])
            writer.writeheader(); writer.writerows(table)
    print(output / 'status.png')


if __name__ == '__main__':
    main()
