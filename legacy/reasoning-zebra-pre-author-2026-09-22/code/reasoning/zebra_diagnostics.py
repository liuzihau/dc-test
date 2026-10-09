"""Descriptive Zebra failure diagnostics; never replace whole-puzzle accuracy."""
import csv
import hashlib
import json
from pathlib import Path

from reasoning.tasks import (_parse_zebra_prompt, _strip_answer, _zebra_relation,
                             score_prediction)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def diagnostic_rows(evaluation):
    """Use only the exact verified test data referenced by this evaluation."""
    data = evaluation.payload
    directory = Path(data['arguments']['data_dir'])
    manifest_path = directory / 'manifest.json'
    if digest(manifest_path) != data['contract']['data_sha256']:
        raise ValueError('dataset manifest SHA256 differs from evaluation contract')
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema_version'] != 1 or manifest['task'] != 'zebra':
        raise ValueError('requires schema-1 Zebra dataset')
    split = manifest['splits']['test']
    if split['filename'] != 'test.jsonl':
        raise ValueError('unexpected test filename')
    path = directory / split['filename']
    if digest(path) != split['sha256']:
        raise ValueError('test split SHA256 mismatch')
    records = [json.loads(line) for line in path.open() if line.strip()]
    if len(records) != split['records']:
        raise ValueError('test split record count mismatch')
    counts = {name: [0, 0] for name in ('wellformed25digits', 'has_eos',
        'all_five_permutations', 'position_accuracy', 'category_C0', 'category_C1',
        'category_C2', 'category_C3', 'category_C4', 'clue_AT', 'clue_SAME',
        'clue_LEFT', 'clue_NEXT')}
    for example in data['examples']:
        index = example['record_index']
        if not isinstance(index, int) or index < 0 or index >= len(records):
            raise ValueError('test record index out of range')
        record = records[index]
        if record['id'] != example['id'] or record['task'] != 'zebra':
            raise ValueError('test record ID/task mismatch')
        if not score_prediction(record, record['answer'])['valid_solution']:
            raise ValueError('test ground truth fails task validator')
        raw = example['predicted_answer_slots']
        if score_prediction(record, raw)['valid_solution'] != example['scores']['valid_solution']:
            raise ValueError('rescored whole-puzzle result differs from saved evaluation')
        answer = _strip_answer(raw)
        wellformed = answer is not None and len(answer) == 25 and all(token in list('12345') for token in answer)
        counts['wellformed25digits'][0] += int(wellformed)
        counts['wellformed25digits'][1] += 1
        counts['has_eos'][0] += int('[EOS]' in raw)
        counts['has_eos'][1] += 1
        if not wellformed:
            continue
        positions = list(map(int, answer))
        correct = [prediction == target for prediction, target in zip(answer, record['answer'])]
        counts['position_accuracy'][0] += sum(correct)
        counts['position_accuracy'][1] += 25
        all_permutations = True
        for category in range(5):
            start = 5 * category
            all_permutations &= set(positions[start:start+5]) == set(range(1, 6))
            counts[f'category_C{category}'][0] += sum(correct[start:start+5])
            counts[f'category_C{category}'][1] += 5
        counts['all_five_permutations'][0] += int(all_permutations)
        counts['all_five_permutations'][1] += 1
        for kind, a, b in _parse_zebra_prompt(record['prompt']):
            satisfied = positions[a] == b if kind == 'AT' else _zebra_relation(kind, positions[a], positions[b])
            counts['clue_' + kind][0] += int(satisfied)
            counts['clue_' + kind][1] += 1
    rows = []
    for metric, (numerator, denominator) in counts.items():
        rows.append(dict(variant=evaluation.variant, step=data['step'], metric=metric,
            numerator=numerator, denominator=denominator,
            percentage=100*numerator/denominator if denominator else '',
            conditioning='all evaluated answers' if metric in ('has_eos', 'wellformed25digits')
                         else 'well-formed 25-digit answers only; malformed answers excluded',
            meaning='descriptive diagnostic, NOT whole-puzzle solving accuracy',
            dataset_sha256=data['contract']['data_sha256'], evaluation=str(evaluation.path)))
    return rows


def summarize_zebra(evaluations, output_dir):
    """Write diagnostic CSV/PNG, returning warnings instead of blocking reports."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, warnings = [], []
    for evaluation in evaluations:
        if evaluation.task != 'zebra':
            continue
        try:
            rows.extend(diagnostic_rows(evaluation))
        except (OSError, ValueError, KeyError, TypeError) as error:
            warnings.append(f'Zebra diagnostics skipped for {evaluation.path}: {error}')
    fields = ('variant', 'step', 'metric', 'numerator', 'denominator', 'percentage',
              'conditioning', 'meaning', 'dataset_sha256', 'evaluation')
    with (output / 'zebra_diagnostics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    groups = [
        ('Answer structure', ('wellformed25digits', 'has_eos', 'all_five_permutations', 'position_accuracy'),
         ('25 digits', 'Has EOS', 'All 5\npermutations*', 'Position\naccuracy*')),
        ('Category position accuracy*', tuple(f'category_C{i}' for i in range(5)), tuple(f'C{i}' for i in range(5))),
        ('Individual clue satisfaction*', ('clue_AT', 'clue_SAME', 'clue_LEFT', 'clue_NEXT'), ('AT', 'SAME', 'LEFT', 'NEXT'))]
    sources = list(dict.fromkeys((row['variant'], row['evaluation']) for row in rows))
    colors = {'vanilla': '#6b7280', 'mdm': '#3b82f6', 'mdm_aux': '#14b8a6', 'both': '#f59e0b', 'both_aux': '#a855f7'}
    for ax, (title, metrics, labels) in zip(axes, groups):
        if sources:
            width = .8 / len(sources)
            for index, (variant, source) in enumerate(sources):
                selected = {row['metric']: row for row in rows if row['evaluation'] == source}
                positions = np.arange(len(metrics)) - .4 + width*(index+.5)
                values = [selected[metric]['percentage'] if selected[metric]['denominator'] else float('nan') for metric in metrics]
                ax.bar(positions, values, width=width, label=variant, color=colors.get(variant, '#6b7280'))
        else:
            ax.text(.5, .5, 'Awaiting verified Zebra test data/results', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9)
        ax.set_xticks(np.arange(len(metrics)), labels)
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Diagnostic percentage (not solve rate)')
    if sources:
        axes[0].legend(fontsize=8)
    fig.suptitle('Zebra failure diagnostics — separate from whole-puzzle accuracy')
    fig.text(.5, .015, '* Conditional on well-formed 25-digit answers; differing subsets can bias comparisons. '
             'Counts/denominators in CSV. No causal or solving claim.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(output / 'zebra_diagnostics.png', dpi=160)
    plt.close(fig)
    return warnings
