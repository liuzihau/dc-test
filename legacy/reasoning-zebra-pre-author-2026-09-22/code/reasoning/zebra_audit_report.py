"""CPU summaries of read-only Zebra audits; diagnostic evidence, not a benchmark claim."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


VARIANTS = ('vanilla', 'mdm', 'mdm_aux', 'both', 'both_aux')


def _one(rows, **wanted):
    matches = [row for row in rows if all(row.get(key) == value for key, value in wanted.items())]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one row matching {wanted}')
    return matches[0]


def _metric(value, *, probability=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('Expected finite numeric metric')
    if probability and not 0 <= value <= 1:
        raise ValueError('Accuracy is outside [0,1]')
    return float(value)


def read_audit(path):
    data = json.loads(path.read_text())
    contract, args, protocol = data['contract'], data['arguments'], data['protocol']
    variant, step = contract['variant'], data['step']
    if data['schema_version'] != 1 or contract['task'] != 'zebra' or variant not in VARIANTS:
        raise ValueError(f'Not a supported Zebra audit: {path}')
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError('Invalid completed step')
    for name in ('checkpoint_sha256', 'data_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', data[name]):
            raise ValueError(f'Missing SHA256 provenance: {name}')
    if (data['data_sha256'] != contract['data_sha256'] or not data['checkpoint']
            or args.get('trust_checkpoint') is not True or protocol.get('no_test_tuning') is not True
            or protocol['splits'] != ['train', 'validation'] or set(data['splits']) != {'train', 'validation'}
            or protocol.get('cold_memory') != 'none; independent first forward'):
        raise ValueError('Missing trusted checkpoint/data/train-validation-only provenance')
    rows, identifiers = [], {}
    for split, section in data['splits'].items():
        cold = section['cold']
        original = _one(cold['summary'], mask_ratio=1., condition='original')['categories']
        destroyed = _one(cold['summary'], mask_ratio=1., condition='clue_content_permuted')['categories']
        shortcut = _one(cold['permutation_shortcut']['summary'], mask_ratio=1.)['categories']
        ids = {}
        for condition in ('original', 'clue_content_permuted'):
            ids[condition] = [row['id'] for row in cold['examples'] if row['mask_ratio'] == 1 and row['condition'] == condition]
        if (ids['original'] != ids['clue_content_permuted'] or len(set(ids['original'])) != section['examples']
                or len(ids['original']) != section['examples']):
            raise ValueError('Unpaired/duplicate/missing cold diagnostic example IDs')
        identifiers[split] = ids['original']
        row = dict(variant=variant, step=step, split=split, examples=section['examples'])
        for name in ('all', 'C0', 'C1', 'C2', 'C3', 'C4'):
            expected = section['examples'] * (25 if name == 'all' else 5)
            if any(group[name]['tokens'] != expected for group in (original, destroyed, shortcut)):
                raise ValueError('Full-mask content denominators do not match example count')
            for label, group in (('original', original), ('permuted', destroyed), ('shortcut', shortcut)):
                row[f'cold_{label}_{name}_nll'] = _metric(group[name]['conditional_nll'])
                row[f'cold_{label}_{name}_accuracy'] = _metric(group[name]['top1_accuracy'], probability=True)
            row[f'cold_{name}_intact_minus_permuted_pp'] = 100 * (row[f'cold_original_{name}_accuracy'] - row[f'cold_permuted_{name}_accuracy'])
            row[f'cold_{name}_nll_minus_shortcut'] = row[f'cold_original_{name}_nll'] - row[f'cold_shortcut_{name}_nll']
        if abs(shortcut['all']['expected_uniform_tie_accuracy'] - .2) > 1e-6:
            raise ValueError('Full-mask clue-blind calibration must equal 20% expected accuracy')
        for condition, generation in section.get('generation', {}).items():
            if condition not in ('correct', 'none'):
                raise ValueError('Unexpected generation condition')
            metrics = generation['metrics']
            if ([example['id'] for example in generation['examples']] != identifiers[split]
                    or metrics['num_examples'] != section['examples'] or metrics['memory_condition'] != condition
                    or metrics['policy'] != 'top_prob' or metrics['candidate_k'] != 8
                    or metrics['tokens_per_step'] != 1 or metrics['seed'] != args['seed']
                    or metrics['token_selection'] != 'paper' or metrics['max_steps'] is not None
                    or metrics['mean_nfe_per_example'] != 26):
                raise ValueError('Generation IDs/protocol disagree with audit')
            row[f'generation_{condition}_solve_accuracy'] = _metric(metrics['valid_solution'], probability=True)
            for name, counts in generation['diagnostics']['values'].items():
                if counts['accuracy'] is not None:
                    row[f'generation_{condition}_{name}'] = _metric(counts['accuracy'], probability=True)
        if all(f'generation_{key}_solve_accuracy' in row for key in ('correct', 'none')):
            row['generation_correct_minus_none_pp'] = 100 * (row['generation_correct_solve_accuracy'] - row['generation_none_solve_accuracy'])
        rows.append(row)
    geometry = {key: contract.get(key) for key in ('global_batch', 'micro_batch', 'world_size', 'seed', 'lr', 'weight_decay', 'warmup_steps', 'grad_clip', 'precision')}
    signature = json.dumps([data['data_sha256'], protocol, {key: args[key] for key in ('seed', 'batch_size', 'examples')}, identifiers, geometry], sort_keys=True)
    return data, rows, signature


def _csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ['variant', 'step', 'split']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(audit_dir, output_dir):
    directory, output = Path(audit_dir).resolve(), Path(output_dir).resolve()
    if output == directory or directory in output.parents:
        raise ValueError('Report output must be outside the input audit directory')
    loaded, all_rows, signature, provenance = {}, [], None, []
    for path in sorted(directory.glob('*.json')):
        data, rows, current_signature = read_audit(path)
        if signature is not None and current_signature != signature:
            raise ValueError(f'Incompatible dataset/protocol/examples/training geometry: {path}')
        signature = current_signature
        key = (data['contract']['variant'], data['step'])
        if key in loaded:
            if loaded[key]['checkpoint_sha256'] != data['checkpoint_sha256'] or loaded[key]['splits'] != data['splits']:
                raise ValueError(f'Conflicting duplicate variant/step: {key}')
            continue
        loaded[key] = data
        all_rows.extend(rows)
        provenance.append(dict(audit=str(path), audit_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               checkpoint=data['checkpoint'], checkpoint_sha256=data['checkpoint_sha256'],
                               variant=key[0], step=key[1], data_sha256=data['data_sha256']))
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / 'audit_metrics.csv', all_rows)
    deltas = []
    for variant in VARIANTS:
        for split in ('train', 'validation'):
            pair = {row['step']: row for row in all_rows if row['variant'] == variant and row['split'] == split}
            if 5000 in pair and 10000 in pair:
                base, final = pair[5000], pair[10000]
                deltas.append(dict(variant=variant, split=split, start_step=5000, end_step=10000,
                    **{key + '_delta_10k_minus_5k': final[key] - base[key] for key in final if key not in ('variant', 'step', 'split', 'examples') and key in base}))
    _csv(output / 'audit_5k_to_10k_deltas.csv', deltas)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    panels = [('cold_original_all_accuracy', 100, 'Cold full-mask content accuracy (%)'),
              ('cold_all_intact_minus_permuted_pp', 1, 'Intact − destroyed prompt accuracy (pp)'),
              ('generation_correct_solve_accuracy', 100, 'Correct-memory whole-puzzle accuracy (%)')]
    for axis, (metric, scale, title) in zip(axes, panels):
        for variant in VARIANTS:
            rows = sorted((row for row in all_rows if row['variant'] == variant and row['split'] == 'validation' and metric in row), key=lambda row: row['step'])
            if rows:
                axis.plot([row['step'] for row in rows], [scale * row[metric] for row in rows], 'o-', label=variant)
        axis.set(title=title, xlabel='Completed optimizer updates')
        axis.grid(alpha=.2)
        if axis.lines:
            axis.legend(fontsize=8)
        else:
            axis.text(.5, .5, 'Pending', ha='center', transform=axis.transAxes)
    axes[0].axhline(20, color='gray', linestyle='--', label='Clue-blind chance 20%')
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color='gray', linestyle='--')
    fig.suptitle('Zebra train/validation diagnostic audit — validation panels')
    fig.text(.5, .015, 'Prompt destruction is out-of-distribution sensitivity, not causal reasoning evidence. Single training seed; no paper-reproduction claim.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .94))
    fig.savefig(output / 'zebra_audit.png', dpi=170)
    plt.close(fig)
    lines = ['# Zebra diagnostic audit', '', 'Train and validation only; no test-set tuning. Figures show validation.', '',
             '| Variant | 5k audit | 10k audit |', '|---|---|---|']
    for variant in VARIANTS:
        lines.append(f'| {variant} | {"Complete" if (variant, 5000) in loaded else "Pending"} | {"Complete" if (variant, 10000) in loaded else "Pending"} |')
    lines += ['', 'Full numeric results (both splits, C0–C4, clue rates, memory correct/none): `audit_metrics.csv`.',
              'Paired checkpoint changes: `audit_5k_to_10k_deltas.csv` (negative NLL delta is better).', '',
              'Full-mask clue-blind calibration is 20% accuracy and ln(5) NLL. The partial-mask calibration in each original audit can improve by exploiting revealed answer permutations without reading clues.',
              'Intact-minus-permuted prompt accuracy measures destructive/OOD sensitivity only. It is not selective causal evidence of relational reasoning.',
              'Generation is evaluated from full masks without teacher forcing. Malformed outputs count wrong in all position/clue diagnostics.',
              'Results use a single training seed and small train/validation subsets. They are diagnostic, not a reproduction of the paper or proof of general reasoning gains.', '',
              'Checkpoint integrity was verified by the audit command; this report checks recorded hashes and comparability metadata, without rereading large remote checkpoints.']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return all_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    rows = summarize(args.audit_dir, args.output_dir)
    print(json.dumps(dict(output_dir=str(Path(args.output_dir).resolve()), split_rows=len(rows),
                          checkpoint_steps=sorted({row['step'] for row in rows}))))


if __name__ == '__main__':
    main()
