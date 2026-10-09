"""Read-only, provenance-checked summaries of completed reasoning evaluations.

These are synthetic-pilot whole-puzzle accuracies, not token accuracies or a
reproduction of the Latent Tokens paper. No model/checkpoint loading is needed.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np


VARIANTS = ('vanilla', 'mdm', 'mdm_aux', 'both', 'both_aux')
CONTRASTS = (('mdm_aux', 'mdm'), ('both_aux', 'both'),
             ('both', 'mdm'), ('both_aux', 'mdm_aux'))
CONTRACT_KEYS = ('task', 'variant', 'model_config', 'data_sha256', 'global_batch',
                 'micro_batch', 'world_size', 'seed', 'lr', 'weight_decay',
                 'warmup_steps', 'grad_clip', 'precision', 'device_type')


@dataclass
class Evaluation:
    path: Path
    run: Path
    payload: dict
    ids: tuple
    successes: np.ndarray

    @property
    def contract(self):
        return self.payload['contract']

    @property
    def task(self):
        return self.contract['task']

    @property
    def variant(self):
        return self.contract['variant']


def wilson_interval(successes, count):
    """Two-sided 95% Wilson interval, returned on the probability scale."""
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError('A nonempty valid binomial count is required')
    z = 1.959963984540054
    p, denominator = successes / count, 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1-p) / count + z*z / (4*count*count)) / denominator
    return (0. if successes == 0 else max(0., center-half),
            1. if successes == count else min(1., center+half))


def paired_interval(condition, reference, *, resamples=2000, seed=2026):
    """Bootstrap paired puzzles, not independent binomial draws per model.

    Differences take only {-1,0,1}, so multinomial resampling their counts is
    exactly equivalent to resampling example indices, with O(resamples) memory.
    """
    if resamples < 100 or len(condition) != len(reference) or not len(condition):
        raise ValueError('Need matching nonempty outcomes and >=100 bootstrap resamples')
    difference = np.asarray(condition, dtype=np.int8) - np.asarray(reference, dtype=np.int8)
    probabilities = np.array([(difference == value).mean() for value in (-1, 0, 1)])
    draws = np.random.default_rng(seed).multinomial(len(difference), probabilities, size=resamples)
    deltas = 100. * (draws[:, 2] - draws[:, 0]) / len(difference)
    low, high = np.quantile(deltas, [.025, .975])
    return dict(mean_delta_pp=100. * float(difference.mean()), ci95_low_pp=float(low),
                ci95_high_pp=float(high), condition_only_correct=int((difference == 1).sum()),
                reference_only_correct=int((difference == -1).sum()),
                both_correct=int(((np.asarray(condition) == 1) & (np.asarray(reference) == 1)).sum()),
                both_wrong=int(((np.asarray(condition) == 0) & (np.asarray(reference) == 0)).sum()),
                bootstrap_degenerate=bool(np.all(difference == difference[0])))


def load_evaluation(path, run, step):
    path, run = Path(path), Path(run)
    data = json.loads(path.read_text())
    if data['step'] != step:
        raise ValueError(f'checkpoint step {data["step"]} is not requested step {step}')
    contract, args, metrics = data['contract'], data['arguments'], data['metrics']
    for key in CONTRACT_KEYS:
        if key not in contract:
            raise ValueError(f'missing training-contract field: {key}')
    task, variant = contract['task'], contract['variant']
    if task not in ('sudoku', 'zebra', 'countdown') or variant not in VARIANTS:
        raise ValueError('unsupported task/model variant')
    sha = contract['data_sha256']
    if len(sha) != 64 or any(char not in '0123456789abcdef' for char in sha):
        raise ValueError('invalid dataset-manifest SHA256')
    if (args['split'] != 'test' or args['protocol'] != 'generate'
            or args['memory_condition'] != 'correct'
            or metrics['evaluation'] != 'closed_loop_generation'
            or metrics['memory_condition'] != 'correct'):
        raise ValueError('requires test/generate/correct-memory evaluation')
    if (args['policy'] != 'top_prob' or metrics['policy'] != 'top_prob'
            or metrics['candidate_k'] != 8 or metrics['token_selection'] != 'paper'
            or metrics['tokens_per_step'] != 1 or metrics['max_steps'] is not None
            or metrics['seed'] != args['seed']):
        raise ValueError('generation protocol differs from fixed candidate-8, one-token paper sampling')
    if args['batch_size'] < 1:
        raise ValueError('invalid evaluation batch size')
    config = contract['model_config']
    if config['neighbors'] != variant.endswith('_aux'):
        raise ValueError('variant and auxiliary-head configuration disagree')
    expected_memory = 'both' if variant.startswith('both') else 'none'
    if config['memory_mode'] != expected_memory:
        raise ValueError('variant and memory configuration disagree')
    if config['attention_mode'] != ('vanilla' if variant == 'vanilla' else 'merged'):
        raise ValueError('unexpected attention architecture')
    if config['trajectory'] != ('single' if variant == 'vanilla' else 'five'):
        raise ValueError('unexpected training trajectory')
    if variant.startswith('both') and (config.get('merged_policy') != 'current_preserving'
            or config['gate_enabled'] or config['cache_only_probability'] != 0
            or config['gradient_mode'] != 'adjacent'):
        raise ValueError('memory result is not the corrected merged/adjacent recipe')
    if contract.get('stress_memory_routes'):
        raise ValueError('smoke-stress checkpoints are not production trials')
    saved_contract = run / 'contract.json'
    if saved_contract.exists() and json.loads(saved_contract.read_text()) != contract:
        raise ValueError('evaluation contract differs from the run contract')
    examples = data['examples']
    if (not examples or len(examples) != args['examples']
            or len(examples) != metrics['num_examples']):
        raise ValueError('incomplete evaluation: requested/reported/record counts differ')
    ids, successes = [], []
    for item in examples:
        if item['task'] != task or not item['all_slots_completed'] or item['remaining_masked_slots'] != 0:
            raise ValueError('wrong-task or incomplete generation record')
        success = item['scores']['valid_solution']
        if not isinstance(success, (bool, int, float)) or success not in (0, 1):
            raise ValueError('valid_solution must be a binary whole-puzzle outcome')
        if not isinstance(item['id'], str) or not isinstance(item['record_index'], int):
            raise ValueError('records must contain stable string IDs and integer indices')
        ids.append((item['id'], item['record_index'], task))
        successes.append(int(success))
    if len({item[0] for item in ids}) != len(ids) or len({item[1] for item in ids}) != len(ids):
        raise ValueError('duplicate puzzle IDs or record indices')
    if not math.isclose(float(metrics['valid_solution']), float(np.mean(successes)), abs_tol=1e-10):
        raise ValueError('reported accuracy disagrees with per-example outcomes')
    return Evaluation(path.resolve(), run.resolve(), data, tuple(ids), np.array(successes, dtype=np.int8))


def protocol_key(evaluation):
    args, metrics = evaluation.payload['arguments'], evaluation.payload['metrics']
    return (evaluation.task, evaluation.contract['data_sha256'], evaluation.payload['step'],
            args['split'], args['protocol'], args['policy'], args['memory_condition'],
            args['seed'], args['batch_size'], args['examples'], metrics['candidate_k'],
            metrics['token_selection'], metrics['tokens_per_step'], metrics['max_steps'])


def contract_differences(a, b, prefix=''):
    result = []
    for key in sorted(set(a) | set(b)):
        name = prefix + key
        if key not in a or key not in b:
            result.append(name)
        elif isinstance(a[key], dict) and isinstance(b[key], dict):
            result.extend(contract_differences(a[key], b[key], name + '.'))
        elif a[key] != b[key]:
            result.append(name)
    return result


def validate_pair(condition, reference):
    if protocol_key(condition) != protocol_key(reference):
        raise ValueError('dataset/step/split/generation seed/protocol/batch/count mismatch')
    if condition.ids != reference.ids:
        raise ValueError('paired puzzle IDs/order differ')
    a, b = copy.deepcopy(condition.contract), copy.deepcopy(reference.contract)
    differences = contract_differences(a, b)
    pair = (condition.variant, reference.variant)
    if pair not in CONTRASTS:
        raise ValueError('not a prespecified contrast')
    if pair in (('mdm_aux', 'mdm'), ('both_aux', 'both')):
        allowed = {'variant', 'model_config.neighbors'}
        description = 'auxiliary-training intervention; same memory, trajectory and optimizer settings'
    else:
        # These flags are explicitly part of the bundled intervention; this is
        # NOT described as a parameter-matched or architecture-only contrast.
        allowed = {'variant', *('model_config.' + key for key in (
            'memory_mode', 'gradient_mode', 'merged_policy', 'gate_enabled',
            'cache_only_probability', 'current_only_probability', 'final_dropout',
            'identity_probability'))}
        description = ('memory + adjacent credit assignment + memory robustness/identity recipe; '
                       'not architecture-only or compute/parameter-matched')
    unexpected = set(differences) - allowed
    if unexpected:
        raise ValueError('uncontrolled training-contract differences: ' + ', '.join(sorted(unexpected)))
    return description, differences


def collect(runs, step, warnings):
    result = []
    for run in dict.fromkeys(Path(run).resolve() for run in runs):
        found = []
        for path in sorted(run.glob('evaluation*.json')):
            try:
                found.append(load_evaluation(path, run, step))
            except (KeyError, TypeError, ValueError, OSError) as error:
                warnings.append(f'Skipped {path}: {error}')
        if not found:
            warnings.append(f'No eligible completed step-{step} evaluation: {run}')
            continue
        # Do not pick whichever seed/retry/test evaluation scores best.
        semantic = lambda item: (protocol_key(item), json.dumps(item.contract, sort_keys=True),
                                 item.ids, item.successes.tobytes(), item.payload['checkpoint'])
        if any(semantic(item) != semantic(found[0]) for item in found[1:]):
            warnings.append(f'Ambiguous multiple eligible evaluations: {run}; none selected')
            continue
        result.append(found[0])
    return sorted(result, key=lambda item: (item.task, VARIANTS.index(item.variant), str(item.run)))


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(runs, output_dir, *, step=5000, resamples=2000, seed=2026):
    if step < 1 or resamples < 100:
        raise ValueError('Positive step and >=100 resamples required')
    output = Path(output_dir).resolve()
    if any(output == Path(run).resolve() for run in runs):
        raise ValueError('Use a dedicated report subdirectory, not a raw run directory')
    warnings = []
    evaluations = collect(runs, step, warnings)
    output.mkdir(parents=True, exist_ok=True)
    from reasoning.zebra_diagnostics import summarize_zebra
    warnings.extend(summarize_zebra(evaluations, output))
    rows, pairs = [], []
    for item in evaluations:
        count, correct = len(item.successes), int(item.successes.sum())
        low, high = wilson_interval(correct, count)
        rows.append(dict(task=item.task, variant=item.variant, step=step, examples=count,
            correct=correct, accuracy_pct=100*correct/count, ci95_low_pct=100*low,
            ci95_high_pct=100*high, train_seed=item.contract['seed'],
            global_batch=item.contract['global_batch'], micro_batch=item.contract['micro_batch'],
            dataset_sha256=item.contract['data_sha256'], evaluation=str(item.path), run=str(item.run)))
    for task in sorted({item.task for item in evaluations}):
        task_evaluations = [item for item in evaluations if item.task == task]
        for condition_name, reference_name in CONTRASTS:
            conditions = [item for item in task_evaluations if item.variant == condition_name]
            references = [item for item in task_evaluations if item.variant == reference_name]
            label = f'{task}: {condition_name} - {reference_name}'
            if len(conditions) != 1 or len(references) != 1:
                warnings.append(f'No unique completed pair for {label}; no paired claim')
                continue
            condition, reference = conditions[0], references[0]
            try:
                description, changes = validate_pair(condition, reference)
            except ValueError as error:
                warnings.append(f'Pair ineligible ({label}): {error}')
                continue
            pairs.append(dict(task=task, condition=condition_name, reference=reference_name,
                step=step, examples=len(condition.successes),
                **paired_interval(condition.successes, reference.successes, resamples=resamples, seed=seed),
                intervention=description, contract_differences=';'.join(changes),
                bootstrap_resamples=resamples, bootstrap_seed=seed,
                condition_evaluation=str(condition.path), reference_evaluation=str(reference.path)))
            if pairs[-1]['bootstrap_degenerate']:
                floor = not condition.successes.any() and not reference.successes.any()
                warnings.append(f'{label}: ' + ('both models solved zero puzzles (floor effect). ' if floor else '')
                    + 'All observed paired differences are identical, so the ordinary bootstrap interval '
                    'is mechanically degenerate; this does NOT establish equivalence or a precisely known '
                    'population difference. Inspect absolute Wilson intervals and collect more informative tests.')
    fields = ('task', 'variant', 'step', 'examples', 'correct', 'accuracy_pct',
              'ci95_low_pct', 'ci95_high_pct', 'train_seed', 'global_batch', 'micro_batch',
              'dataset_sha256', 'evaluation', 'run')
    pair_fields = ('task', 'condition', 'reference', 'step', 'examples', 'mean_delta_pp',
        'ci95_low_pp', 'ci95_high_pp', 'condition_only_correct', 'reference_only_correct',
        'both_correct', 'both_wrong', 'bootstrap_degenerate', 'intervention', 'contract_differences',
        'bootstrap_resamples', 'bootstrap_seed', 'condition_evaluation', 'reference_evaluation')
    write_csv(output / 'accuracy.csv', rows, fields)
    write_csv(output / 'paired_deltas.csv', pairs, pair_fields)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for task in ('sudoku', 'zebra', 'countdown'):
        subset = [row for row in rows if row['task'] == task]
        write_csv(output / f'{task}_accuracy.csv', subset, fields)
        task_pairs = [row for row in pairs if row['task'] == task]
        write_csv(output / f'{task}_paired_deltas.csv', task_pairs, pair_fields)
        # Always refresh placeholders too: no stale success plot if inputs later fail checks.
        fig, ax = plt.subplots(figsize=(8, 4.8))
        if subset:
            x = np.arange(len(subset))
            y = np.array([row['accuracy_pct'] for row in subset])
            errors = np.array([[row['accuracy_pct']-row['ci95_low_pct'] for row in subset],
                               [row['ci95_high_pct']-row['accuracy_pct'] for row in subset]])
            ax.bar(x, y, color=['#6b7280', '#3b82f6', '#14b8a6', '#f59e0b', '#a855f7'][:len(subset)]
                   if len(subset) <= 5 else '#3b82f6', alpha=.85)
            ax.errorbar(x, y, yerr=np.maximum(errors, 0), fmt='none', ecolor='black', capsize=5)
            ax.set_xticks(x, [f'{row["variant"]}\nn={row["examples"]}' for row in subset])
            for position, value in zip(x, y):
                ax.text(position, min(96, value+4), f'{value:.1f}%', ha='center', fontsize=9)
        else:
            ax.text(.5, .5, 'Awaiting eligible completed evaluations', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
        ax.set_ylim(0, 100)
        ax.set_ylabel('Whole-puzzle valid solution (%)')
        ax.set_title(f'{task.title()} — fixed step {step}, 95% Wilson intervals')
        ax.grid(axis='y', alpha=.2)
        fig.text(.5, .015, 'Synthetic pilot; one training seed. Paired comparisons/provenance: summary.md.',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(output / f'{task}_accuracy.png', dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        if task_pairs:
            x = np.arange(len(task_pairs))
            means = np.array([row['mean_delta_pp'] for row in task_pairs])
            errors = np.array([[row['mean_delta_pp']-row['ci95_low_pp'] for row in task_pairs],
                               [row['ci95_high_pp']-row['mean_delta_pp'] for row in task_pairs]])
            ax.errorbar(x, means, yerr=np.maximum(errors, 0), fmt='o', color='#2563eb', capsize=5)
            ax.set_xticks(x, [f'{row["condition"]}\n− {row["reference"]}'
                             + (' *' if row['bootstrap_degenerate'] else '') for row in task_pairs])
        else:
            ax.text(.5, .5, 'Awaiting matched completed pairs', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_ylabel('Whole-puzzle accuracy difference (percentage points)')
        ax.set_title(f'{task.title()} — paired test-puzzle bootstrap, step {step}')
        ax.grid(axis='y', alpha=.2)
        fig.text(.5, .015, '* Degenerate bootstrap ≠ equivalence. No training-seed uncertainty/multiplicity correction.',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(output / f'{task}_paired_deltas.png', dpi=160)
        plt.close(fig)
    lines = [f'# Reasoning pilot — optimizer step {step}', '',
        'Whole-puzzle **valid_solution** accuracy, not token accuracy. Predictions are generated '
        'without teacher forcing; fixed candidate-8 top-prob position selection and paper token sampling.', '',
        '## Completed evaluations', '', '| Task | Model | Solved / tested | Accuracy (95% Wilson CI) |',
        '| --- | --- | --- | --- |']
    for row in rows:
        lines.append(f'| {row["task"]} | {row["variant"]} | {row["correct"]} / {row["examples"]} | '
                     f'{row["accuracy_pct"]:.2f}% [{row["ci95_low_pct"]:.2f}, {row["ci95_high_pct"]:.2f}] |')
    if not rows:
        lines += ['', 'No eligible completed evaluation yet.']
    lines += ['', '## Prespecified paired contrasts', '',
        'Positive deltas favor the first model. Paired bootstrap intervals resample the same puzzles; '
        'wins/losses count puzzles solved by only the first/second model. No selection by test accuracy.', '',
        '| Task | Contrast | Accuracy delta (pp; 95% CI) | Wins / losses | Bootstrap note |',
        '| --- | --- | --- | --- | --- |']
    for row in pairs:
        lines.append(f'| {row["task"]} | {row["condition"]} − {row["reference"]} | '
                     f'{row["mean_delta_pp"]:+.2f} [{row["ci95_low_pp"]:+.2f}, {row["ci95_high_pp"]:+.2f}] | '
                     f'{row["condition_only_correct"]} / {row["reference_only_correct"]} | '
                     + ('Degenerate; not evidence of equivalence' if row['bootstrap_degenerate'] else 'Nondegenerate') + ' |')
    lines += ['', '## Claim scope', '',
        '- Higher held-out whole-puzzle accuracy supports improved solving on these synthetic Sudoku/Zebra '
        'distributions at this fixed training budget; it does not establish general reasoning ability or superiority to the paper.',
        '- `mdm_aux − mdm` and `both_aux − both` isolate the auxiliary-training intervention only when the '
        'full contracts match except auxiliary heads. The latter includes the same memory recipe in both models.',
        '- `both − mdm` and `both_aux − mdm_aux` are bundled memory + adjacent gradients + robustness/identity '
        'training comparisons, NOT architecture-only, compute-matched, or parameter-matched controls.',
        '- These confidence intervals quantify test-puzzle sampling only, not variation across training seeds. '
        'Multiple contrasts are exploratory and intervals are not multiplicity-adjusted. Replication across training '
        'seeds and additional distributions is still needed for broad claims.',
        '- With zero successes or identical per-puzzle outcomes, ordinary paired bootstrap intervals can '
        'collapse mechanically (for example, [0, 0] when both solve no puzzles). This is not an equivalence '
        'test or evidence of no population gap. Zero successes still have a nonzero Wilson upper bound; '
        'floor-effect results are inconclusive about which model reasons better.',
        '- Five thousand updates are a pilot budget, not proof that data/training are sufficient for all claims. '
        'A better auxiliary model does not by itself establish a reusable workspace; correct/absent/shuffled-memory '
        'interventions and causal controls are still needed.',
        '- All evaluations use the prespecified checkpoint step, not a checkpoint picked using test accuracy. '
        'No different-step comparisons or mismatched training/evaluation protocols enter paired deltas.', '',
        'Zebra diagnostic figure/table: `zebra_diagnostics.png` / `zebra_diagnostics.csv`. These separate '
        'formatting, category permutation/position accuracy and individual clue satisfaction. Except formatting/EOS, '
        'they condition on well-formed outputs; counts and denominators are explicit. They do not replace solve '
        'accuracy or establish relational reasoning on their own.', '',
        '## Warnings / pending work', '']
    lines += ['- ' + warning for warning in warnings] or ['- None.']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    metadata = dict(step=step, bootstrap_resamples=resamples, bootstrap_seed=seed,
        evaluations=[dict(path=str(item.path), sha256=hashlib.sha256(item.path.read_bytes()).hexdigest())
                     for item in evaluations], warnings=warnings, comparisons=len(pairs))
    (output / 'report_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for warning in warnings:
        print('WARNING:', warning, flush=True)
    print(f'Reported {len(rows)} evaluations and {len(pairs)} eligible paired contrasts: {output}', flush=True)
    return rows, pairs, warnings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--step', type=int, default=5000)
    parser.add_argument('--bootstrap-resamples', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=2026)
    args = parser.parse_args(argv)
    summarize(args.runs, args.output_dir, step=args.step,
              resamples=args.bootstrap_resamples, seed=args.bootstrap_seed)
