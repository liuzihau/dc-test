"""CPU reports from saved paired predictions; no checkpoint/model evaluation.

One bootstrap resamples packed rows for every head, direction and condition.
Token-level distributions are descriptive; oracle selection uses true labels.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import unicodedata

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np

from owt.research import ROOT, atomic_write, read_json, record_event
from owt.reveal_corruption import make_canvas, nearest_different_sources
from owt.reveal_sweep import digest, exposed_target_mask
from owt.post_diagnostic_canvases import source_pairs

VARIANTS = ['mdm', 'mdm_np_zero_init']
BINS = [-np.inf, -4., -2., -1., -.5, -.1, 0., .1, .5, 1., 2., 4., np.inf]
CASES = ['both_correct', 'rescue', 'main_only_correct', 'both_wrong_same', 'both_wrong_different']


def bootstrap_weights(rows, draws=2000, seed=20261005):
    indices = np.random.default_rng(seed).integers(0, rows, size=(draws, rows))
    offsets = np.arange(draws)[:, None] * rows
    return np.bincount((indices + offsets).ravel(), minlength=draws * rows).reshape(draws, rows).astype(float)


def ratio_summary(numerator, denominator, boot_num, boot_den):
    total = float(np.sum(denominator))
    valid = boot_den > 0
    return dict(value=float(np.sum(numerator) / total) if total else None,
        row_bootstrap_95=np.quantile(boot_num[valid] / boot_den[valid], [.025, .975]).tolist() if valid.any() else None,
        defined_bootstrap_replicates=int(valid.sum()))


def pair_report(clean, first, second, logp_first, logp_second, selected, weights,
                first_confidence=None, second_confidence=None):
    """First=main/reference, second=auxiliary/NP; rescue means second fixes first."""
    if not all(a.shape == clean.shape for a in (first, second, logp_first, logp_second, selected)):
        raise ValueError('Paired prediction shapes differ')
    if weights.shape[1] != len(clean):
        raise ValueError('Bootstrap row population differs')
    if not np.isfinite(logp_first[selected]).all() or not np.isfinite(logp_second[selected]).all():
        raise ValueError('Selected true-token log probabilities are missing')
    mc, ac = first == clean, second == clean
    case_masks = dict(both_correct=mc & ac, rescue=~mc & ac,
        main_only_correct=mc & ~ac, both_wrong_same=~mc & ~ac & (first == second),
        both_wrong_different=~mc & ~ac & (first != second))
    stats = {'targets': selected.sum(1).astype(float)}
    stats.update({name: (mask & selected).sum(1).astype(float) for name, mask in case_masks.items()})
    stats.update(main_correct=(mc & selected).sum(1).astype(float),
        alternative_correct=(ac & selected).sum(1).astype(float),
        main_errors=(~mc & selected).sum(1).astype(float),
        agreement=((first == second) & selected).sum(1).astype(float),
        main_ce_sum=np.where(selected, -logp_first.astype(float), 0).sum(1),
        alternative_ce_sum=np.where(selected, -logp_second.astype(float), 0).sum(1),
        alternative_true_probability_better=((logp_second > logp_first) & selected).sum(1).astype(float))
    stats['ce_difference_sum'] = stats['alternative_ce_sum'] - stats['main_ce_sum']
    stats['accuracy_difference_count'] = stats['alternative_correct'] - stats['main_correct']
    assert np.array_equal(sum(stats[k] for k in CASES), stats['targets'])
    assert np.array_equal(stats['accuracy_difference_count'], stats['rescue'] - stats['main_only_correct'])
    for name, confidence in [('main', first_confidence), ('alternative', second_confidence)]:
        if confidence is not None:
            if not np.isfinite(confidence[selected]).all():
                raise ValueError('Selected confidence is missing')
            stats[name + '_confidence_sum'] = np.where(selected, confidence, 0).sum(1)
    keys = list(stats)
    sums = weights @ np.column_stack([stats[key] for key in keys])
    boot = dict(zip(keys, sums.T))
    definitions = dict(main_accuracy=('main_correct', 'targets'),
        alternative_accuracy=('alternative_correct', 'targets'),
        accuracy_difference=('accuracy_difference_count', 'targets'),
        agreement=('agreement', 'targets'), absolute_rescue=('rescue', 'targets'),
        conditional_rescue=('rescue', 'main_errors'), main_ce=('main_ce_sum', 'targets'),
        alternative_ce=('alternative_ce_sum', 'targets'),
        ce_difference=('ce_difference_sum', 'targets'),
        alternative_true_probability_better=('alternative_true_probability_better', 'targets'))
    definitions.update({case + '_rate': (case, 'targets') for case in CASES})
    for name in ('main', 'alternative'):
        if name + '_confidence_sum' in stats:
            definitions[name + '_mean_top1_probability'] = (name + '_confidence_sum', 'targets')
    metrics = {name: ratio_summary(stats[num], stats[den], boot[num], boot[den])
               for name, (num, den) in definitions.items()}
    delta = (logp_second.astype(float) - logp_first.astype(float))[selected]
    histogram = np.histogram(delta, bins=BINS)[0]
    result = dict(targets=int(stats['targets'].sum()), rows_with_targets=int((stats['targets'] > 0).sum()),
        counts={key: int(stats[key].sum()) for key in CASES}, metrics=metrics,
        logp_difference_distribution=dict(definition='log p_second(Y) - log p_first(Y); positive favors second',
            quantile_probabilities=[.05, .25, .5, .75, .95],
            quantiles=np.quantile(delta, [.05, .25, .5, .75, .95]).tolist() if len(delta) else None,
            histogram_edges=['-inf', *BINS[1:-1], 'inf'], histogram_counts=histogram.tolist()),
        same_top1_targets=int(((first == second) & selected).sum()),
        same_top1_second_probability_better=int(((first == second) & (logp_second > logp_first) & selected).sum()))
    return result, stats, case_masks


def union_report(clean, arrays, selected, weights):
    main = arrays['main_prediction'] == clean
    left = arrays['left_prediction'] == clean
    right = arrays['right_prediction'] == clean
    rescued = ~main & (left | right)
    stats = dict(targets=selected.sum(1).astype(float), main_correct=(main & selected).sum(1).astype(float),
        main_errors=(~main & selected).sum(1).astype(float), rescue=(rescued & selected).sum(1).astype(float),
        oracle_correct=((main | left | right) & selected).sum(1).astype(float),
        auxiliary_agreement=((arrays['left_prediction'] == arrays['right_prediction']) & selected).sum(1).astype(float),
        both_auxiliary_wrong=(~left & ~right & selected).sum(1).astype(float))
    keys = list(stats); boot = dict(zip(keys, (weights @ np.column_stack(list(stats.values()))).T))
    definitions = dict(main_accuracy=('main_correct', 'targets'), absolute_rescue=('rescue', 'targets'),
        conditional_rescue=('rescue', 'main_errors'), oracle_top1_selector_accuracy=('oracle_correct', 'targets'),
        auxiliary_agreement=('auxiliary_agreement', 'targets'), both_auxiliary_wrong=('both_auxiliary_wrong', 'targets'))
    return dict(targets=int(stats['targets'].sum()),
        metrics={k: ratio_summary(stats[n], stats[d], boot[n], boot[d]) for k, (n, d) in definitions.items()},
        limitation='Oracle chooses among three top1 tokens using labels; this is not deployable accuracy.'), stats


def load_predictions(path):
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    ids = arrays['row_ids']
    if len(set(ids.tolist())) != len(ids) or not np.array_equal(ids, np.sort(ids)):
        raise ValueError('Prediction rows must be unique and sorted')
    for key, values in arrays.items():
        if key != 'row_ids' and values.shape != arrays['clean'].shape:
            raise ValueError('Prediction array shape differs: ' + key)
    if arrays['scored'][:, 0].any():
        raise ValueError('Input zero is not a scored target')
    for head in ('main', 'left', 'right'):
        if head + '_prediction' not in arrays:
            continue
        chosen = arrays['scored'] if head == 'main' else arrays[head + '_prediction'] >= 0
        if ((arrays[head + '_prediction'][chosen] >= 50257) | (arrays[head + '_prediction'][chosen] < 0)).any():
            raise ValueError('MASK or an invalid token appears in a selected prediction')
        true_logp = arrays[head + '_true_logp'][chosen]
        if not np.isfinite(true_logp).all() or (true_logp > 1e-6).any():
            raise ValueError('Invalid selected true-token score')
        confidence = arrays.get(head + '_top1_probability')
        if confidence is not None:
            confidence = confidence[chosen]
            correct = arrays[head + '_prediction'][chosen] == arrays['clean'][chosen]
            if (not np.isfinite(confidence).all() or (confidence <= 0).any() or (confidence > 1).any()
                    or not np.allclose(confidence[correct], np.exp(true_logp[correct]), rtol=2e-5, atol=2e-6)
                    or (np.exp(true_logp[~correct]) > confidence[~correct] + 2e-6).any()):
                raise ValueError('Top1 confidence is inconsistent with predictions and true-token scores')
    return arrays


def verify_pair(first, second):
    for key in ('row_ids', 'clean', 'scored'):
        if not np.array_equal(first[key], second[key]):
            raise ValueError('Model prediction pairing differs: ' + key)


def cached_tokenizer():
    from tokenizers import Tokenizer
    candidates = sorted((Path.home() / '.cache/huggingface/hub').glob('models--gpt2/snapshots/*/tokenizer.json'))
    if not candidates:
        raise RuntimeError('Cached GPT2 tokenizer is required; this report never downloads assets')
    return Tokenizer.from_file(str(candidates[0])), candidates[0]


def token_strata(clean, tokenizer):
    tokens, counts = np.unique(clean, return_counts=True)
    categories, frequencies = {}, {}
    for token, count in zip(tokens, counts):
        text = tokenizer.decode([int(token)], skip_special_tokens=False).strip()
        categories[int(token)] = ('special' if token in (50256, 50257) else
            'punctuation' if text and all(unicodedata.category(c).startswith('P') for c in text) else 'content_or_other')
        frequencies[int(token)] = '1_to_10' if count <= 10 else '11_to_100' if count <= 100 else 'over_100'
    groups = {}
    for kind, mapping in [('token_category', categories), ('cohort_token_frequency', frequencies)]:
        for name in sorted(set(mapping.values())):
            groups[kind + '_' + name] = np.isin(clean, [key for key, value in mapping.items() if value == name])
    return groups


def illustrations(clean, first, second, selected, masks, ids, tokenizer, label):
    result = []
    # Small fixed samples within each counted case, not selected by loss magnitude.
    seed = int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    for case, mask in masks.items():
        positions = np.argwhere(mask & selected)
        if not len(positions):
            continue
        for k in rng.choice(len(positions), min(3, len(positions)), replace=False):
            row, target = positions[k]
            result.append(dict(population=label, case=case, row_id=int(ids[row]), target_index=int(target),
                truth_id=int(clean[row, target]), first_id=int(first[row, target]), second_id=int(second[row, target]),
                truth=tokenizer.decode([int(clean[row, target])], skip_special_tokens=False),
                first=tokenizer.decode([int(first[row, target])], skip_special_tokens=False),
                second=tokenizer.decode([int(second[row, target])], skip_special_tokens=False)))
    return result


def read_observations(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def reconstruct_cell(clean, ids, ratio, correctness, rows, anchored=False, source_cache=None):
    masks, wrong, sources = [], [], []
    for x, row_id in zip(clean.astype(np.int64), ids):
        source = (source_cache[int(row_id)] if source_cache is not None else
                  nearest_different_sources(x, (50256, 50257)))
        canvas, mask, errors, _ = make_canvas(x, ratio, correctness, 20261004, int(row_id),
            None, 50257, (50256,), nearest_sources=source)
        if anchored:
            canvas[0] = x[0]; mask[0] = False; errors[0] = False
        row = rows[int(row_id)]
        if digest(canvas) != row['input_sha256'] or digest(x) != row['clean_sha256'] or digest(mask) != row['mask_sha256']:
            raise ValueError('Saved observations disagree with reconstructed clean/input/mask')
        masks.append(mask); wrong.append(errors); sources.append(source)
    masked, wrong, sources = map(np.stack, (masks, wrong, sources))
    scored = masked.copy(); scored[:, 0] = False
    return scored, exposed_target_mask(masked, wrong, sources), masked, wrong


def verify_source_states(arrays, masked, wrong):
    content = ~np.isin(arrays['clean'], (50256, 50257))
    for direction in ('left', 'right'):
        target = slice(1, None) if direction == 'left' else slice(None, -1)
        source = slice(None, -1) if direction == 'left' else slice(1, None)
        eligible = arrays['scored'][:, target] & content[:, :-1] & content[:, 1:]
        expected = np.full(masked.shape, 255, dtype=np.uint8)
        expected[:, target] = np.where(eligible,
            np.where(masked[:, source], 0, np.where(wrong[:, source], 2, 1)), 255)
        if not np.array_equal(expected, arrays[direction + '_source_state']):
            raise ValueError('Saved auxiliary source states differ from actual input states')
        if not np.array_equal(expected != 255, arrays[direction + '_prediction'] >= 0):
            raise ValueError('Saved auxiliary eligibility differs from the selected content population')


def cell_report(first, second, rows, weights, exposure, frequency_groups, tokenizer, label, output):
    verify_pair(first, second)
    clean, selected, ids = first['clean'], first['scored'], first['row_ids']
    statistics, examples, reports = {}, [], {}
    def pair(name, a, b, pa, pb, choose, ca=None, cb=None, sample=False):
        result, stats, cases = pair_report(clean, a, b, pa, pb, choose, weights, ca, cb)
        result['coverage_of_all_scored_targets'] = float(choose.sum() / selected.sum())
        reports[name] = result
        statistics.update({name + '__' + key: value for key, value in stats.items()})
        if sample:
            examples.extend(illustrations(clean, a, b, choose, cases, ids, tokenizer, label + '/' + name))
        return stats
    populations = dict(all=selected, copied=selected & exposure, not_copied=selected & ~exposure)
    populations.update({name: selected & membership for name, membership in frequency_groups.items()})
    for name, choose in populations.items():
        pair('cross_model_' + name, first['main_prediction'], second['main_prediction'],
             first['main_true_logp'], second['main_true_logp'], choose,
             first.get('main_top1_probability'), second.get('main_top1_probability'), sample=name == 'all')
    if 'left_prediction' in second:
        common = selected & (second['left_prediction'] >= 0) & (second['right_prediction'] >= 0)
        for direction in ('left', 'right'):
            eligible = selected & (second[direction + '_prediction'] >= 0)
            for group, state in [('all', None), ('masked', 0), ('correct_revealed', 1), ('wrong_revealed', 2)]:
                choose = eligible if state is None else eligible & (second[direction + '_source_state'] == state)
                for partition, subset in [('all', choose), ('copied', choose & exposure), ('not_copied', choose & ~exposure)]:
                    name = 'main_' + direction + '_' + group + '_' + partition
                    stats = pair(name, second['main_prediction'], second[direction + '_prediction'],
                        second['main_true_logp'], second[direction + '_true_logp'], subset,
                        second.get('main_top1_probability'), second.get(direction + '_top1_probability'),
                        sample=group == partition == 'all')
                    if group == partition == 'all':
                        prefix = f'head_{direction}_all_'
                        for index, row_id in enumerate(ids):
                            for key in ('targets', *CASES):
                                if prefix + key in rows[int(row_id)]:
                                    assert int(stats[key][index]) == int(float(rows[int(row_id)][prefix + key])), 'Saved head counts disagree'
            pair('common_main_' + direction, second['main_prediction'], second[direction + '_prediction'],
                 second['main_true_logp'], second[direction + '_true_logp'], common)
        pair('common_auxiliary_pair', second['left_prediction'], second['right_prediction'],
             second['left_true_logp'], second['right_true_logp'], common,
             second.get('left_top1_probability'), second.get('right_top1_probability'))
        for partition, choose in [('all', common), ('copied', common & exposure), ('not_copied', common & ~exposure)]:
            result, stats = union_report(clean, second, choose, weights)
            result['coverage_of_all_scored_targets'] = float(choose.sum() / selected.sum())
            name = 'three_head_union_' + partition; reports[name] = result
            statistics.update({name + '__' + key: value for key, value in stats.items()})
        left_masked = second['left_source_state'] == 0
        right_masked = second['right_source_state'] == 0
        mixed = common & ((left_masked & (second['right_source_state'] == 1)) |
                          (right_masked & (second['left_source_state'] == 1)))
        visible_prediction = np.where(left_masked, second['right_prediction'], second['left_prediction'])
        masked_prediction = np.where(left_masked, second['left_prediction'], second['right_prediction'])
        visible_logp = np.where(left_masked, second['right_true_logp'], second['left_true_logp'])
        masked_logp = np.where(left_masked, second['left_true_logp'], second['right_true_logp'])
        pair('mixed_main_visible_source', second['main_prediction'], visible_prediction,
             second['main_true_logp'], visible_logp, mixed)
        pair('mixed_main_masked_source', second['main_prediction'], masked_prediction,
             second['main_true_logp'], masked_logp, mixed)
        pair('mixed_masked_vs_visible_source', masked_prediction, visible_prediction,
             masked_logp, visible_logp, mixed)
    np.savez_compressed(output / (label + '_row_statistics.npz'), row_ids=ids, **statistics)
    return dict(cell=label, rows=len(ids), scored_targets=int(selected.sum()), populations=reports), examples


def paired_change(first_before, first_after, second_before, second_after, selected, weights):
    for a, b in [(first_before, first_after), (second_before, second_after), (first_before, second_before)]:
        verify_pair(a, b)
    # Positive degradation=after CE minus before CE; interaction=NP change minus MDM change.
    base = np.where(selected, first_before['main_true_logp'].astype(float) - first_after['main_true_logp'].astype(float), 0).sum(1)
    np_change = np.where(selected, second_before['main_true_logp'].astype(float) - second_after['main_true_logp'].astype(float), 0).sum(1)
    count = selected.sum(1).astype(float)
    values = dict(mdm_ce_change=base, zero_np_ce_change=np_change, np_minus_mdm_change=np_change - base)
    boot_count = weights @ count
    return dict(targets=int(count.sum()), metrics={name: ratio_summary(v, count, weights @ v, boot_count) for name, v in values.items()})


def verify_source_observations(rows, ids, clean, source_ids):
    clean_by_id = dict(zip(ids.tolist(), clean.astype(np.int64)))
    expected = {}
    for row_id in source_ids:
        for ratio in (.1, .2, .6):
            for c in source_pairs(clean_by_id[row_id], ratio, row_id):
                for variant in VARIANTS:
                    expected[variant, row_id, ratio, c['direction'], c['source_state']] = c
    seen = set()
    for row in rows:
        key = row['variant'], int(row['row_id']), float(row['mask_ratio']), int(row['direction']), row['source_state']
        if key in seen or key not in expected:
            raise ValueError('Duplicate or unexpected source observation')
        seen.add(key); c = expected[key]; x = clean_by_id[int(row['row_id'])]
        if (int(row['target_index']) != c['target_index'] or int(row['source_index']) != c['source_index'] or
                int(row['target_token']) != x[c['target_index']] or row['clean_sha256'] != digest(x) or
                row['input_sha256'] != digest(c['canvas'])):
            raise ValueError('Source observation differs from the registered paired input')
    if seen != expected.keys():
        raise ValueError('Source observations have incomplete case coverage')


def source_report(rows, ids, tokenizer, output, bootstraps):
    """Retain both directions and all source conditions in the same row draws."""
    weights = bootstrap_weights(len(ids), bootstraps)
    lookup = {}
    for row in rows:
        key = row['variant'], int(row['row_id']), float(row['mask_ratio']), int(row['direction']), row['source_state']
        if key in lookup:
            raise ValueError('Duplicate source intervention record')
        lookup[key] = row
    result = []
    for ratio in (.1, .2, .6):
        states = {}
        for state in ('masked', 'revealed'):
            states[state] = {}
            for variant in VARIANTS:
                arrays = dict(row_ids=np.asarray(ids), clean=np.zeros((len(ids), 2), dtype=np.int32),
                    scored=np.zeros((len(ids), 2), dtype=bool), main_prediction=np.zeros((len(ids), 2), dtype=np.int32),
                    main_true_logp=np.full((len(ids), 2), np.nan), main_top1_probability=np.full((len(ids), 2), np.nan))
                if variant != 'mdm':
                    arrays.update(auxiliary_prediction=np.zeros((len(ids), 2), dtype=np.int32),
                        auxiliary_true_logp=np.full((len(ids), 2), np.nan), auxiliary_top1_probability=np.full((len(ids), 2), np.nan))
                for i, row_id in enumerate(ids):
                    for j, direction in enumerate((-1, 1)):
                        key = variant, int(row_id), ratio, direction, state
                        if key not in lookup:
                            continue
                        row = lookup[key]; base = lookup['mdm', int(row_id), ratio, direction, state]
                        for field in ('target_index', 'source_index', 'target_token', 'clean_sha256', 'input_sha256'):
                            if row[field] != base[field]:
                                raise ValueError('Source intervention rows differ between models')
                        arrays['clean'][i, j] = int(row['target_token']); arrays['scored'][i, j] = True
                        for field in ('main_prediction', 'main_true_logp', 'main_top1_probability'):
                            arrays[field][i, j] = float(row[field])
                        if variant != 'mdm':
                            for field in ('auxiliary_prediction', 'auxiliary_true_logp', 'auxiliary_top1_probability'):
                                arrays[field][i, j] = float(row[field])
                states[state][variant] = arrays
            verify_pair(states[state]['mdm'], states[state]['mdm_np_zero_init'])
            for direction in ('both', -1, 1):
                a, b = states[state]['mdm'], states[state]['mdm_np_zero_init']
                choose = a['scored'].copy()
                if direction != 'both':
                    choose[:, 1 if direction == -1 else 0] = False
                report, stats, _ = pair_report(a['clean'], a['main_prediction'], b['main_prediction'],
                    a['main_true_logp'], b['main_true_logp'], choose, weights,
                    a['main_top1_probability'], b['main_top1_probability'])
                aux, aux_stats, _ = pair_report(b['clean'], b['main_prediction'], b['auxiliary_prediction'],
                    b['main_true_logp'], b['auxiliary_true_logp'], choose, weights,
                    b['main_top1_probability'], b['auxiliary_top1_probability'])
                label = f'source_mask{round(ratio * 100):03d}_{state}_{direction}'
                np.savez_compressed(output / (label + '_row_statistics.npz'), row_ids=np.asarray(ids),
                    **{'cross__' + k: v for k, v in stats.items()}, **{'main_auxiliary__' + k: v for k, v in aux_stats.items()})
                result.append(dict(mask_ratio=ratio, source_state=state, direction=direction,
                    cross_model=report, main_auxiliary=aux))
        for direction in ('both', -1, 1):
            visible, masked = states['revealed'], states['masked']
            choose = visible['mdm']['scored'].copy()
            if direction != 'both':
                choose[:, 1 if direction == -1 else 0] = False
            effect = paired_change(visible['mdm'], masked['mdm'], visible['mdm_np_zero_init'],
                masked['mdm_np_zero_init'], choose, weights)
            result.append(dict(mask_ratio=ratio, direction=direction,
                definition='Source reveal benefit = CE(source masked) - CE(source revealed)', paired_benefit=effect))
    return result


def write_tex(path, cells, changes, preflight):
    lines = [r'\section{Paired head errors and context controls}',
        'Collection preflight only.' if preflight else 'Final EMA zero NP versus MDM; one training seed.',
        'All rates use identical masked targets within each comparison. Intervals resample packed rows; they do not estimate training-seed uncertainty.',
        r'\begin{center}\scriptsize', r'\begin{tabular}{l|r|r|r|r}',
        r'Condition & NP--MDM CE & Accuracy gap (pp) & Aux rescue & Oracle accuracy \\ \hline']
    for cell in cells:
        cross = cell['populations']['cross_model_all']['metrics']
        union = cell['populations'].get('three_head_union_all', {}).get('metrics', {})
        def show(metric, percent=False):
            if not metric or metric['value'] is None: return '--'
            scale = 100 if percent else 1
            return f"{metric['value'] * scale:+.3f}" if not percent else f"{metric['value'] * scale:.2f}\\%"
        label = cell['cell'].replace('_', r'\_')
        accuracy = '--' if cross['accuracy_difference']['value'] is None else f"{cross['accuracy_difference']['value'] * 100:+.2f}"
        lines.append(label + ' & ' + ' & '.join([show(cross['ce_difference']), accuracy,
            show(union.get('absolute_rescue'), True), show(union.get('oracle_top1_selector_accuracy'), True)]) + r' \\')
    lines.extend([r'\end{tabular}\end{center}',
        'Auxiliary rescue is the fraction of targets where main is wrong and either auxiliary is correct, on the common two-source population. Oracle accuracy uses labels and is not deployable performance.',
        'The JSON report includes the five error cases, source/exposure strata, matched mixed-source comparisons, soft-score distributions and intervals. Token-frequency strata count occurrences in the fixed monitoring cohort; they are exploratory, not training-corpus frequencies.',
        'Correct versus incorrect source classes are descriptive unless explicitly paired. Wrong-context exposure comparisons do not isolate the causal effect of copying.'])
    atomic_write(path, '\n'.join(lines) + '\n')


def plot_reports(output, cells, sources, preflight=False):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/runtime/analysis/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    names = [f'mask{m:03d}_correct100' for m in (100, 80, 60, 40, 20)] + ['mask10_unclamped']
    lookup = {c['cell']: c for c in cells}
    conditions = [lookup[name] for name in names]
    x = np.arange(len(conditions))
    def curve(ax, metrics, color, label, scale=1):
        indices = [i for i, m in enumerate(metrics) if m['value'] is not None]
        if not indices:
            return
        y = np.asarray([metrics[i]['value'] * scale for i in indices])
        intervals = np.asarray([metrics[i]['row_bootstrap_95'] for i in indices]) * scale
        error = np.maximum(0, np.array([y - intervals[:, 0], intervals[:, 1] - y]))
        ax.errorbar(np.asarray(indices), y, yerr=error, fmt='o-', color=color, label=label, capsize=3, lw=1.5)
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.5))
    curve(axes[0], [c['populations']['cross_model_all']['metrics']['ce_difference'] for c in conditions], '#2166ac', 'Zero NP minus MDM')
    axes[0].axhline(0, color='#777777', ls='--', lw=1)
    axes[0].set(title='Main CE gap', ylabel='CE difference (nats/masked target)')
    for name, color, label in [('common_main_left', '#d95f0e', 'Left source'),
                                ('common_main_right', '#238b45', 'Right source'),
                                ('three_head_union_all', '#756bb1', 'Either auxiliary')]:
        curve(axes[1], [c['populations'][name]['metrics']['absolute_rescue'] for c in conditions], color, label, 100)
    axes[1].set(title='Auxiliary rescue on common targets', ylabel='Main wrong, auxiliary correct (%)')
    for direction, color in [('left', '#d95f0e'), ('right', '#238b45')]:
        curve(axes[2], [c['populations']['common_main_' + direction]['metrics']['ce_difference'] for c in conditions], color, direction.title() + ' source')
    axes[2].axhline(0, color='#777777', ls='--', lw=1)
    axes[2].set(title='Auxiliary CE minus main CE', ylabel='CE difference (nats/common target)')
    for ax in axes:
        ax.set(xticks=x, xticklabels=['100', '80', '60', '40', '20', '10'], xlabel='Masked inputs (%)')
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Synthetic report preflight: not research evidence' if preflight else
                 'Fully correct revealed context; final EMA checkpoints', fontsize=12)
    fig.text(.5, .012, 'Paired packed-row 95% intervals; one training seed. Rescue/readout curves share the two-source eligible population.\n'
        'Oracle selection is not deployed; native auxiliary readouts do not prove downstream use.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, .94))
    for extension in ('pdf', 'png'):
        fig.savefig(output / ('head_complementarity.' + extension), dpi=180)
    plt.close(fig)
    paired = {s['mask_ratio']: s['paired_benefit']['metrics'] for s in sources if s.get('direction') == 'both' and 'paired_benefit' in s}
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.4))
    if preflight:
        fig.suptitle('Synthetic report preflight: not research evidence', fontsize=11)
    for metric, color, label in [('mdm_ce_change', '#555555', 'MDM'), ('zero_np_ce_change', '#2166ac', 'Zero NP')]:
        curve(axes[0], [paired[r][metric] for r in (.1, .2, .6)], color, label)
    curve(axes[1], [paired[r]['np_minus_mdm_change'] for r in (.1, .2, .6)], '#2166ac', 'Zero NP minus MDM benefit')
    for ax in axes:
        ax.axhline(0, color='#777777', ls='--', lw=1)
        ax.set(xticks=range(3), xticklabels=['10', '20', '60'], xlabel='Background masked inputs (%)')
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
    axes[0].set(title='Benefit from revealing the same source', ylabel='CE(masked source) minus CE(revealed source)')
    axes[1].set(title='Difference in source-reveal benefit', ylabel='NP benefit minus MDM benefit')
    fig.text(.5, .012, 'Directions remain paired within each row; input zero is clean. Intervals resample rows, not seeds.\n'
        'A frozen input intervention does not identify the responsible training component.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .085, 1, 1))
    for extension in ('pdf', 'png'):
        fig.savefig(output / ('source_reveal_benefit.' + extension), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, default=ROOT / 'outputs/analysis/owt-reveal-sweep-5000')
    parser.add_argument('--supplement', type=Path, default=ROOT / 'outputs/analysis/owt-post-diagnostics-5000')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/analysis/owt-head-complementarity-5000')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--bootstraps', type=int, default=2000)
    args = parser.parse_args()
    if args.bootstraps < 1 or (not args.preflight and args.bootstraps != 2000):
        parser.error('Final reports require 2,000 paired row-bootstrap draws')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'summary.json').exists():
        parser.error('Completed report exists; use a fresh directory')
    core, supplement = read_json(args.core / 'summary.json'), read_json(args.supplement / 'summary.json')
    if not core or not supplement:
        parser.error('Both core and supplement collections must be complete')
    if core['protocol']['variants'] != VARIANTS or supplement['protocol']['variants'] != VARIANTS:
        parser.error('Reports require zero NP versus MDM only')
    ids = np.asarray(core['protocol']['row_ids'])
    if not np.array_equal(ids, supplement['protocol']['row_ids']) or (not args.preflight and (len(ids) != 1024 or supplement['protocol']['preflight'])):
        parser.error('Final report requires matched full cohorts and a final supplement')
    if core['protocol']['seed'] != 20261004 or core['protocol']['anchor'] != 'none':
        parser.error('Unexpected core corruption protocol')
    if not args.preflight:
        for collection in (core, supplement):
            if (collection['protocol']['optimizer_step'] != 5000 or collection['protocol']['parameter_state'] != 'EMA'
                    or 'FP32' not in collection['protocol']['precision']):
                parser.error('Final report requires 5,000-step FP32 EMA collections')
        expected_sources = sorted(np.random.default_rng(20261005).choice(ids, 128, replace=False).tolist())
        if supplement['protocol']['source_row_ids'] != expected_sources:
            parser.error('Prespecified 128-row source cohort changed')
    for variant in VARIANTS:
        for field in ('checkpoint', 'checkpoint_size', 'checkpoint_mtime_ns', 'config_sha256', 'parameter_state', 'optimizer_step'):
            if core['provenance'][variant][field] != supplement['provenance'][variant][field]:
                parser.error('Core and supplement checkpoint identities differ')
    tokenizer, tokenizer_path = cached_tokenizer()
    weights = bootstrap_weights(len(ids), args.bootstraps)
    core_rows = read_observations(args.core / 'observations.csv')
    near_rows = read_observations(args.supplement / 'near_clean_observations.csv')
    stats_dir = args.output / 'row_statistics'; stats_dir.mkdir(exist_ok=True)
    cells, examples, changes, reliable = [], [], [], {}
    frequency_groups = None
    canonical_clean, source_cache = None, None
    for ratio in (1., .8, .6, .4, .2):
        for correctness in (1., .8, .6):
            label = f'mask{round(ratio * 100):03d}_correct{round(correctness * 100):03d}'
            arrays = {v: load_predictions(args.core / 'head_predictions' / f'{v}_{label}.npz') for v in VARIANTS}
            verify_pair(arrays['mdm'], arrays['mdm_np_zero_init'])
            if not np.array_equal(arrays['mdm']['row_ids'], ids):
                raise ValueError('Core row population changed')
            rows = {v: {int(r['row_id']): r for r in core_rows if r['variant'] == v and
                float(r['mask_ratio']) == ratio and float(r['correct_fraction']) == correctness} for v in VARIANTS}
            if canonical_clean is None:
                canonical_clean = arrays['mdm']['clean']
                source_cache = {int(i): nearest_different_sources(x.astype(np.int64), (50256, 50257)) for i, x in zip(ids, canonical_clean)}
            elif not np.array_equal(arrays['mdm']['clean'], canonical_clean):
                raise ValueError('Clean-token cohort changed between conditions')
            scored, exposed, actual_mask, actual_wrong = reconstruct_cell(arrays['mdm']['clean'], ids, ratio, correctness, rows['mdm'], source_cache=source_cache)
            if not np.array_equal(scored, arrays['mdm']['scored']):
                raise ValueError('Reconstructed scored targets differ')
            verify_source_states(arrays['mdm_np_zero_init'], actual_mask, actual_wrong)
            for row_id in ids:
                assert rows['mdm'][int(row_id)]['input_sha256'] == rows['mdm_np_zero_init'][int(row_id)]['input_sha256']
            for variant in VARIANTS:
                for k, row_id in enumerate(ids):
                    r = rows[variant][int(row_id)]; a = arrays[variant]
                    if (int(float(r['masked_targets'])) != int(scored[k].sum()) or
                            int(float(r['masked_correct_count'])) != int(((a['main_prediction'][k] == a['clean'][k]) & scored[k]).sum()) or
                            abs(float(r['masked_ce_sum']) + a['main_true_logp'][k, scored[k]].astype(float).sum()) > 1e-8):
                        raise ValueError('Saved predictions do not reconstruct primary observations')
            if frequency_groups is None:
                frequency_groups = token_strata(arrays['mdm']['clean'], tokenizer)
            cell, sampled = cell_report(arrays['mdm'], arrays['mdm_np_zero_init'], rows['mdm_np_zero_init'],
                weights, exposed, frequency_groups, tokenizer, label, stats_dir)
            cells.append(cell); examples.extend(sampled)
            if correctness == 1.:
                reliable[ratio] = arrays
            else:
                reference = reliable[ratio]
                for partition, choose in [('all', scored), ('copied', scored & exposed), ('not_copied', scored & ~exposed)]:
                    changes.append(dict(cell=label, intervention='wrong_reveals', partition=partition,
                        **paired_change(reference['mdm'], arrays['mdm'], reference['mdm_np_zero_init'],
                                        arrays['mdm_np_zero_init'], choose, weights)))
            atomic_write(args.output / 'progress.json', json.dumps(dict(completed_cells=len(cells), cell=label)) + '\n')
    near_arrays = {}
    for condition in ('mask10_unclamped', 'mask10_first_clean', 'mask20_unclamped', 'mask20_first_clean'):
        arrays = {v: load_predictions(args.supplement / 'head_predictions' / f'{v}_{condition}.npz') for v in VARIANTS}
        verify_pair(arrays['mdm'], arrays['mdm_np_zero_init'])
        if not np.array_equal(arrays['mdm']['row_ids'], ids):
            raise ValueError('Supplement row population changed')
        rows = {v: {int(r['row_id']): r for r in near_rows if r['variant'] == v and r['condition'] == condition} for v in VARIANTS}
        ratio = .1 if condition.startswith('mask10') else .2
        if not np.array_equal(arrays['mdm']['clean'], canonical_clean):
            raise ValueError('Supplement clean tokens differ from the core')
        scored, exposed, actual_mask, actual_wrong = reconstruct_cell(arrays['mdm']['clean'], ids, ratio, 1., rows['mdm'], anchored=condition.endswith('first_clean'), source_cache=source_cache)
        if not np.array_equal(scored, arrays['mdm']['scored']):
            raise ValueError('Supplement scored targets differ')
        verify_source_states(arrays['mdm_np_zero_init'], actual_mask, actual_wrong)
        for row_id in ids:
            assert rows['mdm'][int(row_id)]['input_sha256'] == rows['mdm_np_zero_init'][int(row_id)]['input_sha256']
        cell, sampled = cell_report(arrays['mdm'], arrays['mdm_np_zero_init'], rows['mdm_np_zero_init'],
            weights, exposed, frequency_groups, tokenizer, condition, stats_dir)
        cells.append(cell); examples.extend(sampled); near_arrays[condition] = arrays
    for ratio in (10, 20):
        before, after = near_arrays[f'mask{ratio}_unclamped'], near_arrays[f'mask{ratio}_first_clean']
        changes.append(dict(cell=f'mask{ratio}_first_input_control', intervention='first_input_clean', partition='all',
            **paired_change(before['mdm'], after['mdm'], before['mdm_np_zero_init'], after['mdm_np_zero_init'], before['mdm']['scored'], weights)))
    source_rows = read_observations(args.supplement / 'source_observations.csv')
    verify_source_observations(source_rows, ids, canonical_clean, supplement['protocol']['source_row_ids'])
    sources = source_report(source_rows, supplement['protocol']['source_row_ids'], tokenizer, stats_dir, args.bootstraps)
    artifact = dict(variants=VARIANTS, row_ids=ids.tolist(), cells=cells, matched_context_changes=changes,
        source_interventions=sources,
        preflight=args.preflight, bootstrap=dict(draws=args.bootstraps, seed=20261005, unit='paired packed row',
            shared_across_heads_directions_conditions=True, training_seed_uncertainty=False),
        frequency_definition='Clean-token occurrence count in the same fixed monitoring cohort; not training frequency',
        tokenizer=dict(path=str(tokenizer_path), sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()),
        evidence=dict(core=str(args.core), supplement=str(args.supplement)),
        checkpoint_provenance=core['provenance'],
        collection_fingerprints={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (args.core / 'summary.json', args.core / 'observations.csv', args.supplement / 'summary.json',
             args.supplement / 'near_clean_observations.csv', args.supplement / 'source_observations.csv')},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['One training seed; row intervals do not account for document dependence or repeated inspection.',
            'Source strata have different target populations unless explicitly mixed-source matched.',
            'Diversity alone does not establish usefulness. Oracle selection uses labels.',
            'Soft-score histograms and quantiles are descriptive token distributions.'],
        example_policy='At most three seeded examples per five-case category on selected all-target populations; not chosen by loss')
    atomic_write(args.output / 'examples.json', json.dumps(examples, indent=2, allow_nan=False) + '\n')
    plot_reports(args.output, cells, sources, args.preflight)
    write_tex(args.output / 'report.tex', cells, changes, args.preflight)
    atomic_write(args.output / 'summary.json', json.dumps(artifact, indent=2, allow_nan=False) + '\n')
    record_event('paired_error_report_' + hashlib.sha256(json.dumps(artifact['evidence'], sort_keys=True).encode()).hexdigest()[:12],
        'Paired prediction error report generated',
        f'Reported {len(cells)} core/supplement conditions for zero NP versus MDM on {len(ids)} paired rows; '
        f'preflight={args.preflight}. Included five-case errors, source/exposure strata, native head rescue and oracle '
        'coverage, matched mixed-source comparisons, soft-score distributions, exploratory cohort-token strata '
        'and shared paired-row bootstrap intervals. No model forward pass or training decision was made.', dict(output=str(args.output)))
    print('COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
