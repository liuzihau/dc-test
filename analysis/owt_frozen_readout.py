"""Registered offline readout check: saved scores only, no model forwards."""
import argparse
import hashlib
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np

from analysis.owt_error_report import bootstrap_weights, load_predictions, verify_pair, ratio_summary
from owt.research import ROOT, atomic_write, record_event, timestamp

HEADS = ('main', 'left', 'right')


def mixed_logp(logp, weight):
    """True-token probability of the convex mixture; last dimension=heads."""
    logp = np.asarray(logp, dtype=float)
    if logp.shape[-1] != 3 or not 0 <= weight <= 1:
        raise ValueError('Expected three heads and a convex weight')
    if weight == 0:
        return logp[..., 0].copy()
    auxiliary = np.logaddexp(logp[..., 1], logp[..., 2]) - np.log(2.)
    if weight == 1:
        return auxiliary
    return np.logaddexp(np.log1p(-weight) + logp[..., 0], np.log(weight) + auxiliary)


def fit_weight(conditions):
    """One convex weight, equal condition weights, target weighting within each."""
    scaled = []
    for scores in conditions:
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2 or scores.shape[1] != 3 or not len(scores) or not np.isfinite(scores).all():
            raise ValueError('Calibration condition must contain finite three-head scores')
        main = scores[:, 0]
        auxiliary = np.logaddexp(scores[:, 1], scores[:, 2]) - np.log(2.)
        maximum = np.maximum(main, auxiliary)
        scaled.append((np.exp(main-maximum), np.exp(auxiliary-maximum)))

    if not scaled:
        raise ValueError('No calibration conditions')

    def derivative(weight):
        with np.errstate(divide='ignore', invalid='raise'):
            return float(np.mean([-np.mean((a-m)/((1-weight)*m+weight*a)) for m, a in scaled]))

    left, right = derivative(0), derivative(1)
    if left >= 0:
        weight = 0.
    elif right <= 0:
        weight = 1.
    else:
        low, high = 0., 1.
        for _ in range(60):
            middle = (low+high)/2
            if derivative(middle) < 0:
                low = middle
            else:
                high = middle
        weight = (low+high)/2
    baseline = float(np.mean([-np.mean(c[:, 0]) for c in conditions]))
    objective = float(np.mean([-np.mean(mixed_logp(c, weight)) for c in conditions]))
    if objective > baseline + 1e-10:
        raise RuntimeError('Convex calibration worsened feasible main-only endpoint')
    return dict(weight=weight, equal_condition_ce=objective, main_only_ce=baseline,
                derivative_at_solution=derivative(weight), derivative_at_zero=left,
                derivative_at_one=right, conditions=len(conditions))


def confidence_route(confidence):
    """No labels; np.argmax ties select main, then left, then right."""
    confidence = np.asarray(confidence)
    if confidence.shape[-1] != 3 or not np.isfinite(confidence).all():
        raise ValueError('Expected finite three-head confidence')
    return np.argmax(confidence, axis=-1)


def row_statistics(mdm, np_arm, indices, weight):
    clean = np_arm['clean'][indices]
    selected = (np_arm['scored'][indices] & (np_arm['left_prediction'][indices] >= 0)
                & (np_arm['right_prediction'][indices] >= 0))
    # Invalid, excluded positions receive dummy finite scores before arithmetic.
    logp = np.stack([np_arm[h+'_true_logp'][indices] for h in HEADS], -1).astype(float)
    confidence = np.stack([np_arm[h+'_top1_probability'][indices] for h in HEADS], -1)
    predictions = np.stack([np_arm[h+'_prediction'][indices] for h in HEADS], -1)
    logp = np.where(selected[..., None], logp, 0.)
    confidence = np.where(selected[..., None], confidence, 0.)
    chosen = confidence_route(confidence)
    chosen_logp = np.take_along_axis(logp, chosen[..., None], -1)[..., 0]
    chosen_prediction = np.take_along_axis(predictions, chosen[..., None], -1)[..., 0]
    sums = dict(targets=selected.sum(1).astype(float),
        mdm_ce_sum=np.where(selected, -mdm['main_true_logp'][indices].astype(float), 0).sum(1),
        np_main_ce_sum=np.where(selected, -logp[..., 0], 0).sum(1),
        mixture_ce_sum=np.where(selected, -mixed_logp(logp, weight), 0).sum(1),
        selector_ce_sum=np.where(selected, -chosen_logp, 0).sum(1),
        mdm_correct=((mdm['main_prediction'][indices] == clean) & selected).sum(1).astype(float),
        np_main_correct=((predictions[..., 0] == clean) & selected).sum(1).astype(float),
        selector_correct=((chosen_prediction == clean) & selected).sum(1).astype(float))
    for k, head in enumerate(HEADS):
        sums['selected_'+head] = ((chosen == k) & selected).sum(1).astype(float)
    for comparator in ('np_main', 'mdm'):
        for readout in ('mixture', 'selector'):
            sums[f'{readout}_minus_{comparator}_ce_sum'] = sums[readout+'_ce_sum']-sums[comparator+'_ce_sum']
        sums['selector_minus_'+comparator+'_correct'] = sums['selector_correct']-sums[comparator+'_correct']
    return sums


def summarize_rows(stats, weights):
    keys = list(stats)
    boot = dict(zip(keys, (weights @ np.column_stack([stats[k] for k in keys])).T))
    metrics = {}
    for name in keys:
        if name == 'targets':
            continue
        metric = name.removesuffix('_sum')
        metrics[metric] = ratio_summary(stats[name], stats['targets'], boot[name], boot['targets'])
    return dict(targets=int(stats['targets'].sum()), rows_with_targets=int((stats['targets'] > 0).sum()),
                metrics=metrics), boot


def macro_report(primary_stats, weights):
    reports = [summarize_rows(s, weights) for s in primary_stats]
    if not reports or any(r['targets'] <= 0 for r, _ in reports):
        raise ValueError('Missing primary condition targets')
    result = {}
    for key in reports[0][1]:
        if key == 'targets':
            continue
        point = np.mean([s[key].sum()/s['targets'].sum() for s in primary_stats])
        draws = np.mean([b[key]/b['targets'] for _, b in reports], axis=0)
        if not np.isfinite(draws).all():
            raise ValueError('Empty primary bootstrap condition')
        result[key.removesuffix('_sum')] = dict(value=float(point),
            row_bootstrap_95=np.quantile(draws, [.025, .975]).tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'outputs/research-notes/frozen_readout_protocol_20261001.json')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    output = ROOT/protocol['output']
    if output.exists():
        parser.error('Output exists; refuse duplicate fitting/scoring')
    calibration_ids = np.asarray(protocol['split']['calibration_row_ids'])
    scoring_ids = np.asarray(protocol['split']['scoring_row_ids'])
    if len(calibration_ids) != 512 or len(scoring_ids) != 512 or set(calibration_ids)&set(scoring_ids):
        parser.error('Invalid registered row split')
    expected_ids = np.sort(np.concatenate([calibration_ids, scoring_ids]))

    def load(cell):
        arrays = {}
        for variant, filename in cell['files'].items():
            path = ROOT/filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != cell['sha256'][variant]:
                raise ValueError('Registered input fingerprint changed')
            arrays[variant] = load_predictions(path)
            if not np.array_equal(arrays[variant]['row_ids'], expected_ids):
                raise ValueError('Input cohort differs from registered split')
        verify_pair(arrays['mdm'], arrays['mdm_np_zero_init'])
        return arrays['mdm'], arrays['mdm_np_zero_init']

    calibration = []
    for label in protocol['primary_conditions']:
        cell = next(c for c in protocol['inputs'] if c['cell'] == label)
        _, np_arm = load(cell)
        indices = np.searchsorted(expected_ids, calibration_ids)
        chosen = (np_arm['scored'][indices] & (np_arm['left_prediction'][indices] >= 0)
                  & (np_arm['right_prediction'][indices] >= 0))
        scores = np.stack([np_arm[h+'_true_logp'][indices] for h in HEADS], -1).astype(float)
        calibration.append(scores[chosen])
    fit = fit_weight(calibration)
    del calibration
    output.mkdir(parents=True)
    atomic_write(output/'fit.json', json.dumps(dict(recorded_at=timestamp(), **fit), indent=2)+'\n')
    scoring_indices = np.searchsorted(expected_ids, scoring_ids)
    boot = protocol['bootstrap']
    weights = bootstrap_weights(len(scoring_ids), boot['draws'], boot['seed'])
    cells, primary = [], []
    for cell in protocol['inputs']:
        mdm, np_arm = load(cell)
        stats = row_statistics(mdm, np_arm, scoring_indices, fit['weight'])
        np.savez_compressed(output/(cell['cell']+'_row_statistics.npz'), row_ids=scoring_ids, **stats)
        result, _ = summarize_rows(stats, weights)
        cells.append(dict(cell=cell['cell'], primary=cell['primary'], **result))
        if cell['primary']:
            primary.append(stats)
    if len(primary) != 5:
        raise ValueError('Primary condition list changed')
    summary = dict(recorded_at=timestamp(), protocol=str(args.protocol.relative_to(ROOT)),
        protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_forward_passes=0, calibration=fit, scoring_rows=len(scoring_ids),
        macro_definition='equal mean of target-weighted scores across five registered correct-context conditions',
        primary_macro=macro_report(primary, weights), cells=cells, limits=protocol['limits'],
        bootstrap=boot, mixture_accuracy=None)
    # JSON disallows nonfinite derivatives; real endpoint values are checked.
    atomic_write(output/'summary.json', json.dumps(summary, indent=2, allow_nan=False)+'\n')
    record_event('frozen_readout_5000_complete_20261001', 'Registered frozen readout completed',
        f"One weight fitted on 512 calibration rows: lambda={fit['weight']:.9f}; scored on 512 disjoint rows. "
        'No model forwards or new training. Monitoring-cohort results remain exploratory.',
        dict(output=str(output.relative_to(ROOT)), primary_macro=summary['primary_macro']))
    print(json.dumps(dict(calibration=fit, primary_macro=summary['primary_macro']), indent=2))


if __name__ == '__main__':
    main()
