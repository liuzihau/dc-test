"""Visibility strata, disjoint error classes, and paired correction accounting."""
import numpy as np

OFFSETS = (-2, -1, 1, 2)
ERROR_FIELDS = ('eligible', 'wrong', 'out_left', 'out_right', 'in_left', 'in_right',
                'in_both', 'in_union', 'swap_left', 'swap_right', 'swap',
                'nonreciprocal', 'other')


def make_canvas(clean, document_ids, seed, probability, mask_id=50257, special_ids=(50256,)):
    clean = np.asarray(clean)
    if clean.ndim != 2 or len(document_ids) != len(clean) or not 0 < probability < 1:
        raise ValueError('Expected rows, document IDs, and mask probability in (0,1)')
    content = ~np.isin(clean, special_ids)
    if (clean == mask_id).any():
        raise ValueError('Ground truth contains a mask token')
    masked = np.zeros_like(clean, dtype=bool)
    for row, doc in enumerate(document_ids):
        rng = np.random.default_rng(np.random.SeedSequence([20261006, int(seed), int(doc)]))
        masked[row] = (rng.random(clean.shape[1]) < probability) & content[row]
    canvas = np.where(masked, mask_id, clean)
    return canvas, masked


def eligible_centers(clean, radius, block_size, special_ids=(50256,)):
    clean = np.asarray(clean)
    if clean.ndim != 2 or block_size < 1 or radius < 1:
        raise ValueError('Invalid neighborhood geometry')
    content = ~np.isin(clean, special_ids)
    centers = np.arange(radius, clean.shape[1] - radius)
    eligible = np.zeros_like(content)
    for i in centers:
        if (i-radius) // block_size == (i+radius) // block_size:
            eligible[:, i] = content[:, i-radius:i+radius+1].all(1)
    return eligible


def visibility_statistics(clean, masked, prediction, nll, block_size=1024, special_ids=(50256,)):
    clean, masked, prediction, nll = map(np.asarray, (clean, masked, prediction, nll))
    if any(x.shape != clean.shape for x in (masked, prediction, nll)):
        raise ValueError('Prediction/canvas geometry differs')
    eligible = eligible_centers(clean, 2, block_size, special_ids) & masked
    patterns = np.zeros_like(clean, dtype=np.int8)
    for offset, bit in zip(OFFSETS, (8, 4, 2, 1)):
        patterns[:, 2:-2] += (~masked[:, 2+offset:clean.shape[1]-2+offset]).astype(np.int8)*bit
    stats = np.zeros((len(clean), 16, 3), dtype=np.float64)
    for pattern in range(16):
        selected = eligible & (patterns == pattern)
        stats[:, pattern, 0] = selected.sum(1)
        stats[:, pattern, 1] = (selected & (prediction == clean)).sum(1)
        stats[:, pattern, 2] = np.where(selected, nll, 0).sum(1)
    if not np.isfinite(stats).all():
        raise ValueError('Nonfinite selected-target losses')
    return stats


def error_events(clean, masked, prediction, block_size=1024, special_ids=(50256,)):
    clean, masked, prediction = map(np.asarray, (clean, masked, prediction))
    if clean.shape != masked.shape or clean.shape != prediction.shape:
        raise ValueError('Prediction/canvas geometry differs')
    eligible = eligible_centers(clean, 1, block_size, special_ids)
    eligible[:, 1:-1] &= masked[:, :-2] & masked[:, 1:-1] & masked[:, 2:]
    eligible[:, 1:-1] &= ((clean[:, :-2] != clean[:, 1:-1]) &
                          (clean[:, 2:] != clean[:, 1:-1]) & (clean[:, :-2] != clean[:, 2:]))
    wrong = eligible & (prediction != clean)
    events = {'eligible': eligible, 'wrong': wrong}
    for direction, offset in (('left', -1), ('right', 1)):
        outgoing, incoming = np.zeros_like(eligible), np.zeros_like(eligible)
        outgoing[:, 1:-1] = prediction[:, 1:-1] == clean[:, 1+offset:clean.shape[1]-1+offset]
        incoming[:, 1:-1] = prediction[:, 1+offset:clean.shape[1]-1+offset] == clean[:, 1:-1]
        events['out_'+direction] = outgoing & wrong
        events['in_'+direction] = incoming & wrong
        events['swap_'+direction] = outgoing & incoming & wrong
    events['in_both'] = events['in_left'] & events['in_right']
    events['in_union'] = events['in_left'] | events['in_right']
    events['swap'] = events['swap_left'] | events['swap_right']
    substitution = events['out_left'] | events['out_right']
    events['nonreciprocal'] = substitution & ~events['swap']
    events['other'] = wrong & ~substitution
    assert np.array_equal(events['swap'] | events['nonreciprocal'] | events['other'], wrong)
    return events


def error_statistics(clean, masked, prediction, **kwargs):
    events = error_events(clean, masked, prediction, **kwargs)
    return np.stack([events[name].sum(1) for name in ERROR_FIELDS], axis=1)


def paired_transitions(clean, masked, baseline, candidate, **kwargs):
    left = error_events(clean, masked, baseline, **kwargs)
    right = error_events(clean, masked, candidate, **kwargs)
    eligible = left['eligible']
    assert np.array_equal(eligible, right['eligible'])
    left_classes = [eligible & ~left['wrong'], left['swap'], left['nonreciprocal'], left['other']]
    right_classes = [eligible & ~right['wrong'], right['swap'], right['nonreciprocal'], right['other']]
    return np.stack([np.stack([(a & b).sum(1) for b in right_classes], axis=1)
                     for a in left_classes], axis=1)


def bootstrap_ratio(numerator, denominator, draws):
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    total = denominator.sum()
    if total == 0:
        return {'value': None, 'ci95': None, 'denominator': 0}
    dn = denominator[draws].sum(1)
    boot = np.divide(numerator[draws].sum(1), dn, out=np.full(len(draws), np.nan), where=dn > 0)
    finite = boot[np.isfinite(boot)]
    return {'value': float(numerator.sum()/total),
            'ci95': np.quantile(finite, [.025, .975]).tolist() if finite.size else None,
            'denominator': int(total)}


def summarize(visibility, errors, bootstrap_seed=20261006, bootstraps=2000):
    """Pool tokens while resampling documents, with all corruption draws together."""
    visibility, errors = np.asarray(visibility), np.asarray(errors)
    # Inputs retain (seed, document, condition, metric) axes.
    vis, err = visibility.sum(0), errors.sum(0)
    draws = np.random.default_rng(bootstrap_seed).integers(0, len(vis), size=(bootstraps, len(vis)))
    patterns = []
    for c in range(16):
        patterns.append({'pattern': format(c, '04b'), 'revealed_neighbors': c.bit_count() if hasattr(c, 'bit_count') else bin(c).count('1'),
            'count': int(vis[:, c, 0].sum()),
            'accuracy': bootstrap_ratio(vis[:, c, 1], vis[:, c, 0], draws),
            'nll': bootstrap_ratio(vis[:, c, 2], vis[:, c, 0], draws)})
    e = {name: err[:, j] for j, name in enumerate(ERROR_FIELDS)}
    categories = {}
    for name in ERROR_FIELDS[2:]:
        categories[name] = {'count': int(e[name].sum()),
            'among_wrong': bootstrap_ratio(e[name], e['wrong'], draws),
            'among_eligible': bootstrap_ratio(e[name], e['eligible'], draws)}
    return {'bootstrap_unit': 'independent document, retaining all corruption seeds',
            'bootstrap_draws': bootstraps, 'visibility': patterns,
            'visibility_overall': {'accuracy': bootstrap_ratio(vis[:, :, 1].sum(1), vis[:, :, 0].sum(1), draws),
                                   'nll': bootstrap_ratio(vis[:, :, 2].sum(1), vis[:, :, 0].sum(1), draws)},
            'error_eligible': int(e['eligible'].sum()), 'wrong_centers': int(e['wrong'].sum()),
            'center_error_rate': bootstrap_ratio(e['wrong'], e['eligible'], draws), 'error_events': categories}
