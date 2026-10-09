import numpy as np
import pytest

from analysis.local_denoising_metrics import (make_canvas, visibility_statistics,
    error_events, eligible_centers, paired_transitions, bootstrap_ratio, summarize,
    error_statistics)


def fixture_rows(n=1):
    return np.tile(np.array([99, 10, 11, 12, 13, 14, 99]), (n, 1))


@pytest.mark.parametrize('pattern', range(16))
def test_all_visibility_codes_have_the_expected_center(pattern):
    clean = fixture_rows()
    masked = np.ones_like(clean, dtype=bool)
    for index, bit in zip((1, 2, 4, 5), (8, 4, 2, 1)):
        masked[:, index] = not bool(pattern & bit)
    stats = visibility_statistics(clean, masked, clean, np.full(clean.shape, 2.),
                                  block_size=7, special_ids=(99,))
    assert stats[0, pattern].tolist() == [1, 1, 2]
    assert stats[0, :, 0].sum() == 1


def test_masks_are_paired_by_document_and_preserve_special_tokens():
    clean = fixture_rows(4)
    ids = np.array([50, 60, 70, 80])
    x, masked = make_canvas(clean, ids, 1, .5, mask_id=100, special_ids=(99,))
    order = np.array([3, 0, 2, 1])
    reordered, _ = make_canvas(clean[order], ids[order], 1, .5, mask_id=100, special_ids=(99,))
    assert np.array_equal(reordered, x[order])
    assert not masked[:, [0, -1]].any()
    assert np.array_equal(x[~masked], clean[~masked])
    assert (x[masked] == 100).all()


@pytest.mark.parametrize('kind', ('swap_left', 'swap_right', 'nonreciprocal', 'other', 'correct'))
def test_error_classes_and_reciprocal_direction(kind):
    clean = fixture_rows()
    prediction = clean.copy()
    if kind == 'swap_left':
        prediction[0, 3], prediction[0, 2] = 11, 12
    elif kind == 'swap_right':
        prediction[0, 3], prediction[0, 4] = 13, 12
    elif kind == 'nonreciprocal':
        prediction[0, 3] = 11
    elif kind == 'other':
        prediction[0, 3] = 88
    events = error_events(clean, np.ones_like(clean, dtype=bool), prediction,
                          block_size=7, special_ids=(99,))
    assert events['eligible'][0, 3]
    assert events['wrong'][0, 3] == (kind != 'correct')
    assert events['swap'][0, 3] == kind.startswith('swap')
    assert events['nonreciprocal'][0, 3] == (kind == 'nonreciprocal')
    assert events['other'][0, 3] == (kind == 'other')
    if kind.startswith('swap'):
        assert events[kind][0, 3]


def test_incoming_overlap_is_counted_once_and_is_not_a_swap():
    clean = fixture_rows()
    prediction = clean.copy()
    prediction[0, 2:5] = [12, 88, 12]
    e = error_events(clean, np.ones_like(clean, dtype=bool), prediction,
                     block_size=7, special_ids=(99,))
    assert e['in_left'][0, 3] and e['in_right'][0, 3]
    assert e['in_union'][0, 3] and e['in_both'][0, 3]
    assert not e['swap'][0, 3]
    assert e['other'][0, 3]


@pytest.mark.parametrize('pair', ((2,3), (3,4), (2,4)))
def test_any_repeated_true_id_excludes_the_triple(pair):
    clean = fixture_rows()
    clean[0, pair[1]] = clean[0, pair[0]]
    e = error_events(clean, np.ones_like(clean, dtype=bool), clean,
                     block_size=7, special_ids=(99,))
    assert not e['eligible'][0, 3]


def test_revealed_neighbor_and_block_crossing_are_excluded():
    clean = fixture_rows()
    masked = np.ones_like(clean, dtype=bool)
    masked[0, 2] = False
    e = error_events(clean, masked, clean, block_size=7, special_ids=(99,))
    assert not e['eligible'][0, 3]
    eligible = eligible_centers(clean, 1, block_size=3, special_ids=(99,))
    assert not eligible[0, 2] and not eligible[0, 3]


def test_paired_gain_sum_and_wrong_to_wrong_migration():
    clean = fixture_rows(2)
    masked = np.ones_like(clean, dtype=bool)
    baseline, candidate = clean.copy(), clean.copy()
    baseline[0, 3], baseline[0, 2] = 11, 12
    # One baseline error becomes another error: no accuracy gain for that center.
    baseline[1, 3], candidate[1, 3] = 88, 11
    table = paired_transitions(clean, masked, baseline, candidate,
                               block_size=7, special_ids=(99,))
    gains = table[:, 1:, 0].sum(0) - table[:, 0, 1:].sum(0)
    e0 = error_events(clean, masked, baseline, block_size=7, special_ids=(99,))
    e1 = error_events(clean, masked, candidate, block_size=7, special_ids=(99,))
    assert gains.sum() == e0['wrong'].sum() - e1['wrong'].sum()
    assert table[1, 3, 2] >= 1


def test_undefined_ratio_and_empty_strata_are_explicit():
    draws = np.zeros((10, 1), dtype=int)
    assert bootstrap_ratio([0], [0], draws)['value'] is None
    clean = fixture_rows()
    masked = np.ones_like(clean, dtype=bool)
    vis = visibility_statistics(clean, masked, clean, np.ones_like(clean), block_size=7, special_ids=(99,))
    err = error_statistics(clean, masked, clean, block_size=7, special_ids=(99,))
    result = summarize(vis[None], err[None], bootstraps=20)
    assert result['visibility'][0]['accuracy']['value'] == 1.
    assert result['visibility'][1]['accuracy']['value'] is None
    assert result['error_events']['swap']['among_wrong']['value'] is None
