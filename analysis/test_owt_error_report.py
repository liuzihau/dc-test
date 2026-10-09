import numpy as np
import pytest

from analysis.owt_error_report import (bootstrap_weights, pair_report, paired_change,
    load_predictions, union_report, verify_pair, verify_source_states,
    verify_source_observations, source_pairs, digest, VARIANTS)


def test_five_cases_exhaust_targets_and_bootstrap_reweights_whole_rows():
    clean = np.zeros((2, 5), dtype=np.int32)
    first = np.array([[0, 1, 0, 1, 1], [0, 0, 0, 0, 0]])
    second = np.array([[0, 0, 1, 1, 2], [1, 0, 0, 0, 0]])
    choose = np.array([[True] * 5, [True, False, False, False, False]])
    lp_first = np.log(np.where(first == clean, .8, .1))
    lp_second = np.log(np.where(second == clean, .8, .1))
    weights = np.array([[2., 0.], [0., 2.], [1., 1.]])
    report, stats, _ = pair_report(clean, first, second, lp_first, lp_second, choose, weights)
    assert report['targets'] == 6
    assert report['counts'] == dict(both_correct=1, rescue=1, main_only_correct=2,
                                   both_wrong_same=1, both_wrong_different=1)
    metrics = report['metrics']
    assert metrics['main_accuracy']['value'] == pytest.approx(.5)
    assert metrics['alternative_accuracy']['value'] == pytest.approx(1 / 3)
    assert metrics['accuracy_difference']['value'] == pytest.approx(-1 / 6)
    assert metrics['absolute_rescue']['value'] == pytest.approx(1 / 6)
    assert metrics['conditional_rescue']['value'] == pytest.approx(1 / 3)
    assert metrics['conditional_rescue']['defined_bootstrap_replicates'] == 2
    assert metrics['conditional_rescue']['row_bootstrap_95'] == pytest.approx([1 / 3, 1 / 3])
    assert metrics['accuracy_difference']['row_bootstrap_95'] == pytest.approx(np.quantile([0., -1., -1 / 6], [.025, .975]))
    assert stats['targets'].tolist() == [5., 1.]
    assert report['logp_difference_distribution']['histogram_counts'] and sum(report['logp_difference_distribution']['histogram_counts']) == 6


def test_empty_stratum_and_zero_main_errors_have_unavailable_conditional_rates():
    clean = np.zeros((2, 3), dtype=np.int32)
    lp = np.full((2, 3), np.nan)
    choose = np.zeros((2, 3), dtype=bool)
    report, _, _ = pair_report(clean, clean, clean, lp, lp, choose, bootstrap_weights(2, 20))
    assert report['targets'] == 0
    assert report['metrics']['conditional_rescue']['value'] is None
    assert report['metrics']['main_accuracy']['row_bootstrap_95'] is None
    report, _, _ = pair_report(clean, clean, clean, np.full_like(lp, -.2), np.full_like(lp, -.2),
                              ~choose, bootstrap_weights(2, 20))
    assert report['metrics']['main_accuracy']['value'] == 1.
    assert report['metrics']['conditional_rescue']['value'] is None


def test_union_rescue_and_oracle_share_the_common_target_population():
    clean = np.zeros((2, 3), dtype=np.int32)
    arrays = dict(main_prediction=np.array([[0, 1, 1], [1, 1, 1]]),
                  left_prediction=np.array([[0, 0, 1], [1, 1, 1]]),
                  right_prediction=np.array([[0, 1, 0], [1, 1, 1]]))
    selected = np.array([[True] * 3, [False] * 3])
    report, _ = union_report(clean, arrays, selected, bootstrap_weights(2, 20))
    assert report['targets'] == 3
    assert report['metrics']['absolute_rescue']['value'] == pytest.approx(2 / 3)
    assert report['metrics']['oracle_top1_selector_accuracy']['value'] == 1.
    assert report['metrics']['main_accuracy']['value'] == pytest.approx(1 / 3)


def arrays(logp):
    return dict(row_ids=np.array([0, 1]), clean=np.zeros((2, 3), dtype=np.int32),
                scored=np.array([[False, True, True], [False, True, True]]),
                main_prediction=np.zeros((2, 3), dtype=np.int32),
                main_true_logp=np.array([[np.nan, logp, logp], [np.nan, logp, logp]], dtype=np.float32))


def test_paired_change_retains_float64_differences_of_saved_fp32_scores():
    a, b = arrays(-10.000001), arrays(-10.)
    c, d = arrays(-10.000002), arrays(-10.)
    report = paired_change(a, b, c, d, a['scored'], bootstrap_weights(2, 20))
    expected = float(c['main_true_logp'][0, 1]) - float(d['main_true_logp'][0, 1])
    expected -= float(a['main_true_logp'][0, 1]) - float(b['main_true_logp'][0, 1])
    assert report['metrics']['np_minus_mdm_change']['value'] == expected
    d['clean'][0, 1] = 1
    with pytest.raises(ValueError, match='pairing'):
        paired_change(a, b, c, d, a['scored'], bootstrap_weights(2, 20))


def test_loading_rejects_missing_scores_and_inconsistent_top1_confidence(tmp_path):
    a = arrays(-.2)
    a['main_top1_probability'] = np.where(a['scored'], np.exp(a['main_true_logp']), np.nan)
    path = tmp_path / 'predictions.npz'
    np.savez_compressed(path, **a)
    assert np.array_equal(load_predictions(path)['row_ids'], [0, 1])
    a['main_top1_probability'][0, 1] = .1
    np.savez_compressed(path, **a)
    with pytest.raises(ValueError, match='confidence'):
        load_predictions(path)
    a.pop('main_top1_probability')
    a['main_true_logp'][0, 1] = np.nan
    np.savez_compressed(path, **a)
    with pytest.raises(ValueError, match='score'):
        load_predictions(path)


def test_source_state_verification_uses_neighbor_not_target_and_boundary_exclusions():
    a = arrays(-.2)
    # Target 1 has left source 0 revealed; target 2 has left source 1 masked.
    a['left_source_state'] = np.array([[255, 1, 0]] * 2, dtype=np.uint8)
    a['right_source_state'] = np.array([[255, 0, 255]] * 2, dtype=np.uint8)
    a['left_prediction'] = np.array([[-1, 0, 0]] * 2)
    a['right_prediction'] = np.array([[-1, 0, -1]] * 2)
    mask = a['scored'].copy(); wrong = np.zeros_like(mask)
    verify_source_states(a, mask, wrong)
    a['left_source_state'][0, 1] = 0
    with pytest.raises(ValueError, match='source states'):
        verify_source_states(a, mask, wrong)


def test_bootstrap_preserves_paired_row_count_and_seed():
    weights = bootstrap_weights(7, 31)
    assert weights.shape == (31, 7)
    assert np.array_equal(weights.sum(1), np.full(31, 7))
    assert np.array_equal(weights, bootstrap_weights(7, 31))


def test_source_intervention_requires_exact_inputs_and_complete_paired_coverage():
    clean = np.arange(1, 1025, dtype=np.int64)
    rows = []
    for ratio in (.1, .2, .6):
        for case in source_pairs(clean, ratio, 50):
            for variant in VARIANTS:
                rows.append(dict(variant=variant, row_id=50, mask_ratio=ratio,
                    direction=case['direction'], source_state=case['source_state'],
                    source_index=case['source_index'], target_index=case['target_index'],
                    target_token=int(clean[case['target_index']]), clean_sha256=digest(clean),
                    input_sha256=digest(case['canvas'])))
    assert len(rows) == 24
    verify_source_observations(rows, np.array([50]), clean[None], [50])
    with pytest.raises(ValueError, match='incomplete'):
        verify_source_observations(rows[:-1], np.array([50]), clean[None], [50])
    with pytest.raises(ValueError, match='Duplicate'):
        verify_source_observations(rows + rows[:1], np.array([50]), clean[None], [50])
    altered = [dict(row) for row in rows]
    altered[0]['input_sha256'] = 'different input'
    with pytest.raises(ValueError, match='registered paired input'):
        verify_source_observations(altered, np.array([50]), clean[None], [50])
