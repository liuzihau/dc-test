import numpy as np
import pytest
from analysis.owt_frozen_readout import fit_weight, mixed_logp, confidence_route, macro_report, row_statistics


def scores(main, auxiliary):
    return np.log(np.column_stack([main, auxiliary, auxiliary]))


def test_symmetric_complementary_optimum_and_mixture():
    values = scores([.1, .9], [.9, .1])
    fit = fit_weight([values])
    assert fit['weight'] == pytest.approx(.5, abs=1e-12)
    assert np.exp(mixed_logp(values, fit['weight'])) == pytest.approx([.5, .5])
    assert abs(fit['derivative_at_solution']) < 1e-12
    assert np.array_equal(mixed_logp(values, 0), values[:, 0])


def test_dominated_heads_select_endpoints():
    assert fit_weight([scores([.8, .7], [.2, .3])])['weight'] == 0
    assert fit_weight([scores([.2, .3], [.8, .7])])['weight'] == 1


def test_equal_condition_weight_not_pooled_tokens():
    a = scores([.1], [.9]); b = scores([.9], [.1])
    assert fit_weight([a, b])['weight'] == pytest.approx(.5)
    assert fit_weight([np.repeat(a, 100, axis=0), b])['weight'] == pytest.approx(.5)


def test_extreme_log_scores_are_stable_interior():
    values = np.array([[-1000, -1, -1], [-1, -1000, -1000]], dtype=float)
    fit = fit_weight([values])
    assert fit['weight'] == pytest.approx(.5)
    assert np.isfinite(mixed_logp(values, .5)).all()


def test_routing_never_needs_labels_and_main_wins_ties():
    confidence = np.array([[.8, .8, .8], [.2, .7, .7], [.1, .2, .9]])
    assert np.array_equal(confidence_route(confidence), [0, 1, 2])
    with pytest.raises(ValueError):
        confidence_route([[np.nan, .2, .3]])


def test_shared_macro_condition_weight_and_paired_rows():
    a = dict(targets=np.array([1., 1.]), mixture_ce_sum=np.array([1., 3.]))
    b = dict(targets=np.array([100., 100.]), mixture_ce_sum=np.array([300., 100.]))
    # Correlated row draws: opposing row effects cancel in the equal-condition macro.
    weights = np.array([[2., 0.], [0., 2.], [1., 1.]])
    result = macro_report([a, b], weights)['mixture_ce']
    assert result['value'] == 2
    assert result['row_bootstrap_95'] == [2, 2]


def test_invalid_calibration_rejected():
    with pytest.raises(ValueError):
        fit_weight([])
    with pytest.raises(ValueError):
        fit_weight([np.empty((0, 3))])
    with pytest.raises(ValueError):
        fit_weight([np.array([[np.nan, 0., 0.]])])


def test_scoring_common_population_truth_independent_routing_and_fit():
    # Position0 excluded; position2 lacks one auxiliary and must not enter any comparator.
    np_arm = dict(clean=np.array([[7, 1, 2]]), scored=np.array([[False, True, True]]))
    for head, pred, prob, lp in [
        ('main', [-1, 1, 2], [np.nan, .4, .9], [np.nan, np.log(.4), np.log(.9)]),
        ('left', [-1, 3, 2], [np.nan, .8, .8], [np.nan, np.log(.1), np.log(.8)]),
        ('right', [-1, 1, -1], [np.nan, .6, np.nan], [np.nan, np.log(.6), np.nan])]:
        np_arm[head+'_prediction']=np.array([pred])
        np_arm[head+'_top1_probability']=np.array([prob])
        np_arm[head+'_true_logp']=np.array([lp])
    mdm = dict(main_prediction=np.array([[-1, 1, 2]]),
               main_true_logp=np.array([[np.nan, np.log(.5), np.log(.9)]]))
    calibration = [scores([.1, .9], [.9, .1])]
    weight=fit_weight(calibration)['weight']
    result=row_statistics(mdm,np_arm,np.array([0]),weight)
    assert result['targets'][0]==1
    assert result['np_main_correct'][0]==1
    assert result['selector_correct'][0]==0
    assert result['selected_left'][0]==1
    assert result['mixture_ce_sum'][0]==pytest.approx(-np.log(.375))
    # Scoring labels can alter correctness but cannot change fitted weight or confidence routing.
    np_arm['clean'][0,1]=3
    np_arm['main_true_logp'][0,1]=np.log(.05)
    changed=row_statistics(mdm,np_arm,np.array([0]),weight)
    assert changed['selected_left'][0]==1
    assert changed['selector_correct'][0]==1
    assert fit_weight(calibration)['weight']==weight
