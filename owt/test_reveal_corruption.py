import numpy as np
import pytest

from owt.reveal_corruption import make_canvas, nearest_different_sources


def canvas(mask,correct,keep_first=False):
    return make_canvas(np.arange(16),mask,correct,123,7,np.arange(17),17,[16],keep_first)


def test_full_mask_has_identical_inputs_for_all_reliability_settings():
    inputs=[canvas(1,r) for r in [1,.8,.6]]
    for xt,masked,wrong,info in inputs:
        assert np.all(xt==17) and masked.all() and not wrong.any()
        assert info['actual_correct_fraction'] is None
    assert all(np.array_equal(inputs[0][0],v[0]) for v in inputs)


def test_reliability_changes_only_revealed_tokens_with_paired_nested_errors():
    clean,mask,wrong,info=canvas(.4,1)
    xt80,mask80,wrong80,info80=canvas(.4,.8)
    xt60,mask60,wrong60,info60=canvas(.4,.6)
    assert np.array_equal(mask,mask80) and np.array_equal(mask,mask60)
    assert np.all(~wrong80|wrong60)
    assert np.array_equal(xt80[wrong80],xt60[wrong80])
    assert not wrong.any() and not (wrong60&mask).any()
    assert np.all(xt60[wrong60]!=np.arange(16)[wrong60])
    assert not np.isin(xt60[wrong60],[16,17]).any()
    assert info80['wrong_revealed_positions']==round(.2*info80['reliability_population'])


def test_mask_levels_are_nested_and_anchor_is_explicit():
    previous=np.ones(16,dtype=bool)
    for ratio in [1,.8,.6,.4,.2]:
        _,masked,_,_=canvas(ratio,.6)
        assert np.all(~masked|previous)
        previous=masked
    xt,masked,wrong,info=canvas(1,.6,keep_first=True)
    assert xt[0]==0 and not masked[0] and not wrong[0]
    assert masked[1:].all() and info['reliability_population']==0


def test_corruption_is_deterministic_without_mutating_inputs_or_global_rng():
    clean=np.arange(16)
    state=np.random.get_state()
    a=canvas(.4,.6); b=canvas(.4,.6)
    assert np.array_equal(a[0],b[0])
    assert np.array_equal(state[1],np.random.get_state()[1])
    assert np.array_equal(clean,np.arange(16))
    with pytest.raises(ValueError,match='at least two'):
        make_canvas(clean,.4,.6,123,7,[1,1,1],17)


def test_nearest_clean_token_skips_identical_runs_and_breaks_ties_left():
    # know . . . , : middle dot takes 'know' on the equal-distance tie.
    clean=np.array([1,2,2,2,3])
    sources=nearest_different_sources(clean)
    assert sources.tolist()==[1,0,0,4,3]
    xt,masked,wrong,info=make_canvas(clean,.2,0.,123,7,None,4,nearest_sources=sources)
    assert np.array_equal(xt[wrong],clean[sources[wrong]])
    assert np.all(xt[wrong]!=clean[wrong])
    assert info['copied_masked_source_occurrences']==int(masked[sources[wrong]].sum())


def test_nearest_clean_source_excludes_special_tokens_and_has_no_random_fallback():
    assert nearest_different_sources([1,9,1,2],[9]).tolist()==[3,0,3,2]
    clean=np.ones(16,dtype=np.int64)
    with pytest.raises(ValueError,match='no different'):
        make_canvas(clean,.4,.6,123,7,None,17,nearest_sources=nearest_different_sources(clean))
