import json
import numpy as np
import pytest

from owt.post_diagnostic_canvases import anchor_pair, source_pairs
from owt.post_diagnostics import parse_core_row, score_targets, summarize_sources
from owt.test_head_diagnostics import FakeNP
from owt.post_schedule import core_ready


def test_core_reuse_preserves_hashes_with_condition_suffixes():
    parsed = parse_core_row(dict(variant='mdm', masked_targets='10',
        reference_copied_mask_sha256_c080='abc123', input_sha256='def456', empty=''))
    assert parsed == dict(variant='mdm', masked_targets=10.,
        reference_copied_mask_sha256_c080='abc123', input_sha256='def456')


def test_anchor_changes_only_input_zero_and_preserves_scored_targets():
    clean = np.arange(32, dtype=np.int64) % 2
    changed = 0
    for row_id in range(20):
        original, anchored = anchor_pair(clean, .2, row_id, mask_id=3, boundaries=(2,))
        assert np.array_equal(original[0][1:], anchored[0][1:])
        assert np.array_equal(original[1][1:], anchored[1][1:])
        assert anchored[0][0] == clean[0]
        assert not anchored[1][0]
        changed += int(original[1][0])
    assert changed > 0  # Exercise an actual clamp, not only identity controls.


def test_source_toggle_keeps_target_other_inputs_and_anchor_identical():
    clean = np.arange(32, dtype=np.int64) % 2
    clean[8] = 2
    before = np.random.get_state()
    for ratio in (.1, .2, .6):
        for row_id in range(10):
            cases = source_pairs(clean, ratio, row_id, mask_id=3, boundaries=(2,))
            if not cases:
                continue
            assert len(cases) == 4
            assert len({c['target_index'] for c in cases}) == 1
            for a, b in zip(cases[::2], cases[1::2]):
                j, i = a['target_index'], a['source_index']
                assert j >= 2 and i == j + a['direction']
                assert clean[i] != 2 and clean[j] != 2
                assert a['source_state'] == 'masked' and b['source_state'] == 'revealed'
                assert a['canvas'][i] == 3 and b['canvas'][i] == clean[i]
                other = np.arange(len(clean)) != i
                assert np.array_equal(a['canvas'][other], b['canvas'][other])
                assert a['canvas'][j] == b['canvas'][j] == 3
                assert a['canvas'][0] == b['canvas'][0] == clean[0]
            again = source_pairs(clean, ratio, row_id, mask_id=3, boundaries=(2,))
            for a, b in zip(cases, again):
                assert a['target_index'] == b['target_index']
                assert np.array_equal(a['canvas'], b['canvas'])
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_missing_source_population_is_recorded_without_replacement():
    assert source_pairs(np.full(16, 2), .6, 3, mask_id=3, boundaries=(2,)) == []


def test_selected_source_uses_correct_offset_and_only_one_backbone_pass():
    model = FakeNP()
    clean = np.array([[0, 0, 0, 0, 0, 0, 2]] * 2)
    cases = [dict(canvas=np.array([0, 3, 3, 3, 3, 3, 2]),
                  target_index=3, source_index=3 + d, direction=d) for d in (-1, 1)]
    rows = score_targets(model, clean, cases, 'cpu')
    assert model.calls == 1
    assert rows[0]['main_prediction'] == rows[1]['main_prediction'] == 0
    assert rows[0]['auxiliary_prediction'] == 1  # +1 head from left source index 2.
    assert rows[1]['auxiliary_prediction'] == 0  # -1 head from right source index 4.
    assert rows[0]['auxiliary_target_correct'] == 0
    assert rows[1]['auxiliary_target_correct'] == 1
    assert not model.backbone.output_layer.linear._forward_pre_hooks


def source_observations():
    rows = []
    for variant in ('mdm', 'mdm_np_zero_init'):
        for row_id in (0, 1):
            for direction in (-1, 1):
                for state in ('masked', 'revealed'):
                    ce = 4. + row_id + (.5 if variant != 'mdm' else 0)
                    if state == 'revealed':
                        ce -= 1.5 if variant != 'mdm' else .5
                    rows.append(dict(variant=variant, row_id=row_id, mask_ratio=.2,
                        direction=direction, source_state=state, target_index=3,
                        source_index=3 + direction, input_sha256=f'{row_id}-{direction}-{state}',
                        clean_sha256=f'clean{row_id}', main_target_ce=ce))
    return rows


def test_source_benefit_interaction_uses_paired_rows_and_correct_sign():
    result = summarize_sources(source_observations(), ['mdm', 'mdm_np_zero_init'], bootstraps=20)
    cell = next(r for r in result if r.get('variant') == 'mdm_np_zero_init' and r['direction'] == 'both')
    assert cell['source_reveal_benefit'] == pytest.approx(1.5)
    assert cell['np_minus_mdm_benefit'] == pytest.approx(1.)
    assert cell['masked_main_ce_gap_vs_mdm'] == pytest.approx(.5)
    assert cell['revealed_main_ce_gap_vs_mdm'] == pytest.approx(-.5)
    assert cell['interaction_row_bootstrap_95'] == pytest.approx([1., 1.])


def test_source_summary_rejects_unmatched_canvases_and_duplicate_cases():
    rows = source_observations()
    rows[-1]['input_sha256'] = 'different input'
    with pytest.raises(AssertionError, match='pairing'):
        summarize_sources(rows, ['mdm', 'mdm_np_zero_init'])
    with pytest.raises(ValueError, match='Duplicate'):
        summarize_sources(source_observations() + source_observations()[:1], ['mdm', 'mdm_np_zero_init'])


def test_follower_requires_core_completion_full_cohort_and_all_figures(tmp_path):
    root, core = tmp_path / 'runs', tmp_path / 'core'
    root.mkdir(); core.mkdir()
    state = root / 'reveal_sweep_queue.json'
    state.write_text(json.dumps(dict(stage='waiting_for_zero_completion')))
    assert not core_ready(root, core)
    state.write_text(json.dumps(dict(stage='complete', output=str(core))))
    artifact = dict(protocol=dict(variants=['mdm', 'mdm_np_zero_init'],
                                 row_ids=list(range(1024)), optimizer_step=5000))
    (core / 'summary.json').write_text(json.dumps(artifact))
    with pytest.raises(RuntimeError, match='figures'):
        core_ready(root, core)
    for m in (100, 80, 60, 40, 20):
        for c in (100, 80, 60):
            (core / f'layer_cosine_mask{m:03d}_correct{c:03d}.pdf').touch()
    assert core_ready(root, core)
    artifact['protocol']['variants'].insert(1, 'mdm_np')
    (core / 'summary.json').write_text(json.dumps(artifact))
    with pytest.raises(RuntimeError, match='two-arm'):
        core_ready(root, core)


def test_follower_does_not_treat_failed_core_as_a_finished_wait(tmp_path):
    (tmp_path / 'reveal_sweep_queue.json').write_text(json.dumps(dict(stage='failed')))
    with pytest.raises(RuntimeError, match='failed'):
        core_ready(tmp_path, tmp_path / 'core')
