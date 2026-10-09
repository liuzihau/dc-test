import copy
import json
from pathlib import Path

import numpy as np
import pytest

from reasoning.reporting import (collect, load_evaluation, paired_interval,
                                  summarize, validate_pair, wilson_interval)


def evaluation_fixture(tmp_path, variant='mdm', values=(1, 0, 1, 0), *, task='sudoku', step=5000,
                       run_name=None):
    memory = variant.startswith('both')
    config = dict(memory_mode='both' if memory else 'none', neighbors=variant.endswith('_aux'),
        attention_mode='vanilla' if variant == 'vanilla' else 'merged',
        trajectory='single' if variant == 'vanilla' else 'five',
        gradient_mode='adjacent' if memory else 'detached', gate_enabled=not memory,
        cache_only_probability=0.0, current_only_probability=.05 if memory else 0.,
        final_dropout=.10 if memory else 0., identity_probability=.25 if memory else 0.,
        weights=[.05, .1, .2, 1., .7], neighbor_weight=.5, kmin=.025, kmax=.1,
        max_mask_ratio=.9975, hidden_size=512, n_heads=8, n_layers=6, max_length=192,
        source_dropout_warmup_steps=1000, identity_weight=.1, identity_margin=.05,
        identity_final_probability=.5, dropout=0.)
    if memory:
        config['merged_policy'] = 'current_preserving'
    contract = dict(task=task, variant=variant, model_config=config, data_sha256='a'*64,
        global_batch=128, micro_batch=128, world_size=1, seed=1, lr=.0003,
        weight_decay=.01, warmup_steps=1000, grad_clip=1., precision='bf16', device_type='cuda')
    run = tmp_path / (run_name or (task + '-' + variant))
    run.mkdir(parents=True)
    checkpoint = str(run / 'checkpoints/last.pt')
    data = dict(step=step, checkpoint=checkpoint, contract=contract,
        arguments=dict(split='test', protocol='generate', memory_condition='correct', policy='top_prob',
            seed=2026, batch_size=8, examples=len(values), checkpoint=checkpoint),
        metrics=dict(evaluation='closed_loop_generation', policy='top_prob', candidate_k=8,
            token_selection='paper', seed=2026, memory_condition='correct', tokens_per_step=1,
            max_steps=None, num_examples=len(values), valid_solution=sum(values)/len(values)),
        examples=[dict(id=f'puzzle-{index}', record_index=index, task=task, all_slots_completed=True,
            remaining_masked_slots=0, scores=dict(valid_solution=value)) for index, value in enumerate(values)])
    path = run / f'evaluation-last-step{step}-fake-n{len(values)}.json'
    path.write_text(json.dumps(data))
    return run, path, data


def rewrite(path, data):
    path.write_text(json.dumps(data))


def test_wilson_endpoints_and_standard_interval():
    assert wilson_interval(0, 100)[0] == 0
    assert wilson_interval(100, 100)[1] == 1
    assert wilson_interval(50, 100) == pytest.approx((.4038315304, .5961684696))
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_paired_bootstrap_is_deterministic_and_counts_discordances():
    values = paired_interval([1, 1, 0, 0], [0, 1, 1, 0], seed=17)
    assert values == paired_interval([1, 1, 0, 0], [0, 1, 1, 0], seed=17)
    assert values['mean_delta_pp'] == 0
    assert values['condition_only_correct'] == values['reference_only_correct'] == 1
    assert values['both_correct'] == values['both_wrong'] == 1
    assert not values['bootstrap_degenerate']
    assert values['ci95_low_pp'] < 0 < values['ci95_high_pp']
    assert paired_interval([1, 0], [1, 0])['ci95_high_pp'] == 0
    assert paired_interval([1, 0], [1, 0])['bootstrap_degenerate']


def test_complete_report_and_aux_memory_classification(tmp_path):
    runs = []
    originals = {}
    for variant, values in [('vanilla', (0, 0, 1, 0)), ('mdm', (1, 0, 1, 0)),
                            ('mdm_aux', (1, 1, 1, 0)), ('both', (1, 1, 1, 0)),
                            ('both_aux', (1, 1, 1, 1))]:
        run, path, _ = evaluation_fixture(tmp_path, variant, values)
        runs.append(run)
        originals[path] = path.read_bytes()
    output = tmp_path / 'report'
    rows, pairs, warnings = summarize(runs, output)
    assert len(rows) == 5 and len(pairs) == 4 and not warnings
    assert all(row['examples'] == 4 for row in rows)
    aux = next(row for row in pairs if row['condition'] == 'both_aux' and row['reference'] == 'both')
    assert aux['mean_delta_pp'] == 25.
    assert aux['condition_only_correct'] == 1
    assert 'auxiliary-training' in aux['intervention']
    memory = next(row for row in pairs if row['condition'] == 'both')
    assert 'not architecture-only' in memory['intervention']
    assert all(path.read_bytes() == original for path, original in originals.items())
    assert (output / 'sudoku_accuracy.png').stat().st_size > 1000
    assert (output / 'accuracy.csv').exists()
    assert (output / 'paired_deltas.csv').exists()
    assert 'not variation across training seeds' in (output / 'summary.md').read_text()
    assert len(json.loads((output / 'report_manifest.json').read_text())['evaluations']) == 5


@pytest.mark.parametrize('change,match', [
    (lambda d: d.update(step=4500), 'checkpoint step'),
    (lambda d: d['arguments'].update(split='validation'), 'test/generate'),
    (lambda d: d['arguments'].update(protocol='cold'), 'test/generate'),
    (lambda d: d['arguments'].update(memory_condition='none'), 'test/generate'),
    (lambda d: d['metrics'].update(memory_condition='shuffle_dcache'), 'test/generate'),
    (lambda d: d['metrics'].update(candidate_k=4), 'generation protocol'),
    (lambda d: d['metrics'].update(seed=3), 'generation protocol'),
    (lambda d: d['metrics'].update(max_steps=2), 'generation protocol'),
    (lambda d: d['metrics'].update(num_examples=3), 'incomplete evaluation'),
    (lambda d: d['arguments'].update(examples=1000), 'incomplete evaluation'),
    (lambda d: d['metrics'].update(valid_solution=.99), 'reported accuracy'),
    (lambda d: d['examples'][1].update(id='puzzle-0'), 'duplicate puzzle'),
    (lambda d: d['examples'][0].update(all_slots_completed=False), 'incomplete generation'),
    (lambda d: d['examples'][0]['scores'].update(valid_solution=.5), 'binary whole-puzzle'),
    (lambda d: d['contract'].update(data_sha256='invalid'), 'SHA256'),
    (lambda d: d['contract']['model_config'].update(neighbors=True), 'auxiliary-head'),
    (lambda d: d['contract'].update(stress_memory_routes=True), 'smoke-stress'),
])
def test_malformed_or_ineligible_evaluations_are_rejected(tmp_path, change, match):
    run, path, data = evaluation_fixture(tmp_path)
    change(data)
    rewrite(path, data)
    with pytest.raises(ValueError, match=match):
        load_evaluation(path, run, 5000)


def test_old_memory_recipe_and_changed_run_contract_rejected(tmp_path):
    run, path, data = evaluation_fixture(tmp_path, 'both')
    data['contract']['model_config']['merged_policy'] = 'legacy'
    rewrite(path, data)
    with pytest.raises(ValueError, match='corrected merged'):
        load_evaluation(path, run, 5000)
    data['contract']['model_config']['merged_policy'] = 'current_preserving'
    rewrite(path, data)
    wrong = copy.deepcopy(data['contract'])
    wrong['seed'] = 100
    (run / 'contract.json').write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match='run contract'):
        load_evaluation(path, run, 5000)


@pytest.mark.parametrize('field,value', [('seed', 2), ('global_batch', 64), ('micro_batch', 64),
    ('lr', .001), ('precision', 'fp32'), ('warmup_steps', 0)])
def test_aux_pairs_reject_other_training_changes(tmp_path, field, value):
    run_a, path_a, a = evaluation_fixture(tmp_path, 'mdm_aux')
    run_b, path_b, _ = evaluation_fixture(tmp_path, 'mdm')
    a['contract'][field] = value
    rewrite(path_a, a)
    with pytest.raises(ValueError, match='uncontrolled training'):
        validate_pair(load_evaluation(path_a, run_a, 5000), load_evaluation(path_b, run_b, 5000))


@pytest.mark.parametrize('change', [
    lambda d: d['contract'].update(data_sha256='b'*64),
    lambda d: d['arguments'].update(batch_size=4),
    lambda d: (d['arguments'].update(seed=13), d['metrics'].update(seed=13)),
    lambda d: d['examples'].reverse(),
    lambda d: d['contract']['model_config'].update(kmax=.08),
])
def test_memory_pair_rejects_data_protocol_order_and_trajectory_changes(tmp_path, change):
    run_a, path_a, data = evaluation_fixture(tmp_path, 'both')
    run_b, path_b, _ = evaluation_fixture(tmp_path, 'mdm')
    change(data)
    rewrite(path_a, data)
    with pytest.raises(ValueError):
        validate_pair(load_evaluation(path_a, run_a, 5000), load_evaluation(path_b, run_b, 5000))


def test_duplicates_not_selected_by_best_test_accuracy(tmp_path):
    run, path, data = evaluation_fixture(tmp_path)
    duplicate = run / 'evaluation-second.json'
    duplicate.write_text(json.dumps(data))
    warnings = []
    assert len(collect([run, run], 5000, warnings)) == 1
    assert not warnings
    data['examples'][0]['scores']['valid_solution'] = 0
    data['metrics']['valid_solution'] = .25
    duplicate.write_text(json.dumps(data))
    assert not collect([run], 5000, warnings)
    assert any('Ambiguous' in warning for warning in warnings)


def test_partial_report_is_explicit_and_does_not_pair_steps(tmp_path):
    run_a, _, _ = evaluation_fixture(tmp_path, 'both_aux', step=5000)
    run_b, _, _ = evaluation_fixture(tmp_path, 'both', step=4500)
    rows, pairs, warnings = summarize([run_a, run_b, tmp_path / 'missing'], tmp_path / 'report')
    assert len(rows) == 1 and not pairs
    assert any('4500' in warning for warning in warnings)
    assert any('No unique completed pair' in warning for warning in warnings)
    text = (tmp_path / 'report/summary.md').read_text()
    assert 'No eligible completed step-5000 evaluation' in text


def test_pair_contrast_does_not_hide_extra_model_depth(tmp_path):
    run_a, path_a, data = evaluation_fixture(tmp_path, 'both')
    run_b, path_b, _ = evaluation_fixture(tmp_path, 'mdm')
    data['contract']['model_config']['n_layers'] = 7
    rewrite(path_a, data)
    with pytest.raises(ValueError, match='n_layers'):
        validate_pair(load_evaluation(path_a, run_a, 5000), load_evaluation(path_b, run_b, 5000))


def test_output_cannot_replace_raw_run(tmp_path):
    run, _, _ = evaluation_fixture(tmp_path)
    with pytest.raises(ValueError, match='dedicated report'):
        summarize([run], run)


def test_all_zero_floor_is_not_reported_as_equivalence(tmp_path):
    runs = [evaluation_fixture(tmp_path, variant, (0, 0, 0, 0), task='zebra')[0]
            for variant in ('mdm', 'mdm_aux')]
    rows, pairs, warnings = summarize(runs, tmp_path / 'report')
    assert all(row['accuracy_pct'] == 0 and row['ci95_high_pct'] > 0 for row in rows)
    assert len(pairs) == 1 and pairs[0]['bootstrap_degenerate']
    assert pairs[0]['ci95_low_pp'] == pairs[0]['ci95_high_pp'] == 0
    assert any('floor effect' in message and 'NOT establish equivalence' in message for message in warnings)
    text = (tmp_path / 'report/summary.md').read_text()
    assert 'Degenerate; not evidence of equivalence' in text
    assert 'not an equivalence test' in text
