"""Semantic checks for anchor probes and paired reveal-schedule experiments."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.eval import dcache_eval_common as common
from test_recurrence_audit import RecordingModel


@pytest.fixture
def evaluator():
  return import_module('scripts.eval.eval_anchor_schedule')


@pytest.fixture
def plotter():
  return import_module('scripts.eval.plot_anchor_schedule')


@pytest.fixture
def batch():
  tokens = (torch.arange(3 * 97).reshape(3, 97) % 15).long()
  attention = torch.ones_like(tokens)
  return tokens, attention, [7, 18, 32]


def test_probes_and_strided_groups_cover_only_initially_masked_tokens(batch, evaluator):
  _, attention, ids = batch
  schedule = evaluator.make_schedule(attention, ids, 0.5, 19, probe_count=8, stride=32)

  assert schedule.probe_mask.sum(-1).tolist() == [8, 8, 8]
  assert torch.all(schedule.probe_mask <= schedule.initial_mask)
  assert not schedule.initial_mask[:, 0].any()
  assert not schedule.probe_mask[:, 0].any()
  positions = torch.arange(attention.shape[1])[None, :]
  revealable = schedule.initial_mask & ~schedule.probe_mask
  strided = schedule.reveal_groups['strided']
  random = schedule.reveal_groups['random']
  assert len(strided) == len(random) == 32
  for slot, group in enumerate(strided):
    assert torch.equal(group, revealable & (positions % 32 == slot))
    assert torch.equal(group.sum(-1), random[slot].sum(-1))
  for groups in [strided, random]:
    visits = torch.stack(groups).sum(0)
    assert torch.equal(visits, revealable.long())
    assert visits.max().item() == 1
  assert torch.any(torch.stack(strided).sum(-1) == 0)


def test_schedule_and_probes_are_document_stable_under_batch_reorder(batch, evaluator):
  _, attention, ids = batch
  original = evaluator.make_schedule(attention, ids, 0.5, 19, probe_count=8, stride=32)
  order = torch.tensor([2, 0, 1])
  changed = evaluator.make_schedule(
    attention[order], [ids[index] for index in order.tolist()],
    0.5, 19, probe_count=8, stride=32)
  assert torch.equal(changed.initial_mask, original.initial_mask[order])
  assert torch.equal(changed.probe_mask, original.probe_mask[order])
  for name in ['strided', 'random']:
    for original_group, changed_group in zip(
        original.reveal_groups[name], changed.reveal_groups[name]):
      assert torch.equal(changed_group, original_group[order])


def test_sampling_never_uses_teacher_values_or_writes_unselected_positions(batch, evaluator):
  tokens, attention, ids = batch
  schedule = evaluator.make_schedule(attention, ids, 0.5, 19, probe_count=8, stride=32)
  state = common.masked_state(tokens, schedule.initial_mask, RecordingModel.mask_index)
  reveal = schedule.initial_mask & ~schedule.probe_mask
  scores = torch.full((*state.shape, 17), -torch.inf)
  scores[..., 15] = 0.0

  actual = evaluator.reveal_from_model(
    state, scores, reveal, ids, seed=12, mask_index=RecordingModel.mask_index)

  assert torch.all(actual[reveal] == 15)
  assert torch.all(actual[reveal] != tokens[reveal])
  assert torch.equal(actual[~reveal], state[~reveal])
  assert torch.all(actual[schedule.probe_mask] == RecordingModel.mask_index)
  assert torch.equal(state, common.masked_state(tokens, schedule.initial_mask, 16))


def test_sampling_randomness_is_per_position_not_reveal_round_or_batch(batch, evaluator):
  tokens, attention, ids = batch
  schedule = evaluator.make_schedule(attention, ids, 0.5, 19, probe_count=8, stride=32)
  initial = common.masked_state(tokens, schedule.initial_mask, RecordingModel.mask_index)
  generator = torch.Generator().manual_seed(222)
  scores = torch.randn((*initial.shape, 17), generator=generator).log_softmax(-1)
  reveal = schedule.initial_mask & ~schedule.probe_mask
  kwargs = dict(seed=91, mask_index=RecordingModel.mask_index)
  all_at_once = evaluator.reveal_from_model(initial, scores, reveal, ids, **kwargs)
  for name in ['strided', 'random']:
    state = initial.clone()
    # Reverse chronological groups as a stronger check on per-position seeds.
    for group in reversed(schedule.reveal_groups[name]):
      state = evaluator.reveal_from_model(state, scores, group, ids, **kwargs)
    assert torch.equal(state, all_at_once)
  order = torch.tensor([1, 2, 0])
  reordered = evaluator.reveal_from_model(
    initial[order], scores[order], reveal[order],
    [ids[index] for index in order.tolist()], **kwargs)
  assert torch.equal(reordered, all_at_once[order])
  assert torch.all(all_at_once[reveal] != RecordingModel.mask_index)


def test_teacher_arms_have_fixed_probes_identical_final_input_and_33_real_forwards(
    batch, evaluator):
  tokens, attention, ids = batch
  schedule = evaluator.make_schedule(attention, ids, 0.5, 19, probe_count=8, stride=32)
  model = RecordingModel()
  rows = evaluator.evaluate_phase(
    model, 'two-forward', tokens, attention, ids,
    evaluator.Phase(0.5, 'teacher'), 19,
    probe_count=8, stride=32, score_rounds=tuple(range(33)))

  assert len(model.calls) == 4 * 33
  assert len(rows) == 3 * 4 * 33
  frame = pd.DataFrame(rows)
  assert frame.masked_tokens.eq(8).all()
  for call_index, call in enumerate(model.calls):
    scored = frame.iloc[call_index * 3:(call_index + 1) * 3]
    assert scored['round'].eq(call_index % 33).all()
    assert scored.remaining_mask_tokens.tolist() == call['state'].eq(model.mask_index).sum(-1).tolist()
  expected_final = common.masked_state(tokens, schedule.probe_mask, model.mask_index)
  arms = [model.calls[start:start + 33] for start in range(0, len(model.calls), 33)]
  absent_arms = 0
  for arm in arms:
    assert arm[0]['dcache'] is arm[0]['final'] is None
    assert torch.equal(arm[-1]['state'], expected_final)
    for step, call in enumerate(arm):
      assert torch.all(call['state'][schedule.probe_mask] == model.mask_index)
      assert torch.equal(call['state'][:, 0], tokens[:, 0])
      mask_counts = call['state'].eq(model.mask_index).sum(-1)
      expected_ratios = mask_counts.float() / (attention.sum(-1) - 1)
      assert torch.allclose(call['sigma'].reshape(-1), expected_ratios)
      if step:
        assert torch.equal(call['final'], arm[step - 1]['output_final'])
        if call['dcache'] is not None:
          assert torch.equal(call['dcache'][0], arm[step - 1]['output_dcache'][0])
    if all(call['dcache'] is None for call in arm):
      absent_arms += 1
  assert absent_arms == 2


def test_model_reveals_are_independent_of_all_initially_hidden_teacher_targets(
    batch, evaluator):
  tokens, attention, ids = batch
  schedule = evaluator.make_schedule(attention, ids, 0.5, 19, probe_count=8, stride=32)
  changed = torch.where(schedule.initial_mask, (tokens + 7) % 15, tokens)
  initial = common.masked_state(tokens, schedule.initial_mask, RecordingModel.mask_index)
  models = [RecordingModel(), RecordingModel()]
  for model, targets in zip(models, [tokens, changed]):
    evaluator.evaluate_phase(
      model, 'two-forward', targets, attention, ids,
      evaluator.Phase(0.5, 'model'), 19,
      probe_count=8, stride=32, score_rounds=(0, 1, 32))
    assert len(model.calls) == 4 * 33
    assert all(torch.all(call['state'][schedule.probe_mask] == model.mask_index)
               for call in model.calls)
    assert all(torch.equal(model.calls[start]['state'], initial)
               for start in range(0, 4 * 33, 33))
  for left, right in zip(models[0].calls, models[1].calls):
    assert torch.equal(left['state'], right['state'])
    assert torch.equal(left['sigma'], right['sigma'])


def test_sampler_rejects_overwriting_a_previously_committed_token(batch, evaluator):
  tokens, _, ids = batch
  scores = torch.zeros((*tokens.shape, 17)).log_softmax(-1)
  reveal = torch.zeros_like(tokens, dtype=torch.bool)
  reveal[:, 1] = True
  with pytest.raises(ValueError, match='already visible'):
    evaluator.reveal_from_model(tokens, scores, reveal, ids, 12, RecordingModel.mask_index)


def test_full_mask_1024_excludes_first_token_and_keeps_128_probes(evaluator):
  attention = torch.ones((1, 1024), dtype=torch.long)
  schedule = evaluator.make_schedule(attention, [42], 1.0, 19)
  assert schedule.initial_mask.sum().item() == 1023
  assert not schedule.initial_mask[0, 0]
  assert schedule.probe_mask.sum().item() == 128
  for name in ['strided', 'random']:
    assert torch.stack(schedule.reveal_groups[name]).sum().item() == 895


@pytest.mark.parametrize('variant', ['vanilla', 'objective'])
def test_no_memory_baselines_also_make_all_33_actual_forwards(batch, evaluator, variant):
  tokens, attention, ids = batch
  model = RecordingModel()
  rows = evaluator.evaluate_phase(
    model, variant, tokens, attention, ids, evaluator.Phase(0.5, 'teacher'), 19,
    probe_count=8, stride=32, score_rounds=(0, 32))
  assert len(model.calls) == 2 * 33
  assert len(rows) == 3 * 2 * 2
  assert all(call['dcache'] is call['final'] is None for call in model.calls)
  assert torch.equal(model.calls[32]['state'], model.calls[65]['state'])


def test_anchor_pause_resume_preserves_parts_and_handles_a_short_final_batch(
    tmp_path, batch, evaluator, monkeypatch):
  tokens, attention, _ = batch
  args = SimpleNamespace(
    examples=4, batch_size=3, variant='two-forward', seed=19, num_workers=0,
    device='cpu', expected_step=5000, data_dir=tmp_path,
    checkpoint=tmp_path / 'mock.ckpt', probe_count=8, stride=32,
    score_rounds=[0, 32], sources=['teacher', 'model'], initial_mask_ratios=[.5])
  phases = evaluator.make_phases(args)
  loader = [{'input_ids': tokens, 'attention_mask': attention},
            {'input_ids': tokens[:1], 'attention_mask': attention[:1]}]
  monkeypatch.setattr(common, 'compose_eval_config', lambda *a, **kw: object())
  monkeypatch.setattr(common, 'load_tokenizer', lambda config: object())
  monkeypatch.setattr(common, 'load_validation_data', lambda *a, **kw: loader)
  models = []

  def load(*args):
    model = RecordingModel()
    models.append(model)
    return model

  monkeypatch.setattr(common, 'load_ema_model', load)
  output = tmp_path / 'resumed'
  output.mkdir()
  stop = evaluator.CooperativeStop()
  evaluate = evaluator.evaluate_phase

  def pause_after_phase(*args, **kwargs):
    rows = evaluate(*args, **kwargs)
    stop.request(15, None)
    return rows

  monkeypatch.setattr(evaluator, 'evaluate_phase', pause_after_phase)
  evaluator.run_evaluation(args, phases, output, stop, 'test-fingerprint')
  first = common.part_path(output, phases[0].name, 0)
  first_bytes, first_mtime = first.read_bytes(), first.stat().st_mtime_ns
  assert not common.part_path(output, phases[1].name, 0).exists()
  monkeypatch.setattr(evaluator, 'evaluate_phase', evaluate)
  evaluator.run_evaluation(args, phases, output, evaluator.CooperativeStop(), 'test-fingerprint')
  assert first.read_bytes() == first_bytes
  assert first.stat().st_mtime_ns == first_mtime
  assert len(models) == 2
  assert len(models[1].calls) == 3 * 4 * 33
  evaluator.run_evaluation(args, phases, output, evaluator.CooperativeStop(), 'test-fingerprint')
  assert len(models) == 2
  assert evaluator.write_reports(args, phases, output) == (True, 4, 4)

  uninterrupted = tmp_path / 'uninterrupted'
  uninterrupted.mkdir()
  evaluator.run_evaluation(
    args, phases, uninterrupted, evaluator.CooperativeStop(), 'test-fingerprint')
  for phase in phases:
    for index in [0, 1]:
      pd.testing.assert_frame_equal(
        pd.read_csv(common.part_path(output, phase.name, index)),
        pd.read_csv(common.part_path(uninterrupted, phase.name, index)))
  partial = pd.read_csv(common.part_path(output, phases[0].name, 1))
  assert partial.example_id.unique().tolist() == [3]
  with pytest.raises(RuntimeError, match='Incomplete result part'):
    evaluator.validate_part(partial.iloc[:-1], args, phases[0], 1)


def test_anchor_resume_refuses_changed_protocol_and_keeps_existing_data(tmp_path, evaluator):
  output = tmp_path / 'evaluation'
  metadata = {'evaluation': 'anchor_schedule', 'stride': 32, 'probe_count': 128}
  fingerprint = evaluator.prepare_output(output, metadata)
  complete = common.part_path(output, 'done', 0)
  common.atomic_write_csv(complete, pd.DataFrame({'keep': [7]}))
  assert evaluator.prepare_output(output, metadata) == fingerprint
  with pytest.raises(RuntimeError, match='fingerprint changed'):
    evaluator.prepare_output(output, dict(metadata, probe_count=64))
  assert pd.read_csv(complete)['keep'].tolist() == [7]


@pytest.mark.parametrize('change', ['source_hash', 'checkpoint_hash'])
def test_comparison_rejects_changed_provenance_between_seed_runs(
    tmp_path, evaluator, plotter, monkeypatch, change):
  monkeypatch.setattr(evaluator, 'checkpoint_metadata', lambda *a: {
    'sha256': 'same-frozen-checkpoint', 'global_step': 5000, 'path': 'mock.ckpt'})
  monkeypatch.setattr(evaluator, 'prepared_data_fingerprint', lambda *a: {'tokens': 'same'})
  monkeypatch.setattr(evaluator, 'source_fingerprints', lambda: {'producer': 'same'})
  args = SimpleNamespace(
    variant='two-forward', expected_step=5000, checkpoint=tmp_path / 'mock.ckpt',
    data_dir=tmp_path, examples=4, batch_size=3, initial_mask_ratios=[.5],
    sources=['teacher'], probe_count=8, stride=32, score_rounds=[0, 32])
  metadata, coverage = [], []
  for seed in [13, 27]:
    args.seed = seed
    value = evaluator.make_metadata(args, evaluator.make_phases(args))
    directory = tmp_path / f'seed-{seed}'
    evaluator.prepare_output(directory, value)
    metadata.append(value)
    coverage.append({
      'variant': 'two-forward', 'seed': seed, 'run_dir': str(directory),
      'parts': 2, 'rows': 32, 'documents': 4, 'complete': True})
  coverage = pd.DataFrame(coverage)
  _, all_present = plotter.validate_manifests(coverage)
  assert all_present
  assert coverage.complete.all()
  assert coverage.expected_parts.tolist() == [2, 2]  # Short last batch counts.
  if change == 'source_hash':
    metadata[1]['source_sha256'] = {'producer': 'changed'}
  else:
    metadata[1]['checkpoint']['sha256'] = 'another-checkpoint'
  common.atomic_write_json(tmp_path / 'seed-27' / 'manifest.json', metadata[1])
  with pytest.raises(ValueError, match='incomparable manifests|Multiple source checkpoints'):
    plotter.validate_manifests(coverage)




def interaction_arms():
  reference = pd.DataFrame({
    'seed': [13, 13, 13], 'example_id': [2, 3, 4],
    'masked_tokens': [4, 8, 12], 'remaining_mask_tokens': [20, 22, 25],
    'nll': [3.0, 4.0, 5.0], 'top1_accuracy': [0.3, 0.4, 0.5],
    'top5_accuracy': [0.6, 0.7, 0.8],
  })
  strided, random = reference.copy(), reference.copy()
  strided['nll'] -= np.array([0.5, 0.2, 0.1])
  random['nll'] -= np.array([0.2, 0.1, 0.02])
  strided['top1_accuracy'] += np.array([0.10, 0.04, 0.02])
  random['top1_accuracy'] += np.array([0.02, 0.01, 0.004])
  return [strided, reference.copy(), random, reference.copy()]


def test_joint_paired_interaction_has_expected_sign_and_block_seed_uncertainty(plotter):
  arms = interaction_arms()
  first = plotter.paired_linear_contrast(arms, [1, -1, -1, 1], 500, 87)
  repeated = [pd.concat([arm.assign(seed=seed) for seed in [13, 27, 91]]) for arm in arms]
  actual = plotter.paired_linear_contrast(
    repeated, [1, -1, -1, 1], 500, 87, expected_seeds=[13, 27, 91])

  assert actual['delta_nll'] == pytest.approx(np.average([-.3, -.1, -.08], weights=[4, 8, 12]))
  assert actual['delta_top1_accuracy_pp'] == pytest.approx(
    100 * np.average([.08, .03, .016], weights=[4, 8, 12]))
  assert actual['documents'] == actual['clusters'] == 3
  assert actual['seeds'] == 3
  assert actual['observations'] == 9
  assert actual['cluster_unit'] == 'packed_validation_block_all_mask_seeds'
  for key in first:
    if key.startswith(('delta_', 'ci95_')):
      assert actual[key] == pytest.approx(first[key])


@pytest.mark.parametrize('change', ['missing_id', 'probe_count', 'remaining_count', 'missing_seed'])
def test_anchor_pairing_rejects_unmatched_samples_or_counts(plotter, change):
  arms = interaction_arms()
  expected_seeds = None
  if change == 'missing_id':
    arms[0] = arms[0].iloc[:-1]
  elif change == 'probe_count':
    arms[0] = arms[0].assign(masked_tokens=1)
  elif change == 'remaining_count':
    arms[0] = arms[0].assign(remaining_mask_tokens=100)
  else:
    expected_seeds = [13, 27]
  with pytest.raises(plotter.PairingError):
    plotter.paired_linear_contrast(
      arms, [1, -1, -1, 1], 100, 12, expected_seeds=expected_seeds)


def test_anchor_pairing_requires_complete_seed_vector_for_every_block(plotter):
  arms = [pd.concat([arm, arm.assign(seed=27).iloc[:-1]]) for arm in interaction_arms()]
  with pytest.raises(plotter.PairingError, match='Incomplete seed/block grid'):
    plotter.paired_linear_contrast(arms, [1, -1, -1, 1], 100, 12)


def test_anchor_plot_command_writes_all_figures_from_partial_raw_parts(tmp_path, plotter):
  for variant in ['five-forward', 'two-forward']:
    frames = []
    for source in ['teacher', 'model']:
      for round_index in [0, 1, 32]:
        for metrics, (schedule, condition) in zip(interaction_arms(), [
            ('strided', 'correct'), ('strided', 'absent_dcache'),
            ('random', 'correct'), ('random', 'absent_dcache')]):
          frames.append(metrics.assign(
            variant=variant, source=source, schedule=schedule, condition=condition,
            initial_mask_ratio=.5, round=round_index, masked_tokens=4,
            remaining_mask_tokens=(4 if round_index == 32 else 48 - round_index)))
    common.atomic_write_csv(
      common.part_path(tmp_path / 'raw' / variant / 'seed-13', 'phase', 0),
      pd.concat(frames, ignore_index=True))
  plotter.main([
    '--input-root', str(tmp_path / 'raw'), '--output-dir', str(tmp_path / 'figures'),
    '--bootstrap-samples', '20', '--expected-variants', 'five-forward', 'two-forward',
    '--expected-seeds', '13'])

  figures = tmp_path / 'figures'
  for name in ['curves_teacher_ratio0.5.png', 'curves_model_ratio0.5.png',
               'schedule_gain.png', 'dcache_contribution.png', 'dcache_schedule_interaction.png']:
    assert (figures / name).stat().st_size > 1000
  assert 'PARTIAL snapshot' in (figures / 'REPORT.md').read_text()
  assert pd.read_csv(figures / 'pairing_issues.csv').empty
  assert not pd.read_csv(figures / 'endpoint_table.csv').empty


def test_overnight_runner_command_preserves_the_requested_evaluator_protocol(evaluator):
  runner = import_module('scripts.eval.run_anchor_schedule')
  args = runner.parse_args(['--gpus', '2', '3', '--hours', '24'])
  runner.validate_protocol(args)
  command = runner.evaluator_command(args, 'two-forward', 20260907, 23.9)
  child = evaluator.parse_args(command[3:])
  assert args.gpus == ['2', '3']
  assert child.variant == 'two-forward'
  assert child.seed == 20260907
  assert child.device == 'cuda:0'
  assert child.max_hours == 23.9
  for name in ['examples', 'batch_size', 'num_workers', 'probe_count', 'stride',
               'sources', 'initial_mask_ratios', 'score_rounds']:
    assert getattr(child, name) == getattr(args, name)
  assert child.output_dir == args.output_root / 'two-forward' / 'seed-20260907'


@pytest.mark.parametrize('arguments', [
  ['--stride', '3'], ['--initial-mask-ratios', '.1'], ['--score-rounds', '0', '16'],
])
def test_overnight_runner_rejects_inconsistent_protocols(arguments):
  runner = import_module('scripts.eval.run_anchor_schedule')
  with pytest.raises(ValueError):
    runner.validate_protocol(runner.parse_args(arguments))
