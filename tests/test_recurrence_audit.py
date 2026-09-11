"""Behavioral checks for the paired recurrence audit, using a tiny CPU model."""

from types import SimpleNamespace
from importlib import import_module

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.eval import dcache_eval_common as common
from scripts.eval import plot_recurrence_audit as plotting


@pytest.fixture
def audit():
  return import_module('scripts.eval.eval_recurrence_audit')


class RecordingModel:
  """Deterministic predictions and memory that depend on the actual input."""

  mask_index = 16

  def __init__(self):
    self.calls = []

  def _sigma_from_p(self, probability):
    return probability.clone()

  def forward(self, state, sigma, previous_step_kv=None,
              previous_final_hidden=None, return_step_kv=False,
              return_dcachehooping=False, **kwargs):
    del kwargs
    hidden = state.float().unsqueeze(-1) + 1.0
    if previous_step_kv is not None:
      hidden = hidden + previous_step_kv[0] * 0.2
    if previous_final_hidden is not None:
      hidden = hidden + previous_final_hidden * 0.07
    prediction = (hidden.squeeze(-1).long() % 15)
    logits = -0.4 * torch.abs(
      torch.arange(17).view(1, 1, -1) - prediction.unsqueeze(-1))
    logits[..., self.mask_index] = -torch.inf
    scores = logits.log_softmax(dim=-1)
    cache = [hidden + 3.0]
    final = hidden + 7.0
    self.calls.append({
      'state': state.clone(), 'sigma': sigma.clone(),
      'dcache': (None if previous_step_kv is None
                 else [entry.clone() for entry in previous_step_kv]),
      'final': (None if previous_final_hidden is None
                else previous_final_hidden.clone()),
      'output_dcache': [entry.clone() for entry in cache],
      'output_final': final.clone(),
    })
    if return_dcachehooping:
      return SimpleNamespace(scores=scores, step_kv=cache, final_hidden=final)
    if return_step_kv:
      return scores, cache
    return scores


@pytest.fixture
def sample_batch():
  tokens = torch.tensor([
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
  ])
  return tokens, torch.ones_like(tokens), [17, 18, 19, 20]


def observe_scores(monkeypatch, model):
  """Record the real model call that supplied each scored condition."""
  observed = []
  original = common.score_masked_tokens

  def record(scores, targets, mask, example_ids, **fields):
    observed.append({
      **fields, 'call': model.calls[-1], 'mask': mask.clone(),
      'targets': targets.clone(), 'example_ids': list(example_ids),
    })
    return original(scores, targets, mask, example_ids, **fields)

  monkeypatch.setattr(common, 'score_masked_tokens', record)
  return observed


def test_reveal_samples_only_new_positions_without_teacher_input(sample_batch, audit):
  tokens, attention, ids = sample_batch
  masks = common.deterministic_nested_masks(
    attention, ids, [0.3, 0.6], seed=71)
  state = common.masked_state(tokens, masks[0.6], RecordingModel.mask_index)
  scores = torch.full((*state.shape, 17), -torch.inf)
  scores[..., 15] = 0.0

  revealed = audit.reveal_from_model(
    state, scores, masks[0.6], masks[0.3], ids,
    seed=12, stream='reveal_test', mask_index=RecordingModel.mask_index)

  newly_visible = masks[0.6] & ~masks[0.3]
  assert torch.all(revealed[newly_visible] == 15)
  assert torch.all(revealed[newly_visible] != tokens[newly_visible])
  assert torch.equal(revealed[~newly_visible], state[~newly_visible])
  assert torch.all(revealed[masks[0.3]] == RecordingModel.mask_index)
  assert torch.equal(state, common.masked_state(tokens, masks[0.6], 16))


def test_reveal_randomness_is_document_stable_when_batch_order_changes(sample_batch, audit):
  tokens, attention, ids = sample_batch
  masks = common.deterministic_nested_masks(
    attention, ids, [0.2, 0.8], seed=123)
  state = common.masked_state(tokens, masks[0.8], RecordingModel.mask_index)
  scores = torch.zeros((*state.shape, 17)).log_softmax(-1)
  kwargs = dict(seed=44, stream='unchanged_stream', mask_index=16)
  full = audit.reveal_from_model(
    state, scores, masks[0.8], masks[0.2], ids, **kwargs)
  order = torch.tensor([2, 0, 3, 1])
  reordered = audit.reveal_from_model(
    state[order], scores[order], masks[0.8][order], masks[0.2][order],
    [ids[index] for index in order.tolist()], **kwargs)
  assert torch.equal(reordered, full[order])
  assert torch.all(full[masks[0.8] & ~masks[0.2]] != 16)


def test_independent_shuffle_uses_distinct_nonself_donors_and_preserves_identity(audit):
  ids = torch.arange(4).reshape(4, 1, 1).float()
  memory = audit.Memory(dcache=[ids.clone()], final=ids.clone() + 100)

  correct = audit.intervene(memory, 'correct')
  independent = audit.intervene(memory, 'shuffle_both_independent')
  coherent = audit.intervene(memory, 'shuffle_both_coherent')

  assert torch.equal(correct.dcache[0], ids)
  assert torch.equal(correct.final, ids + 100)
  cache_donor = independent.dcache[0].flatten()
  final_donor = independent.final.flatten() - 100
  recipient = ids.flatten()
  assert torch.all(cache_donor != recipient)
  assert torch.all(final_donor != recipient)
  assert torch.all(cache_donor != final_donor)
  assert torch.equal(coherent.dcache[0], coherent.final - 100)
  assert torch.equal(memory.dcache[0], ids)
  assert torch.equal(memory.final, ids + 100)


@pytest.mark.parametrize('condition,keep_dcache,keep_final', [
  ('correct', True, True),
  ('absent_dcache', False, True),
  ('absent_final', True, False),
  ('absent_both', False, False),
])
def test_absence_removes_only_the_requested_memory(condition, keep_dcache, keep_final, audit):
  memory = audit.Memory(dcache=[torch.ones(4, 2, 1)], final=torch.ones(4, 2, 1))
  actual = audit.intervene(memory, condition)
  assert (actual.dcache is not None) == keep_dcache
  assert (actual.final is not None) == keep_final


@pytest.mark.parametrize('variant', ['five-forward', 'two-forward'])
@pytest.mark.parametrize('protocol', ['no_warmup', 'full_warmup'])
def test_transition_initialization_and_paired_input_are_exact(
    sample_batch, audit, monkeypatch, variant, protocol):
  tokens, attention, ids = sample_batch
  model = RecordingModel()
  observed = observe_scores(monkeypatch, model)
  phase = audit.Phase('transitions', protocol, 0.6, 0.3)
  rows = audit.evaluate_transition(model, variant, tokens, attention, ids, phase, 71)

  assert model.calls[0]['dcache'] is None
  assert model.calls[0]['final'] is None
  source_index = int(protocol == 'full_warmup')
  if source_index:
    assert torch.all(model.calls[0]['state'][:, 1:] == model.mask_index)
    assert torch.equal(model.calls[1]['dcache'][0], model.calls[0]['output_dcache'][0])
    assert torch.equal(model.calls[1]['final'], model.calls[0]['output_final'])
  source = model.calls[source_index]
  assert (source['state'] == model.mask_index).sum(dim=-1).tolist() == [6] * 4
  assert len(rows) == 4 * 8
  assert len(observed) == 8
  correct = next(item for item in observed if item['condition'] == 'correct')
  assert torch.equal(correct['call']['dcache'][0], source['output_dcache'][0])
  assert torch.equal(correct['call']['final'], source['output_final'])
  for item in observed:
    assert torch.equal(item['call']['state'], correct['call']['state'])
    assert torch.equal(item['mask'], correct['mask'])
    assert torch.equal(item['targets'], tokens)
    assert torch.equal(item['call']['state'][:, 0], tokens[:, 0])
    assert item['mask'].sum(dim=-1).tolist() == [3] * 4


def test_recurrence_reuses_exact_input_but_each_arm_keeps_its_own_history(
    sample_batch, audit, monkeypatch):
  tokens, attention, ids = sample_batch
  model = RecordingModel()
  observed = observe_scores(monkeypatch, model)
  phase = audit.Phase('repeat', 'same_state', 0.3, 0.3)
  rows = audit.evaluate_repeat(
    model, 'two-forward', tokens, attention, ids, phase, 71, [1, 2, 3, 4])
  first = model.calls[0]
  assert first['dcache'] is first['final'] is None
  for call in model.calls:
    assert torch.equal(call['state'], first['state'])
    assert torch.equal(call['sigma'], first['sigma'])
  for condition in audit.condition_names('two-forward', 'repeat'):
    last = first
    for item in [item for item in observed if item.get('condition') == condition]:
      expected = audit.intervene(
        audit.Memory(last['output_dcache'], last['output_final']), condition)
      call = item['call']
      assert (call['dcache'] is None) == (expected.dcache is None)
      assert (call['final'] is None) == (expected.final is None)
      if expected.dcache is not None:
        assert torch.equal(call['dcache'][0], expected.dcache[0])
      if expected.final is not None:
        assert torch.equal(call['final'], expected.final)
      last = call
  frame = pd.DataFrame(rows)
  first_rows = frame[frame.repeat_step == 1]
  assert (first_rows.groupby('example_id').nll.nunique() == 1).all()
  absent = frame[frame.condition == 'absent_both']
  assert (absent.groupby('example_id').nll.nunique() == 1).all()


def test_generated_context_never_reads_hidden_teacher_tokens(
    sample_batch, audit, monkeypatch):
  tokens, attention, ids = sample_batch
  phase = audit.Phase('generated', 'generated', 0.8, 0.2)
  model = RecordingModel()
  observed = observe_scores(monkeypatch, model)
  audit.evaluate_generated(
    model, 'two-forward', tokens, attention, ids, phase, 71, 3, 1.0)
  for step in [1, 2, 3]:
    masks = [item['mask'] for item in observed if item.get('repeat_step') == step]
    assert len(masks) == 5
    assert all(torch.equal(mask, masks[0]) for mask in masks)
  first_revealed = [item['call']['state'] for item in observed
                    if item.get('repeat_step') == 1]
  assert all(torch.equal(state, first_revealed[0]) for state in first_revealed)

  hidden = common.deterministic_nested_masks(attention, ids, [0.8], 71)[0.8]
  changed_targets = torch.where(hidden, (tokens + 5) % 15, tokens)
  other_model = RecordingModel()
  audit.evaluate_generated(
    other_model, 'two-forward', changed_targets, attention, ids, phase, 71, 3, 1.0)
  assert len(model.calls) == len(other_model.calls)
  assert all(torch.equal(left['state'], right['state'])
             for left, right in zip(model.calls, other_model.calls))
  for condition in audit.condition_names('two-forward', 'generated'):
    arm = [item for item in observed if item.get('condition') == condition][:3]
    for previous, current in zip(arm, arm[1:]):
      assert torch.equal(current['call']['state'][~previous['mask']],
                         previous['call']['state'][~previous['mask']])


def test_interrupted_evaluation_resumes_without_recomputing_completed_parts(
    tmp_path, sample_batch, audit, monkeypatch):
  tokens, attention, _ = sample_batch
  args = SimpleNamespace(
    examples=4, batch_size=4, variant='two-forward', seed=71,
    num_workers=0, device='cpu', expected_step=5000,
    data_dir=tmp_path, checkpoint=tmp_path / 'mock.ckpt',
    repeat_steps=[1, 2], generated_steps=2, temperature=1.0)
  phases = [audit.Phase('transitions', 'no_warmup', 0.6, 0.3),
            audit.Phase('repeat', 'same_state', 0.3, 0.3)]
  monkeypatch.setattr(common, 'compose_eval_config', lambda *a, **kw: object())
  monkeypatch.setattr(common, 'load_tokenizer', lambda config: object())
  monkeypatch.setattr(common, 'load_validation_data', lambda *a, **kw: [
    {'input_ids': tokens, 'attention_mask': attention}])
  loaded = []

  def load(*args):
    model = RecordingModel()
    loaded.append(model)
    return model

  monkeypatch.setattr(common, 'load_ema_model', load)
  resumed_dir = tmp_path / 'resumed'
  resumed_dir.mkdir()
  stop = audit.CooperativeStop()
  evaluate_transition = audit.evaluate_transition

  def pause_after_phase(*args, **kwargs):
    rows = evaluate_transition(*args, **kwargs)
    stop.request(15, None)
    return rows

  monkeypatch.setattr(audit, 'evaluate_transition', pause_after_phase)
  audit.run_evaluation(args, phases, resumed_dir, stop, 'test-fingerprint')
  saved = common.part_path(resumed_dir, phases[0].name, 0)
  saved_bytes, saved_mtime = saved.read_bytes(), saved.stat().st_mtime_ns
  assert not common.part_path(resumed_dir, phases[1].name, 0).exists()

  monkeypatch.setattr(audit, 'evaluate_transition', evaluate_transition)
  audit.run_evaluation(
    args, phases, resumed_dir, audit.CooperativeStop(), 'test-fingerprint')
  assert saved.read_bytes() == saved_bytes
  assert saved.stat().st_mtime_ns == saved_mtime
  assert len(loaded) == 2
  assert len(loaded[1].calls) == 6  # First pass + one per repeat arm; no transition.
  audit.run_evaluation(
    args, phases, resumed_dir, audit.CooperativeStop(), 'test-fingerprint')
  assert len(loaded) == 2  # Complete output skips model loading entirely.

  clean_dir = tmp_path / 'uninterrupted'
  clean_dir.mkdir()
  audit.run_evaluation(
    args, phases, clean_dir, audit.CooperativeStop(), 'test-fingerprint')
  for phase in phases:
    pd.testing.assert_frame_equal(
      pd.read_csv(common.part_path(resumed_dir, phase.name, 0)),
      pd.read_csv(common.part_path(clean_dir, phase.name, 0)))


def test_resume_refuses_protocol_changes_without_removing_results(tmp_path, audit):
  output = tmp_path / 'results'
  metadata = {'checkpoint': 'frozen-example', 'seed': 17, 'temperature': 1.0}
  initial = audit.prepare_output(output, metadata)
  common.atomic_write_csv(
    common.part_path(output, 'already_complete', 0), pd.DataFrame({'keep': [1]}))
  assert audit.prepare_output(output, metadata) == initial
  with pytest.raises(RuntimeError, match='fingerprint changed'):
    audit.prepare_output(output, dict(metadata, temperature=0.8))
  assert common.part_path(output, 'already_complete', 0).is_file()




def paired_frames():
  reference = pd.DataFrame({
    'seed': [13, 13, 13], 'example_id': [2, 3, 4],
    'masked_tokens': [1, 3, 2], 'nll': [2.0, 3.0, 4.0],
    'top1_accuracy': [0.3, 0.4, 0.5],
    'top5_accuracy': [0.5, 0.6, 0.7],
  })
  condition = reference.copy()
  condition['nll'] += np.array([-0.4, -0.2, -0.1])
  condition['top1_accuracy'] += np.array([0.2, 0.1, 0.05])
  return condition, reference


def test_bootstrap_keeps_document_repeats_across_mask_seeds_in_one_cluster():
  condition, reference = paired_frames()
  first = plotting.paired_document_bootstrap(condition, reference, 1000, 77)
  repeated_condition = pd.concat([
    condition.assign(seed=seed) for seed in [13, 27, 40]], ignore_index=True)
  repeated_reference = pd.concat([
    reference.assign(seed=seed) for seed in [13, 27, 40]], ignore_index=True)
  repeated = plotting.paired_document_bootstrap(
    repeated_condition, repeated_reference, 1000, 77)

  assert repeated['documents'] == first['documents'] == 3
  assert repeated['seeds'] == 3
  assert repeated['observations'] == 9
  for key in first:
    if key.startswith(('delta_', 'ci95_')):
      assert repeated[key] == pytest.approx(first[key])
  assert first['delta_nll'] == pytest.approx(-0.2)
  assert first['delta_top1_accuracy_pp'] == pytest.approx(10.0)


def test_shuffle_bootstrap_clusters_shared_donor_batches_across_mask_seeds():
  condition, reference = paired_frames()
  first = plotting.paired_document_bootstrap(
    condition, reference, 1000, 77, cluster_size=2)
  repeated = plotting.paired_document_bootstrap(
    pd.concat([condition.assign(seed=seed) for seed in [13, 27, 40]]),
    pd.concat([reference.assign(seed=seed) for seed in [13, 27, 40]]),
    1000, 77, cluster_size=2)
  assert first['documents'] == repeated['documents'] == 3
  assert first['clusters'] == repeated['clusters'] == 2
  assert repeated['cluster_unit'] == 'evaluation_batch'
  for key in first:
    if key.startswith(('delta_', 'ci95_')):
      assert repeated[key] == pytest.approx(first[key])



def test_pairing_rejects_missing_documents_and_mismatched_masks():
  condition, reference = paired_frames()
  with pytest.raises(plotting.PairingError, match='Different'):
    plotting.paired_document_bootstrap(condition.iloc[:-1], reference, 100, 77)
  with pytest.raises(plotting.PairingError, match='counts differ'):
    plotting.paired_document_bootstrap(
      condition.assign(masked_tokens=100), reference, 100, 77)


def test_only_complete_atomic_parts_are_loaded_when_run_is_partial(tmp_path):
  condition, _ = paired_frames()
  raw = condition.assign(
    variant='five-forward', condition='correct', family='transition',
    protocol='no_warmup', s_mask_ratio=0.4, t_mask_ratio=0.3, repeat_step=2)
  run = tmp_path / 'five-forward' / 'seed-13'
  path = common.part_path(run, 'first_phase', 0)
  common.atomic_write_csv(path, raw)
  # A crash before atomic rename must not publish this half-written part.
  common.atomic_write_csv(path.with_suffix('.csv.tmp'), raw.iloc[:1])
  common.atomic_write_csv(run / 'summary.csv', raw)

  loaded, coverage = plotting.load_parts(tmp_path)

  assert len(loaded) == 3
  assert len(coverage) == 1
  assert not bool(coverage.iloc[0]['complete'])
  assert coverage.iloc[0]['parts'] == 1
  assert plotting.summarize(loaded).iloc[0].conditional_nll == pytest.approx(
    np.average(raw.nll, weights=raw.masked_tokens))
  common.atomic_write_csv(common.part_path(run, 'duplicate_phase', 0), raw)
  with pytest.raises(ValueError, match='Duplicate'):
    plotting.load_parts(tmp_path)


def test_plot_command_handles_partial_results_from_all_protocols(tmp_path):
  condition, reference = paired_frames()
  for variant in ['five-forward', 'two-forward']:
    pieces = []
    for name, metric_frame in [('correct', condition), ('absent_both', reference)]:
      for protocol in ['no_warmup', 'full_warmup', 'same_state', 'generated']:
        for step in ([1, 2] if protocol in ['same_state', 'generated'] else [2]):
          pieces.append(metric_frame.assign(
            variant=variant, condition=name,
            family=('transition' if 'warmup' in protocol else protocol),
            protocol=protocol, s_mask_ratio=(0.3 if protocol == 'same_state' else 0.4),
            t_mask_ratio=0.3, repeat_step=step))
    common.atomic_write_csv(
      common.part_path(tmp_path / 'raw' / variant / 'seed-13', 'phase', 0),
      pd.concat(pieces, ignore_index=True))

  plotting.main([
    '--input-root', str(tmp_path / 'raw'), '--output-dir', str(tmp_path / 'figures'),
    '--bootstrap-samples', '20', '--expected-variants', 'five-forward', 'two-forward'])

  figures = tmp_path / 'figures'
  for name in ['transition_quality_no_warmup.png',
               'transition_quality_full_warmup.png',
               'warmup_sensitivity.png', 'same_state_recurrence.png',
               'generated_trajectory.png']:
    assert (figures / name).stat().st_size > 1000
  assert 'PARTIAL snapshot' in (figures / 'REPORT.md').read_text()
  assert pd.read_csv(figures / 'pairing_issues.csv').empty
