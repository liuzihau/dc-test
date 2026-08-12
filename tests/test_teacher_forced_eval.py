import numpy as np
import pandas as pd
import pytest
import torch

from scripts.eval import dcache_eval_common as common


def test_deterministic_masks_have_exact_counts_and_are_nested():
  attention = torch.ones((2, 11), dtype=torch.long)
  ratios = [0.2, 0.5, 1.0]

  first = common.deterministic_nested_masks(
    attention, [7, 8], ratios, seed=123)
  second = common.deterministic_nested_masks(
    attention, [7, 8], ratios, seed=123)

  for ratio in ratios:
    assert torch.equal(first[ratio], second[ratio])
    assert torch.all(~first[ratio][:, 0])
    assert first[ratio].sum(dim=-1).tolist() == [round(10 * ratio)] * 2
  assert torch.all(first[0.2] <= first[0.5])
  assert torch.all(first[0.5] <= first[1.0])


def test_masked_token_scoring_ignores_visible_positions():
  logits = torch.tensor([[[8.0, 0.0, 0.0],
                          [0.0, 5.0, 0.0],
                          [0.0, 0.0, 5.0]]])
  log_probs = logits.log_softmax(dim=-1)
  targets = torch.tensor([[2, 1, 2]])
  mask = torch.tensor([[False, True, True]])

  rows = common.score_masked_tokens(
    log_probs, targets, mask, [42], condition='test')

  assert len(rows) == 1
  assert rows[0]['example_id'] == 42
  assert rows[0]['masked_tokens'] == 2
  assert rows[0]['top1_accuracy'] == 1.0
  assert rows[0]['top5_accuracy'] == 1.0
  assert rows[0]['nll'] < 0.02


def test_aggregate_is_token_weighted_and_ppl_is_exp_nll():
  frame = pd.DataFrame([
    {'condition': 'a', 'masked_tokens': 1, 'nll': 1.0,
     'top1_accuracy': 1.0, 'top5_accuracy': 1.0},
    {'condition': 'a', 'masked_tokens': 3, 'nll': 3.0,
     'top1_accuracy': 0.0, 'top5_accuracy': 1.0},
  ])

  summary = common.aggregate_metrics(frame, ['condition']).iloc[0]

  assert summary.conditional_nll == 2.5
  assert np.isclose(summary.conditional_ppl, np.exp(2.5))
  assert summary.top1_accuracy == 0.25
  assert summary.top5_accuracy == 1.0


def test_paired_delta_sign_is_condition_minus_reference():
  rows = []
  for example_id, (correct, no_cache) in enumerate([(1.0, 1.4), (2.0, 2.2)]):
    for condition, nll in [
        ('dcache_correct', correct), ('dcache_no_cache', no_cache)]:
      rows.append({
        'example_id': example_id,
        'mask_ratio': 0.5,
        'condition': condition,
        'nll': nll,
      })
  paired = common.paired_condition_differences(
    pd.DataFrame(rows),
    group_columns=['mask_ratio'],
    comparisons=[('dcache_no_cache', 'dcache_correct')],
    seed=9,
    bootstrap_samples=200)

  assert len(paired) == 1
  assert np.isclose(paired.iloc[0].mean_delta_nll, 0.3)
  assert paired.iloc[0].ci95_low > 0


def test_cache_controls_preserve_shapes_and_do_not_alias_values():
  cache = [torch.arange(24).reshape(3, 2, 4)]

  shuffled = common.roll_cache_batch(cache)
  zeroed = common.zero_cache(cache)

  assert shuffled[0].shape == cache[0].shape
  assert torch.equal(shuffled[0][0], cache[0][-1])
  assert torch.count_nonzero(zeroed[0]) == 0
  assert zeroed[0].data_ptr() != cache[0].data_ptr()


def test_restart_manifest_reuses_only_identical_protocol(tmp_path):
  output = tmp_path / 'evaluation'

  common.prepare_output(output, {'protocol': 'a', 'examples': 8}, force=False)
  common.prepare_output(output, {'protocol': 'a', 'examples': 8}, force=False)

  with pytest.raises(RuntimeError, match='metadata differs'):
    common.prepare_output(
      output, {'protocol': 'a', 'examples': 9}, force=False)
