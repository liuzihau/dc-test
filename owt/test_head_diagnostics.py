from types import SimpleNamespace

import numpy as np
import pytest
import torch

from owt.head_diagnostics import summarize
from owt.reveal_sweep import score_batch


class FakeNP(torch.nn.Module):
    def __init__(self, enabled=True):
        super().__init__()
        self.np_config = SimpleNamespace(enabled=enabled)
        self.mask_index = 3
        self.boundary_ids = [2]
        self.calls = 0
        self.backbone = torch.nn.Module()
        self.backbone.output_layer = torch.nn.Module()
        self.backbone.output_layer.linear = self.projection([0, 0, 1, 0, 1, 1, 2])
        self.backbone.neighbor_heads = torch.nn.Module()
        self.backbone.neighbor_heads.offsets = (1, -1)
        self.backbone.neighbor_heads.heads = torch.nn.ModuleList([
            self.projection([0, 0, 1, 1, 2, 0, 0]),
            self.projection([0, 0, 0, 0, 0, 0, 0])])

    @staticmethod
    def projection(predictions):
        layer = torch.nn.Linear(7, 4, bias=False)
        with torch.no_grad():
            layer.weight.zero_()
            for index, prediction in enumerate(predictions):
                layer.weight[prediction, index] = 4
            layer.weight[3] = 100  # MASK must be excluded, even if it wins raw logits.
        return layer

    def forward(self, inputs, sigma):
        self.calls += 1
        hidden = torch.eye(7)[None].expand(len(inputs), -1, -1)
        logits = self.backbone.output_layer.linear(hidden)
        logits[..., 3] = -torch.inf
        return logits.log_softmax(-1)


def inputs():
    clean = np.array([[0, 0, 0, 0, 0, 0, 2]])
    canvas = np.full_like(clean, 3)
    canvas[:, 0] = 0
    return clean, canvas, canvas == 3, np.zeros_like(canvas, dtype=bool)


def test_same_forward_preserves_primary_scores_and_aligns_neighbor_targets():
    model = FakeNP()
    clean, canvas, masked, wrong = inputs()
    baseline = score_batch(model, clean, canvas, masked, wrong, 'cpu')[0]
    saved = {}
    row = score_batch(model, clean, canvas, masked, wrong, 'cpu',
                      capture_heads=True, prediction_output=saved)[0]
    assert model.calls == 2  # One backbone pass per call, including both auxiliary heads.
    assert {key: row[key] for key in baseline} == baseline
    assert saved['left_prediction'][0].tolist() == [-1, 0, 0, 1, 1, 2, -1]
    assert saved['right_prediction'][0].tolist() == [-1, 0, 0, 0, 0, -1, -1]
    assert np.isnan(saved['main_top1_probability'][0, 0])
    assert np.isfinite(saved['main_top1_probability'][saved['scored']]).all()
    for direction in ['left', 'right']:
        eligible = saved[direction + '_prediction'] >= 0
        confidence = saved[direction + '_top1_probability']
        assert np.isnan(confidence[~eligible]).all()
        assert ((confidence[eligible] > 0) & (confidence[eligible] <= 1)).all()
    assert not model.backbone.output_layer.linear._forward_pre_hooks
    assert row['head_left_all_targets'] == 5
    for case in ['both_correct', 'rescue', 'main_only_correct', 'both_wrong_same', 'both_wrong_different']:
        assert row['head_left_all_' + case] == 1
    assert row['head_left_correct_revealed_targets'] == 1
    assert row['head_left_masked_targets'] == 4
    assert row['head_left_wrong_revealed_targets'] == 0
    assert row['head_union_targets'] == 4
    assert row['head_union_rescue'] == 2
    cell = summarize([dict(variant='mdm_np', mask_ratio=.8, correct_fraction=1., **row)])[0]
    metrics = cell['groups']['left_all']
    assert metrics['agreement'] == pytest.approx(.4)
    assert metrics['absolute_rescue'] == pytest.approx(.2)
    assert metrics['conditional_rescue'] == pytest.approx(1 / 3)
    assert metrics['main_accuracy'] == metrics['auxiliary_accuracy'] == pytest.approx(.4)
    assert metrics['main_accuracy'] - metrics['auxiliary_accuracy'] == pytest.approx(
        (row['head_left_all_main_only_correct'] - row['head_left_all_rescue']) / 5)
    assert cell['oracle_top1_selector_accuracy'] == 1.


def test_wrong_revealed_source_stratum_uses_source_index_not_target_index():
    model = FakeNP()
    clean, canvas, masked, wrong = inputs()
    canvas[0, 0] = 1
    wrong[0, 0] = True
    row = score_batch(model, clean, canvas, masked, wrong, 'cpu', capture_heads=True)[0]
    assert row['head_left_wrong_revealed_targets'] == 1
    assert row['head_left_wrong_revealed_both_correct'] == 1
    assert row['head_left_correct_revealed_targets'] == 0
    assert row['head_right_wrong_revealed_targets'] == 0


def test_empty_eligible_population_is_unavailable_not_fake_zero_accuracy():
    model = FakeNP()
    clean, canvas, masked, wrong = inputs()
    clean[:] = 2
    row = score_batch(model, clean, canvas, masked, wrong, 'cpu', capture_heads=True)[0]
    cell = summarize([dict(variant='mdm_np', mask_ratio=1., correct_fraction=1., **row)])[0]
    assert cell['groups']['left_all']['targets'] == 0
    assert cell['groups']['left_all']['auxiliary_ce'] is None
    assert cell['groups']['left_all']['conditional_rescue'] is None
    assert cell['oracle_top1_selector_accuracy'] is None


def test_mdm_saves_main_predictions_without_inventing_an_auxiliary_head():
    model = FakeNP(enabled=False)
    saved = {}
    clean, canvas, masked, wrong = inputs()
    row = score_batch(model, clean, canvas, masked, wrong, 'cpu', capture_heads=True, prediction_output=saved)[0]
    assert 'main_prediction' in saved and 'left_prediction' not in saved
    assert not any(key.startswith('head_') for key in row)
    assert summarize([dict(variant='mdm', mask_ratio=1., correct_fraction=1., **row)]) == []
