"""Actual small-BD3 branch, gradient, RNG, loss and resume checks on CPU."""
import copy
import csv
import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import numpy as np
import torch

from owt.model import OWTMDM
from owt.neighbor import NeighborHeads, neighbor_terms
from owt.source_pairing import pairing_terms
from owt.test_initialization import config, model
from owt.transformer_np import NeighborTransformer, process_branch, transformer_neighbor_terms
from owt.transformer_np_metrics import TransformerLocalMetrics, transformer_gradient_norms
from owt.transformer_np_model import TransformerNPMDM
from models.dit import Rotary

torch.set_num_threads(2)


def make_model(policy='masked_source', dropout=.2, checkpoint=True):
    cfg = config('mdm_np_zero_init_transformer_' + policy)
    cfg.model.dropout = dropout
    cfg.mechanisms.np.branch_checkpoint = checkpoint
    tokenizer = SimpleNamespace(vocab_size=7, mask_token=None, all_special_ids=[])
    with patch('diffusion.metrics.Metrics', return_value=torch.nn.Module()):
        return TransformerNPMDM(cfg, tokenizer).train()


def batch():
    x = torch.tensor([[1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4],
                      [6,5,4,3,2,1,6,5,4,3,2,1,6,5,4,3]])
    return x, torch.ones_like(x)


@pytest.mark.parametrize('policy', ['target_only', 'masked_source'])
def test_loss_statistics_and_gradients_match_existing_linear_rules(policy):
    torch.manual_seed(21)
    heads = NeighborHeads(4, 8, [-1, 1]).double()
    reference = copy.deepcopy(heads)
    hidden = torch.randn(2, 6, 4, dtype=torch.float64, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    clean = torch.tensor([[1,2,3,4,5,1], [2,1,5,0,3,4]])
    noisy = torch.tensor([[1,7,7,4,7,7], [7,7,5,0,7,7]])
    valid = torch.ones_like(clean)
    factors = torch.tensor([[2.], [3.]], dtype=torch.float64)
    values, stats = transformer_neighbor_terms(heads, {-1:hidden, 1:hidden}, clean, noisy,
        valid, 7, [0], factors, valid.sum(), policy, chunk_size=1)
    if policy == 'masked_source':
        expected, old_stats = pairing_terms(reference, other, clean, noisy, valid, 7, [0],
            factors, valid.sum(), policy, chunk_size=1)
        for offset in (-1, 1):
            torch.testing.assert_close(stats[offset], old_stats[offset], rtol=0, atol=0)
    else:
        expected, counts = neighbor_terms(reference, other, clean, noisy, valid, 7, [0],
            factors, valid.sum(), chunk_size=1)
        for offset in (-1, 1):
            assert stats[offset][:, 2].sum() == counts[offset]
            assert stats[offset][:, 2].sum() == stats[offset][:, 0].sum()
    for offset in (-1, 1):
        torch.testing.assert_close(values[offset], expected[offset], rtol=0, atol=0)
    sum(values.values()).backward()
    sum(expected.values()).backward()
    torch.testing.assert_close(hidden.grad, other.grad, rtol=0, atol=0)
    for a, b in zip(heads.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


@pytest.mark.parametrize('policy', ['target_only', 'masked_source'])
def test_main_initialization_rng_first_loss_and_ema_match_baseline(policy):
    cfg = config('mdm_np_zero_init_low_weight')
    cfg.model.dropout = .2
    torch.manual_seed(71)
    baseline = model(cfg)
    initial_rng = torch.get_rng_state().clone()
    torch.manual_seed(71)
    new = make_model(policy)
    torch.testing.assert_close(torch.get_rng_state(), initial_rng, rtol=0, atol=0)
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
    assert len(list(new._get_parameters())) == len(new.ema.shadow_params)
    for value, shadow in zip(new._get_parameters(), new.ema.shadow_params):
        torch.testing.assert_close(value, shadow, rtol=0, atol=0)
    x, valid = batch()
    torch.manual_seed(42)
    baseline._loss(x, valid.clone()).loss.backward()
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    result = new._loss(x, valid.clone())
    result.loss.backward()
    torch.testing.assert_close(new._last_components['main_elbo'], baseline._last_components['main_elbo'], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
    assert torch.isfinite(result.loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())
    a, b = new.backbone.neighbor_branches
    assert a.block.attn_qkv.weight.data_ptr() != b.block.attn_qkv.weight.data_ptr()
    assert a.block.attn_qkv.weight.any() and b.block.attn_qkv.weight.any()
    assert all(not head[-1].weight.any() for head in new.backbone.neighbor_heads.heads)


def test_checkpointed_dropout_replays_outputs_gradients_and_preserves_rng():
    torch.manual_seed(19)
    branch = NeighborTransformer(config('mdm_np_zero_init_transformer_target_only')).train()
    branch.block.dropout = .4
    with torch.no_grad():
        branch.block.adaLN_modulation.bias.fill_(.3)
    reference = copy.deepcopy(branch)
    hidden = torch.randn(2, 16, 32, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    c = torch.randn(2, 16)
    rotary = Rotary(8)(hidden)
    rng = torch.get_rng_state().clone()
    value = process_branch(branch, hidden, rotary, c, seed=314159)
    value.square().sum().backward()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    expected = process_branch(reference, other, rotary, c, seed=314159, use_checkpoint=False)
    expected.square().sum().backward()
    torch.testing.assert_close(value, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, other.grad, rtol=0, atol=0)
    for a, b in zip(branch.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    changed = process_branch(reference, other, rotary, c, seed=314160, use_checkpoint=False)
    assert not torch.equal(changed, expected)


def test_np_learns_then_reaches_branch_attention_and_shared_trunk_before_last_block():
    torch.manual_seed(7)
    new = make_model(dropout=0.)
    # Remove the main loss only in this gradient-route probe.
    new.config.objective.current_weight = 0.
    optimizer = torch.optim.AdamW(new._get_parameters(), lr=1e-3, weight_decay=0.)
    x, valid = batch()
    with patch.object(new, 'q_xt', side_effect=lambda clean, p, **kw: torch.full_like(clean, new.mask_index)):
        for step in range(3):
            optimizer.zero_grad()
            torch.manual_seed(42)
            new._loss(x, valid.clone()).loss.backward()
            assert all(p.grad is not None for p in new.parameters())
            assert all(not p.grad.any() for p in new.backbone.blocks[-1].parameters())
            if step == 0:
                assert all(not p.grad.any() for p in new.backbone.neighbor_branches.parameters())
                assert all(head[-1].weight.grad.abs().sum() > 0 for head in new.backbone.neighbor_heads.heads)
            if step == 2:
                assert all(branch.block.attn_qkv.weight.grad.abs().sum() > 0 for branch in new.backbone.neighbor_branches)
                assert new.backbone.vocab_embed.embedding.grad.abs().sum() > 0
            optimizer.step()


def test_empty_pairs_keep_processing_and_readouts_in_ddp_graph():
    new = make_model()
    x, valid = batch()
    def isolated_masks(clean, p, **kw):
        noisy = clean.clone()
        noisy[:, 2] = new.mask_index
        return noisy
    with patch.object(new, 'q_xt', side_effect=isolated_masks):
        loss = new._loss(x, valid.clone()).loss
    assert all(s[:, 2].sum() == 0 for s in new._last_pair_statistics.values())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())
    assert all(not p.grad.any() for p in new.backbone.neighbor_branches.parameters())


def test_validation_uses_only_main_branch_and_preserves_existing_scores():
    torch.manual_seed(71)
    new = make_model().eval()
    baseline = model(config('mdm_np_zero_init_low_weight')).eval()
    baseline.load_state_dict({k:v for k,v in new.state_dict().items() if k in baseline.state_dict()})
    x, valid = batch()
    def unexpected(*args):
        raise AssertionError('Auxiliary processing must not run during main validation')
    handles = [branch.register_forward_pre_hook(unexpected) for branch in new.backbone.neighbor_branches]
    try:
        torch.manual_seed(42)
        expected = baseline._loss(x, valid.clone()).loss
        torch.manual_seed(42)
        actual = new._loss(x, valid.clone()).loss
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert new.branch_calls == 0
    finally:
        for handle in handles:
            handle.remove()


def test_model_optimizer_and_private_dropout_counter_roundtrip():
    new = make_model()
    optimizer = torch.optim.AdamW(new._get_parameters(), lr=1e-3)
    x, valid = batch()
    for _ in range(2):
        optimizer.zero_grad()
        new._loss(x, valid.clone()).loss.backward()
        optimizer.step()
    saved_model, saved_optimizer = copy.deepcopy(new.state_dict()), copy.deepcopy(optimizer.state_dict())
    receipt = {}
    with patch.object(OWTMDM, 'on_save_checkpoint'):
        new.on_save_checkpoint(receipt)
    rng = torch.get_rng_state().clone()
    restored = make_model()
    restored.load_state_dict(saved_model)
    other_optimizer = torch.optim.AdamW(restored._get_parameters(), lr=1e-3)
    other_optimizer.load_state_dict(saved_optimizer)
    with patch.object(OWTMDM, 'on_load_checkpoint'):
        restored.on_load_checkpoint(receipt)
    for mdl, opt in [(new, optimizer), (restored, other_optimizer)]:
        torch.set_rng_state(rng)
        opt.zero_grad()
        loss = mdl._loss(x, valid.clone()).loss
        loss.backward()
        opt.step()
    assert new.branch_calls == restored.branch_calls == 3
    for name, value in new.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    restored.np_config.weights = [float(weight) / 2 for weight in restored.np_config.weights]
    with pytest.raises(ValueError, match='identical transformer'):
        restored.on_load_checkpoint(receipt)


def test_joint_gradient_partition_includes_auxiliary_processing_separately():
    new = make_model()
    for parameter in new.parameters():
        parameter.grad = torch.ones_like(parameter)
    result = transformer_gradient_norms(new, 1.)
    auxiliary = sum(p.numel() for p in new.backbone.neighbor_branches.parameters())
    assert result['neighbor_processing_l2'] == pytest.approx(auxiliary ** .5, rel=1e-6)
    assert result['joint_l2'] ** 2 == pytest.approx(sum(result[k] ** 2 for k in
        ['shared_trunk_l2', 'main_readout_l2', 'neighbor_readouts_l2', 'neighbor_processing_l2']))


def test_clean_input_cross_attention_is_rejected():
    cfg = config('mdm_np_zero_init_transformer_target_only')
    cfg.algo.cross_attn = True
    with pytest.raises(ValueError, match='clean-input'):
        TransformerNPMDM(cfg, SimpleNamespace(vocab_size=7, mask_token=None, all_special_ids=[]))


def test_diagnostic_forward_uses_whole_canvas_and_branch_features_without_labels():
    new = make_model().eval()
    new.backbone.force_fp32_eval = True
    x, _ = batch()
    noisy = x.clone()
    noisy[:, 2:5] = new.mask_index
    with torch.inference_mode():
        expected = new.forward(noisy, sigma=torch.ones(2, 1))
        main, features = new.diagnostic_forward(noisy, sigma=torch.ones(2, 1))
    torch.testing.assert_close(main, expected, rtol=0, atol=0)
    assert set(features) == {-1, 1}
    assert all(h.shape == (2, 16, 32) and h.dtype == torch.float32 for h in features.values())
    assert new.branch_calls == 0
    assert not new.backbone.blocks[-1]._forward_pre_hooks
    new.train()
    with pytest.raises(ValueError, match='eval mode'):
        new.diagnostic_forward(noisy, sigma=torch.ones(2, 1))


def test_checkpoint_matches_uncheckpointed_branch_under_bfloat16_autocast():
    torch.manual_seed(9)
    branch = NeighborTransformer(config('mdm_np_zero_init_transformer_target_only')).train()
    with torch.no_grad():
        branch.block.adaLN_modulation.bias.fill_(.3)
    other = copy.deepcopy(branch)
    h = torch.randn(2, 16, 32, requires_grad=True)
    reference = h.detach().clone().requires_grad_()
    c = torch.randn(2, 16)
    rotary = Rotary(8)(h)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = process_branch(branch, h, rotary, c, 19)
        expected = process_branch(other, reference, rotary, c, 19, use_checkpoint=False)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(h.grad, reference.grad, rtol=0, atol=0)
    for a, b in zip(branch.parameters(), other.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_fixed_validation_restores_training_rng_and_records_actual_scores(tmp_path):
    callback = TransformerLocalMetrics(tmp_path)
    trainer = SimpleNamespace(global_rank=0, global_step=500, is_global_zero=True,
        sanity_checking=False, callback_metrics={'val/nll': 4., 'val/ppl': 54., 'val/bpd': 5.7})
    module = SimpleNamespace(config=SimpleNamespace(seed=1))
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    callback.on_validation_start(trainer, module)
    random.random()
    np.random.rand()
    torch.rand(5)
    callback.on_validation_end(trainer, module)
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
    with (tmp_path / 'local_metrics/validation.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]['optimizer_step'] == '500' and float(rows[0]['val_nll']) == 4.
    assert callback.rng is None
