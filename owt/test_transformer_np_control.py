"""B mechanics: native feature routes, matched exposure, private RNG and resume."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from owt.model import OWTMDM
from owt.neighbor import NeighborHeads
from owt.source_pairing import pairing_terms, private_pair_generator
from owt.test_initialization import config
from owt.test_transformer_np import make_model as make_a, batch
from owt.transformer_np import transformer_neighbor_terms
from owt.transformer_np_control import MatchedTransformerNPMDM, matched_transformer_terms

torch.set_num_threads(2)


def make_model(dropout=.2):
    cfg = config('mdm_np_zero_init_transformer_pair_count_control')
    cfg.model.dropout = dropout
    tokenizer = SimpleNamespace(vocab_size=7, mask_token=None, all_special_ids=[])
    with patch('diffusion.metrics.Metrics', return_value=torch.nn.Module()):
        return MatchedTransformerNPMDM(cfg, tokenizer).train()


def canvas():
    clean = torch.tensor([[1,2,3,4,5,1], [2,1,5,0,3,4], [1,2,3,4,5,6], [2,3,4,5,6,1]])
    noisy = torch.tensor([[1,7,7,4,7,7], [7,7,5,0,7,7], [7,7,7,7,7,7], [2,3,7,5,6,1]])
    return clean, noisy, torch.ones_like(clean), torch.tensor([[2.],[3.],[4.],[5.]], dtype=torch.float64)


def test_native_loss_and_gradients_equal_established_count_control():
    torch.manual_seed(21)
    heads = NeighborHeads(4, 8, [-1, 1]).double(); reference = copy.deepcopy(heads)
    hidden = torch.randn(4, 6, 4, dtype=torch.float64, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    clean, noisy, valid, factors = canvas()
    rng = torch.get_rng_state().clone()
    value, stats = matched_transformer_terms(heads, {-1:hidden, 1:hidden}, clean, noisy, valid,
        7, [0], factors, valid.sum(), 'matched_pair_count', private_pair_generator(271828, 3, 1), chunk_size=1)
    expected, old_stats = pairing_terms(reference, other, clean, noisy, valid, 7, [0], factors,
        valid.sum(), 'matched_pair_count', private_pair_generator(271828, 3, 1), chunk_size=1)
    for offset in (-1, 1):
        torch.testing.assert_close(value[offset], expected[offset], rtol=0, atol=0)
        torch.testing.assert_close(stats[offset], old_stats[offset], rtol=0, atol=0)
    sum(value.values()).backward(); sum(expected.values()).backward()
    torch.testing.assert_close(hidden.grad, other.grad, rtol=0, atol=0)
    for a, b in zip(heads.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_counts_and_weight_mass_match_a_with_visible_sources_selected():
    heads = NeighborHeads(4, 8, [-1, 1]).double()
    hidden = torch.randn(4, 6, 4, dtype=torch.float64)
    clean, noisy, valid, factors = canvas()
    _, a = transformer_neighbor_terms(heads, {-1:hidden, 1:hidden}, clean, noisy, valid,
        7, [0], factors, valid.sum(), 'masked_source')
    visible = 0
    for calls in range(8):
        _, b = matched_transformer_terms(heads, {-1:hidden, 1:hidden}, clean, noisy, valid,
            7, [0], factors, valid.sum(), 'matched_pair_count', private_pair_generator(271828, calls, 0))
        for offset in (-1, 1):
            torch.testing.assert_close(a[offset][:, [0,1,2,4,5,6]], b[offset][:, [0,1,2,4,5,6]], rtol=0, atol=0)
            visible += float((b[offset][:, 2]-b[offset][:, 3]).sum())
    assert visible > 0
    # Assert each row separately, including boundary and isolated-mask rows.
    for row in range(len(clean)):
        sl = slice(row, row+1)
        _, a_row = transformer_neighbor_terms(heads, {-1:hidden[sl], 1:hidden[sl]}, clean[sl], noisy[sl], valid[sl],
            7, [0], factors[sl], valid[sl].sum(), 'masked_source')
        _, b_row = matched_transformer_terms(heads, {-1:hidden[sl], 1:hidden[sl]}, clean[sl], noisy[sl], valid[sl],
            7, [0], factors[sl], valid[sl].sum(), 'matched_pair_count', private_pair_generator(271828, 0, 0))
        for offset in (-1, 1):
            torch.testing.assert_close(a_row[offset][:, [0,1,2,4,5,6]], b_row[offset][:, [0,1,2,4,5,6]], rtol=0, atol=0)
    bad = factors.expand_as(clean).clone(); bad[0, 2] += 1
    with pytest.raises(ValueError, match='row-constant'):
        matched_transformer_terms(heads, {-1:hidden, 1:hidden}, clean, noisy, valid, 7, [0], bad,
            valid.sum(), 'matched_pair_count', private_pair_generator(271828, 0, 0))


def test_direction_uses_its_own_native_features():
    heads = NeighborHeads(4, 8, [-1, 1]).double()
    left = torch.randn(4, 6, 4, dtype=torch.float64, requires_grad=True)
    right = torch.randn(4, 6, 4, dtype=torch.float64, requires_grad=True)
    clean, noisy, valid, factors = canvas()
    values, _ = matched_transformer_terms(heads, {-1:left, 1:right}, clean, noisy, valid,
        7, [0], factors, valid.sum(), 'matched_pair_count', private_pair_generator(271828, 0, 0))
    values[-1].backward()
    assert left.grad.abs().sum() > 0 and right.grad is None


def test_a_b_initialization_main_loss_and_rng_match():
    torch.manual_seed(71); a = make_a(); rng = torch.get_rng_state().clone()
    torch.manual_seed(71); b = make_model()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
    assert b.config.mechanisms.np.source_policy == b.np_config.source_policy == 'matched_pair_count'
    assert b.hparams.config.mechanisms.np.source_policy == 'matched_pair_count'
    x, valid = batch()
    torch.manual_seed(42); a._loss(x, valid.clone()).loss.backward(); after_a = torch.get_rng_state().clone()
    torch.manual_seed(42); b._loss(x, valid.clone()).loss.backward()
    torch.testing.assert_close(a._last_components['main_elbo'], b._last_components['main_elbo'], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), after_a, rtol=0, atol=0)
    for offset in (-1, 1):
        torch.testing.assert_close(a._last_pair_statistics[offset][:, [0,1,2,4,5,6]],
            b._last_pair_statistics[offset][:, [0,1,2,4,5,6]], rtol=0, atol=0)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.parameters())
    assert all(torch.equal(p, shadow) for p, shadow in zip(b._get_parameters(), b.ema.shadow_params))


def test_empty_pairs_connect_every_parameter_and_validation_is_main_only():
    b = make_model(); x, valid = batch()
    def corrupt(clean, p, **kw):
        noisy = clean.clone(); noisy[:, 2] = b.mask_index; return noisy
    with patch.object(b, 'q_xt', side_effect=corrupt):
        loss = b._loss(x, valid.clone()).loss
    loss.backward()
    assert all(s[:, 2].sum() == 0 for s in b._last_pair_statistics.values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.parameters())
    assert all(not p.grad.any() for p in b.backbone.neighbor_branches.parameters())
    b.eval()
    with patch.object(b.backbone.neighbor_branches[0], 'forward', side_effect=AssertionError('unexpected branch')):
        assert torch.isfinite(b._loss(x, valid.clone()).loss)
    assert b.branch_calls == 1


def test_model_optimizer_counter_resume_is_exact_and_changed_seed_rejected():
    b = make_model(); optimizer = torch.optim.AdamW(b._get_parameters(), lr=1e-3)
    x, valid = batch()
    for _ in range(2):
        optimizer.zero_grad(); b._loss(x, valid.clone()).loss.backward(); optimizer.step()
    weights, optim = copy.deepcopy(b.state_dict()), copy.deepcopy(optimizer.state_dict())
    receipt = {}
    with patch.object(OWTMDM, 'on_save_checkpoint'): b.on_save_checkpoint(receipt)
    rng = torch.get_rng_state().clone()
    restored = make_model(); restored.load_state_dict(weights)
    second = torch.optim.AdamW(restored._get_parameters(), lr=1e-3); second.load_state_dict(optim)
    with patch.object(OWTMDM, 'on_load_checkpoint'): restored.on_load_checkpoint(receipt)
    for model, opt in [(b, optimizer), (restored, second)]:
        torch.set_rng_state(rng); opt.zero_grad(); model._loss(x, valid.clone()).loss.backward(); opt.step()
    for name, value in b.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    assert b.branch_calls == restored.branch_calls == 3
    restored.np_config.pair_selection_seed += 1
    with pytest.raises(ValueError, match='identical transformer'): restored.on_load_checkpoint(receipt)


def test_b_recipe_changes_only_source_policy_and_private_pair_seed():
    from omegaconf import OmegaConf
    a = config('mdm_np_zero_init_transformer_masked_source')
    b = config('mdm_np_zero_init_transformer_pair_count_control')
    assert list(b.mechanisms.np.weights) == [.25, .25]
    b.mechanisms.np.source_policy = a.mechanisms.np.source_policy
    assert b.mechanisms.np.pop('pair_selection_seed') == 271828
    assert OmegaConf.to_container(a.mechanisms) == OmegaConf.to_container(b.mechanisms)
    assert a.objective == b.objective


@pytest.mark.parametrize('field,value', [('np_weight_per_direction', .05), ('scientific_review_complete', False)])
def test_schedule_rejects_changed_design_before_launch(tmp_path, field, value):
    import json
    from owt.transformer_np_control_schedule import verify_selection, B, RUN_ROOT
    selection = dict(selected_variant=B, execution_ready=True, scientific_review_complete=True,
        np_weight_per_direction=.25, source_policy='matched_pair_count', optimizer_steps=5000,
        run_root=str(RUN_ROOT), user_instruction='do B first and during that we can derive some idea')
    selection[field] = value
    path = tmp_path / 'selection.json'; path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match='reviewed user-selected'):
        verify_selection(path)


def test_busy_gpu_lock_prevents_worker_creation(tmp_path, monkeypatch):
    import fcntl
    import owt.transformer_np_control_schedule as schedule
    monkeypatch.setattr(schedule, 'ROOT', tmp_path)
    monkeypatch.setattr(schedule, 'LOCK', Path('gpu.lock'))
    monkeypatch.setattr(schedule, 'verify_selection', lambda *args: {'np_weight_per_direction': .25})
    monkeypatch.setattr(schedule.sys, 'argv', ['schedule', '--selection', 'unused', '--variant', schedule.B])
    with (tmp_path / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with patch.object(schedule.subprocess, 'Popen') as worker:
            with pytest.raises(BlockingIOError): schedule.main()
            worker.assert_not_called()
