"""The corrected merged policy keeps current KV and does not attenuate past V."""
import copy
from types import SimpleNamespace

import pytest
import torch

from diffusion import Diffusion
from models.dit import DDiTBlock, DIT, apply_denoising_rope_2d
from test_merged_attention import merged_model


def model_pair(ema=0):
    original, tokenizer = merged_model(adjacent=False, ema=ema)
    config = copy.deepcopy(original.config)
    config.step_memory.merged_policy = 'current_preserving'
    config.step_memory.gate.enabled = False
    config.step_memory.pretrain.source_dropout.cache_only_probability = 0.0
    config.step_memory.pretrain.source_dropout.current_only_probability = 0.05
    detached = Diffusion(config, tokenizer=tokenizer)
    # Use nontrivial normal residuals, as the existing credit-horizon tests do.
    state = {k: v for k, v in original.state_dict().items()
             if not k.endswith('step_memory_gate')}
    detached.load_state_dict(state, strict=True)
    config = copy.deepcopy(config)
    config.step_memory.detach_between_steps = False
    config.dcachehooping.adjacent_grad.enabled = True
    adjacent = Diffusion(config, tokenizer=tokenizer)
    adjacent.load_state_dict(detached.state_dict(), strict=True)
    return detached, adjacent, tokenizer


def block():
    return DDiTBlock(n=8, dim=32, n_heads=4, adaLN=False, block_size=8,
                     dropout=0, attn_backend='sdpa', step_memory_enabled=True,
                     attention_mode='merged', merged_policy='current_preserving')


def test_unattenuated_previous_values_match_one_joint_softmax(monkeypatch):
    m = block()
    x = torch.randn(2, 8, 32)
    cache = torch.randn(2, 8, 2, 4, 8)
    calls = []
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def tracked(*args, **kwargs):
        calls.append(1)
        return sdpa(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, 'scaled_dot_product_attention', tracked)
    actual, _ = m.merged_attention(x, cache, False, None)
    raw = m.attn_qkv(m.norm1(x)).reshape(2, 8, 3, 4, 8).permute(0, 3, 2, 1, 4)
    pos = torch.arange(8)
    q = apply_denoising_rope_2d(raw[:, :, 0], pos, 1, 6)
    ck = apply_denoising_rope_2d(raw[:, :, 1], pos, 1, 6)
    pk = apply_denoising_rope_2d(cache[:, :, 0].transpose(1, 2), pos, 0, 6)
    k = torch.cat((pk, ck), dim=-2)
    v = torch.cat((cache[:, :, 1].transpose(1, 2), raw[:, :, 2]), dim=-2)
    expected = ((q @ k.transpose(-1, -2) / 8 ** 0.5).softmax(-1) @ v)
    torch.testing.assert_close(actual, expected.transpose(1, 2).reshape(2, 8, 32))
    assert len(calls) == 1
    assert m.step_memory_gate is None
    assert not any('step_memory_gate' in name for name in m.state_dict())


def test_current_only_removes_keys_and_values_and_cache_gradient():
    m = block()
    x = torch.randn(2, 8, 32)
    cache = torch.randn(2, 8, 2, 4, 8, requires_grad=True)
    source = torch.full((2, 8), 2, dtype=torch.int8)
    cold, _ = m.merged_attention(x, None, False, None)
    current, _ = m.merged_attention(x, cache, False, source)
    torch.testing.assert_close(current, cold)
    current.square().sum().backward()
    assert cache.grad is not None and torch.count_nonzero(cache.grad) == 0
    altered, _ = m.merged_attention(x, cache.detach() * 10, False, source)
    torch.testing.assert_close(altered, current)


def test_cache_only_is_rejected_instead_of_silently_ignoring_contract():
    with pytest.raises(ValueError, match='forbids cache-only'):
        block().merged_attention(torch.randn(2, 8, 32),
            torch.randn(2, 8, 2, 4, 8), False, torch.ones(2, 8, dtype=torch.int8))


@pytest.mark.parametrize('invalid', ['gate', 'cache_only', 'separate', 'unknown'])
def test_model_construction_rejects_conflicting_policy(invalid):
    _, model, _ = model_pair()
    config = copy.deepcopy(model.config)
    if invalid == 'gate':
        config.step_memory.gate.enabled = True
    elif invalid == 'cache_only':
        config.step_memory.pretrain.source_dropout.cache_only_probability = 0.20
    elif invalid == 'separate':
        config.step_memory.attention_mode = 'separate'
    else:
        config.step_memory.merged_policy = 'typo'
    with pytest.raises(ValueError, match='current_preserving|merged_policy'):
        DIT(config, model.vocab_size)


@pytest.mark.parametrize('step', [0, 500, 1000, 1500])
def test_source_probabilities_keep_rng_draws_and_only_drop_previous(step, monkeypatch):
    _, model, _ = model_pair()
    model._trainer = SimpleNamespace(global_step=step)
    mask = torch.ones(1, 100, dtype=torch.bool)
    mask[:, -1] = False
    draws = (torch.arange(100).float() + 0.5).reshape(1, 100) / 100
    calls = []

    def fixed_rand(shape, **kwargs):
        assert shape == mask.shape
        calls.append(1)
        return draws

    monkeypatch.setattr(torch, 'rand', fixed_rand)
    source, metrics = model._source_dropout_mask(mask)
    assert len(calls) == 1
    assert (source == 1).sum() == 0
    assert (source == 2).sum() == 5
    assert source[0, -1] == 0  # Revealed queries are never dropped.
    assert metrics['cache_only_probability'] == 0
    assert metrics['cache_only_fraction'] == 0
    assert model._step_memory_gate_mean() == 1
    model.eval()
    source, _ = model._source_dropout_mask(mask)
    assert source is None and len(calls) == 1


@pytest.mark.parametrize('identity', [False, True])
def test_five_forwards_objective_rng_identity_and_final_detachment(monkeypatch, identity):
    import test_adjacent_five_forward as suite
    monkeypatch.setattr(suite, 'make_pair', model_pair)
    suite.test_adjacent_mode_preserves_losses_rng_and_five_canonical_forwards(identity)


@pytest.mark.parametrize('target_step', [1, 2, 3, 4])
def test_credit_still_reaches_only_one_previous_forward(monkeypatch, target_step):
    import test_adjacent_five_forward as suite
    monkeypatch.setattr(suite, 'make_pair', model_pair)
    suite.test_actual_transformer_loss_reaches_only_immediately_previous_hidden(target_step)


def test_validation_no_vjps_and_same_mask_sequence_as_legacy(monkeypatch):
    import test_adjacent_five_forward as suite
    monkeypatch.setattr(suite, 'make_pair', model_pair)
    suite.test_validation_never_requests_gradient_vjps(monkeypatch)
    old, tokenizer = merged_model()
    _, new, _ = model_pair()
    x, attention = suite.inputs(tokenizer)
    results = []
    for model in (old, new):
        model.eval()
        torch.manual_seed(937)
        with torch.inference_mode():
            _, _, metrics = model._dcachehooping_pretrain_loss(x, attention)
        results.append((metrics, torch.get_rng_state()))
    torch.testing.assert_close(results[0][1], results[1][1])
    for state in ('t0', 't1', 't2', 't3'):
        torch.testing.assert_close(results[0][0]['mask_ratio_' + state],
                                   results[1][0]['mask_ratio_' + state])


def test_real_cpu_train_validate_ema_reload(tmp_path, monkeypatch):
    import test_adjacent_five_forward as suite
    monkeypatch.setattr(suite, 'make_pair', model_pair)
    suite.test_lightning_accumulation_validation_and_checkpoint_ema_reload(tmp_path)


@pytest.mark.parametrize('adjacent', [False, True])
def test_optional_neighbors_keep_primary_loss_and_credit(monkeypatch, adjacent):
    import test_neighbor_training as suite

    def pair(adjacent=True, ema=0):
        detached, connected, tokenizer = model_pair(ema=ema)
        base = connected if adjacent else detached
        config = copy.deepcopy(base.config)
        config.neighbor_prediction.enabled = True
        config.neighbor_prediction.chunk_size = 3
        neighbor = Diffusion(config, tokenizer=tokenizer)
        missing, unexpected = neighbor.load_state_dict(base.state_dict(), strict=False)
        assert missing and all('neighbor_heads' in name for name in missing)
        assert not unexpected
        return base, neighbor, tokenizer

    monkeypatch.setattr(suite, 'model_pair', pair)
    suite.test_auxiliary_preserves_primary_losses_rng_and_identity(adjacent)


def test_legacy_metadata_absence_preserves_old_state_and_outputs():
    legacy, tokenizer = merged_model()
    config = copy.deepcopy(legacy.config)
    del config.step_memory.merged_policy
    restored = Diffusion(config, tokenizer=tokenizer)
    restored.load_state_dict(legacy.state_dict(), strict=True)
    legacy.eval()
    restored.eval()
    x = torch.randint(1, tokenizer.vocab_size, (2, 8))
    with torch.no_grad():
        expected = legacy.backbone(x, sigma=torch.zeros(2), return_step_kv=True)
        actual = restored.backbone(x, sigma=torch.zeros(2), return_step_kv=True)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert all(layer.merged_policy == 'legacy' for layer in restored.backbone.blocks)


@pytest.mark.parametrize('reverse', [False, True])
def test_fit_resume_cannot_relabel_policy_even_with_compatible_shapes(reverse):
    _, model, _ = model_pair()
    saved_config = copy.deepcopy(model.config)
    if reverse:
        model.config.step_memory.merged_policy = 'legacy'
    else:
        del saved_config.step_memory.merged_policy
    model._trainer = SimpleNamespace(state=SimpleNamespace(fn='fit'))
    with pytest.raises(ValueError, match='Cannot resume training across merged policies'):
        model.on_load_checkpoint({'global_step': 1,
                                  'hyper_parameters': {'config': saved_config}})
