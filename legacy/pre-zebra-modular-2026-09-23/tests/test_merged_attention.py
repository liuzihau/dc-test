"""Single-softmax math, cache provenance, and actual five-forward integration."""
import copy

import pytest
import torch

from diffusion import Diffusion
from models.dit import DDiTBlock, apply_denoising_rope_2d
from test_dcachehooping import make_model


def block(gate=False):
    return DDiTBlock(n=8, dim=32, n_heads=4, adaLN=False, block_size=8,
                     dropout=0, attn_backend='sdpa', step_memory_enabled=True,
                     step_memory_gate_enabled=gate, attention_mode='merged')


@pytest.mark.parametrize('mode', [None, 0, 1, 2])
@pytest.mark.parametrize('gate', [False, True])
def test_one_softmax_matches_explicit_reference(mode, gate, monkeypatch):
    m = block(gate)
    x = torch.randn(2, 8, 32)
    cache = torch.randn(2, 8, 2, 4, 8) if mode is not None else None
    source = torch.full((2, 8), mode, dtype=torch.long) if mode is not None else None
    calls = []
    sdpa = torch.nn.functional.scaled_dot_product_attention
    def tracked(*args, **kwargs):
        calls.append(1)
        return sdpa(*args, **kwargs)
    monkeypatch.setattr(torch.nn.functional, 'scaled_dot_product_attention', tracked)
    actual, entry = m.merged_attention(x, cache, False, source)
    assert len(calls) == 1
    raw = m.attn_qkv(m.norm1(x)).reshape(2, 8, 3, 4, 8).permute(0, 3, 2, 1, 4)
    torch.testing.assert_close(entry, raw[:, :, 1:].permute(0, 3, 2, 1, 4))
    pos = torch.arange(8)
    q = apply_denoising_rope_2d(raw[:, :, 0], pos, 1, 6)
    k = apply_denoising_rope_2d(raw[:, :, 1], pos, 1, 6)
    v = raw[:, :, 2]
    if cache is not None:
        pk = apply_denoising_rope_2d(cache[:, :, 0].transpose(1, 2), pos, 0, 6)
        pv = cache[:, :, 1].transpose(1, 2)
        if gate:
            pv = pv * torch.tanh(m.step_memory_gate)
        k, v = torch.cat((pk, k), -2), torch.cat((pv, v), -2)
    scores = q @ k.transpose(-1, -2) / (8 ** 0.5)
    if mode == 1:
        scores[..., 8:] = -torch.inf
    elif mode == 2:
        scores[..., :8] = -torch.inf
    expected = (scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 8, 32)
    torch.testing.assert_close(actual, expected)
    assert m.dc_qkv is None and m.dc_norm is None and m.dc_attn_out is None
    assert not any('dc_qkv' in k for k in m.state_dict())


def test_detach_preserves_writer_gradient_not_previous_hidden():
    m = block()
    x = torch.randn(2, 8, 32, requires_grad=True)
    _, entry = m.merged_attention(x, None, True, None)
    entry.square().sum().backward()
    assert x.grad is None
    assert m.attn_qkv.weight.grad.abs().sum() > 0


def merged_model(adjacent=True, ema=0):
    original, tok = make_model(core_only=True, auxiliary_probability=0.0, ema=ema)
    config = copy.deepcopy(original.config)
    config.step_memory.attention_mode = 'merged'
    config.step_memory.detach_between_steps = not adjacent
    config.dcachehooping.adjacent_grad.enabled = adjacent
    model = Diffusion(config, tokenizer=tok)
    torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.02)
    # BD3 AdaLN initializes the normal attention/MLP residual gates to zero.
    # Probe connectivity after opening them, rather than mistaking an initial
    # zero multiplier for a detached graph. Production initialization is intact.
    with torch.no_grad():
        for layer in model.backbone.blocks:
            layer.adaLN_modulation.bias.chunk(6)[2].fill_(0.1)
            layer.adaLN_modulation.bias.chunk(6)[5].fill_(0.1)
    return model, tok


@pytest.mark.parametrize('identity', [False, True])
def test_actual_five_forward_objective_and_one_hop(monkeypatch, identity):
    import test_adjacent_five_forward as suite
    def pair(ema=0):
        legacy, tok = merged_model(adjacent=False)
        config = copy.deepcopy(legacy.config)
        config.step_memory.detach_between_steps = False
        config.dcachehooping.adjacent_grad.enabled = True
        adjacent = Diffusion(config, tokenizer=tok)
        adjacent.load_state_dict(legacy.state_dict(), strict=True)
        return legacy, adjacent, tok
    monkeypatch.setattr(suite, 'make_pair', pair)
    suite.test_adjacent_mode_preserves_losses_rng_and_five_canonical_forwards(identity)


@pytest.mark.parametrize('target_step', [1, 2, 3, 4])
def test_no_gradient_reaches_two_predecessors(monkeypatch, target_step):
    import test_adjacent_five_forward as suite
    def pair(ema=0):
        model, tok = merged_model()
        return None, model, tok
    monkeypatch.setattr(suite, 'make_pair', pair)
    suite.test_actual_transformer_loss_reaches_only_immediately_previous_hidden(target_step)


def test_shifted_cache_and_checkpoint_roundtrip():
    model, tok = merged_model()
    x = torch.randint(1, tok.vocab_size, (2, 8))
    entries = []
    hook = model.backbone.blocks[1].register_forward_hook(
        lambda m, a, out: entries.append(out[1]))
    model.eval()
    out = model.backbone(x, sigma=None, return_step_kv=True)
    hook.remove()
    torch.testing.assert_close(out[1][0], entries[0])
    clone, _ = merged_model()
    clone.load_state_dict(model.state_dict(), strict=True)
    clone.eval()
    again = clone.backbone(x, sigma=None, return_step_kv=True)
    torch.testing.assert_close(out[0], again[0])


def test_unsupported_prefix_is_rejected():
    m = block()
    with pytest.raises(ValueError, match='prefix'):
        m(torch.randn(2, 8, 32), None, None, store_kv=True)
    with pytest.raises(ValueError, match='Source dropout'):
        m.merged_attention(torch.randn(2, 8, 32), None, False,
                           torch.zeros(2, 8, dtype=torch.long))
    from scripts.cloud.check_h100_resume import scientific_config
    with pytest.raises(Exception, match='separate attention'):
        scientific_config({'step_memory': {'attention_mode': 'merged'}})


def test_cpu_training_validation_ema_checkpoint(tmp_path, monkeypatch):
    import test_adjacent_five_forward as suite
    def pair(ema=0):
        model, tok = merged_model(ema=ema)
        return None, model, tok
    monkeypatch.setattr(suite, 'make_pair', pair)
    suite.test_lightning_accumulation_validation_and_checkpoint_ema_reload(tmp_path)


def test_new_launcher_recipe_and_compact_guard(tmp_path):
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import hydra
    root = Path(__file__).resolve().parents[1]
    target = tmp_path / 'scripts/train'
    target.mkdir(parents=True)
    for name in ('train_owt_dcache_merged_adjacent_5k.sh',
                 'train_owt_dcache_final_state_adjacent_5k_2x3090.sh',
                 'train_owt_dcache_final_state_5k_2x3090.sh',
                 'train_owt_dcache_pretrain_100k.sh'):
        shutil.copyfile(root / 'scripts/train' / name, target / name)
    recorder = tmp_path / 'python'
    recorder.write_text(f'#!{sys.executable}\nimport sys,json\n'
                        'print(2 if sys.argv[1]=="-c" else json.dumps(sys.argv[1:]))\n')
    recorder.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith('DCACHE_')}
    env['DCACHE_PYTHON'] = str(recorder)
    cmd = ['bash', str(target / 'train_owt_dcache_merged_adjacent_5k.sh')]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout.splitlines()[-1])
    # Resolver registration and Hydra config loading use the same test factory.
    make_model(core_only=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(root / 'configs')):
        config = hydra.compose(config_name='config', overrides=args[2:])
    assert config.step_memory.attention_mode == 'merged'
    assert config.dcachehooping.adjacent_grad.enabled
    assert not config.step_memory.detach_between_steps
    assert config.trainer.max_steps == 5000
    assert config.trainer.val_check_interval == 64000
    assert config.step_memory.pretrain.identity.batch_probability == 0.25
    assert config.step_memory.pretrain.source_dropout.cache_only_probability == 0.20
    assert config.dcachehooping.latent_dropout_probability == 0.10
    assert config.callbacks.checkpoint_every_n_steps.save_top_k == 3
    assert 'merged' in config.checkpointing.save_dir
    bad = subprocess.run(cmd, env={**env, 'DCACHE_RESUME_CKPT': 'old.ckpt'},
                         capture_output=True, text=True, timeout=30)
    assert bad.returncode == 2 and 'NEW architecture' in bad.stderr
