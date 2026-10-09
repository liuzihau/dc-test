"""Actual BD3 initialization and first-update gradient checks on a small CPU model."""
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from owt.entrypoint import ROOT, make_config
from owt.model import OWTMDM


def config(variant):
    args = SimpleNamespace(variant=variant, data_dir=ROOT/'.cache/huggingface',
        microbatch=8, global_batch=512, workers=1, steps=5000, interval=500,
        resume=None, run=ROOT/'.cache/runtime/owt/init-test', val_examples=1024)
    result = make_config(args)
    result.model.hidden_size = 32
    result.model.cond_dim = 16
    result.model.n_blocks = 2
    result.model.n_heads = 4
    result.model.length = result.block_size = 16
    result.model.dropout = 0
    return result


def model(cfg):
    tokenizer = SimpleNamespace(vocab_size=7, mask_token=None, all_special_ids=[])
    with patch('diffusion.metrics.Metrics', return_value=torch.nn.Module()):
        return OWTMDM(cfg, tokenizer).train()


def test_zero_recipe_only_changes_auxiliary_initialization():
    random, zero = config('mdm_np'), config('mdm_np_zero_init')
    zero.mechanisms.np.pop('initialization')
    assert OmegaConf.to_container(random, resolve=True) == OmegaConf.to_container(zero, resolve=True)


def test_low_weight_recipe_only_changes_the_two_auxiliary_coefficients():
    standard, low = config('mdm_np_zero_init'), config('mdm_np_zero_init_low_weight')
    assert list(low.mechanisms.np.weights) == [0.05, 0.05]
    low.mechanisms.np.weights = standard.mechanisms.np.weights
    assert OmegaConf.to_container(standard, resolve=True) == OmegaConf.to_container(low, resolve=True)
    torch.manual_seed(71)
    a = model(config('mdm_np_zero_init'))
    expected_rng = torch.get_rng_state()
    torch.manual_seed(71)
    b = model(config('mdm_np_zero_init_low_weight'))
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    for name, tensor in a.state_dict().items():
        torch.testing.assert_close(tensor, b.state_dict()[name], rtol=0, atol=0)


def test_shared_parameters_rng_and_ema_unchanged_by_zero_auxiliary_heads():
    torch.manual_seed(71)
    baseline = model(config('mdm'))
    expected_rng = torch.get_rng_state()
    torch.manual_seed(71)
    augmented = model(config('mdm_np_zero_init'))
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    for name, tensor in baseline.state_dict().items():
        torch.testing.assert_close(tensor, augmented.state_dict()[name], rtol=0, atol=0)
    assert augmented.backbone.adaLN
    for block in augmented.backbone.blocks:
        assert not block.adaLN_modulation.weight.any()
        assert not block.adaLN_modulation.bias.any()
    assert not augmented.backbone.output_layer.linear.weight.any()
    assert not augmented.backbone.output_layer.linear.bias.any()
    for head in augmented.backbone.neighbor_heads.heads:
        assert not head[-1].weight.any() and not head[-1].bias.any()
    params = list(augmented._get_parameters())
    assert len(params) == len(augmented.ema.shadow_params)
    for p, shadow in zip(params, augmented.ema.shadow_params):
        torch.testing.assert_close(p, shadow, rtol=0, atol=0)


def test_zero_heads_learn_before_sending_gradients_to_shared_features():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    augmented = model(config('mdm_np_zero_init'))
    x = torch.tensor([[1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4],
                      [6,5,4,3,2,1,6,5,4,3,2,1,6,5,4,3]])
    valid = torch.ones_like(x)
    optimizer = torch.optim.AdamW(augmented._get_parameters(), lr=3e-4)
    torch.manual_seed(42)
    augmented._loss(x, valid.clone()).loss.backward()
    assert augmented.backbone.vocab_embed.embedding.grad.abs().sum() == 0
    for head in augmented.backbone.neighbor_heads.heads:
        assert head[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad()
    torch.manual_seed(42)
    augmented._loss(x, valid.clone()).loss.backward()
    assert augmented.backbone.vocab_embed.embedding.grad.abs().sum() > 0
    assert any(block.adaLN_modulation.bias.grad.abs().sum() > 0
               for block in augmented.backbone.blocks)
    # Existing random-head behavior enters the shared backbone immediately.
    torch.manual_seed(1)
    random = model(config('mdm_np'))
    torch.manual_seed(42)
    random._loss(x, valid.clone()).loss.backward()
    assert random.backbone.vocab_embed.embedding.grad.abs().sum() > 0


def test_frozen_fp32_override_disables_internal_amp_but_preserves_training():
    # Exercise the actual backbone with CPU AMP standing in for CUDA AMP.
    # No CUDA allocation is allowed while the production training owns GPUs 2/3.
    backbone=model(config('mdm')).backbone
    tokens=torch.ones((2,16),dtype=torch.long)
    sigma=torch.zeros(2)
    original=torch.amp.autocast

    def cpu_amp(device_type, dtype=None, enabled=True, **kwargs):
        assert device_type=='cuda'
        return original('cpu',dtype=dtype or torch.bfloat16,enabled=enabled,**kwargs)

    with patch('torch.amp.autocast',side_effect=cpu_amp):
        with torch.inference_mode():
            backbone.eval()
            assert backbone(tokens,sigma).dtype==torch.bfloat16
            backbone.force_fp32_eval=True
            assert backbone(tokens,sigma).dtype==torch.float32
            backbone.train()
            assert backbone(tokens,sigma).dtype==torch.bfloat16
