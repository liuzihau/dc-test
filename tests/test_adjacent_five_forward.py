"""Integration guarantees for the actual five-state DCache objective."""

import copy

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

from diffusion import Diffusion
from test_dcachehooping import make_model


def make_pair(ema=0):
  legacy, tokenizer = make_model(
    enabled=True, auxiliary_probability=0.0, core_only=True, ema=ema)
  torch.nn.init.normal_(legacy.backbone.output_layer.linear.weight, std=0.02)
  config = copy.deepcopy(legacy.config)
  config.step_memory.detach_between_steps = False
  config.dcachehooping.adjacent_grad.enabled = True
  adjacent = Diffusion(config, tokenizer=tokenizer)
  adjacent.load_state_dict(legacy.state_dict(), strict=True)
  return legacy, adjacent, tokenizer


def inputs(tokenizer):
  generator = torch.Generator().manual_seed(1901)
  x0 = torch.randint(1, tokenizer.vocab_size, (2, 8), generator=generator)
  return x0, torch.ones_like(x0)


@pytest.mark.parametrize('identity', [False, True])
def test_adjacent_mode_preserves_losses_rng_and_five_canonical_forwards(identity):
  legacy, adjacent, tokenizer = make_pair()
  for model in (legacy, adjacent):
    model.config.step_memory.pretrain.identity.enabled = identity
    model.config.step_memory.pretrain.identity.batch_probability = 1.0
    model.config.step_memory.pretrain.identity.margin = 10.0
  x0, attention = inputs(tokenizer)
  calls = []
  forward_calls = []
  original = adjacent._dcachehooping_state_loss

  def inspected(*args, **kwargs):
    calls.append(kwargs.copy())
    return original(*args, **kwargs)

  adjacent._dcachehooping_state_loss = inspected
  hook = adjacent.backbone.register_forward_hook(
    lambda module, args, output: forward_calls.append(torch.is_grad_enabled()))
  torch.manual_seed(1937)
  expected, expected_ref, expected_metrics = legacy._dcachehooping_pretrain_loss(
    x0, attention)
  expected_rng = torch.get_rng_state()
  torch.manual_seed(1937)
  actual, actual_ref, actual_metrics = adjacent._dcachehooping_pretrain_loss(
    x0, attention)
  hook.remove()

  torch.testing.assert_close(actual.detach(), expected.detach(), rtol=0, atol=0)
  torch.testing.assert_close(actual_ref.nlls, expected_ref.nlls, rtol=0, atol=0)
  torch.testing.assert_close(torch.get_rng_state(), expected_rng)
  for key, value in expected_metrics.items():
    torch.testing.assert_close(actual_metrics[key], value, rtol=0, atol=0)
  assert len(calls) == 5
  assert forward_calls == ([True] * 5 + ([False] if identity else []))
  assert calls[0]['previous_step_kv'] is None
  for call in calls[1:]:
    assert all(entry.is_leaf and entry.requires_grad
               for entry in call['previous_step_kv'])
    assert not call['previous_final_hidden'].requires_grad
  assert actual_metrics['adjacent_gradient_edges'] == 4
  assert actual_metrics['adjacent_gradient_horizon'] == 1
  assert all(parameter.grad is None for parameter in adjacent.parameters())
  actual.backward()
  assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
             for parameter in adjacent.parameters())


@pytest.mark.parametrize('target_step', [1, 2, 3, 4])
def test_actual_transformer_loss_reaches_only_immediately_previous_hidden(target_step):
  _, model, tokenizer = make_pair()
  pretrain = model.config.step_memory.pretrain
  pretrain.identity.enabled = False
  pretrain.source_dropout.enabled = False
  names = ['full_loss_weight', 't0_loss_weight', 't1_loss_weight',
           't2_loss_weight', 't3_loss_weight']
  for index, name in enumerate(names):
    setattr(pretrain, name, float(index == target_step))
  hidden_states = []

  def retain_hidden(module, args, output):
    # Last block's returned hidden is an ancestor of the dedicated final KV
    # writer and logits; the returned final_hidden view is only a sibling.
    hidden = output[0] if isinstance(output, tuple) else output
    hidden.retain_grad()
    hidden_states.append(hidden)

  hook = model.backbone.blocks[-1].register_forward_hook(retain_hidden)
  x0, attention = inputs(tokenizer)
  total, _, _ = model._dcachehooping_pretrain_loss(x0, attention)
  hook.remove()
  total.backward()
  assert len(hidden_states) == 5
  for index, hidden in enumerate(hidden_states):
    has_grad = hidden.grad is not None and torch.count_nonzero(hidden.grad) > 0
    assert bool(has_grad) == (index in (target_step - 1, target_step)), index


def test_validation_never_requests_gradient_vjps(monkeypatch):
  legacy, adjacent, tokenizer = make_pair()
  legacy.eval()
  adjacent.eval()
  x0, attention = inputs(tokenizer)

  def forbidden(*args, **kwargs):
    raise AssertionError('Validation must not compute cotangents')

  monkeypatch.setattr(torch.autograd, 'grad', forbidden)
  with torch.inference_mode():
    torch.manual_seed(100)
    expected, _, expected_metrics = legacy._dcachehooping_pretrain_loss(x0, attention)
    torch.manual_seed(100)
    actual, _, actual_metrics = adjacent._dcachehooping_pretrain_loss(x0, attention)
  torch.testing.assert_close(actual, expected, rtol=0, atol=0)
  for key, value in expected_metrics.items():
    torch.testing.assert_close(actual_metrics[key], value, rtol=0, atol=0)


@pytest.mark.parametrize('invalid', ['detached', 'two_forward', 'fp16', 'tentative'])
def test_conflicting_adjacent_config_is_rejected(invalid):
  _, model, tokenizer = make_pair()
  config = copy.deepcopy(model.config)
  if invalid == 'detached':
    config.step_memory.detach_between_steps = True
  elif invalid == 'two_forward':
    config.dcachehooping.two_forward.enabled = True
  elif invalid == 'fp16':
    config.trainer.precision = '16-mixed'
  else:
    config.dcachehooping.tentative.enabled = True
  with pytest.raises(AssertionError):
    Diffusion(config, tokenizer=tokenizer)


def test_lightning_accumulation_validation_and_checkpoint_ema_reload(tmp_path):
  _, model, tokenizer = make_pair(ema=0.9)
  x0, attention = inputs(tokenizer)
  samples = [{'input_ids': x0[index], 'attention_mask': attention[index]}
             for _ in range(4) for index in range(2)]
  loader = DataLoader(samples, batch_size=2)
  trainer = L.Trainer(
    accelerator='cpu', devices=1, max_steps=2, accumulate_grad_batches=2,
    val_check_interval=2, limit_val_batches=1, num_sanity_val_steps=0,
    logger=False, enable_checkpointing=False, enable_progress_bar=False,
    enable_model_summary=False, default_root_dir=tmp_path)
  trainer.fit(model, loader, loader)
  assert trainer.global_step == 2
  assert torch.isfinite(trainer.callback_metrics['val/loss_t2'])
  checkpoint = tmp_path / 'adjacent.ckpt'
  trainer.save_checkpoint(checkpoint)
  restored = Diffusion.load_from_checkpoint(
    checkpoint, tokenizer=tokenizer, strict=True, weights_only=False)
  assert restored.config.dcachehooping.adjacent_grad.enabled
  assert restored.ema is not None
  # No new model parameters: the original final-state architecture can also
  # load these weights strictly for existing inference-only interventions.
  final_only_config = copy.deepcopy(restored.config)
  final_only_config.dcachehooping.adjacent_grad.enabled = False
  final_only_config.step_memory.detach_between_steps = True
  legacy = Diffusion.load_from_checkpoint(
    checkpoint, tokenizer=tokenizer, config=final_only_config,
    strict=True, weights_only=False)
  assert set(legacy.state_dict()) == set(restored.state_dict())
