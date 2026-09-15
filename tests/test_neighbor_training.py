"""Neighbor-only loss credit and preservation of the primary experiment."""
import copy

import pytest
import torch

from diffusion import Diffusion
from test_dcachehooping import make_model


def model_pair(adjacent=True, ema=0):
  reference, tokenizer = make_model(core_only=True, auxiliary_probability=0, ema=ema)
  config = copy.deepcopy(reference.config)
  config.step_memory.attention_mode = 'merged'
  config.step_memory.detach_between_steps = not adjacent
  config.dcachehooping.adjacent_grad.enabled = adjacent
  torch.manual_seed(4001)
  base = Diffusion(config, tokenizer=tokenizer)
  base_rng = torch.get_rng_state()
  config = copy.deepcopy(config)
  config.neighbor_prediction.enabled = True
  config.neighbor_prediction.chunk_size = 3
  torch.manual_seed(4001)
  neighbor = Diffusion(config, tokenizer=tokenizer)
  assert torch.equal(base_rng, torch.get_rng_state())
  for name, weight in base.state_dict().items():
    torch.testing.assert_close(neighbor.state_dict()[name], weight, rtol=0, atol=0)
  return base, neighbor, tokenizer


def data(tokenizer):
  # Text8 special IDs are 0..4; use normal characters only.
  return torch.randint(5, tokenizer.vocab_size, (2, 8)), torch.ones(2, 8)


def open_residuals(model):
  with torch.no_grad():
    for layer in model.backbone.blocks:
      layer.adaLN_modulation.bias.chunk(6)[2].fill_(0.1)
      layer.adaLN_modulation.bias.chunk(6)[5].fill_(0.1)


@pytest.mark.parametrize('adjacent', [False, True])
def test_auxiliary_preserves_primary_losses_rng_and_identity(adjacent):
  base, neighbor, tok = model_pair(adjacent)
  x, attention = data(tok)
  calls = []
  hook = neighbor.backbone.register_forward_hook(lambda *args: calls.append(torch.is_grad_enabled()))
  torch.manual_seed(901)
  expected, ref, diag = base._dcachehooping_pretrain_loss(x, attention)
  expected_rng = torch.get_rng_state()
  torch.manual_seed(901)
  actual, actual_ref, metrics = neighbor._dcachehooping_pretrain_loss(x, attention)
  hook.remove()
  assert torch.equal(expected_rng, torch.get_rng_state())
  assert calls.count(True) == 5
  torch.testing.assert_close(ref.nlls, actual_ref.nlls, rtol=0, atol=0)
  for name in ('loss_full', 'loss_t0', 'loss_t1', 'loss_t2', 'loss_t3',
               'identity_loss', 'mask_ratio_t2', 'loss_base'):
    torch.testing.assert_close(diag[name], metrics[name], rtol=0, atol=0)
  torch.testing.assert_close(actual.detach(), expected.detach() + metrics['neighbor_weighted_loss'])
  weights = [0.05, 0.10, 0.20, 1.0, 0.70]
  expected_neighbor = sum(w * metrics[f'neighbor_{s}_loss']
    for w, s in zip(weights, ('full', 't0', 't1', 't2', 't3'))) / sum(weights)
  torch.testing.assert_close(metrics['neighbor_weighted_loss'], expected_neighbor * 0.5)
  actual.backward()
  for name, parameter in neighbor.backbone.neighbor_heads.named_parameters():
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
  assert any(p.grad.abs().sum() > 0 for p in neighbor.backbone.neighbor_heads.parameters())


@pytest.mark.parametrize('adjacent', [False, True])
@pytest.mark.parametrize('target', [1, 2, 3, 4])
def test_neighbor_only_credit_has_correct_horizon(monkeypatch, target, adjacent):
  _, model, tok = model_pair(adjacent)
  open_residuals(model)
  model.config.step_memory.pretrain.identity.enabled = False
  model.config.step_memory.pretrain.source_dropout.enabled = False
  model.config.dcachehooping.latent_dropout_probability = 0
  names = ('full_loss_weight', 't0_loss_weight', 't1_loss_weight', 't2_loss_weight', 't3_loss_weight')
  for index, name in enumerate(names):
    setattr(model.config.step_memory.pretrain, name, float(index == target))
  original = model._dcachehooping_state_loss
  def auxiliary_only(*args, **kwargs):
    loss, output = original(*args, **kwargs)
    loss.loss = loss.loss * 0
    return loss, output
  monkeypatch.setattr(model, '_dcachehooping_state_loss', auxiliary_only)
  hidden = []
  def retain(module, args, output):
    tensor = output[0] if isinstance(output, tuple) else output
    tensor.retain_grad()
    hidden.append(tensor)
  hook = model.backbone.blocks[-1].register_forward_hook(retain)
  x, attention = data(tok)
  loss, _, _ = model._dcachehooping_pretrain_loss(x, attention)
  hook.remove()
  loss.backward()
  assert len(hidden) == 5
  for index, tensor in enumerate(hidden):
    active = tensor.grad is not None and tensor.grad.abs().sum() > 0
    expected = index == target or (adjacent and index == target - 1)
    assert bool(active) == expected, (index, target, adjacent)


def test_inference_does_not_compute_auxiliary(monkeypatch):
  base, model, tok = model_pair()
  x, _ = data(tok)
  def forbidden(*args, **kwargs):
    raise AssertionError('Inference must not invoke neighbor heads')
  for head in model.backbone.neighbor_heads.heads.values():
    monkeypatch.setattr(head, 'forward', forbidden)
  base.eval()
  model.eval()
  with torch.inference_mode():
    a = base.backbone(x, sigma=None)
    b = model.backbone(x, sigma=None)
  torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_training_validation_and_ema_roundtrip(tmp_path, monkeypatch):
  import test_adjacent_five_forward as suite
  def pair(ema=0):
    base, model, tok = model_pair(ema=ema)
    return base, model, tok
  monkeypatch.setattr(suite, 'make_pair', pair)
  suite.test_lightning_accumulation_validation_and_checkpoint_ema_reload(tmp_path)


def test_validation_records_auxiliaries_but_keeps_primary_nll(tmp_path):
  import lightning as L
  base, model, tok = model_pair()
  x, attention = data(tok)
  loader = torch.utils.data.DataLoader([
    {'input_ids': ids, 'attention_mask': mask}
    for ids, mask in zip(x, attention)], batch_size=2)
  results = []
  for network in (base, model):
    trainer = L.Trainer(
      accelerator='cpu', devices=1, logger=False, enable_checkpointing=False,
      enable_progress_bar=False, num_sanity_val_steps=0, limit_val_batches=1,
      default_root_dir=tmp_path)
    results.append(trainer.validate(network, loader, verbose=False)[0])
  before, after = results
  for name in ('val/nll', 'val/loss_t2', 'val/loss_t0', 'val/loss_t1',
               'val/loss_t3', 'val/loss_base', 'val/mask_ratio_t2'):
    assert before[name] == after[name], name
  for state in ('full', 't0', 't1', 't2', 't3'):
    for direction in ('prev', 'next'):
      for metric in ('loss', 'accuracy', 'count'):
        assert torch.isfinite(torch.tensor(
          after[f'val/neighbor_{state}_{direction}_{metric}']))
  assert after['val/loss_total'] == pytest.approx(
    before['val/loss_total'] + after['val/neighbor_weighted_loss'], abs=1e-6)


def test_requires_merged_and_core_recipe():
  base, _, tok = model_pair()
  config = copy.deepcopy(base.config)
  config.neighbor_prediction.enabled = True
  config.step_memory.attention_mode = 'separate'
  with pytest.raises(ValueError, match='merged'):
    Diffusion(config, tokenizer=tok)
  config.step_memory.attention_mode = 'merged'
  config.neighbor_prediction.weight = -0.5
  with pytest.raises(ValueError, match='weight'):
    Diffusion(config, tokenizer=tok)
