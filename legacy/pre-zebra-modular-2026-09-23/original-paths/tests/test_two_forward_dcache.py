import pathlib

import hydra
import lightning as L
import pytest
import torch
from omegaconf import OmegaConf

import dataloader
from diffusion import Diffusion


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def make_model(ema=0, two_forward=True):
  for name, resolver in {
      'cwd': lambda: str(REPO_ROOT),
      'device_count': lambda: 1,
      'eval': eval,
      'div_up': lambda x, y: (x + y - 1) // y,
  }.items():
    if not OmegaConf.has_resolver(name):
      OmegaConf.register_new_resolver(name, resolver)
  with hydra.initialize_config_dir(
      version_base=None, config_dir=str(REPO_ROOT / 'configs')):
    config = hydra.compose(
      config_name='config',
      overrides=[
        'algo=mdlm',
        'model=tiny',
        'model.length=8',
        'model.hidden_size=32',
        'model.cond_dim=16',
        'model.n_blocks=2',
        'model.n_heads=4',
        'model.dropout=0.0',
        'model.attn_backend=sdpa',
        'block_size=8',
        'loader.global_batch_size=2',
        'loader.eval_global_batch_size=2',
        'loader.batch_size=2',
        'loader.eval_batch_size=2',
        'trainer.devices=1',
        f'training.ema={ema}',
        'step_memory.enabled=true',
        'step_memory.use_previous_kv=true',
        ('step_memory.detach_between_steps=false'
         if two_forward else 'step_memory.detach_between_steps=true'),
        'step_memory.gate.enabled=true',
        'step_memory.gate.init=0.1',
        'step_memory.pretrain.enabled=true',
        'step_memory.pretrain.source_dropout.enabled=true',
        'step_memory.pretrain.source_dropout.cache_only_probability=0.20',
        'step_memory.pretrain.source_dropout.current_only_probability=0.05',
        'step_memory.pretrain.source_dropout.warmup_steps=1',
        'step_memory.pretrain.identity.enabled=false',
        'dcachehooping.enabled=true',
        ('dcachehooping.two_forward.enabled=true'
         if two_forward else 'dcachehooping.two_forward.enabled=false'),
        'dcachehooping.two_forward.max_first_mask_ratio=1.0',
        'dcachehooping.two_forward.first_loss_weight=0.80',
        'dcachehooping.two_forward.second_loss_weight=1.25',
        'dcachehooping.status_embedding.enabled=false',
        'dcachehooping.latent_dropout_probability=0.0',
        'dcachehooping.latent_mask_probability=0.0',
        'dcachehooping.latent_mask_loss_weight=0.0',
        'dcachehooping.tentative.enabled=false',
        'dcachehooping.tentative.batch_probability=0.0',
        'dcachehooping.tentative.loss_weight=0.0',
        'dcachehooping.confidence.enabled=false',
        'dcachehooping.confidence.loss_weight=0.0',
        'step_memory.rollout.enabled=false',
        'wandb=null',
      ])
  tokenizer = dataloader.Text8Tokenizer()
  return Diffusion(config, tokenizer=tokenizer), tokenizer


def test_two_state_sampler_is_nested_and_permits_full_first_state(monkeypatch):
  torch.manual_seed(201)
  model, tokenizer = make_model()
  x0 = torch.randint(1, tokenizer.vocab_size, (4, 8))
  attention = torch.ones_like(x0)

  # Force t to its upper endpoint. Then s=t+k=max_first=1 and integer
  # correction must retain the fully masked first state.
  monkeypatch.setattr(
    torch, 'rand_like', lambda value: torch.ones_like(value))
  trajectory = model._sample_two_forward_trajectory(x0, attention)

  counts = trajectory['mask_counts']
  eligible_counts = trajectory['eligible'].sum(dim=-1)
  assert torch.equal(counts[:, 0], eligible_counts)
  assert torch.all(counts[:, 1] < counts[:, 0])
  assert torch.all(counts[:, 1] >= 1)
  first, second = trajectory['masks']
  assert torch.all(~second | first)
  assert torch.all(
    trajectory['sampled_ratios'][:, 0]
    - trajectory['sampled_ratios'][:, 1] >= 0.025 - 1e-6)


def test_none_and_explicit_zero_final_state_are_identical():
  torch.manual_seed(202)
  model, tokenizer = make_model()
  model.eval()
  with torch.no_grad():
    model.backbone.dcachehooping_latent_norm.weight.fill_(0.3)
    model.backbone.dcachehooping_latent_norm.bias.fill_(0.2)
  x = torch.randint(1, tokenizer.vocab_size, (2, 8))
  sigma = torch.full((2, 1), 0.5)
  zeros = torch.zeros(2, 8, 32)

  with torch.no_grad():
    implicit = model.forward(x, sigma=sigma, sample_mode=True)
    explicit = model.forward(
      x, sigma=sigma, sample_mode=True,
      previous_final_hidden=zeros)
  torch.testing.assert_close(implicit, explicit, rtol=0, atol=0)


def test_objective_uses_exactly_two_forwards_and_requested_gradient_routes():
  torch.manual_seed(203)
  model, tokenizer = make_model()
  torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.02)
  x0 = torch.randint(1, tokenizer.vocab_size, (2, 8))
  attention = torch.ones_like(x0)
  calls = []
  first_cache = []
  original = model._dcachehooping_state_loss

  def inspected(*args, **kwargs):
    calls.append(kwargs.copy())
    result = original(*args, **kwargs)
    if len(calls) == 1:
      for entry in result[1].step_kv:
        entry.retain_grad()
        first_cache.append(entry)
    return result

  model._dcachehooping_state_loss = inspected
  total, reference, diagnostics = model._dcachehooping_two_forward_loss(
    x0, attention)

  assert len(calls) == 2
  assert calls[0]['previous_step_kv'] is None
  assert calls[0]['previous_final_hidden'] is None
  assert calls[0]['return_step_kv'] is True
  assert calls[0]['detach_cache_backbone'] is False
  assert calls[1]['previous_step_kv'] is not None
  assert calls[1]['previous_final_hidden'] is not None
  assert calls[1]['previous_final_hidden'].requires_grad is False
  assert calls[1]['return_step_kv'] is False
  assert calls[1]['detach_cache_backbone'] is False
  assert diagnostics['num_forwards'] == 2
  assert diagnostics['loss_weight_sum'] == pytest.approx(2.05)
  expected = (
    0.80 * diagnostics['loss_first']
    + 1.25 * diagnostics['loss_second']) / 2.05
  torch.testing.assert_close(total.detach(), expected)
  torch.testing.assert_close(reference.loss, diagnostics['loss_second'])

  total.backward()
  # Returned cache tensors are not used by the first loss. Their gradients
  # therefore prove that the second loss crosses the forward boundary.
  assert all(entry.grad is not None for entry in first_cache)
  assert any(torch.count_nonzero(entry.grad) > 0 for entry in first_cache)
  latent_norm = model.backbone.dcachehooping_latent_norm
  assert latent_norm.weight.grad is not None
  assert latent_norm.bias.grad is not None


def test_source_and_final_state_dropout_apply_only_to_second_forward():
  torch.manual_seed(204)
  model, tokenizer = make_model()
  model.config.dcachehooping.latent_dropout_probability = 1.0
  sentinel = torch.ones((2, 8), dtype=torch.int8)
  calls = []
  original_state_loss = model._dcachehooping_state_loss

  def inspected(*args, **kwargs):
    calls.append(kwargs.copy())
    return original_state_loss(*args, **kwargs)

  model._dcachehooping_state_loss = inspected
  model._source_dropout_mask = lambda mask: (
    sentinel,
    {
      'cache_only_probability': torch.tensor(0.2),
      'cache_only_fraction': torch.tensor(1.0),
      'current_only_fraction': torch.tensor(0.0),
    })
  x0 = torch.randint(1, tokenizer.vocab_size, (2, 8))
  attention = torch.ones_like(x0)
  _, _, diagnostics = model._dcachehooping_two_forward_loss(x0, attention)

  assert calls[0].get('step_memory_source_mask') is None
  assert calls[1]['step_memory_source_mask'] is sentinel
  assert calls[1]['previous_final_hidden'] is None
  assert diagnostics['final_state_dropout_fraction'] == 1


def test_two_forward_validation_logs_comparable_second_state_metrics():
  torch.manual_seed(205)
  model, tokenizer = make_model()
  examples = [{
    'input_ids': torch.randint(1, tokenizer.vocab_size, (8,)),
    'attention_mask': torch.ones(8, dtype=torch.long),
  } for _ in range(2)]
  loader = torch.utils.data.DataLoader(examples, batch_size=2)
  trainer = L.Trainer(
    accelerator='cpu', devices=1, logger=False, enable_checkpointing=False,
    enable_progress_bar=False, num_sanity_val_steps=0, limit_val_batches=1)

  result = trainer.validate(model, loader, verbose=False)[0]

  for name in [
      'val/nll', 'val/loss_s', 'val/loss_t', 'val/loss_first',
      'val/loss_second', 'val/loss_total', 'val/mask_ratio_s',
      'val/mask_ratio_t', 'val/num_forwards', 'val/loss_weight_sum']:
    assert name in result
    assert torch.isfinite(torch.tensor(result[name]))
  assert result['val/num_forwards'] == pytest.approx(2.0)
  assert result['val/loss_weight_sum'] == pytest.approx(2.05)


def test_two_forward_checkpoint_and_ema_reload_strictly(tmp_path):
  torch.manual_seed(206)
  model, tokenizer = make_model(ema=0.9)
  examples = [{
    'input_ids': torch.randint(1, tokenizer.vocab_size, (8,)),
    'attention_mask': torch.ones(8, dtype=torch.long),
  } for _ in range(2)]
  loader = torch.utils.data.DataLoader(examples, batch_size=2)
  trainer = L.Trainer(
    accelerator='cpu', devices=1, logger=False, enable_checkpointing=False,
    enable_progress_bar=False, max_steps=1, limit_train_batches=1,
    limit_val_batches=0, num_sanity_val_steps=0)
  trainer.fit(model, loader)
  checkpoint = tmp_path / 'two-forward.ckpt'
  trainer.save_checkpoint(checkpoint)

  loaded = Diffusion.load_from_checkpoint(
    checkpoint, config=model.config, tokenizer=tokenizer,
    strict=True, weights_only=False)
  assert isinstance(
    loaded.backbone.dcachehooping_latent_norm, torch.nn.LayerNorm)
  assert loaded.backbone.dcachehooping_latent_norm.bias is not None
  assert loaded.ema is not None
  assert len(loaded.ema.shadow_params) == len([
    parameter for parameter in loaded.parameters()
    if parameter.requires_grad])
  loaded.ema.copy_to(loaded._get_parameters())


def test_legacy_final_state_mode_keeps_bias_free_normalization():
  new_model, _ = make_model(two_forward=True)
  legacy_model, _ = make_model(two_forward=False)
  assert isinstance(
    new_model.backbone.dcachehooping_latent_norm, torch.nn.LayerNorm)
  assert new_model.backbone.dcachehooping_latent_norm.bias is not None
  # The legacy custom normalization is deliberately bias-free, preserving its
  # old checkpoint parameter names and shapes.
  assert not hasattr(legacy_model.backbone.dcachehooping_latent_norm, 'bias')
