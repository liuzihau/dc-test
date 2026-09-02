import pathlib

import hydra
import lightning as L
import pytest
import torch
from omegaconf import OmegaConf

import dataloader
from diffusion import Diffusion


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def make_model(objective_enabled=True):
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
        'training.ema=0',
        f'training.objective_matched_multistate.enabled={str(objective_enabled).lower()}',
        'step_memory.enabled=false',
        'step_memory.use_previous_kv=false',
        'step_memory.gate.enabled=false',
        'step_memory.pretrain.enabled=false',
        'step_memory.pretrain.teacher_token_probability=1.0',
        'step_memory.pretrain.source_dropout.enabled=false',
        'step_memory.pretrain.identity.enabled=false',
        'step_memory.rollout.enabled=false',
        'wandb=null',
      ])
  tokenizer = dataloader.Text8Tokenizer()
  return Diffusion(config, tokenizer=tokenizer), tokenizer


def test_objective_matched_model_is_parameter_identical_to_vanilla():
  torch.manual_seed(21)
  objective_model, _ = make_model(objective_enabled=True)
  torch.manual_seed(21)
  vanilla_model, _ = make_model(objective_enabled=False)

  objective_parameters = dict(objective_model.named_parameters())
  vanilla_parameters = dict(vanilla_model.named_parameters())
  assert objective_parameters.keys() == vanilla_parameters.keys()
  assert sum(p.numel() for p in objective_parameters.values()) == \
    sum(p.numel() for p in vanilla_parameters.values())
  assert not objective_model.backbone.step_memory_enabled
  assert objective_model.backbone.dc_final_writer is None
  assert all(block.dc_qkv is None for block in objective_model.backbone.blocks)
  assert all('dc_' not in name for name in objective_parameters)
  assert all('step_memory_gate' not in name for name in objective_parameters)


def test_objective_matched_loss_uses_five_cache_free_forwards_and_normalizes(
    monkeypatch):
  torch.manual_seed(22)
  model, tokenizer = make_model()
  torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.02)
  x0 = torch.randint(1, tokenizer.vocab_size, (2, 8))
  attention = torch.ones_like(x0)

  calls = []
  original_forward = model.forward

  def recording_forward(*args, **kwargs):
    calls.append(dict(kwargs))
    return original_forward(*args, **kwargs)

  monkeypatch.setattr(model, 'forward', recording_forward)
  total, reference, diagnostics = \
    model._objective_matched_multistate_loss(x0, attention)

  assert len(calls) == 5
  for call in calls:
    assert call.get('sample_mode') is True
    assert 'previous_step_kv' not in call
    assert 'return_step_kv' not in call
    assert 'step_memory_source_mask' not in call
  assert diagnostics['num_forwards'] == 5
  assert diagnostics['loss_weight_sum'] == pytest.approx(2.05)
  expected = (
    0.05 * diagnostics['loss_full']
    + 0.10 * diagnostics['loss_t0']
    + 0.20 * diagnostics['loss_t1']
    + 1.00 * diagnostics['loss_t2']
    + 0.70 * diagnostics['loss_t3']) / 2.05
  torch.testing.assert_close(total.detach(), expected)
  torch.testing.assert_close(reference.loss.detach(), diagnostics['loss_t2'])

  total.backward()
  gradient = model.backbone.output_layer.linear.weight.grad
  assert gradient is not None
  assert torch.count_nonzero(gradient) > 0


def test_objective_matched_trajectory_is_strictly_nested_and_teacher_forced():
  torch.manual_seed(23)
  model, tokenizer = make_model()
  x0 = torch.randint(1, tokenizer.vocab_size, (64, 8))
  attention = torch.ones_like(x0)
  trajectory = model._sample_local_step_trajectory(x0, attention)

  sampled = trajectory['sampled_ratios']
  assert torch.all(sampled[:, 0] <= 0.9975)
  assert torch.all(sampled[:, 3] >= 0)
  steps = sampled[:, :-1] - sampled[:, 1:]
  assert torch.all(steps >= 0.025 - 1e-6)
  assert torch.all(steps <= 0.10 + 1e-6)
  assert torch.all(trajectory['mask_counts'][:, :-1]
                   > trajectory['mask_counts'][:, 1:])
  for state, mask in zip(trajectory['states'], trajectory['masks']):
    assert torch.equal(state[~mask], x0[~mask])
    assert torch.all(state[mask] == model.mask_index)


def test_objective_matched_validation_logs_all_five_states():
  torch.manual_seed(24)
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
      'val/nll', 'val/loss_total', 'val/loss_full', 'val/loss_t0',
      'val/loss_t1', 'val/loss_t2', 'val/loss_t3',
      'val/num_forwards', 'val/loss_weight_sum']:
    assert name in result
    assert torch.isfinite(torch.tensor(result[name]))
  assert result['val/num_forwards'] == pytest.approx(5.0)
  assert result['val/loss_weight_sum'] == pytest.approx(2.05)


def test_objective_matched_lightning_training_step_runs():
  torch.manual_seed(25)
  model, tokenizer = make_model()
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

  assert trainer.global_step == 1
