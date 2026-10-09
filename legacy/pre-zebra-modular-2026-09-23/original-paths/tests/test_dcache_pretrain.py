import pathlib

import hydra
import lightning as L
import torch
from omegaconf import OmegaConf

import dataloader
from diffusion import Diffusion


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def make_pretrain_model(ema=0):
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
        'step_memory.detach_between_steps=true',
        'step_memory.gate.enabled=true',
        'step_memory.gate.init=0.1',
        'step_memory.pretrain.enabled=true',
        'step_memory.pretrain.teacher_token_probability=1.0',
        'step_memory.pretrain.source_dropout.enabled=true',
        'step_memory.pretrain.identity.enabled=true',
        'step_memory.pretrain.identity.batch_probability=1.0',
        'step_memory.rollout.enabled=false',
        'wandb=null',
      ])
  tokenizer = dataloader.Text8Tokenizer()
  return Diffusion(config, tokenizer=tokenizer), tokenizer


def test_local_trajectory_ratios_and_exact_masks_are_strictly_nested():
  torch.manual_seed(13)
  model, tokenizer = make_pretrain_model()
  x0 = torch.randint(1, tokenizer.vocab_size, (64, 8))
  attention = torch.ones_like(x0)

  trajectory = model._sample_local_step_trajectory(x0, attention)

  sampled = trajectory['sampled_ratios']
  assert torch.all(sampled[:, 0] <= 0.9975)
  assert torch.all(sampled[:, 3] >= 0)
  sampled_steps = sampled[:, :-1] - sampled[:, 1:]
  assert torch.all(sampled_steps >= 0.025 - 1e-6)
  assert torch.all(sampled_steps <= 0.10 + 1e-6)
  counts = trajectory['mask_counts']
  assert torch.all(counts[:, 0] <= 6)  # position zero is not maskable
  assert torch.all(counts[:, 3] >= 1)
  assert torch.all(counts[:, :-1] > counts[:, 1:])
  for earlier, later in zip(
      trajectory['masks'][:-1], trajectory['masks'][1:]):
    assert torch.all(~later | earlier)


def test_five_forward_pretrain_loss_is_finite_and_trains_cache_and_gate():
  torch.manual_seed(11)
  model, tokenizer = make_pretrain_model()
  torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.02)
  x0 = torch.randint(1, tokenizer.vocab_size, (2, 8))
  attention = torch.ones_like(x0)

  total, s_loss, diagnostics = model._step_memory_pretrain_loss(x0, attention)
  assert torch.isfinite(total)
  assert torch.isfinite(s_loss.loss)
  for name in [
      'loss_full', 'loss_t0', 'loss_t1', 'loss_t2', 'loss_t3',
      'mask_ratio_t0', 'mask_ratio_t1', 'mask_ratio_t2', 'mask_ratio_t3',
      'identity_loss', 'identity_gain', 'gate_mean']:
    assert name in diagnostics
    assert torch.isfinite(diagnostics[name])
  assert diagnostics['identity_applied'] == 1
  total.backward()
  gradient = model.backbone.dc_final_writer.kv.weight.grad
  assert gradient is not None
  assert torch.count_nonzero(gradient) > 0
  gate_gradient = model.backbone.blocks[0].step_memory_gate.grad
  assert gate_gradient is not None
  assert torch.count_nonzero(gate_gradient) > 0


def test_validation_logs_cache_assisted_s_loss_and_standard_nll():
  torch.manual_seed(12)
  model, tokenizer = make_pretrain_model()
  examples = []
  for _ in range(2):
    examples.append({
      'input_ids': torch.randint(1, tokenizer.vocab_size, (8,)),
      'attention_mask': torch.ones(8, dtype=torch.long),
    })
  loader = torch.utils.data.DataLoader(examples, batch_size=2)
  trainer = L.Trainer(
    accelerator='cpu', devices=1, logger=False, enable_checkpointing=False,
    enable_progress_bar=False, num_sanity_val_steps=0, limit_val_batches=1)

  result = trainer.validate(model, loader, verbose=False)[0]

  assert torch.isfinite(torch.tensor(result['val/nll']))
  assert torch.isfinite(torch.tensor(result['val/loss_s']))
  assert torch.isfinite(torch.tensor(result['val/loss_t']))
  assert torch.isfinite(torch.tensor(result['val/loss_t0']))
  assert torch.isfinite(torch.tensor(result['val/loss_t1']))
  assert torch.isfinite(torch.tensor(result['val/loss_t2']))
  assert torch.isfinite(torch.tensor(result['val/loss_t3']))
  assert torch.isfinite(torch.tensor(result['val/loss_full']))
  assert torch.isfinite(torch.tensor(result['val/loss_total']))


def test_gated_v2_checkpoint_and_ema_reload_strictly(tmp_path):
  torch.manual_seed(14)
  model, tokenizer = make_pretrain_model(ema=0.9)
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
  checkpoint = tmp_path / 'dcache-v2.ckpt'
  trainer.save_checkpoint(checkpoint)

  loaded = Diffusion.load_from_checkpoint(
    checkpoint, config=model.config, tokenizer=tokenizer,
    strict=True, weights_only=False)
  assert loaded.backbone.blocks[0].step_memory_gate is not None
  assert loaded.ema is not None
  assert len(loaded.ema.shadow_params) == len([
    parameter for parameter in loaded.parameters() if parameter.requires_grad])
  loaded.ema.copy_to(loaded._get_parameters())
