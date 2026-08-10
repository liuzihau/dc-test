import pathlib

import hydra
import lightning as L
import torch
from omegaconf import OmegaConf

import dataloader
from diffusion import Diffusion


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def make_pretrain_model():
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
        'step_memory.enabled=true',
        'step_memory.use_previous_kv=true',
        'step_memory.detach_between_steps=true',
        'step_memory.pretrain.enabled=true',
        'step_memory.pretrain.teacher_token_probability=1.0',
        'step_memory.rollout.enabled=false',
        'wandb=null',
      ])
  tokenizer = dataloader.Text8Tokenizer()
  return Diffusion(config, tokenizer=tokenizer), tokenizer


def test_nested_pretrain_states_reveal_and_keep_at_least_one_token():
  torch.manual_seed(10)
  model, tokenizer = make_pretrain_model()
  x0 = torch.randint(1, tokenizer.vocab_size, (3, 8))
  attention = torch.ones_like(x0)
  s_time = torch.tensor([[0.2], [0.6], [0.95]])

  x_s, x_t, _, s_mask, t_mask = model._ensure_nested_transition(
    x0, attention, s_time)

  assert torch.all(~t_mask | s_mask)
  assert torch.all(s_mask.sum(dim=-1) >= 2)
  assert torch.all(t_mask.sum(dim=-1) >= 1)
  assert torch.all(t_mask.sum(dim=-1) < s_mask.sum(dim=-1))
  assert torch.equal(x_s[s_mask], torch.full_like(x_s[s_mask], model.mask_index))
  assert torch.equal(x_t[~t_mask], x0[~t_mask])


def test_three_pass_pretrain_loss_is_finite_and_trains_final_writer():
  torch.manual_seed(11)
  model, tokenizer = make_pretrain_model()
  torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.02)
  x0 = torch.randint(1, tokenizer.vocab_size, (2, 8))
  attention = torch.ones_like(x0)

  total, s_loss, diagnostics = model._step_memory_pretrain_loss(x0, attention)
  assert torch.isfinite(total)
  assert torch.isfinite(s_loss.loss)
  assert set(diagnostics) == {
    'loss_full', 'loss_s', 'loss_t', 's_mask_ratio', 't_mask_ratio',
    'revealed_tokens', 'remaining_masks'}
  total.backward()
  gradient = model.backbone.dc_final_writer.kv.weight.grad
  assert gradient is not None
  assert torch.count_nonzero(gradient) > 0


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
  assert torch.isfinite(torch.tensor(result['val/loss_full']))
  assert torch.isfinite(torch.tensor(result['val/loss_total']))
