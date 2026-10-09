import pathlib

import hydra
import lightning as L
import torch
from omegaconf import OmegaConf

import dataloader
from diffusion import DcachehoopingOutput, Diffusion


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def make_model(enabled=True, auxiliary_probability=1.0, ema=0,
               core_only=False):
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
        'step_memory.pretrain.source_dropout.enabled=true',
        'step_memory.pretrain.identity.enabled=true',
        'step_memory.pretrain.identity.batch_probability=1.0',
        f'dcachehooping.enabled={str(enabled).lower()}',
        # The full-objective unit test deliberately exercises both auxiliary
        # graphs together at tiny scale. Production uses exclusive routes.
        'dcachehooping.exclusive_auxiliary_routes=false',
        'dcachehooping.latent_dropout_probability=0.0',
        f'dcachehooping.latent_mask_probability={auxiliary_probability}',
        f'dcachehooping.tentative.batch_probability={auxiliary_probability}',
        *([
          'dcachehooping.status_embedding.enabled=false',
          'dcachehooping.latent_mask_probability=0.0',
          'dcachehooping.latent_mask_loss_weight=0.0',
          'dcachehooping.tentative.enabled=false',
          'dcachehooping.tentative.batch_probability=0.0',
          'dcachehooping.tentative.loss_weight=0.0',
          'dcachehooping.confidence.enabled=false',
          'dcachehooping.confidence.loss_weight=0.0',
        ] if core_only else []),
        'wandb=null',
      ])
  tokenizer = dataloader.Text8Tokenizer()
  return Diffusion(config, tokenizer=tokenizer), tokenizer


def test_neutral_initialization_preserves_dcache_v2_logits():
  torch.manual_seed(101)
  base, tokenizer = make_model(enabled=False)
  torch.manual_seed(102)
  hooping, _ = make_model(enabled=True)
  missing, unexpected = hooping.load_state_dict(base.state_dict(), strict=False)
  assert not unexpected
  assert all('dcachehooping_' in name for name in missing)
  base.eval()
  hooping.eval()
  x = torch.randint(1, tokenizer.vocab_size, (2, 8))
  sigma = torch.ones(2, 1)
  previous_hidden = torch.randn(2, 8, 32)
  status = torch.randint(0, 3, (2, 8))

  with torch.no_grad():
    expected = base.forward(x, sigma=sigma, sample_mode=True)
    actual = hooping.forward(
      x, sigma=sigma, sample_mode=True,
      previous_final_hidden=previous_hidden,
      token_status=status,
      return_dcachehooping=True).scores

  torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_editable_log_probs_do_not_copy_clamp_tentative_tokens():
  torch.manual_seed(103)
  model, tokenizer = make_model(enabled=True)
  torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.02)
  x = torch.randint(1, tokenizer.vocab_size, (2, 8))
  sigma = torch.ones(2, 1)
  status = torch.full_like(x, 2)

  output = model.forward(
    x, sigma=sigma, sample_mode=True, token_status=status,
    return_dcachehooping=True,
    return_editable_log_probs=True,
    return_confidence_logits=True)

  copied = torch.gather(output.scores, -1, x[:, :, None]).squeeze(-1)
  editable = torch.gather(
    output.editable_log_probs, -1, x[:, :, None]).squeeze(-1)
  visible = x.ne(model.mask_index)
  assert torch.all(copied[visible] == 0)
  assert torch.all(torch.isfinite(editable[visible]))
  assert torch.all(editable[visible] < 0)


def test_core_mode_omits_tentative_parameters_and_large_optional_outputs():
  torch.manual_seed(109)
  model, tokenizer = make_model(
    enabled=True, auxiliary_probability=0.0, core_only=True)
  torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.02)
  assert model.backbone.dcachehooping_status_embed is None
  assert model.backbone.dcachehooping_confidence_head is None

  x = torch.randint(1, tokenizer.vocab_size, (2, 8))
  sigma = torch.ones(2, 1)
  output = model.forward(
    x, sigma=sigma, sample_mode=True, return_step_kv=True,
    return_dcachehooping=True)
  assert output.editable_log_probs is None
  assert output.confidence_logits is None

  attention = torch.ones_like(x)
  total, _, diagnostics = model._dcachehooping_pretrain_loss(x, attention)
  assert diagnostics['tentative_applied'] == 0
  assert diagnostics['latent_mask_applied'] == 0
  total.backward()
  assert torch.count_nonzero(
    model.backbone.dcachehooping_latent_norm.weight.grad) > 0


def test_sampling_update_propagates_both_recurrent_sources():
  torch.manual_seed(110)
  model, _ = make_model(
    enabled=True, auxiliary_probability=0.0, core_only=True)
  model.eval()
  batch_size = 2
  x = torch.full((batch_size, 8), model.mask_index, dtype=torch.long)
  previous_cache = [torch.randn(batch_size, 4, 8, 8)]
  previous_hidden = torch.randn(batch_size, 8, 32)
  next_cache = [torch.randn(batch_size, 4, 8, 8)]
  next_hidden = torch.randn(batch_size, 8, 32, requires_grad=True)
  calls = []

  def fake_forward(indices, sigma, **kwargs):
    calls.append(kwargs)
    logits = torch.randn(batch_size, 8, model.vocab_size)
    return DcachehoopingOutput(
      scores=logits.log_softmax(-1),
      editable_log_probs=None,
      step_kv=next_cache,
      final_hidden=next_hidden,
      confidence_logits=None)

  model.forward = fake_forward
  _, _, returned_cache, returned_hidden = model._ddpm_caching_update(
    x=x,
    t=torch.full((batch_size, 1), 0.8),
    dt=0.1,
    previous_step_kv=previous_cache,
    previous_final_hidden=previous_hidden)

  assert len(calls) == 1
  assert calls[0]['previous_step_kv'] is previous_cache
  assert calls[0]['previous_final_hidden'] is previous_hidden
  assert calls[0]['return_dcachehooping'] is True
  assert returned_cache is next_cache
  torch.testing.assert_close(returned_hidden, next_hidden)
  assert not returned_hidden.requires_grad


def test_cached_logits_do_not_advance_recurrent_sources():
  torch.manual_seed(111)
  model, _ = make_model(
    enabled=True, auxiliary_probability=0.0, core_only=True)
  model.eval()
  batch_size = 2
  x = torch.full((batch_size, 8), model.mask_index, dtype=torch.long)
  previous_cache = [torch.randn(batch_size, 4, 8, 8)]
  previous_hidden = torch.randn(batch_size, 8, 32)
  probabilities = torch.rand(batch_size, 8, model.vocab_size)
  probabilities /= probabilities.sum(-1, keepdim=True)

  def forbidden_forward(*args, **kwargs):
    raise AssertionError('Cached logits must not trigger a model forward')

  model.forward = forbidden_forward
  _, _, returned_cache, returned_hidden = model._ddpm_caching_update(
    x=x,
    t=torch.full((batch_size, 1), 0.8),
    dt=0.1,
    p_x0=probabilities,
    previous_step_kv=previous_cache,
    previous_final_hidden=previous_hidden)

  assert returned_cache is previous_cache
  assert returned_hidden is previous_hidden


def test_dcachehooping_full_objective_is_finite_and_trains_new_paths():
  torch.manual_seed(104)
  model, tokenizer = make_model(enabled=True)
  torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.02)
  x0 = torch.randint(1, tokenizer.vocab_size, (2, 8))
  attention = torch.ones_like(x0)

  total, reference, diagnostics = model._dcachehooping_pretrain_loss(
    x0, attention)

  assert torch.isfinite(total)
  assert torch.isfinite(reference.loss)
  for name in [
      'loss_base', 'latent_mask_loss', 'tentative_loss', 'confidence_loss',
      'tentative_before_accuracy', 'tentative_after_accuracy',
      'tentative_wrong_fix_rate', 'tentative_correct_keep_rate',
      'confidence_brier', 'identity_loss', 'identity_gain']:
    assert name in diagnostics
    assert torch.isfinite(diagnostics[name])
  assert diagnostics['latent_mask_applied'] == 1
  assert diagnostics['tentative_applied'] == 1
  assert diagnostics['identity_applied'] == 1
  assert diagnostics['tentative_count'] > 0

  total.backward()
  assert torch.count_nonzero(
    model.backbone.dcachehooping_latent_norm.weight.grad) > 0
  assert torch.count_nonzero(
    model.backbone.dcachehooping_status_embed.weight.grad) > 0
  assert torch.count_nonzero(
    model.backbone.dcachehooping_confidence_head.weight.grad) > 0


def test_exclusive_auxiliary_routes_preserve_each_route_without_overlap():
  torch.manual_seed(108)
  model, tokenizer = make_model(enabled=True, auxiliary_probability=0.0)
  model.config.dcachehooping.exclusive_auxiliary_routes = True
  model.config.dcachehooping.latent_mask_probability = 0.10
  model.config.dcachehooping.tentative.batch_probability = 0.25
  model.config.step_memory.pretrain.identity.enabled = False
  x0 = torch.randint(1, tokenizer.vocab_size, (2, 8))
  attention = torch.ones_like(x0)

  # The first interval [0, .25) is tentative correction.
  model._sample_synchronized_uniform = lambda device: 0.10
  _, _, tentative = model._dcachehooping_pretrain_loss(x0, attention)
  assert tentative['tentative_applied'] == 1
  assert tentative['latent_mask_applied'] == 0

  # The next interval [.25, .35) is latent-mask robustness.
  model._sample_synchronized_uniform = lambda device: 0.30
  _, _, latent_mask = model._dcachehooping_pretrain_loss(x0, attention)
  assert latent_mask['tentative_applied'] == 0
  assert latent_mask['latent_mask_applied'] == 1


def test_full_mask_robustness_keeps_prefix_and_original_loss_mask():
  torch.manual_seed(105)
  model, tokenizer = make_model(enabled=True)
  x0 = torch.randint(1, tokenizer.vocab_size, (2, 8))
  attention = torch.ones_like(x0)
  attention[:, -1] = 0
  eligible = attention.bool()
  eligible[:, 0] = False
  lexical_mask_state = torch.where(eligible, model.mask_index, x0)

  assert torch.equal(lexical_mask_state[:, 0], x0[:, 0])
  assert torch.equal(lexical_mask_state[:, -1], x0[:, -1])
  assert torch.all(lexical_mask_state[eligible] == model.mask_index)


def test_dcache_v2_checkpoint_and_ema_migrate_to_dcachehooping(tmp_path):
  torch.manual_seed(106)
  old_model, tokenizer = make_model(
    enabled=False, auxiliary_probability=0.0, ema=0.9)
  examples = [{
    'input_ids': torch.randint(1, tokenizer.vocab_size, (8,)),
    'attention_mask': torch.ones(8, dtype=torch.long),
  } for _ in range(2)]
  loader = torch.utils.data.DataLoader(examples, batch_size=2)
  trainer = L.Trainer(
    accelerator='cpu', devices=1, logger=False, enable_checkpointing=False,
    enable_progress_bar=False, max_steps=1, limit_train_batches=1,
    limit_val_batches=0, num_sanity_val_steps=0)
  trainer.fit(old_model, loader)
  checkpoint = tmp_path / 'dcache-v2.ckpt'
  trainer.save_checkpoint(checkpoint)

  new_model, _ = make_model(enabled=True, ema=0.9)
  migrated = Diffusion.load_from_checkpoint(
    checkpoint, config=new_model.config, tokenizer=tokenizer,
    strict=False, weights_only=False)

  assert migrated.backbone.dcachehooping_latent_norm is not None
  assert torch.count_nonzero(
    migrated.backbone.dcachehooping_latent_norm.weight) == 0
  assert len(migrated.ema.shadow_params) == len([
    parameter for parameter in migrated.parameters()
    if parameter.requires_grad])


def test_optional_parameters_remain_in_graph_when_auxiliaries_are_absent():
  torch.manual_seed(107)
  model, tokenizer = make_model(enabled=True, auxiliary_probability=0.0)
  model.config.dcachehooping.latent_dropout_probability = 1.0
  torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.02)
  x0 = torch.randint(1, tokenizer.vocab_size, (2, 8))
  attention = torch.ones_like(x0)

  total, _, diagnostics = model._dcachehooping_pretrain_loss(x0, attention)
  assert diagnostics['tentative_applied'] == 0
  assert diagnostics['latent_mask_applied'] == 0
  total.backward()

  confidence_weight_grad = (
    model.backbone.dcachehooping_confidence_head.weight.grad)
  latent_weight_grad = model.backbone.dcachehooping_latent_norm.weight.grad
  assert confidence_weight_grad is not None
  assert latent_weight_grad is not None
  assert torch.count_nonzero(confidence_weight_grad) == 0
  assert torch.count_nonzero(latent_weight_grad) == 0
