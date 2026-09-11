"""CPU-only resume guards. Run with TMPDIR and pytest basetemp inside .cache."""

import copy
import hashlib
import itertools
from types import SimpleNamespace

import lightning as L
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from checkpoint_resume import GEOMETRY_KEY, batch_geometry, migrate_batch_geometry
from dataloader import FaultTolerantDistributedSampler
from diffusion import Diffusion


def config(world=2, micro=2, global_batch=512):
  return {
    'trainer': {'devices': world, 'num_nodes': 1,
                'accumulate_grad_batches': global_batch // (world * micro)},
    'loader': {'batch_size': micro, 'global_batch_size': global_batch,
               'num_workers': 1, 'pin_memory': False},
    'strategy': {'_target_': 'lightning.pytorch.strategies.DDPStrategy'},
    'checkpointing': {'allow_batch_geometry_change': True},
  }


def checkpoint():
  progress = dict.fromkeys(('ready', 'started', 'processed', 'completed'), 192000)
  return {
    'global_step': 1500, 'epoch': 0,
    'hyper_parameters': {'config': config()},
    'state_dict': {'weight': torch.ones(1)},
    'ema': {'num_updates': 1500, 'shadow_params': [torch.ones(1)]},
    'optimizer_states': [{'state': {'moment': torch.tensor([1.5])}}],
    'lr_schedulers': [{'last_epoch': 1500}],
    'sampler': {'random_state': None},
    'loops': {'fit_loop': {
      'epoch_progress': {'current': {'completed': 0}},
      'epoch_loop.batch_progress': {
        'current': dict(progress), 'total': dict(progress), 'is_last_batch': False},
      'epoch_loop.automatic_optimization.optim_progress': {'optimizer': {'step': {
        'current': {'ready': 1500, 'completed': 1500},
        'total': {'ready': 1500, 'completed': 1500}}}},
      'epoch_loop.state_dict': {'_batches_that_stepped': 1500}}},
  }


def test_two_gpu_to_one_preserves_global_cursor_and_next_documents():
  saved = checkpoint()
  original = copy.deepcopy(saved)
  report = migrate_batch_geometry(
    saved, batch_geometry(config(world=1)), allow_change=True)
  assert report['global_samples_in_epoch'] == 768000
  assert report['rank_samples_in_epoch'] == 768000
  assert report['epoch'] == 0
  progress = saved['loops']['fit_loop']['epoch_loop.batch_progress']
  for scope in ('current', 'total'):
    assert set(progress[scope].values()) == {384000}
  assert saved['global_step'] == 1500
  assert saved['lr_schedulers'] == original['lr_schedulers']
  assert saved['optimizer_states'] == original['optimizer_states']
  assert saved['ema'] == original['ema']
  assert saved['hyper_parameters'] == original['hyper_parameters']
  assert saved['sampler'] == original['sampler']

  dataset = range(800512)
  old_next = []
  for rank in (0, 1):
    sampler = FaultTolerantDistributedSampler(dataset, num_replicas=2, rank=rank)
    sampler.load_state_dict({'epoch': 0, 'counter': 384000})
    old_next.append(list(itertools.islice(iter(sampler), 256)))
  expected = [index for pair in zip(*old_next) for index in pair]
  sampler = FaultTolerantDistributedSampler(dataset, num_replicas=1, rank=0)
  sampler.load_state_dict({'epoch': 0, 'counter': report['rank_samples_in_epoch']})
  assert list(itertools.islice(iter(sampler), 512)) == expected


def test_same_geometry_is_unchanged_even_without_opt_in():
  saved = checkpoint()
  loops = saved['loops']['fit_loop']
  assert migrate_batch_geometry(saved, batch_geometry(config())) is None
  assert saved['loops']['fit_loop'] is loops
  assert GEOMETRY_KEY not in saved


def test_migration_does_not_modify_the_source_checkpoint_file(tmp_path):
  path = tmp_path / 'source.ckpt'
  torch.save(checkpoint(), path)
  before = hashlib.sha256(path.read_bytes()).hexdigest()
  loaded = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
  migrate_batch_geometry(loaded, batch_geometry(config(world=1)), allow_change=True)
  assert hashlib.sha256(path.read_bytes()).hexdigest() == before
  reloaded = torch.load(path, map_location='cpu', weights_only=False)
  assert reloaded['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed'] == 192000


@pytest.mark.parametrize('fault', [
  'no_opt_in', 'different_global_batch', 'non_ddp', 'partial_microbatch',
  'partial_optimizer', 'wrong_step', 'wrong_batch_steps', 'last_batch',
  'missing_optimizer', 'missing_scheduler', 'inconsistent_geometry',
])
def test_invalid_migration_is_rejected_without_partial_mutation(fault):
  saved = checkpoint()
  target = batch_geometry(config(world=1))
  allow = True
  fit = saved['loops']['fit_loop']
  if fault == 'no_opt_in':
    allow = False
  elif fault == 'different_global_batch':
    target = batch_geometry(config(world=1, global_batch=256))
  elif fault == 'non_ddp':
    target['distributed_sampler'] = False
  elif fault == 'partial_microbatch':
    fit['epoch_loop.batch_progress']['current']['ready'] += 1
  elif fault == 'partial_optimizer':
    fit['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']['current']['ready'] += 1
  elif fault == 'wrong_step':
    saved['global_step'] += 1
  elif fault == 'wrong_batch_steps':
    fit['epoch_loop.state_dict']['_batches_that_stepped'] -= 1
  elif fault == 'last_batch':
    fit['epoch_loop.batch_progress']['is_last_batch'] = True
  elif fault == 'missing_optimizer':
    saved.pop('optimizer_states')
  elif fault == 'missing_scheduler':
    saved.pop('lr_schedulers')
  elif fault == 'inconsistent_geometry':
    target['accumulation'] = 128
  before = copy.deepcopy(fit)
  with pytest.raises(ValueError):
    migrate_batch_geometry(saved, target, allow_change=allow)
  assert saved['loops']['fit_loop'] == before
  assert GEOMETRY_KEY not in saved


def test_runtime_geometry_must_match_config_when_resuming():
  with pytest.raises(ValueError, match='world size'):
    batch_geometry(config(), world_size=1)
  with pytest.raises(ValueError, match='accumulation'):
    batch_geometry(config(world=1), accumulation=128)
  # Saves record actual runtime settings, even if a programmatic tiny test
  # deliberately overrides the configuration passed to the Trainer.
  actual = batch_geometry(config(), world_size=1, accumulation=2,
                          distributed_sampler=False, verify_config=False)
  assert actual['global_batch'] == 4


def test_migrated_metadata_is_used_for_later_resumes():
  saved = checkpoint()
  target = batch_geometry(config(world=1))
  migrate_batch_geometry(saved, target, allow_change=True)
  loops = saved['loops']['fit_loop']
  assert migrate_batch_geometry(saved, target, allow_change=False) is None
  assert saved['loops']['fit_loop'] is loops


@pytest.mark.parametrize('attached', [False, True])
def test_standalone_evaluation_skips_geometry_migration(attached):
  saved = checkpoint()
  model = SimpleNamespace(ema=None, config=config(world=1))
  model._trainer = SimpleNamespace(state=SimpleNamespace(fn='validate')) if attached else None
  Diffusion.on_load_checkpoint(model, saved)
  assert model.fast_forward_batches == 192000
  assert GEOMETRY_KEY not in saved


class TinyResumeModel(L.LightningModule):
  """Use real Diffusion checkpoint/sampler hooks without language-model work."""

  on_load_checkpoint = Diffusion.on_load_checkpoint
  on_save_checkpoint = Diffusion.on_save_checkpoint
  on_train_start = Diffusion.on_train_start
  on_validation_model_zero_grad = Diffusion.on_validation_model_zero_grad

  def __init__(self, micro):
    super().__init__()
    self.config = OmegaConf.create(config(world=1, micro=micro, global_batch=8))
    self.save_hyperparameters({'config': self.config})
    self.weight = torch.nn.Parameter(torch.tensor(0.1))
    self.ema = None
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self.valid_steps = []
    self.loaded_state = None

  def training_step(self, batch, batch_idx):
    if self.loaded_state is None:
      optimizer = self.trainer.optimizers[0]
      self.loaded_state = {
        'step': self.global_step,
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'scheduler': copy.deepcopy(self.trainer.lr_scheduler_configs[0].scheduler.state_dict()),
      }
    return (self.weight * batch[0] - 1).square().mean()

  def validation_step(self, batch, batch_idx):
    self.valid_steps.append(self.global_step)

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step + 1) / 100)
    return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


def test_socket_free_first_resumed_validation_is_not_suppressed(tmp_path, monkeypatch):
  # A normal single-device Trainer exercises Lightning's restored-loop and
  # validation hooks without initializing distributed sockets/worker IPC.
  # Distributed cursor migration itself is covered separately above and below.
  monkeypatch.setattr(TinyResumeModel, 'on_train_start', lambda self: None)
  data = TensorDataset(torch.arange(1, 129, dtype=torch.float32))

  def trainer(steps):
    return L.Trainer(
      accelerator='cpu', devices=1, max_steps=steps,
      accumulate_grad_batches=2, val_check_interval=2,
      limit_val_batches=1, num_sanity_val_steps=0,
      logger=False, enable_checkpointing=False, enable_progress_bar=False,
      enable_model_summary=False, default_root_dir=tmp_path)

  first = TinyResumeModel(micro=4)
  initial_trainer = trainer(2)
  initial_trainer.fit(first, DataLoader(data, batch_size=4), DataLoader(data, batch_size=4))
  path = tmp_path / 'single-device.ckpt'
  initial_trainer.save_checkpoint(path)
  second = TinyResumeModel(micro=4)
  resumed_trainer = trainer(3)
  resumed_trainer.fit(second, DataLoader(data, batch_size=4), DataLoader(data, batch_size=4), ckpt_path=path)
  assert resumed_trainer.global_step == 3
  assert 3 in second.valid_steps
  assert second.loaded_state['step'] == 2
  assert second._batch_geometry_migration is None


def test_real_cpu_lightning_resume_restores_states_and_first_validation(tmp_path):
  from lightning.pytorch.strategies import DDPStrategy

  def trainer(micro, steps):
    return L.Trainer(
      accelerator='cpu', devices=1, strategy=DDPStrategy(process_group_backend='gloo'),
      max_steps=steps, accumulate_grad_batches=8 // micro,
      val_check_interval=8 // micro, limit_val_batches=1, num_sanity_val_steps=0,
      logger=False, enable_checkpointing=False, enable_progress_bar=False,
      enable_model_summary=False, default_root_dir=tmp_path)

  data = TensorDataset(torch.arange(1, 129, dtype=torch.float32))
  first = TinyResumeModel(micro=4)
  initial_trainer = trainer(micro=4, steps=2)
  initial_trainer.fit(first, DataLoader(data, batch_size=4), DataLoader(data, batch_size=4))
  path = tmp_path / 'tiny.ckpt'
  initial_trainer.save_checkpoint(path)
  saved = torch.load(path, map_location='cpu', weights_only=False)
  before = hashlib.sha256(path.read_bytes()).hexdigest()

  second = TinyResumeModel(micro=2)
  resumed_trainer = trainer(micro=2, steps=3)
  resumed_trainer.fit(second, DataLoader(data, batch_size=2), DataLoader(data, batch_size=2), ckpt_path=path)

  assert second.loaded_state['step'] == 2
  assert second.loaded_state['scheduler'] == saved['lr_schedulers'][0]
  expected_optimizer = saved['optimizer_states'][0]
  actual_optimizer = second.loaded_state['optimizer']
  assert actual_optimizer['param_groups'] == expected_optimizer['param_groups']
  for parameter_id, values in expected_optimizer['state'].items():
    for name, value in values.items():
      torch.testing.assert_close(actual_optimizer['state'][parameter_id][name], value, rtol=0, atol=0)
  assert second.fast_forward_batches == 8
  assert second._batch_geometry_migration['global_samples_in_epoch'] == 16
  assert resumed_trainer.global_step == 3
  assert 3 in second.valid_steps  # First scheduled post-resume validation is not skipped.
  assert resumed_trainer.lr_scheduler_configs[0].scheduler.last_epoch == 3
  assert hashlib.sha256(path.read_bytes()).hexdigest() == before
