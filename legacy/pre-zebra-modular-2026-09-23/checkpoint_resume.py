"""Opt-in, optimizer-boundary migration of this repo's DDP resume counters.

Only the loaded checkpoint dictionary is changed; checkpoint files, optimizer
state, scheduler state, EMA, model weights, and the denoising recipe are not.
This preserves a global permutation cursor, not random draws or bitwise results.
"""

import copy
from numbers import Integral


GEOMETRY_KEY = 'dcache_batch_geometry'
_COUNTERS = ('ready', 'started', 'processed', 'completed')


def _integer(value, name, minimum=1):
  if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
    raise ValueError(f'{name} must be an integer >= {minimum}, got {value!r}')
  return int(value)


def _validate_geometry(geometry):
  if geometry.get('version') != 1:
    raise ValueError('Unsupported checkpoint batch-geometry metadata version')
  result = {'version': 1}
  for key in ('world_size', 'micro_batch', 'global_batch', 'accumulation'):
    result[key] = _integer(geometry[key], key)
  if not isinstance(geometry.get('distributed_sampler'), bool):
    raise ValueError('Batch geometry must identify its distributed sampler')
  result['distributed_sampler'] = geometry['distributed_sampler']
  if (result['world_size'] * result['micro_batch'] * result['accumulation']
      != result['global_batch']):
    raise ValueError('global_batch must equal world_size * micro_batch * accumulation')
  return result


def batch_geometry(config, *, world_size=None, accumulation=None,
                   distributed_sampler=None, verify_config=True):
  """Read resolved config values, optionally verified against a live Trainer."""
  trainer = config['trainer']
  configured_world = (
    _integer(trainer['devices'], 'trainer.devices')
    * _integer(trainer.get('num_nodes', 1), 'trainer.num_nodes'))
  if verify_config and world_size is not None and configured_world != world_size:
    raise ValueError('Configured world size does not match the attached Trainer')
  configured_accumulation = _integer(
    trainer['accumulate_grad_batches'], 'trainer.accumulate_grad_batches')
  if verify_config and accumulation is not None and configured_accumulation != accumulation:
    raise ValueError('Configured accumulation does not match the attached Trainer')
  if distributed_sampler is None:
    strategy = config.get('strategy', {})
    distributed_sampler = (
      strategy.get('_target_') == 'lightning.pytorch.strategies.DDPStrategy')
  actual_world = configured_world if world_size is None else world_size
  actual_accumulation = configured_accumulation if accumulation is None else accumulation
  micro_batch = config['loader']['batch_size']
  return _validate_geometry({
    'version': 1,
    'world_size': actual_world,
    'micro_batch': micro_batch,
    # Saving must record actual Trainer overrides (including small CPU tests),
    # not a stale declaration in hyper_parameters. Resume validates the config.
    'global_batch': (config['loader']['global_batch_size'] if verify_config
                     else actual_world * micro_batch * actual_accumulation),
    'accumulation': actual_accumulation,
    'distributed_sampler': distributed_sampler,
  })


def migrate_batch_geometry(checkpoint, new_geometry, *, allow_change=False):
  """Remap completed DDP microbatch counters, or refuse an unsafe resume.

  Requires constant global batch and complete optimizer boundaries, with one
  optimizer and an unchanged dataset/permutation seed/order. End-of-epoch/padded
  checkpoints and partial accumulation are deliberately unsupported. The caller
  must call this only for Trainer.fit, before Lightning restores loop state.
  Same-geometry resumes are left byte-for-byte unchanged in memory.
  """
  new = _validate_geometry(new_geometry)
  if GEOMETRY_KEY in checkpoint:
    old = _validate_geometry(checkpoint[GEOMETRY_KEY])
  else:
    old = batch_geometry(checkpoint['hyper_parameters']['config'])
  if old == new:
    return None
  if not allow_change:
    raise ValueError(
      'Checkpoint batch geometry changed. Set '
      'checkpointing.allow_batch_geometry_change=true only for a verified '
      'same-global-batch DDP migration.')
  if old['global_batch'] != new['global_batch']:
    raise ValueError('Batch-geometry migration requires unchanged global batch size')
  if not old['distributed_sampler'] or not new['distributed_sampler']:
    raise ValueError('Batch-geometry migration requires explicit DDP on both runs, even for one GPU')
  if len(checkpoint.get('optimizer_states', [])) != 1:
    raise ValueError('Batch-geometry migration requires one saved optimizer')
  if len(checkpoint.get('lr_schedulers', [])) != 1:
    raise ValueError('Batch-geometry migration requires one saved step scheduler')

  # Validate everything before replacing any in-memory state. Do not deepcopy
  # model/Adam/EMA tensors: those can occupy several GB even with mmap loading.
  fit = copy.deepcopy(checkpoint['loops']['fit_loop'])
  batch = fit['epoch_loop.batch_progress']
  if batch.get('is_last_batch', False):
    raise ValueError('End-of-epoch/padded checkpoints need a separate sampler migration')
  epoch = _integer(fit['epoch_progress']['current']['completed'], 'epoch', 0)
  step = fit['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']
  global_step = _integer(checkpoint['global_step'], 'global_step', 0)
  if step['total']['completed'] != global_step:
    raise ValueError('Checkpoint global step and optimizer step disagree')
  if fit['epoch_loop.state_dict']['_batches_that_stepped'] != global_step:
    raise ValueError('Checkpoint step counter and global step disagree')
  if step['current']['completed'] > global_step:
    raise ValueError('Current-epoch optimizer steps exceed total optimizer steps')
  for scope in ('total', 'current'):
    updates = _integer(step[scope]['completed'], f'{scope} optimizer steps', 0)
    if step[scope]['ready'] != updates:
      raise ValueError('Cannot migrate a partially completed optimizer step')
    expected = updates * old['accumulation']
    if any(batch[scope][key] != expected for key in _COUNTERS):
      raise ValueError(
        f'Cannot migrate {scope} partial/inconsistent microbatch counters; '
        'checkpoint must be saved at a complete optimizer boundary')
    for key in _COUNTERS:
      batch[scope][key] = updates * new['accumulation']

  global_offset = step['current']['completed'] * old['global_batch']
  if (batch['current']['completed'] * new['micro_batch'] * new['world_size']
      != global_offset):
    raise ValueError('Internal error: migrated global sample cursor changed')
  report = {
    'from': old,
    'to': new,
    'global_step': global_step,
    'epoch': epoch,
    'global_samples_in_epoch': global_offset,
    'rank_samples_in_epoch': global_offset // new['world_size'],
  }
  checkpoint['loops']['fit_loop'] = fit
  checkpoint[GEOMETRY_KEY] = new
  return report
