#!/usr/bin/env python3
"""Bounded synthetic Lightning/EMA/DDP smoke test for adjacent-only DCache.

Examples (physical devices are selected by the caller, not this script):
  CUDA_VISIBLE_DEVICES=2,3 python scripts/profile_adjacent_gradients.py \
    --devices 2 --steps 3 --accumulate 2 --force-identity \
    --work-dir outputs/smoke-adjacent-real-unique
  python scripts/profile_adjacent_gradients.py --tiny --accelerator cpu \
    --devices 2 --work-dir outputs/smoke-adjacent-cpu-unique

This is a synthetic correctness/resource test, not a research training run.
It never loads a training dataset or checkpoint. No checkpoint is saved unless
--save-checkpoint is explicitly requested. All task scratch stays in the repo.
"""

import argparse
import contextlib
import copy
import json
import os
import pathlib
import sys
import time


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO_ROOT / '.cache' / 'runtime' / 'adjacent-smoke'
for _key, _directory in {
    'TMPDIR': RUNTIME_ROOT / 'tmp',
    'TMP': RUNTIME_ROOT / 'tmp',
    'TEMP': RUNTIME_ROOT / 'tmp',
    'MPLCONFIGDIR': REPO_ROOT / '.cache' / 'matplotlib',
    'TORCHINDUCTOR_CACHE_DIR': REPO_ROOT / '.cache' / 'torchinductor',
    'TRITON_CACHE_DIR': REPO_ROOT / '.cache' / 'triton',
    'CUDA_CACHE_PATH': REPO_ROOT / '.cache' / 'cuda',
    'XDG_CACHE_HOME': REPO_ROOT / '.cache',
    'HF_HOME': REPO_ROOT / '.cache' / 'huggingface',
    'HF_HUB_CACHE': REPO_ROOT / '.cache' / 'huggingface' / 'hub',
    'HF_DATASETS_CACHE': REPO_ROOT / '.cache' / 'huggingface' / 'datasets',
    'TORCH_HOME': REPO_ROOT / '.cache' / 'torch',
    'NUMBA_CACHE_DIR': REPO_ROOT / '.cache' / 'numba',
}.items():
  _directory.mkdir(parents=True, exist_ok=True)
  os.environ[_key] = str(_directory)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
sys.path.insert(0, str(REPO_ROOT))

import hydra  # noqa: E402
import lightning as L  # noqa: E402
from lightning.pytorch.strategies import DDPStrategy  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import dataloader  # noqa: E402
import diffusion  # noqa: E402
from recurrent_gradients import AdjacentCacheGradients  # noqa: E402


def atomic_json(path, value):
  temporary = path.with_suffix(path.suffix + '.partial')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  temporary.replace(path)


class ToyRecurrence(torch.nn.Module):
  """Small independent model used only to check DDP gradient arithmetic."""

  weights = (0.05, 0.10, 0.20, 1.00, 0.70)

  def __init__(self):
    super().__init__()
    self.input_map = torch.nn.Linear(4, 4)
    self.reader = torch.nn.Linear(4, 4)
    self.writer = torch.nn.Linear(4, 4)
    self.output_map = torch.nn.Linear(4, 2)

  def state(self, inputs, memory, final_hidden):
    value = self.input_map(inputs)
    if memory is not None:
      value = value + self.reader(memory)
    if final_hidden is not None:
      value = value + 0.13 * final_hidden.detach()
    return torch.tanh(value)

  def local_loss(self, hidden, target):
    return (self.output_map(hidden) - target).square().mean()

  def forward(self, inputs, targets):
    bridge = AdjacentCacheGradients()
    memory = final_hidden = None
    losses = []
    for step, weight in enumerate(self.weights):
      hidden = self.state(inputs[:, step], memory, final_hidden)
      losses.append(weight * self.local_loss(hidden, targets[:, step]))
      if step < 4:
        memory = bridge.consume([self.writer(hidden)])[0]
      final_hidden = hidden.detach()
    return bridge.attach(sum(losses) / sum(self.weights))

  def reference_loss(self, inputs, targets):
    # Materialize the trajectory values, then independently replay each loss
    # and at most its one predecessor. No bridge/helper is used here.
    memories, final_hiddens = [None], [None]
    with torch.no_grad():
      for step in range(5):
        hidden = self.state(
          inputs[:, step], memories[-1], final_hiddens[-1])
        memories.append(self.writer(hidden))
        final_hiddens.append(hidden)
    losses = []
    for step, weight in enumerate(self.weights):
      memory = None
      if step:
        producer = self.state(
          inputs[:, step - 1], memories[step - 1],
          final_hiddens[step - 1])
        memory = self.writer(producer)
      hidden = self.state(inputs[:, step], memory, final_hiddens[step])
      losses.append(weight * self.local_loss(hidden, targets[:, step]))
    return sum(losses) / sum(self.weights)


def toy_parity_worker(rank, work_dir):
  work_dir = pathlib.Path(work_dir)
  world_size, accumulation, micro_batch = 2, 2, 2
  torch.set_num_threads(2)
  torch.distributed.init_process_group(
    'gloo', init_method='file://' + str(work_dir / 'rendezvous'),
    rank=rank, world_size=world_size)
  try:
    torch.manual_seed(123)
    model = torch.nn.parallel.DistributedDataParallel(
      ToyRecurrence().double(), find_unused_parameters=False)
    reference = copy.deepcopy(model.module)
    generator = torch.Generator().manual_seed(89)
    inputs = torch.randn(8, 5, 4, generator=generator, dtype=torch.float64)
    targets = torch.randn(8, 5, 2, generator=generator, dtype=torch.float64)
    for micro_step in range(accumulation):
      offset = (micro_step * world_size + rank) * micro_batch
      synchronize = micro_step == accumulation - 1
      with (contextlib.nullcontext() if synchronize else model.no_sync()):
        loss = model(inputs[offset:offset + micro_batch],
                     targets[offset:offset + micro_batch]) / accumulation
        loss.backward()
    reference.reference_loss(inputs, targets).backward()
    max_gradient_difference = 0.0
    for actual, expected in zip(model.module.parameters(),
                                reference.parameters()):
      if actual.grad is None or expected.grad is None:
        raise RuntimeError('Toy parity has missing parameter gradients')
      torch.testing.assert_close(
        actual.grad, expected.grad, rtol=1e-10, atol=1e-12)
      max_gradient_difference = max(max_gradient_difference, float(
        (actual.grad - expected.grad).abs().max()))
    torch.optim.SGD(model.parameters(), lr=0.03).step()
    torch.optim.SGD(reference.parameters(), lr=0.03).step()
    max_parameter_difference = 0.0
    for actual, expected in zip(model.module.parameters(),
                                reference.parameters()):
      torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
      max_parameter_difference = max(max_parameter_difference, float(
        (actual.detach() - expected.detach()).abs().max()))
    atomic_json(work_dir / f'parity-rank-{rank}.json', {
      'status': 'passed', 'rank': rank, 'world_size': world_size,
      'accumulate': accumulation,
      'maximum_gradient_error': max_gradient_difference,
      'maximum_parameter_error': max_parameter_difference,
      'reference': 'Independent per-loss one-predecessor replay, global batch',
    })
    torch.distributed.barrier()
  finally:
    torch.distributed.destroy_process_group()


class SyntheticTokens(Dataset):
  def __init__(self, tokenizer, size, length, seed):
    generator = torch.Generator().manual_seed(seed)
    valid_ids = torch.arange(tokenizer.vocab_size)
    if tokenizer.mask_token_id is not None:
      valid_ids = valid_ids[valid_ids.ne(tokenizer.mask_token_id)]
    self.tokens = valid_ids[torch.randint(
      0, len(valid_ids), (size, length), generator=generator)]
    self.tokens[:, 0] = tokenizer.bos_token_id

  def __len__(self):
    return len(self.tokens)

  def __getitem__(self, index):
    tokens = self.tokens[index]
    return {'input_ids': tokens, 'attention_mask': torch.ones_like(tokens)}


def compose_config(args):
  for name, resolver in {
      'cwd': lambda: str(REPO_ROOT),
      'device_count': lambda: args.devices,
      'eval': eval,
      'div_up': lambda x, y: (x + y - 1) // y,
  }.items():
    if not OmegaConf.has_resolver(name):
      OmegaConf.register_new_resolver(name, resolver)
  global_batch = args.micro_batch * args.devices * args.accumulate
  overrides = [
    'algo=mdlm', 'data=openwebtext-split',
    f'model={"tiny" if args.tiny else args.model}',
    f'model.length={args.length}', 'model.attn_backend=sdpa',
    f'block_size={args.length}',
    f'loader.global_batch_size={global_batch}',
    f'loader.eval_global_batch_size={global_batch}',
    f'loader.batch_size={args.micro_batch}',
    f'loader.eval_batch_size={args.micro_batch}', 'loader.num_workers=1',
    f'trainer.devices={args.devices}',
    f'trainer.accumulate_grad_batches={args.accumulate}',
    f'trainer.max_steps={args.steps}',
    f'trainer.precision={args.precision}',
    'data.insert_train_special=false', 'data.insert_valid_special=false',
    'data.insert_valid_eos=false', 'training.from_pretrained=null',
    'training.resample=false', f'training.ema={args.ema}',
    'step_memory.enabled=true', 'step_memory.use_previous_kv=true',
    'step_memory.detach_between_steps=false',
    'step_memory.gate.enabled=true', 'step_memory.gate.init=0.1',
    'step_memory.pretrain.enabled=true',
    'step_memory.pretrain.teacher_token_probability=1.0',
    'step_memory.pretrain.source_dropout.enabled=true',
    f'step_memory.pretrain.source_dropout.warmup_steps={args.source_warmup}',
    'step_memory.pretrain.identity.enabled=true',
    ('step_memory.pretrain.identity.batch_probability=1.0'
     if args.force_identity else
     'step_memory.pretrain.identity.batch_probability=0.25'),
    'dcachehooping.enabled=true', 'dcachehooping.adjacent_grad.enabled=true',
    'dcachehooping.two_forward.enabled=false',
    'dcachehooping.status_embedding.enabled=false',
    'dcachehooping.latent_dropout_probability=0.10',
    'dcachehooping.latent_mask_probability=0.0',
    'dcachehooping.latent_mask_loss_weight=0.0',
    'dcachehooping.tentative.enabled=false',
    'dcachehooping.tentative.batch_probability=0.0',
    'dcachehooping.tentative.loss_weight=0.0',
    'dcachehooping.confidence.enabled=false',
    'dcachehooping.confidence.loss_weight=0.0',
    'dcachehooping.identity_final_probability=0.50',
    'step_memory.rollout.enabled=false', 'wandb=null',
  ]
  if args.tiny:
    overrides.extend([
      'model.hidden_size=32', 'model.cond_dim=16', 'model.n_blocks=2',
      'model.n_heads=4', 'model.dropout=0.0'])
  if args.source_mode != 'mixed':
    overrides.extend([
      'step_memory.pretrain.source_dropout.cache_only_probability='
      + ('1.0' if args.source_mode == 'cache-only' else '0.0'),
      'step_memory.pretrain.source_dropout.current_only_probability='
      + ('1.0' if args.source_mode == 'current-only' else '0.0')])
  with hydra.initialize_config_dir(
      version_base=None, config_dir=str(REPO_ROOT / 'configs')):
    return hydra.compose(config_name='config', overrides=overrides)


class HealthChecks(L.Callback):
  def __init__(self, args):
    self.args = args
    self.optimizer_steps = 0
    self.backward_calls = 0
    self.validation_batches = 0
    self.started = None

  def on_fit_start(self, trainer, pl_module):
    self.started = time.perf_counter()
    if pl_module.device.type == 'cuda':
      torch.cuda.reset_peak_memory_stats(pl_module.device)

  def on_after_backward(self, trainer, pl_module):
    self.backward_calls += 1

  def on_before_optimizer_step(self, trainer, pl_module, optimizer):
    gradients = [
      parameter.grad for parameter in pl_module.parameters()
      if parameter.grad is not None]
    if not gradients or not bool(torch.stack([
        torch.isfinite(gradient).all() for gradient in gradients]).all()):
      raise RuntimeError('Missing or nonfinite optimizer gradients')
    self.optimizer_steps += 1

  def on_validation_batch_end(
      self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
    self.validation_batches += 1

  def on_fit_end(self, trainer, pl_module):
    if trainer.global_step != self.args.steps:
      raise RuntimeError('Trainer stopped before requested optimizer steps')
    if self.optimizer_steps != self.args.steps:
      raise RuntimeError('Optimizer callback count does not match max_steps')
    if self.backward_calls != self.args.steps * self.args.accumulate:
      raise RuntimeError('Gradient accumulation did not execute as requested')
    if self.validation_batches < self.args.val_batches:
      raise RuntimeError('EMA validation did not run')
    metrics = {}
    for key, value in trainer.callback_metrics.items():
      if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
          raise RuntimeError(f'Nonfinite logged metric: {key}')
        if value.numel() == 1:
          metrics[key] = float(value.detach())
    for key in ['trainer/loss', 'val/loss_total', 'val/loss_t2']:
      if key not in metrics:
        raise RuntimeError(f'Required metric missing: {key}')
    if metrics.get('trainer/adjacent_gradient_edges') != 4:
      raise RuntimeError('Training did not register all four adjacent edges')
    if (self.args.steps >= 2 and metrics.get(
        'trainer/adjacent_input_gradient_norm', 0.0) <= 0.0):
      raise RuntimeError('No nonzero adjacent cotangent on the final update')
    is_cuda = pl_module.device.type == 'cuda'
    if is_cuda:
      torch.cuda.synchronize(pl_module.device)
    report = {
      'status': 'passed', 'rank': trainer.global_rank,
      'world_size': trainer.world_size, 'device': str(pl_module.device),
      'gpu_name': (torch.cuda.get_device_name(pl_module.device)
                   if is_cuda else None),
      'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
      'tmpdir': os.environ['TMPDIR'], 'tiny': self.args.tiny,
      'length': self.args.length, 'micro_batch': self.args.micro_batch,
      'optimizer_steps': self.optimizer_steps,
      'backward_calls': self.backward_calls,
      'validation_batches': self.validation_batches,
      'accumulate_grad_batches': self.args.accumulate,
      'precision': self.args.precision,
      'ema_decay': self.args.ema,
      'ema_updates': getattr(pl_module.ema, 'num_updates', None),
      'parameters': sum(p.numel() for p in pl_module.parameters()),
      'elapsed_seconds': time.perf_counter() - self.started,
      'peak_allocated_gib': (
        torch.cuda.max_memory_allocated(pl_module.device) / 1024 ** 3
        if is_cuda else None),
      'peak_reserved_gib': (
        torch.cuda.max_memory_reserved(pl_module.device) / 1024 ** 3
        if is_cuda else None),
      'metrics': metrics,
    }
    atomic_json(self.args.work_dir / f'rank-{trainer.global_rank}.json', report)
    if self.args.save_checkpoint:
      trainer.save_checkpoint(str(self.args.work_dir / 'smoke.ckpt'))
    trainer.strategy.barrier()
    if trainer.is_global_zero:
      reports = [json.loads((
        self.args.work_dir / f'rank-{rank}.json').read_text())
        for rank in range(trainer.world_size)]
      atomic_json(self.args.work_dir / 'RESULT.json', {
        'status': 'passed', 'ranks': reports,
        'note': 'Synthetic smoke only; not a training or validation benchmark.'})
      print('ADJACENT_GRADIENT_PROFILE_OK', flush=True)
      print(json.dumps(reports, indent=2, sort_keys=True), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--devices', type=int, choices=[1, 2], default=2)
  parser.add_argument('--accelerator', choices=['auto', 'cpu', 'gpu'],
                      default='auto')
  parser.add_argument('--micro-batch', type=int, default=2)
  parser.add_argument('--steps', type=int, default=3)
  parser.add_argument('--accumulate', type=int, default=2)
  parser.add_argument('--val-batches', type=int, default=2)
  parser.add_argument('--length', type=int)
  parser.add_argument('--model', default='small')
  parser.add_argument('--tiny', action='store_true')
  parser.add_argument('--ema', type=float, default=0.9999)
  parser.add_argument('--force-identity', action='store_true')
  parser.add_argument('--source-mode', choices=[
    'mixed', 'joint', 'cache-only', 'current-only'], default='mixed')
  parser.add_argument('--source-warmup', type=int, default=1000)
  parser.add_argument('--save-checkpoint', action='store_true')
  parser.add_argument('--toy-ddp-parity', action='store_true',
                      help='Run independent CPU-only 2-rank gradient parity')
  parser.add_argument('--work-dir', required=True, type=pathlib.Path)
  args = parser.parse_args()
  if min(args.micro_batch, args.steps, args.accumulate,
         args.val_batches, args.source_warmup) < 1:
    parser.error('Batch, update, accumulation, validation and warmup counts > 0')
  if args.steps > 10:
    parser.error('This bounded smoke harness permits at most 10 updates')
  if args.force_identity and args.micro_batch < 2:
    parser.error('--force-identity requires --micro-batch >= 2')
  args.length = args.length or (32 if args.tiny else 1024)
  if args.length < 8:
    parser.error('The five-state trajectory needs length >= 8')
  args.work_dir = args.work_dir.resolve()
  args.work_dir.mkdir(parents=True, exist_ok=True)
  if (args.work_dir / 'RESULT.json').exists():
    parser.error('Choose a new work-dir; a completed smoke result exists here')
  if args.toy_ddp_parity:
    torch.multiprocessing.spawn(
      toy_parity_worker, args=(str(args.work_dir),), nprocs=2, join=True)
    reports = [json.loads((
      args.work_dir / f'parity-rank-{rank}.json').read_text())
      for rank in range(2)]
    atomic_json(args.work_dir / 'RESULT.json', {
      'status': 'passed', 'test': 'toy-ddp-gradient-parity', 'ranks': reports})
    print('ADJACENT_TOY_DDP_PARITY_OK', flush=True)
    print(json.dumps(reports, indent=2, sort_keys=True), flush=True)
    return
  accelerator = args.accelerator
  if accelerator == 'auto':
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
  if accelerator == 'cpu' and not args.tiny:
    parser.error('Real-size profile requires GPU; use --tiny for CPU checks')
  if accelerator == 'gpu' and torch.cuda.device_count() < args.devices:
    parser.error('Fewer visible CUDA devices than --devices')
  args.precision = 'bf16-mixed' if accelerator == 'gpu' else '32-true'
  torch.set_float32_matmul_precision('high')
  torch.multiprocessing.set_start_method('spawn', force=True)
  L.seed_everything(37, workers=True)
  config = compose_config(args)
  tokenizer = (dataloader.Text8Tokenizer() if args.tiny else
               dataloader.get_tokenizer(config))
  model = diffusion.Diffusion(config, tokenizer=tokenizer)
  # Each rank receives this many batches after DistributedSampler sharding.
  train_batches = args.steps * args.accumulate
  train_loader = DataLoader(SyntheticTokens(
    tokenizer, train_batches * args.devices * args.micro_batch,
    args.length, seed=37), batch_size=args.micro_batch, num_workers=0)
  valid_loader = DataLoader(SyntheticTokens(
    tokenizer, args.val_batches * args.devices * args.micro_batch,
    args.length, seed=41), batch_size=args.micro_batch, num_workers=0)
  strategy = (DDPStrategy(find_unused_parameters=False,
                          process_group_backend=(
                            'nccl' if accelerator == 'gpu' else 'gloo'))
              if args.devices > 1 else 'auto')
  trainer = L.Trainer(
    accelerator=accelerator, devices=args.devices, strategy=strategy,
    precision=args.precision, max_steps=args.steps,
    accumulate_grad_batches=args.accumulate, gradient_clip_val=1.0,
    limit_train_batches=train_batches, limit_val_batches=args.val_batches,
    val_check_interval=train_batches, num_sanity_val_steps=0,
    log_every_n_steps=1, logger=False, enable_checkpointing=False,
    enable_progress_bar=False, enable_model_summary=False,
    default_root_dir=str(args.work_dir), callbacks=[HealthChecks(args)])
  if int(os.environ.get('LOCAL_RANK', '0')) == 0:
    atomic_json(args.work_dir / 'config.json',
                OmegaConf.to_container(config, resolve=True))
  trainer.fit(model, train_loader, valid_loader)


if __name__ == '__main__':
  main()
