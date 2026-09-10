#!/usr/bin/env python3
"""Resumable, paired diagnostics of denoising recurrence using frozen EMA weights.

All mask trajectories are deterministic per document.  ``generated`` starts
with teacher-provided context, then reveals only model-sampled tokens; its NLL
is remaining-mask reconstruction NLL, not free-generation perplexity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
import pathlib
import signal
import sys
import time
from typing import Any, Optional

import pandas as pd
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.eval import dcache_eval_common as common  # noqa: E402


VARIANTS = ('five-forward', 'two-forward', 'dcache-v2', 'objective', 'vanilla')
DUAL_VARIANTS = ('five-forward', 'two-forward')
FAMILIES = ('transitions', 'repeat', 'generated')
GROUP_COLUMNS = [
  'family', 'variant', 'seed', 'protocol', 's_mask_ratio', 't_mask_ratio',
  'repeat_step', 'condition',
]
RAW_COLUMNS = GROUP_COLUMNS + [
  'example_id', 'masked_tokens', 'nll', 'top1_accuracy', 'top5_accuracy',
]
PAUSED_EXIT_CODE = 75


@dataclass(frozen=True)
class Memory:
  dcache: Any = None
  final: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class Phase:
  family: str
  protocol: str
  s_ratio: float
  t_ratio: float

  @property
  def name(self) -> str:
    s_tag = format(self.s_ratio, '.10f').rstrip('0').rstrip('.')
    t_tag = format(self.t_ratio, '.10f').rstrip('0').rstrip('.')
    return f'{self.family}_{self.protocol}_s{s_tag}_t{t_tag}'


def ratio_key(value: float) -> float:
  return round(float(value), 10)


def condition_names(variant: str, family: str) -> tuple[str, ...]:
  if variant not in VARIANTS or family not in FAMILIES:
    raise ValueError(f'Unknown variant/family: {variant}/{family}')
  if variant in ('vanilla', 'objective'):
    return ('absent_both',)
  if variant == 'dcache-v2':
    return ('correct', 'absent_both', 'shuffle_dcache')
  if family == 'transitions':
    return (
      'correct', 'absent_dcache', 'absent_final', 'absent_both',
      'shuffle_dcache', 'shuffle_final', 'shuffle_both_coherent',
      'shuffle_both_independent',
    )
  return (
    'correct', 'absent_dcache', 'absent_final', 'absent_both',
    'shuffle_both_coherent',
  )


def intervene(memory: Memory, condition: str) -> Memory:
  """Apply a read intervention, without modifying the stored source memory.

  The independent double shuffle uses *different* wrong donors: D rolls by
  one document and F rolls by two.  Coherent shuffle gives both sources from
  the same wrong document.  Repeated shuffling is a whole-trajectory stress
  test: cyclic rolls can eventually carry a document's information back.
  """
  if condition == 'correct':
    return memory
  if condition == 'absent_both':
    return Memory()
  if condition == 'absent_dcache':
    return Memory(None, memory.final)
  if condition == 'absent_final':
    return Memory(memory.dcache, None)
  if condition not in (
      'shuffle_dcache', 'shuffle_final', 'shuffle_both_coherent',
      'shuffle_both_independent'):
    raise ValueError(f'Unknown condition: {condition}')
  dcache, final = memory.dcache, memory.final
  if condition in (
      'shuffle_dcache', 'shuffle_both_coherent', 'shuffle_both_independent'):
    if dcache is not None:
      if dcache[0].shape[0] < 3:
        raise ValueError('Shuffling requires batches of at least three documents')
      dcache = common.roll_cache_batch(dcache, shift=1)
  if condition in (
      'shuffle_final', 'shuffle_both_coherent', 'shuffle_both_independent'):
    if final is not None:
      if final.shape[0] < 3:
        raise ValueError('Shuffling requires batches of at least three documents')
      shift = 2 if condition == 'shuffle_both_independent' else 1
      final = final.roll(shifts=shift, dims=0)
  return Memory(dcache, final)


def forward_with_memory(model, variant, state, sigma, memory=Memory(),
                        return_memory=True):
  """Normalize the three existing forward-output interfaces.

  A missing final latent uses the checkpoint's own trained initialization:
  two-forward maps None to a zero tensor *before* its affine LayerNorm;
  five-forward skips the latent addition.  We do not zero the LayerNorm bias.
  """
  recurrent = variant not in ('vanilla', 'objective')
  dual = variant in DUAL_VARIANTS
  kwargs = dict(sigma=sigma, sample_mode=True)
  if recurrent:
    kwargs.update(previous_step_kv=memory.dcache,
                  return_step_kv=bool(return_memory))
  if dual:
    kwargs.update(previous_final_hidden=memory.final,
                  return_dcachehooping=bool(return_memory))
  output = model.forward(state, **kwargs)
  if return_memory and dual:
    return output.scores, Memory(output.step_kv, output.final_hidden)
  if return_memory and recurrent:
    scores, dcache = output
    return scores, Memory(dcache, None)
  return output, Memory()


def deterministic_seed(seed: int, example_id: int, stream: str) -> int:
  payload = f'{int(seed)}:{int(example_id)}:{stream}'.encode('utf-8')
  return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') % (2**63 - 1)


def reveal_from_model(state, log_probs, previous_mask, next_mask,
                      example_ids, seed, stream, mask_index, temperature=1.0):
  """Reveal only newly selected positions, without accepting ground truth.

  A separate CPU RNG per document/stage couples the random draws across
  conditions and variants, and reproduces them exactly when resuming.  Mask
  selection has its own independent seed stream (the common mask helper).
  """
  if not math.isfinite(temperature) or temperature <= 0:
    raise ValueError('Sampling temperature must be finite and positive')
  previous_mask = previous_mask.to(state.device).bool()
  next_mask = next_mask.to(state.device).bool()
  if not torch.all(next_mask <= previous_mask):
    raise ValueError('The next mask must be nested inside the previous mask')
  if len(example_ids) != state.shape[0]:
    raise ValueError('Example IDs must match the batch size')
  reveal = previous_mask & ~next_mask
  if not torch.all(reveal.sum(dim=1) > 0):
    raise ValueError('Every reveal step must reveal at least one token')
  result = state.clone()
  for row, example_id in enumerate(example_ids):
    selected = reveal[row].nonzero(as_tuple=False).flatten()
    values = log_probs[row, selected].detach().float().cpu() / temperature
    values[:, int(mask_index)] = -float('inf')
    probabilities = values.softmax(dim=-1)
    if not torch.isfinite(probabilities).all():
      raise ValueError('Nonfinite model token probabilities during rollout')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(deterministic_seed(seed, example_id, stream))
    sampled = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
    result[row, selected] = sampled.to(state.device)
  return result


def fields_for(variant, phase, seed, condition, repeat_step=0, t_ratio=None):
  return {
    'family': phase.family, 'variant': variant, 'seed': int(seed),
    'protocol': phase.protocol, 's_mask_ratio': float(phase.s_ratio),
    't_mask_ratio': float(phase.t_ratio if t_ratio is None else t_ratio),
    'repeat_step': int(repeat_step), 'condition': condition,
  }


def evaluate_transition(model, variant, tokens, attention, example_ids,
                        phase, seed):
  masks = common.deterministic_nested_masks(
    attention, example_ids, [phase.s_ratio, phase.t_ratio], seed)
  s_mask = masks[phase.s_ratio].to(tokens.device)
  t_mask = masks[phase.t_ratio].to(tokens.device)
  if not torch.all(t_mask <= s_mask):
    raise RuntimeError('Non-nested transition masks')
  if not torch.all((s_mask & ~t_mask).sum(dim=1) > 0):
    raise ValueError('Transition ratios must reveal at least one token')
  t_state = common.masked_state(tokens, t_mask, model.mask_index)
  t_sigma = common.sigma_for_ratio(
    model, phase.t_ratio, len(example_ids), tokens.device)
  if variant in ('vanilla', 'objective'):
    scores, _ = forward_with_memory(
      model, variant, t_state, t_sigma, return_memory=False)
    return common.score_masked_tokens(
      scores, tokens, t_mask, example_ids,
      **fields_for(variant, phase, seed, 'absent_both'))

  memory = Memory()
  if phase.protocol == 'full_warmup':
    eligible = common.eligible_positions(attention).to(tokens.device)
    full_state = common.masked_state(tokens, eligible, model.mask_index)
    full_sigma = common.sigma_for_ratio(
      model, 1.0, len(example_ids), tokens.device)
    scores, memory = forward_with_memory(model, variant, full_state, full_sigma)
    del scores, full_state, full_sigma
  elif phase.protocol != 'no_warmup':
    raise ValueError(f'Unknown transition protocol: {phase.protocol}')
  s_state = common.masked_state(tokens, s_mask, model.mask_index)
  s_sigma = common.sigma_for_ratio(
    model, phase.s_ratio, len(example_ids), tokens.device)
  scores, source = forward_with_memory(model, variant, s_state, s_sigma, memory)
  del scores, memory, s_state, s_sigma
  rows = []
  for condition in condition_names(variant, phase.family):
    read_memory = intervene(source, condition)
    scores, _ = forward_with_memory(
      model, variant, t_state, t_sigma, read_memory, return_memory=False)
    rows.extend(common.score_masked_tokens(
      scores, tokens, t_mask, example_ids,
      **fields_for(variant, phase, seed, condition)))
    del scores, read_memory
  return rows


def evaluate_repeat(model, variant, tokens, attention, example_ids,
                    phase, seed, repeat_steps):
  mask = common.deterministic_nested_masks(
    attention, example_ids, [phase.t_ratio], seed)[phase.t_ratio].to(tokens.device)
  state = common.masked_state(tokens, mask, model.mask_index)
  sigma = common.sigma_for_ratio(
    model, phase.t_ratio, len(example_ids), tokens.device)
  scores, first_memory = forward_with_memory(model, variant, state, sigma)
  first_rows = common.score_masked_tokens(scores, tokens, mask, example_ids)
  del scores
  rows = []
  steps = set(repeat_steps)
  for condition in condition_names(variant, phase.family):
    # Each arm starts from the same immutable first output.  Subsequent
    # memories belong only to that arm, including the shuffled trajectory.
    memory = first_memory
    if 1 in steps:
      rows.extend(dict(row, **fields_for(
        variant, phase, seed, condition, repeat_step=1)) for row in first_rows)
    for step in range(2, max(steps) + 1):
      if variant in ('vanilla', 'objective'):
        if step in steps:
          rows.extend(dict(row, **fields_for(
            variant, phase, seed, condition, repeat_step=step))
            for row in first_rows)
        continue
      read_memory = intervene(memory, condition)
      scores, memory = forward_with_memory(model, variant, state, sigma, read_memory)
      if step in steps:
        rows.extend(common.score_masked_tokens(
          scores, tokens, mask, example_ids,
          **fields_for(variant, phase, seed, condition, repeat_step=step)))
      del scores, read_memory
  return rows


def generated_ratios(phase: Phase, generated_steps: int) -> list[float]:
  if generated_steps < 1 or phase.s_ratio <= phase.t_ratio:
    raise ValueError('Generated rollout requires positive steps and s > t')
  return [ratio_key(phase.s_ratio + (phase.t_ratio - phase.s_ratio) * index
                    / generated_steps) for index in range(generated_steps + 1)]


def evaluate_generated(model, variant, tokens, attention, example_ids,
                       phase, seed, generated_steps, temperature):
  ratios = generated_ratios(phase, generated_steps)
  masks = {ratio: mask.to(tokens.device) for ratio, mask in
           common.deterministic_nested_masks(
             attention, example_ids, ratios, seed).items()}
  initial = common.masked_state(tokens, masks[ratios[0]], model.mask_index)
  sigma = common.sigma_for_ratio(
    model, ratios[0], len(example_ids), tokens.device)
  scores, first_memory = forward_with_memory(model, variant, initial, sigma)
  stream_base = f'generated:s{phase.s_ratio:.10f}:t{phase.t_ratio:.10f}'
  # The first no-memory prediction and its sampled reveal are identical
  # across conditions, so share them and release the vocabulary tensor now.
  first_state = reveal_from_model(
    initial, scores, masks[ratios[0]], masks[ratios[1]], example_ids,
    seed, f'{stream_base}:reveal1', model.mask_index, temperature)
  del scores, initial, sigma
  rows = []
  for condition in condition_names(variant, phase.family):
    state, memory = first_state.clone(), first_memory
    for step in range(1, generated_steps + 1):
      ratio = ratios[step]
      sigma = common.sigma_for_ratio(model, ratio, len(example_ids), tokens.device)
      read_memory = intervene(memory, condition)
      scores, memory = forward_with_memory(model, variant, state, sigma, read_memory)
      rows.extend(common.score_masked_tokens(
        scores, tokens, masks[ratio], example_ids,
        **fields_for(variant, phase, seed, condition,
                     repeat_step=step, t_ratio=ratio)))
      if step < generated_steps:
        state = reveal_from_model(
          state, scores, masks[ratio], masks[ratios[step + 1]], example_ids,
          seed, f'{stream_base}:reveal{step + 1}', model.mask_index, temperature)
      del scores, sigma, read_memory
  return rows


def make_phases(args) -> list[Phase]:
  phases = []
  for family in args.families:
    for t_ratio in args.ratios:
      if family == 'transitions':
        for jump in args.jumps:
          s_ratio = ratio_key(t_ratio + jump)
          if s_ratio > 1:
            continue
          for protocol in ('no_warmup', 'full_warmup'):
            phases.append(Phase(family, protocol, s_ratio, t_ratio))
      elif family == 'repeat':
        phases.append(Phase(family, 'same_state', t_ratio, t_ratio))
      elif family == 'generated':
        s_ratio = ratio_key(t_ratio + args.generated_jump)
        if s_ratio <= 1:
          phases.append(Phase(family, 'generated', s_ratio, t_ratio))
      else:
        raise ValueError(f'Unknown family: {family}')
  if not phases:
    raise ValueError('Requested grid produces no valid phases')
  return phases


def expected_part_rows(args, phase, batch_size=None):
  steps = (len(args.repeat_steps) if phase.family == 'repeat'
           else args.generated_steps if phase.family == 'generated' else 1)
  return ((args.batch_size if batch_size is None else batch_size)
          * len(condition_names(args.variant, phase.family)) * steps)


def validate_part(frame, args, phase, batch_index):
  if set(frame.columns) != set(RAW_COLUMNS):
    raise RuntimeError(f'Unexpected result schema for {phase.name}/{batch_index}')
  if len(frame) != expected_part_rows(args, phase):
    raise RuntimeError(f'Incomplete result part for {phase.name}/{batch_index}')
  if frame.duplicated(GROUP_COLUMNS + ['example_id']).any():
    raise RuntimeError(f'Duplicate result rows for {phase.name}/{batch_index}')
  expected_ids = set(common.batch_ids(batch_index, args.batch_size, args.batch_size))
  if set(frame.example_id) != expected_ids:
    raise RuntimeError(f'Wrong example IDs for {phase.name}/{batch_index}')
  expected_conditions = set(condition_names(args.variant, phase.family))
  if set(frame.condition) != expected_conditions:
    raise RuntimeError(f'Wrong conditions for {phase.name}/{batch_index}')
  for name, value in [('family', phase.family), ('protocol', phase.protocol),
                      ('variant', args.variant), ('seed', args.seed),
                      ('s_mask_ratio', phase.s_ratio)]:
    column = frame[name].map(ratio_key) if name == 's_mask_ratio' else frame[name]
    if not column.eq(value).all():
      raise RuntimeError(f'Wrong {name} for {phase.name}/{batch_index}')
  numeric = frame[['masked_tokens', 'nll', 'top1_accuracy', 'top5_accuracy']]
  if not all(math.isfinite(float(value)) for value in numeric.to_numpy().flat):
    raise RuntimeError(f'Nonfinite metric in {phase.name}/{batch_index}')
  if not frame.masked_tokens.gt(0).all() or not frame.nll.ge(-1e-5).all():
    raise RuntimeError(f'Invalid token count or NLL in {phase.name}/{batch_index}')
  if not (frame.top1_accuracy.between(0, 1).all()
          and frame.top5_accuracy.between(0, 1).all()):
    raise RuntimeError(f'Invalid accuracy in {phase.name}/{batch_index}')
  if phase.family == 'transitions':
    expected_pairs = {(0, phase.t_ratio)}
  elif phase.family == 'repeat':
    expected_pairs = {(step, phase.t_ratio) for step in args.repeat_steps}
  else:
    expected_pairs = set(enumerate(generated_ratios(phase, args.generated_steps)[1:], 1))
  actual_pairs = {(int(row.repeat_step), ratio_key(row.t_mask_ratio))
                  for row in frame.itertuples()}
  if actual_pairs != expected_pairs:
    raise RuntimeError(f'Wrong step/ratio grid for {phase.name}/{batch_index}')
  for _, group in frame.groupby(['condition', 'repeat_step', 't_mask_ratio']):
    if set(group.example_id) != expected_ids:
      raise RuntimeError(f'Incomplete paired documents in {phase.name}/{batch_index}')


def file_digest(path):
  digest = hashlib.sha256()
  with pathlib.Path(path).open('rb') as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def source_fingerprints():
  paths = [
    pathlib.Path(__file__).resolve(), REPO_ROOT / 'scripts/eval/dcache_eval_common.py',
    REPO_ROOT / 'diffusion.py', REPO_ROOT / 'models/dit.py',
    REPO_ROOT / 'dataloader.py', REPO_ROOT / 'noise_schedule.py',
    REPO_ROOT / 'rollout_utils.py', REPO_ROOT / 'utils.py',
    REPO_ROOT / 'metrics.py',
  ]
  paths.extend(sorted((REPO_ROOT / 'models').rglob('*.py')))
  paths.extend(sorted((REPO_ROOT / 'configs').rglob('*.yaml')))
  return {str(path.relative_to(REPO_ROOT)): file_digest(path)
          for path in paths if path.is_file()}


def prepared_data_fingerprint(data_dir):
  root = data_dir / 'openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat'
  files = {}
  for path in sorted(root.rglob('*')):
    if not path.is_file():
      continue
    stat = path.stat()
    entry = {'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if path.suffix == '.json':
      entry['sha256'] = file_digest(path)
    files[str(path.relative_to(root))] = entry
  if not files:
    raise ValueError(f'Prepared validation data directory is empty: {root}')
  return {'path': str(root), 'files': files,
          'content_policy': 'JSON SHA256; Arrow size/mtime (not a full dataset hash)'}


def checkpoint_metadata(path, expected_step):
  before = common.checkpoint_fingerprint(path)
  # mmap reads checkpoint metadata without materializing another parameter
  # copy.  The common loader below still performs the strict Lightning/EMA
  # load, including the repository's migration hooks.
  checkpoint = torch.load(str(path), map_location='cpu', weights_only=False, mmap=True)
  step = int(checkpoint.get('global_step', -1))
  if step != expected_step:
    raise ValueError(f'Expected checkpoint global_step={expected_step}, found {step}: {path}')
  if 'ema' not in checkpoint:
    raise ValueError(f'Checkpoint contains no EMA state: {path}')
  del checkpoint
  digest = file_digest(path)
  after = common.checkpoint_fingerprint(path)
  if before != after:
    raise RuntimeError('Checkpoint changed while being inspected; retry after it is stable')
  return {**after, 'sha256': digest, 'global_step': step}


def make_metadata(args, phases):
  return {
    'evaluation': 'recurrence_audit', 'schema_version': 1,
    'variant': args.variant, 'weights': 'EMA', 'expected_step': args.expected_step,
    'checkpoint': checkpoint_metadata(args.checkpoint, args.expected_step),
    'dataset': 'openwebtext-split', 'split': 'validation',
    'data': prepared_data_fingerprint(args.data_dir), 'length': 1024,
    'examples': args.examples, 'batch_size': args.batch_size, 'seed': args.seed,
    'families': args.families, 'ratios': args.ratios, 'jumps': args.jumps,
    'repeat_steps': args.repeat_steps, 'generated_steps': args.generated_steps,
    'generated_jump': args.generated_jump, 'temperature': args.temperature,
    'phases': [phase.__dict__ for phase in phases],
    'conditions': {family: condition_names(args.variant, family) for family in args.families},
    'mask_protocol': 'Exact-count nested masks, seed+document ID; exclude first token; same x_t across all models, jumps, arms and warmup protocols.',
    'protocols': {
      'no_warmup': 'x_s forward with absent memories -> identical teacher-forced x_t under each memory read intervention.',
      'full_warmup': 'All eligible tokens masked -> x_s with both correct available memories -> identical x_t under each intervention.',
      'same_state': 'Tokens, mask and sigma fixed; each arm maintains its own recurrent history after common first no-memory pass.',
      'generated': 'Teacher context only in initial x_s; fixed random nested reveals use model-sampled tokens at temperature; each intervention has its own generated tokens and memory history. Score remaining masked targets; this is NOT free-generation perplexity.',
    },
    'baseline_protocol_note': 'Vanilla/objective have no recurrent sources. Their redundant source/warmup forwards are omitted, so full_warmup and no_warmup predict the same x_t. Same-state later outputs equal their first output.',
    'repeat_initialization': 'Step 1 has no memory for all arms and is computed once; its equal metrics are copied to every arm. Later steps have separate per-arm memory histories.',
    'absent_final': 'None: native checkpoint initialization; two-forward inserts zero before affine latent LayerNorm (learned bias retained), five-forward omits latent addition.',
    'shuffles': {
      'shuffle_dcache': 'D donor cyclic roll 1, F correct if present',
      'shuffle_final': 'D correct if present, F donor cyclic roll 1',
      'shuffle_both_coherent': 'D and F from same wrong donor (roll 1)',
      'shuffle_both_independent': 'D from roll 1, F from distinct wrong donor roll 2; distinct donors, NOT probabilistically independent random permutations',
      'trajectory_caveat': 'Repeated cyclic shuffling is a contaminated-history stress test; information can return through donor cycles. It is not an independent per-round causal effect.',
    },
    'sampling_rng': 'CPU categorical draws, SHA256(seed, document ID, generated initial/final ratio and reveal index); excludes variant/condition for common random numbers; independent of mask RNG.',
    'source_sha256': source_fingerprints(),
    'torch_version': str(torch.__version__),
  }


def utc_now():
  return datetime.now(timezone.utc).isoformat()


def prepare_output(output_dir, metadata):
  """Allow resume only for an identical scientific configuration; never delete."""
  output_dir.mkdir(parents=True, exist_ok=True)
  encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()
  fingerprint = hashlib.sha256(encoded).hexdigest()
  manifest_path = output_dir / 'manifest.json'
  if manifest_path.exists():
    existing = json.loads(manifest_path.read_text())
    if existing.get('fingerprint') != fingerprint:
      raise RuntimeError(
        f'Evaluation fingerprint changed at {manifest_path}. Use a new output '
        'directory; existing results have been preserved.')
  else:
    unexpected = [path for path in output_dir.iterdir() if path.name != '.audit.lock']
    if unexpected:
      raise RuntimeError(f'Refusing to adopt a nonempty output directory without a manifest: {output_dir}')
    common.atomic_write_json(manifest_path, {
      **metadata, 'fingerprint': fingerprint, 'created_at': utc_now(),
      'git_revision': common.git_revision(),
    })
  (output_dir / 'parts').mkdir(exist_ok=True)
  return fingerprint


def write_reports(args, phases, output_dir):
  frames = []
  complete_parts = 0
  for phase in phases:
    for batch_index in range(args.examples // args.batch_size):
      path = common.part_path(output_dir, phase.name, batch_index)
      if not path.is_file():
        continue
      frame = pd.read_csv(path)
      validate_part(frame, args, phase, batch_index)
      frames.append(frame)
      complete_parts += 1
  expected_parts = len(phases) * (args.examples // args.batch_size)
  complete = complete_parts == expected_parts
  if frames:
    raw = pd.concat(frames, ignore_index=True).sort_values(GROUP_COLUMNS + ['example_id'])
    if raw.duplicated(GROUP_COLUMNS + ['example_id']).any():
      raise RuntimeError('Duplicate rows across completed phase parts')
    summary = common.aggregate_metrics(raw, GROUP_COLUMNS)
    common.atomic_write_csv(output_dir / 'summary.csv', summary)
    if complete:
      expected_rows = sum(expected_part_rows(args, phase) for phase in phases) * (args.examples // args.batch_size)
      if len(raw) != expected_rows:
        raise RuntimeError(f'Expected {expected_rows} total rows, found {len(raw)}')
      common.atomic_write_csv(output_dir / 'per_document_metrics.csv', raw)
  return complete, complete_parts, expected_parts


class CooperativeStop:
  def __init__(self, max_hours=None):
    self.started = time.monotonic()
    self.deadline = None if max_hours is None else self.started + max_hours * 3600
    self.signal_number = None

  def request(self, number, _frame):
    self.signal_number = number

  @property
  def reason(self):
    if self.signal_number is not None:
      return f'signal_{self.signal_number}'
    if self.deadline is not None and time.monotonic() >= self.deadline:
      return 'max_hours'
    return None


def run_evaluation(args, phases, output_dir, stop, fingerprint):
  expected_parts = len(phases) * (args.examples // args.batch_size)
  done = set()
  for phase in phases:
    for batch_index in range(args.examples // args.batch_size):
      path = common.part_path(output_dir, phase.name, batch_index)
      if path.is_file():
        validate_part(pd.read_csv(path), args, phase, batch_index)
        done.add((phase.name, batch_index))

  def status(state, **extra):
    elapsed = time.monotonic() - stop.started
    common.atomic_write_json(output_dir / 'STATUS.json', {
      'status': state, 'updated_at': utc_now(), 'variant': args.variant,
      'seed': args.seed, 'fingerprint': fingerprint,
      'completed_parts': len(done), 'expected_parts': expected_parts,
      'elapsed_seconds_this_invocation': elapsed,
      'device': args.device, 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
      'peak_allocated_gib': (torch.cuda.max_memory_allocated(torch.device(args.device)) / 2**30
                             if str(args.device).startswith('cuda') else 0.0),
      **extra,
    })

  if len(done) == expected_parts:
    status('finalizing')
    return status
  if stop.reason:
    status('paused', reason=stop.reason)
    return status
  status('loading')
  recurrent = args.variant not in ('vanilla', 'objective')
  config = common.compose_eval_config(
    args.batch_size, args.data_dir, recurrent=recurrent,
    num_workers=args.num_workers, gate_enabled=recurrent,
    final_state_enabled=args.variant in DUAL_VARIANTS,
    two_forward_enabled=args.variant == 'two-forward')
  tokenizer = common.load_tokenizer(config)
  loader = common.load_validation_data(
    config, tokenizer, args.examples, args.batch_size, args.num_workers)
  device = torch.device(args.device)
  common.print_device_banner(device)
  model = common.load_ema_model(args.checkpoint, config, tokenizer, device)
  initial_done = len(done)
  print(f'Loaded {args.variant} EMA, step {args.expected_step}; '
        f'{expected_parts - initial_done}/{expected_parts} phase/batch parts remain.', flush=True)
  eval_start = time.monotonic()
  try:
    with torch.inference_mode():
      for batch_index, batch in enumerate(loader):
        if stop.reason:
          break
        if all((phase.name, batch_index) in done for phase in phases):
          continue
        tokens = batch['input_ids'].long().to(device, non_blocking=True)
        attention = batch['attention_mask']
        ids = common.batch_ids(batch_index, args.batch_size, tokens.shape[0])
        for phase in phases:
          if (phase.name, batch_index) in done:
            continue
          if stop.reason:
            break
          if phase.family == 'transitions':
            rows = evaluate_transition(model, args.variant, tokens, attention, ids, phase, args.seed)
          elif phase.family == 'repeat':
            rows = evaluate_repeat(model, args.variant, tokens, attention, ids, phase, args.seed, args.repeat_steps)
          else:
            rows = evaluate_generated(model, args.variant, tokens, attention, ids, phase, args.seed, args.generated_steps, args.temperature)
          frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
          validate_part(frame, args, phase, batch_index)
          common.atomic_write_csv(common.part_path(output_dir, phase.name, batch_index), frame)
          done.add((phase.name, batch_index))
          count = len(done) - initial_done
          seconds_per_part = (time.monotonic() - eval_start) / count
          eta_hours = seconds_per_part * (expected_parts - len(done)) / 3600
          status('running', last_phase=phase.name, last_batch=batch_index,
                 estimated_remaining_hours=eta_hours)
          if count % 10 == 0 or len(done) == expected_parts:
            print(f'[{args.variant} seed={args.seed}] parts {len(done)}/{expected_parts}, '
                  f'batch {batch_index + 1}/{len(loader)}, {phase.name}; '
                  f'elapsed={(time.monotonic() - stop.started)/3600:.2f}h, '
                  f'rough ETA={eta_hours:.2f}h', flush=True)
        del tokens
  finally:
    del model
    if device.type == 'cuda':
      torch.cuda.empty_cache()
  status('finalizing' if len(done) == expected_parts else 'paused', reason=stop.reason)
  return status


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--variant', choices=VARIANTS, required=True)
  parser.add_argument('--checkpoint', type=pathlib.Path, required=True)
  parser.add_argument('--expected-step', type=int, default=5000)
  parser.add_argument('--data-dir', type=pathlib.Path, required=True)
  parser.add_argument('--output-dir', type=pathlib.Path, required=True)
  parser.add_argument('--examples', type=int, default=800)
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--seed', type=int, default=20260812)
  parser.add_argument('--num-workers', type=int, default=2)
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--families', choices=FAMILIES, nargs='+', default=list(FAMILIES))
  parser.add_argument('--ratios', type=float, nargs='+', default=[.05, .10, .20, .30, .50, .70])
  parser.add_argument('--jumps', type=float, nargs='+', default=[.025, .05, .10, .20])
  parser.add_argument('--repeat-steps', type=int, nargs='+', default=[1, 2, 4, 8])
  parser.add_argument('--generated-jump', type=float, default=.10)
  parser.add_argument('--generated-steps', type=int, default=4)
  parser.add_argument('--temperature', type=float, default=1.0)
  parser.add_argument('--max-hours', type=float,
                      help='Cooperative wall-clock limit, stopping after an atomic phase/batch; exit 75 means safely paused.')
  return parser.parse_args(argv)


def validate_args(args):
  if args.examples <= 0 or args.batch_size < 3 or args.examples % args.batch_size:
    raise ValueError('Use examples > 0 divisible by batch-size >= 3 (distinct donor controls)')
  if args.num_workers < 0 or args.expected_step < 0:
    raise ValueError('Worker count and expected step must be nonnegative')
  args.ratios = sorted(set(ratio_key(value) for value in args.ratios))
  args.jumps = sorted(set(ratio_key(value) for value in args.jumps))
  args.repeat_steps = sorted(set(args.repeat_steps))
  args.families = list(dict.fromkeys(args.families))
  if not args.ratios or not all(math.isfinite(value) and 0 < value < 1 for value in args.ratios):
    raise ValueError('Target mask ratios must be strictly between zero and one')
  if not args.jumps or not all(math.isfinite(value) and 0 < value <= 1 for value in args.jumps):
    raise ValueError('Transition jumps must be in (0, 1]')
  if not args.repeat_steps or min(args.repeat_steps) < 1:
    raise ValueError('Repeat steps must be positive')
  if args.generated_steps < 1 or not 0 < args.generated_jump <= 1:
    raise ValueError('Invalid generated step count or jump')
  if not math.isfinite(args.temperature) or args.temperature <= 0:
    raise ValueError('Temperature must be finite and positive')
  if args.max_hours is not None and (not math.isfinite(args.max_hours) or args.max_hours <= 0):
    raise ValueError('--max-hours must be finite and positive')
  args.generated_jump = ratio_key(args.generated_jump)
  for field in ('checkpoint', 'data_dir', 'output_dir'):
    setattr(args, field, getattr(args, field).expanduser().resolve())
  if not args.checkpoint.is_file():
    raise FileNotFoundError(args.checkpoint)
  prepared = args.data_dir / 'openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat'
  if not prepared.is_dir():
    raise FileNotFoundError(prepared)
  if str(args.device).startswith('cuda') and not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested but is unavailable')


def main(argv=None):
  args = parse_args(argv)
  validate_args(args)
  stop = CooperativeStop(args.max_hours)
  for number in (signal.SIGINT, signal.SIGTERM):
    signal.signal(number, stop.request)
  torch.set_float32_matmul_precision('high')
  phases = make_phases(args)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  with (args.output_dir / '.audit.lock').open('a') as lock:
    try:
      fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
      raise RuntimeError(f'Another evaluator owns {args.output_dir}') from error
    metadata = make_metadata(args, phases)
    fingerprint = prepare_output(args.output_dir, metadata)
    try:
      update_status = run_evaluation(args, phases, args.output_dir, stop, fingerprint)
      complete, done, expected = write_reports(args, phases, args.output_dir)
      if complete:
        common.atomic_write_json(args.output_dir / 'completion.json', {
          'status': 'complete', 'completed_at': utc_now(), 'fingerprint': fingerprint,
          'completed_parts': done, 'expected_parts': expected,
          'examples_per_condition': args.examples,
        })
        update_status('complete')
        print(f'Complete: {args.output_dir}', flush=True)
        return 0
      update_status('paused', reason=stop.reason)
      print(f'Safely paused ({done}/{expected} parts): {args.output_dir}. '
            'Run the same command again to resume.', flush=True)
      return PAUSED_EXIT_CODE
    except Exception as error:
      common.atomic_write_json(args.output_dir / 'STATUS.json', {
        'status': 'failed', 'updated_at': utc_now(), 'fingerprint': fingerprint,
        'error_type': type(error).__name__, 'error': str(error),
        'completed_parts_preserved': True,
      })
      raise


if __name__ == '__main__':
  raise SystemExit(main())
