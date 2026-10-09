#!/usr/bin/env python3
"""Paired random-versus-strided reveal diagnostics with permanently masked probes.

This is an inference-only, fixed-probe diagnostic, NOT free generation, a soft
anchor mechanism, token repair, or a training change. Teacher/model commits are
irreversible; only the same initially masked probe tokens are scored each round.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import json
import math
import os
import pathlib
import signal
import sys
import time

import pandas as pd
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.eval import dcache_eval_common as common  # noqa: E402
from scripts.eval import eval_recurrence_audit as recurrence  # noqa: E402
from scripts.eval.eval_recurrence_audit import (  # noqa: E402
  CooperativeStop, DUAL_VARIANTS, Memory, PAUSED_EXIT_CODE, VARIANTS,
  checkpoint_metadata, deterministic_seed, forward_with_memory, intervene,
  prepared_data_fingerprint, prepare_output, ratio_key, utc_now,
)

SOURCES = ('teacher', 'model')
SCHEDULES = ('strided', 'random')
DEFAULT_SCORE_ROUNDS = (0, 1, 2, 4, 8, 16, 24, 32)
GROUP_COLUMNS = [
  'variant', 'seed', 'source', 'schedule', 'condition', 'initial_mask_ratio',
  'round',
]
RAW_COLUMNS = GROUP_COLUMNS + [
  'example_id', 'masked_tokens', 'remaining_mask_tokens', 'nll',
  'top1_accuracy', 'top5_accuracy',
]


@dataclass(frozen=True)
class Phase:
  initial_mask_ratio: float
  source: str

  @property
  def name(self):
    ratio = format(self.initial_mask_ratio, '.10f').rstrip('0').rstrip('.')
    return f'ratio{ratio}_{self.source}'


@dataclass(frozen=True)
class Schedule:
  initial_mask: torch.Tensor
  probe_mask: torch.Tensor
  reveal_groups: dict[str, tuple[torch.Tensor, ...]]


def condition_names(variant):
  if variant not in VARIANTS:
    raise ValueError(f'Unknown variant: {variant}')
  if variant in ('objective', 'vanilla'):
    return ('absent_dcache',)
  return ('correct', 'absent_dcache')


def make_schedule(attention, example_ids, initial_mask_ratio, seed,
                  probe_count=128, stride=32):
  """Construct matched CPU masks with actual original-position residue groups.

  Zero-based slot j selects positions j, j+stride, ... (one-based j+1,
  j+1+stride, ...), intersected with initial masks and excluding fixed probes.
  Random order is split using exactly those per-document/per-slot counts,
  including empty slots. Probe and schedule RNG streams are independent of
  each other and of the shared initial-mask RNG.
  """
  if probe_count < 1 or stride < 1:
    raise ValueError('probe_count and stride must be positive')
  attention = attention.detach().cpu()
  if attention.ndim != 2 or attention.shape[1] < 2:
    raise ValueError('Attention mask must be [batch, length >= 2]')
  initial = common.deterministic_nested_masks(
    attention, example_ids, [initial_mask_ratio], seed)[initial_mask_ratio]
  probes = torch.zeros_like(initial)
  ratio_tag = f'{float(initial_mask_ratio):.10f}'
  for row, example_id in enumerate(example_ids):
    positions = initial[row].nonzero(as_tuple=False).flatten()
    if positions.numel() < probe_count:
      raise ValueError(
        f'Example {example_id}: initial mask contains {positions.numel()} '
        f'tokens, fewer than requested {probe_count} fixed probes')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(deterministic_seed(
      seed, example_id, f'anchor_probes:ratio{ratio_tag}'))
    probes[row, positions[torch.randperm(
      positions.numel(), generator=generator)[:probe_count]]] = True
  revealable = initial & ~probes
  residue = torch.arange(initial.shape[1]) % stride
  strided = tuple(revealable & (residue == slot) for slot in range(stride))
  random = tuple(torch.zeros_like(initial) for _ in range(stride))
  for row, example_id in enumerate(example_ids):
    positions = revealable[row].nonzero(as_tuple=False).flatten()
    generator = torch.Generator(device='cpu')
    generator.manual_seed(deterministic_seed(
      seed, example_id, f'anchor_random_schedule:ratio{ratio_tag}:stride{stride}'))
    shuffled = positions[torch.randperm(positions.numel(), generator=generator)]
    offset = 0
    for slot, group in enumerate(strided):
      count = int(group[row].sum())
      random[slot][row, shuffled[offset:offset + count]] = True
      offset += count
    if offset != positions.numel():
      raise RuntimeError('Schedule did not cover every revealable position')
  return Schedule(initial, probes, {'strided': strided, 'random': random})


def reveal_from_model(state, log_probs, reveal_mask, example_ids, seed,
                      mask_index):
  """Commit sampled tokens only at selected, currently masked positions.

  This function cannot read hidden targets. A position-specific uniform draw
  drives inverse-CDF categorical sampling, so changing its reveal round does
  not change its random draw. Variant, ratio, source, condition, and schedule
  are deliberately absent from token RNG keys. Sampling excludes mask_index.
  Empty per-document groups are valid and do not alter state.
  """
  if state.ndim != 2 or log_probs.shape[:2] != state.shape:
    raise ValueError('State and scores must have matching [batch, length] axes')
  if len(example_ids) != state.shape[0] or reveal_mask.shape != state.shape:
    raise ValueError('Example IDs/reveal mask must match state')
  if not 0 <= int(mask_index) < log_probs.shape[-1]:
    raise ValueError('Mask token index is outside the vocabulary')
  reveal_mask = reveal_mask.to(state.device).bool()
  if not torch.all(state[reveal_mask].eq(int(mask_index))):
    raise ValueError('A reveal must never overwrite an already visible token')
  result = state.clone()
  for row, example_id in enumerate(example_ids):
    positions = reveal_mask[row].nonzero(as_tuple=False).flatten()
    if not positions.numel():
      continue
    values = log_probs[row, positions].detach().float().cpu().clone()
    values[:, int(mask_index)] = -float('inf')
    probabilities = values.softmax(dim=-1)
    if not torch.isfinite(probabilities).all():
      raise ValueError('Nonfinite model token probabilities during rollout')
    # Double-precision CDF avoids rounding near-one mass to one early, and
    # right=True skips zero-probability plateaus even when the uniform is 0.
    cumulative = probabilities.double().cumsum(dim=-1)
    cumulative /= cumulative[:, -1:].clone()
    cumulative[:, -1] = 1.0
    uniforms = []
    for position in positions.cpu().tolist():
      generator = torch.Generator(device='cpu')
      generator.manual_seed(deterministic_seed(
        seed, example_id, f'anchor_token:position{int(position)}'))
      uniforms.append(torch.rand((), generator=generator, dtype=torch.float64))
    sampled = torch.searchsorted(
      cumulative, torch.stack(uniforms).unsqueeze(-1), right=True).squeeze(-1)
    if torch.any(sampled.eq(int(mask_index))):
      raise RuntimeError('Categorical sampler returned the excluded mask token')
    result[row, positions] = sampled.to(state.device)
  return result


def sigma_for_mask(model, mask, attention, device):
  """Use each document's actual remaining mask fraction, excluding token one."""
  counts = mask.detach().cpu().sum(dim=1)
  eligible = common.eligible_positions(attention.detach().cpu()).sum(dim=1)
  if torch.any(eligible <= 0):
    raise ValueError('Every document must have eligible masked positions')
  return torch.cat([
    common.sigma_for_ratio(model, float(count) / int(total), 1, device)
    for count, total in zip(counts, eligible)
  ], dim=0)


def evaluate_phase(model, variant, tokens, attention, example_ids, phase, seed,
                   probe_count=128, stride=32,
                   score_rounds=DEFAULT_SCORE_ROUNDS):
  """Evaluate independent histories for both schedules and every condition.

  Each arm makes stride+1 actual forwards, including round zero and empty
  reveal slots. In dual-source models, absent_dcache continues to read that
  arm's own evolving final hidden state; it does not borrow a control latent.
  """
  if phase.source not in SOURCES:
    raise ValueError(f'Unknown token source: {phase.source}')
  score_rounds = set(score_rounds)
  if not score_rounds or min(score_rounds) < 0 or max(score_rounds) > stride:
    raise ValueError('Scoring rounds must be between zero and stride inclusive')
  plan = make_schedule(attention, example_ids, phase.initial_mask_ratio,
                       seed, probe_count, stride)
  probes = plan.probe_mask.to(tokens.device)
  initial = common.masked_state(tokens, plan.initial_mask, model.mask_index)
  rows = []
  for schedule_name in SCHEDULES:
    groups = plan.reveal_groups[schedule_name]
    for condition in condition_names(variant):
      state = initial.clone()
      remaining = plan.initial_mask.clone()
      memory = Memory()
      for round_index in range(stride + 1):
        sigma = sigma_for_mask(model, remaining, attention, tokens.device)
        scores, memory = forward_with_memory(
          model, variant, state, sigma, intervene(memory, condition))
        if round_index in score_rounds:
          scored = common.score_masked_tokens(
            scores, tokens, probes, example_ids, variant=variant, seed=int(seed),
            source=phase.source, schedule=schedule_name, condition=condition,
            initial_mask_ratio=float(phase.initial_mask_ratio), round=round_index)
          counts = remaining.sum(dim=1).tolist()
          rows.extend(dict(row, remaining_mask_tokens=int(count))
                      for row, count in zip(scored, counts))
        if round_index < stride:
          group = groups[round_index]
          if torch.any(group & (~remaining | plan.probe_mask)):
            raise RuntimeError('Reveal schedule modifies visible tokens or fixed probes')
          if phase.source == 'teacher':
            state = torch.where(group.to(tokens.device), tokens, state)
          else:
            state = reveal_from_model(
              state, scores, group, example_ids, seed, model.mask_index)
          remaining = remaining & ~group
        del scores, sigma
      if not torch.equal(remaining, plan.probe_mask):
        raise RuntimeError('Final remaining mask must consist exactly of fixed probes')
      if not torch.all(state[probes].eq(model.mask_index)):
        raise RuntimeError('Fixed probe tokens were accidentally revealed')
      del memory, state
  return rows


def make_phases(args):
  return [Phase(ratio, source) for ratio in args.initial_mask_ratios
          for source in args.sources]


def expected_batch_size(args, batch_index):
  return min(args.batch_size, args.examples - batch_index * args.batch_size)


def number_of_batches(args):
  return math.ceil(args.examples / args.batch_size)


def expected_part_rows(args, phase=None, batch_size=None):
  del phase
  count = args.batch_size if batch_size is None else batch_size
  return count * len(SCHEDULES) * len(condition_names(args.variant)) * len(args.score_rounds)


def validate_part(frame, args, phase, batch_index):
  label = f'{phase.name}/{batch_index}'
  batch_size = expected_batch_size(args, batch_index)
  if batch_size <= 0 or set(frame.columns) != set(RAW_COLUMNS):
    raise RuntimeError(f'Unexpected result schema/batch for {label}')
  if len(frame) != expected_part_rows(args, phase, batch_size):
    raise RuntimeError(f'Incomplete result part for {label}')
  if frame.duplicated(GROUP_COLUMNS + ['example_id']).any():
    raise RuntimeError(f'Duplicate result rows for {label}')
  expected_ids = set(common.batch_ids(batch_index, args.batch_size, batch_size))
  for column in ('seed', 'round', 'example_id', 'masked_tokens', 'remaining_mask_tokens'):
    values = pd.to_numeric(frame[column], errors='coerce')
    if not values.map(lambda value: math.isfinite(float(value))).all():
      raise RuntimeError(f'Nonfinite {column} in {label}')
    if not values.eq(values.round()).all():
      raise RuntimeError(f'Noninteger {column} in {label}')
  for name, value in [('variant', args.variant), ('seed', args.seed),
                      ('source', phase.source),
                      ('initial_mask_ratio', phase.initial_mask_ratio)]:
    values = frame[name].map(ratio_key) if name == 'initial_mask_ratio' else frame[name]
    if not values.eq(value).all():
      raise RuntimeError(f'Wrong {name} for {label}')
  expected_grid = {(schedule, condition, step, example_id)
                   for schedule in SCHEDULES
                   for condition in condition_names(args.variant)
                   for step in args.score_rounds for example_id in expected_ids}
  actual_grid = set(frame[['schedule', 'condition', 'round', 'example_id']]
                    .itertuples(index=False, name=None))
  if actual_grid != expected_grid:
    raise RuntimeError(f'Incomplete paired schedule/condition/round/document grid in {label}')
  for column in ('nll', 'top1_accuracy', 'top5_accuracy'):
    if not frame[column].map(lambda value: math.isfinite(float(value))).all():
      raise RuntimeError(f'Nonfinite metric {column} in {label}')
  if not frame.masked_tokens.eq(args.probe_count).all():
    raise RuntimeError(f'Wrong fixed probe count in {label}')
  if not frame.remaining_mask_tokens.between(args.probe_count, 1023).all():
    raise RuntimeError(f'Invalid remaining mask count in {label}')
  if not frame.nll.ge(-1e-5).all():
    raise RuntimeError(f'Invalid NLL in {label}')
  if not (frame.top1_accuracy.between(0, 1).all()
          and frame.top5_accuracy.between(0, 1).all()
          and (frame.top1_accuracy <= frame.top5_accuracy + 1e-6).all()):
    raise RuntimeError(f'Invalid accuracy in {label}')
  paired_counts = frame.groupby(['example_id', 'round']).remaining_mask_tokens.nunique()
  if not paired_counts.eq(1).all():
    raise RuntimeError(f'Mismatched per-round mask counts across paired arms in {label}')
  for _, group in frame.groupby(['example_id', 'schedule', 'condition']):
    if group.sort_values('round').remaining_mask_tokens.diff().dropna().gt(0).any():
      raise RuntimeError(f'Mask counts increase across rounds in {label}')
  final = frame[frame['round'] == args.stride]
  if not final.remaining_mask_tokens.eq(args.probe_count).all():
    raise RuntimeError(f'Final scored mask does not equal the probes in {label}')


def source_fingerprints():
  fingerprints = recurrence.source_fingerprints()
  own_path = pathlib.Path(__file__).resolve()
  fingerprints[str(own_path.relative_to(REPO_ROOT))] = recurrence.file_digest(own_path)
  return fingerprints


def make_metadata(args, phases):
  return {
    'evaluation': 'anchor_schedule', 'schema_version': 1,
    'variant': args.variant, 'weights': 'EMA', 'expected_step': args.expected_step,
    'checkpoint': checkpoint_metadata(args.checkpoint, args.expected_step),
    'dataset': 'openwebtext-split', 'split': 'validation',
    'data': prepared_data_fingerprint(args.data_dir), 'length': 1024,
    'examples': args.examples, 'batch_size': args.batch_size, 'seed': args.seed,
    'initial_mask_ratios': args.initial_mask_ratios, 'sources': args.sources,
    'schedules': list(SCHEDULES), 'probe_count': args.probe_count,
    'stride': args.stride, 'score_rounds': args.score_rounds,
    'forwards_per_trajectory': args.stride + 1,
    'conditions': list(condition_names(args.variant)),
    'phases': [phase.__dict__ for phase in phases],
    'mask_protocol': 'Shared exact-count initial masks from common.deterministic_nested_masks; seed+document ID; eligible attention excludes first token, which stays visible. Pairing is within each initial mask ratio.',
    'probe_protocol': 'Exactly probe_count positions sampled without replacement FROM each initial mask using an independent SHA256 seed stream; fixed across sources/schedules/conditions/variants and all rounds at that initial ratio. Probes remain masked and are the ONLY primary scored targets.',
    'schedule_protocol': 'Strided uses actual one-based groups {r, r+stride, ...}, r=1..stride, excluding initial visible positions and probes. Random permutes all revealable positions and splits them using the exact SAME per-document, per-slot strided counts. Empty slots still receive a forward. No rebalancing or reindexing of residue groups.',
    'sigma_protocol': 'At every forward use actual per-document remaining_mask_tokens divided by attention-eligible positions excluding first token; common.sigma_for_ratio then checkpoint-native sigma conversion. Counts and sigma match across paired schedules/conditions at every round.',
    'teacher_protocol': 'Only newly selected reveal positions receive true tokens. At final round all schedule/condition token inputs match exactly for a document and initial ratio, isolating effects of separately accumulated memory histories.',
    'model_protocol': 'Initial visible context is teacher-provided; subsequently selected positions receive model categorical samples only. Reveals cannot read hidden targets or revise previously visible tokens. Each arm owns its sampled-token and recurrent history.',
    'memory_protocol': 'Reset both memories for EVERY condition/schedule/source/initial-ratio trajectory. correct reads its own DCache and final state; absent_dcache passes D=None but retains its OWN evolving final state when the model has that route. Final-route enabled does not mean final tensor values are equal across conditions.',
    'baseline_protocol': 'Objective/vanilla use only absent_dcache (no DCache/final route). They still execute EVERY scheduled forward, including round zero and empty reveal groups, matching recurrent NFE.',
    'sampling_rng': 'CPU inverse-CDF categorical, mask token probability forced to zero; one independent float64 uniform keyed by SHA256(seed, document ID, anchor_token:position{zero_based_position}), excluding reveal round/schedule/source/ratio/condition/variant. Mask, probe and reveal-order RNGs are independent.',
    'interpretation': 'Artificial permanently masked fixed-probe diagnostic, NOT free generation, soft anchors, revisable commitments, repair, or a training change. Intermediate random/strided visible context differs intentionally; primary scored target positions do not. NLL is conditional probe reconstruction NLL, not free-generation perplexity.',
    'source_sha256': source_fingerprints(), 'torch_version': str(torch.__version__),
  }


def write_reports(args, phases, output_dir):
  frames = []
  done = 0
  expected = len(phases) * number_of_batches(args)
  for phase in phases:
    for batch_index in range(number_of_batches(args)):
      path = common.part_path(output_dir, phase.name, batch_index)
      if not path.is_file():
        continue
      frame = pd.read_csv(path)
      validate_part(frame, args, phase, batch_index)
      frames.append(frame)
      done += 1
  if frames:
    raw = pd.concat(frames, ignore_index=True).sort_values(GROUP_COLUMNS + ['example_id'])
    if raw.duplicated(GROUP_COLUMNS + ['example_id']).any():
      raise RuntimeError('Duplicate rows across completed parts')
    summary = common.aggregate_metrics(raw, GROUP_COLUMNS)
    counts = raw.groupby(GROUP_COLUMNS, as_index=False).agg(
      mean_remaining_mask_tokens=('remaining_mask_tokens', 'mean'),
      min_remaining_mask_tokens=('remaining_mask_tokens', 'min'),
      max_remaining_mask_tokens=('remaining_mask_tokens', 'max'))
    summary = summary.merge(counts, on=GROUP_COLUMNS, validate='one_to_one')
    common.atomic_write_csv(output_dir / 'summary.csv', summary)
    # A fresh partial raw file is useful for inspection and carries only
    # complete, validated phase/batch units; completion.json is authoritative.
    common.atomic_write_csv(output_dir / 'per_document_metrics.csv', raw)
  return done == expected, done, expected


def run_evaluation(args, phases, output_dir, stop, fingerprint):
  expected_parts = len(phases) * number_of_batches(args)
  expected_paths = {common.part_path(output_dir, phase.name, index)
                    for phase in phases for index in range(number_of_batches(args))}
  unexpected = set((output_dir / 'parts').glob('*/*.csv')) - expected_paths
  if unexpected:
    raise RuntimeError(f'Unexpected result part files: {sorted(map(str, unexpected))}')
  done = set()
  for phase in phases:
    for batch_index in range(number_of_batches(args)):
      path = common.part_path(output_dir, phase.name, batch_index)
      if path.is_file():
        validate_part(pd.read_csv(path), args, phase, batch_index)
        done.add((phase.name, batch_index))
  completion_path = output_dir / 'completion.json'
  if completion_path.is_file():
    completion = json.loads(completion_path.read_text())
    if (completion.get('status') != 'complete'
        or completion.get('fingerprint') != fingerprint
        or completion.get('completed_parts') != expected_parts
        or completion.get('expected_parts') != expected_parts
        or len(done) != expected_parts):
      raise RuntimeError('Completion marker does not match the validated result parts')
  device = torch.device(args.device)
  invocation_started = utc_now()
  previous_elapsed = 0.0
  previous_peak = 0.0
  prior_status = output_dir / 'STATUS.json'
  if prior_status.is_file():
    prior = json.loads(prior_status.read_text())
    if prior.get('fingerprint') == fingerprint:
      previous_elapsed = float(prior.get('elapsed_seconds_total', 0.0))
      previous_peak = float(prior.get('peak_allocated_gib', 0.0))

  def status(state, **extra):
    elapsed = time.monotonic() - stop.started
    peak = (torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == 'cuda' else 0.0)
    common.atomic_write_json(output_dir / 'STATUS.json', {
      'status': state, 'updated_at': utc_now(), 'invocation_started_at': invocation_started,
      'variant': args.variant, 'seed': args.seed, 'fingerprint': fingerprint,
      'completed_parts': len(done), 'expected_parts': expected_parts,
      'elapsed_seconds_this_invocation': elapsed,
      'elapsed_seconds_total': previous_elapsed + elapsed,
      'device': args.device, 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
      'peak_allocated_gib': max(previous_peak, peak),
      'peak_allocated_gib_this_invocation': peak, **extra,
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
  common.print_device_banner(device)
  if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)
  model = common.load_ema_model(args.checkpoint, config, tokenizer, device)
  initial_done = len(done)
  print(f'Loaded {args.variant} EMA, step {args.expected_step}; '
        f'{expected_parts - initial_done}/{expected_parts} anchor phase/batch parts remain.',
        flush=True)
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
          status('running', active_phase=phase.name, active_batch=batch_index)
          part_started = time.monotonic()
          rows = evaluate_phase(
            model, args.variant, tokens, attention, ids, phase, args.seed,
            args.probe_count, args.stride, args.score_rounds)
          frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
          validate_part(frame, args, phase, batch_index)
          common.atomic_write_csv(common.part_path(output_dir, phase.name, batch_index), frame)
          done.add((phase.name, batch_index))
          count = len(done) - initial_done
          seconds_per_part = (time.monotonic() - eval_start) / count
          eta_hours = seconds_per_part * (expected_parts - len(done)) / 3600
          status('running', last_phase=phase.name, last_batch=batch_index,
                 last_part_seconds=time.monotonic() - part_started,
                 estimated_remaining_hours=eta_hours)
          print(f'[{args.variant} seed={args.seed}] parts {len(done)}/{expected_parts}, '
                f'batch {batch_index + 1}/{number_of_batches(args)}, {phase.name}; '
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
  parser.add_argument('--num-workers', type=int, default=2)
  parser.add_argument('--seed', type=int, default=20260812)
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--initial-mask-ratios', type=float, nargs='+', default=[.5, 1.0])
  parser.add_argument('--sources', choices=SOURCES, nargs='+', default=list(SOURCES))
  parser.add_argument('--probe-count', type=int, default=128)
  parser.add_argument('--stride', type=int, default=32)
  parser.add_argument('--score-rounds', type=int, nargs='+', default=list(DEFAULT_SCORE_ROUNDS))
  parser.add_argument('--max-hours', type=float,
                      help='Cooperative wall-clock limit; finish the active atomic phase/batch, then exit 75. Rerun the same command to resume.')
  return parser.parse_args(argv)


def validate_args(args):
  if args.examples <= 0 or args.batch_size <= 0:
    raise ValueError('examples and batch-size must be positive')
  if args.num_workers < 0 or args.expected_step < 0:
    raise ValueError('Worker count and expected step must be nonnegative')
  if not 1 <= args.probe_count <= 1023 or not 1 <= args.stride <= 1024:
    raise ValueError('Use probe-count in [1,1023] and stride in [1,1024]')
  args.initial_mask_ratios = sorted(set(ratio_key(value) for value in args.initial_mask_ratios))
  args.sources = list(dict.fromkeys(args.sources))
  args.score_rounds = sorted(set(args.score_rounds))
  if not args.initial_mask_ratios or not all(
      math.isfinite(value) and 0 < value <= 1 for value in args.initial_mask_ratios):
    raise ValueError('Initial mask ratios must be finite and in (0,1]')
  if any(max(1, round(ratio * 1023)) < args.probe_count for ratio in args.initial_mask_ratios):
    raise ValueError('Initial mask is too small for requested fixed probe count at length 1024')
  if not args.sources or any(source not in SOURCES for source in args.sources):
    raise ValueError('Use at least one supported reveal source')
  if (not args.score_rounds or min(args.score_rounds) < 0
      or max(args.score_rounds) > args.stride):
    raise ValueError('Scoring rounds must be in [0,stride]')
  if args.max_hours is not None and (not math.isfinite(args.max_hours) or args.max_hours <= 0):
    raise ValueError('--max-hours must be finite and positive')
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
        update_status('complete')
        performance = json.loads((args.output_dir / 'STATUS.json').read_text())
        common.atomic_write_json(args.output_dir / 'completion.json', {
          'status': 'complete', 'completed_at': utc_now(), 'fingerprint': fingerprint,
          'completed_parts': done, 'expected_parts': expected,
          'examples_per_condition': args.examples,
          'elapsed_seconds_total': performance['elapsed_seconds_total'],
          'peak_allocated_gib': performance['peak_allocated_gib'],
        })
        print(f'Complete: {args.output_dir}', flush=True)
        return 0
      update_status('paused', reason=stop.reason)
      print(f'Safely paused ({done}/{expected} parts): {args.output_dir}. '
            'Run the same command again to resume.', flush=True)
      return PAUSED_EXIT_CODE
    except Exception as error:
      prior = args.output_dir / 'STATUS.json'
      status = json.loads(prior.read_text()) if prior.is_file() else {}
      common.atomic_write_json(prior, {
        **status, 'status': 'failed', 'updated_at': utc_now(),
        'fingerprint': fingerprint, 'error_type': type(error).__name__,
        'error': str(error), 'completed_parts_preserved': True,
      })
      raise


if __name__ == '__main__':
  raise SystemExit(main())
