#!/usr/bin/env python3
"""Evaluate whether one DCache recurrence improves an unchanged noisy state."""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import dcache_eval_common as common  # noqa: E402


DEFAULT_MASK_RATIOS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70)
FIRST = 'first_no_cache'
CORRECT = 'second_correct_cache'
SHUFFLED = 'second_shuffled_cache'


def parse_ratio(value: str) -> float:
  try:
    ratio = float(value)
  except ValueError as error:
    raise argparse.ArgumentTypeError(
      f'Expected a mask ratio such as 0.30; got {value!r}') from error
  if not 0 < ratio <= 1:
    raise argparse.ArgumentTypeError(
      f'Mask ratio must satisfy 0 < ratio <= 1; got {value!r}')
  return ratio


def parse_args():
  parser = argparse.ArgumentParser(
    description=(
      'Run an unchanged corrupted state once without cache, then again with '
      'the first pass cache or a batch-shuffled copy of that cache.'))
  parser.add_argument('--dcache-checkpoint', type=pathlib.Path, required=True)
  parser.add_argument('--data-dir', type=pathlib.Path, required=True)
  parser.add_argument('--output-dir', type=pathlib.Path, required=True)
  parser.add_argument(
    '--mask-ratios', type=parse_ratio, nargs='+',
    default=DEFAULT_MASK_RATIOS)
  parser.add_argument('--examples', type=int, default=800)
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--num-workers', type=int, default=2)
  parser.add_argument('--seed', type=int, default=20260812)
  parser.add_argument('--bootstrap-samples', type=int, default=10_000)
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--dcache-gate-enabled', action='store_true')
  parser.add_argument('--force', action='store_true')
  return parser.parse_args()


def validate_args(args) -> None:
  if args.examples <= 0:
    raise ValueError('--examples must be positive')
  if args.batch_size < 2:
    raise ValueError('--batch-size must be at least 2 for shuffled cache')
  if args.examples % args.batch_size != 0:
    raise ValueError(
      '--examples must be divisible by --batch-size so no incomplete batch '
      'weakens the shuffled-cache control')
  if args.num_workers < 0:
    raise ValueError('--num-workers cannot be negative')
  if args.bootstrap_samples <= 0:
    raise ValueError('--bootstrap-samples must be positive')
  if str(args.device).startswith('cuda') and not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested but torch.cuda.is_available() is false')
  args.mask_ratios = sorted(set(float(ratio) for ratio in args.mask_ratios))
  args.dcache_checkpoint = args.dcache_checkpoint.expanduser().resolve()
  args.data_dir = args.data_dir.expanduser().resolve()
  if not args.dcache_checkpoint.is_file():
    raise FileNotFoundError(args.dcache_checkpoint)
  expected_data = (
    args.data_dir
    / 'openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat')
  if not expected_data.is_dir():
    raise FileNotFoundError(
      f'Prepared validation dataset not found: {expected_data}')


def run_evaluation(args, loader, tokenizer, output_dir, device) -> None:
  phase = 'same_state_recurrence'
  missing = [
    index for index in range(len(loader))
    if not common.part_path(output_dir, phase, index).is_file()
  ]
  if not missing:
    print('All batches are already complete; skipping model load.', flush=True)
    return

  config = common.compose_eval_config(
    args.batch_size,
    args.data_dir,
    recurrent=True,
    num_workers=args.num_workers,
    gate_enabled=args.dcache_gate_enabled)
  model = common.load_ema_model(
    args.dcache_checkpoint, config, tokenizer, device)
  print(f'Loaded DCache EMA; {len(missing)}/{len(loader)} batches remain.',
        flush=True)

  with torch.inference_mode():
    for batch_index, batch in enumerate(loader):
      path = common.part_path(output_dir, phase, batch_index)
      if path.is_file():
        continue
      tokens_cpu = batch['input_ids'].long()
      attention = batch['attention_mask']
      ids = common.batch_ids(batch_index, args.batch_size, tokens_cpu.shape[0])
      masks = common.deterministic_nested_masks(
        attention, ids, args.mask_ratios, args.seed, ignore_first=True)
      tokens = tokens_cpu.to(device, non_blocking=True)
      rows = []

      for ratio in args.mask_ratios:
        mask = masks[ratio].to(device, non_blocking=True)
        state = common.masked_state(tokens, mask, model.mask_index)
        sigma = common.sigma_for_ratio(model, ratio, len(ids), device)

        first_log_probs, first_cache = model.forward(
          state,
          sigma=sigma,
          sample_mode=True,
          previous_step_kv=None,
          return_step_kv=True)
        rows.extend(common.score_masked_tokens(
          first_log_probs, tokens, mask, ids,
          condition=FIRST, mask_ratio=ratio))

        correct_log_probs = model.forward(
          state,
          sigma=sigma,
          sample_mode=True,
          previous_step_kv=first_cache)
        rows.extend(common.score_masked_tokens(
          correct_log_probs, tokens, mask, ids,
          condition=CORRECT, mask_ratio=ratio))

        shuffled_cache = common.roll_cache_batch(first_cache)
        shuffled_log_probs = model.forward(
          state,
          sigma=sigma,
          sample_mode=True,
          previous_step_kv=shuffled_cache)
        rows.extend(common.score_masked_tokens(
          shuffled_log_probs, tokens, mask, ids,
          condition=SHUFFLED, mask_ratio=ratio))

        del first_log_probs, correct_log_probs, shuffled_log_probs
        del first_cache, shuffled_cache, state, sigma, mask

      common.write_part(path, rows)
      if (batch_index + 1) % 10 == 0 or batch_index + 1 == len(loader):
        print(f'Completed batch {batch_index + 1}/{len(loader)}', flush=True)

  common.unload_model(model)


def paired_accuracy_differences(
    raw: pd.DataFrame,
    seed: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
  """Return condition-minus-reference top-1 differences in percentage points."""
  rows = []
  comparisons = (
    (CORRECT, FIRST),
    (SHUFFLED, FIRST),
    (CORRECT, SHUFFLED),
  )
  for ratio_index, (ratio, group) in enumerate(
      raw.groupby('mask_ratio', sort=True)):
    for comparison_index, (condition, reference) in enumerate(comparisons):
      left = group[group.condition == condition][
        ['example_id', 'top1_accuracy']]
      right = group[group.condition == reference][
        ['example_id', 'top1_accuracy']]
      paired = left.merge(
        right, on='example_id', suffixes=('_condition', '_reference'),
        validate='one_to_one')
      if len(paired) != len(left) or len(paired) != len(right):
        raise RuntimeError(
          f'Incomplete accuracy pairing for ratio={ratio}, '
          f'{condition} vs {reference}')
      delta_pp = 100.0 * (
        paired.top1_accuracy_condition.to_numpy(dtype=np.float64)
        - paired.top1_accuracy_reference.to_numpy(dtype=np.float64))
      mean, low, high = common.paired_bootstrap_mean(
        delta_pp,
        seed=seed + ratio_index * 100 + comparison_index,
        samples=bootstrap_samples)
      rows.append({
        'mask_ratio': float(ratio),
        'condition': condition,
        'reference': reference,
        'n_examples': int(len(paired)),
        'mean_delta_top1_pp': mean,
        'ci95_low': low,
        'ci95_high': high,
      })
  return pd.DataFrame(rows)


def save_plot(summary: pd.DataFrame, paired_accuracy: pd.DataFrame,
              path: pathlib.Path) -> None:
  colors = {
    FIRST: '#1f77b4',
    CORRECT: '#2ca02c',
    SHUFFLED: '#d62728',
  }
  labels = {
    FIRST: 'first pass (no cache)',
    CORRECT: 'second pass (correct self-cache)',
    SHUFFLED: 'second pass (shuffled self-cache)',
  }
  if summary.mask_ratio.nunique() == 1:
    _save_single_ratio_plot(
      summary, paired_accuracy, path, colors=colors, labels=labels)
    return

  fig, axes = plt.subplots(1, 3, figsize=(18, 5))
  for condition in (FIRST, CORRECT, SHUFFLED):
    group = summary[summary.condition == condition].sort_values('mask_ratio')
    x = group.mask_ratio.to_numpy() * 100
    axes[0].plot(
      x, group.conditional_nll, marker='o', linewidth=2,
      color=colors[condition], label=labels[condition])
    axes[1].plot(
      x, group.top1_accuracy * 100, marker='o', linewidth=2,
      color=colors[condition], label=labels[condition])

  for condition in (CORRECT, SHUFFLED):
    group = paired_accuracy[
      (paired_accuracy.condition == condition)
      & (paired_accuracy.reference == FIRST)].sort_values('mask_ratio')
    x = group.mask_ratio.to_numpy() * 100
    y = group.mean_delta_top1_pp.to_numpy()
    axes[2].plot(
      x, y, marker='o', linewidth=2, color=colors[condition],
      label=f'{labels[condition]} - first pass')
    axes[2].fill_between(
      x, group.ci95_low.to_numpy(), group.ci95_high.to_numpy(),
      color=colors[condition], alpha=0.15)

  axes[0].set_ylabel('Conditional masked-token NLL')
  axes[1].set_ylabel('Masked-token top-1 accuracy (%)')
  axes[2].set_ylabel('Paired top-1 change vs first pass (pp)')
  for axis in axes:
    axis.set_xlabel('Mask ratio of unchanged input (%)')
    axis.grid(alpha=0.25)
  axes[0].legend(fontsize=8)
  axes[2].legend(fontsize=8)
  axes[2].axhline(0, color='black', linewidth=0.8)
  fig.suptitle('Same-state DCache recurrence: x → cache(x) → x again')
  fig.tight_layout()
  fig.savefig(path, dpi=180, bbox_inches='tight')
  plt.close(fig)


def _save_single_ratio_plot(
    summary: pd.DataFrame,
    paired_accuracy: pd.DataFrame,
    path: pathlib.Path,
    colors: dict[str, str],
    labels: dict[str, str],
) -> None:
  """Use readable bars instead of three isolated line-plot points."""
  conditions = (FIRST, CORRECT, SHUFFLED)
  indexed = summary.set_index('condition')
  missing = set(conditions) - set(indexed.index)
  if missing:
    raise ValueError(f'Missing conditions for single-ratio plot: {missing}')
  ratio = float(summary.mask_ratio.iloc[0])
  short_labels = ('First\nno cache', 'Second\ncorrect cache',
                  'Second\nshuffled cache')
  bar_colors = [colors[condition] for condition in conditions]

  nll = np.array([
    float(indexed.loc[condition, 'conditional_nll'])
    for condition in conditions])
  accuracy = 100.0 * np.array([
    float(indexed.loc[condition, 'top1_accuracy'])
    for condition in conditions])

  delta = paired_accuracy[
    (paired_accuracy.reference == FIRST)
    & paired_accuracy.condition.isin((CORRECT, SHUFFLED))].set_index(
      'condition')
  delta_conditions = (CORRECT, SHUFFLED)
  delta_values = np.array([
    float(delta.loc[condition, 'mean_delta_top1_pp'])
    for condition in delta_conditions])
  delta_low = np.array([
    float(delta.loc[condition, 'ci95_low'])
    for condition in delta_conditions])
  delta_high = np.array([
    float(delta.loc[condition, 'ci95_high'])
    for condition in delta_conditions])
  errors = np.vstack((delta_values - delta_low, delta_high - delta_values))

  fig, axes = plt.subplots(1, 3, figsize=(16, 5))
  positions = np.arange(len(conditions))
  axes[0].bar(positions, nll, color=bar_colors, width=0.68)
  axes[1].bar(positions, accuracy, color=bar_colors, width=0.68)
  axes[0].set_xticks(positions, short_labels)
  axes[1].set_xticks(positions, short_labels)

  for axis, values, precision in (
      (axes[0], nll, 3),
      (axes[1], accuracy, 2)):
    span = max(float(values.max() - values.min()), 0.01)
    margin = span * 0.45
    axis.set_ylim(float(values.min() - margin), float(values.max() + margin))
    for position, value in enumerate(values):
      axis.text(
        position, value + span * 0.06, f'{value:.{precision}f}',
        ha='center', va='bottom', fontsize=10, fontweight='bold')

  delta_positions = np.arange(len(delta_conditions))
  axes[2].bar(
    delta_positions, delta_values, yerr=errors,
    color=[colors[condition] for condition in delta_conditions],
    width=0.60, capsize=5)
  axes[2].set_xticks(
    delta_positions, ('Correct cache', 'Shuffled cache'))
  axes[2].axhline(0, color='black', linewidth=0.9)
  axes[2].set_ylim(
    min(-0.12, float(delta_low.min() - 0.10)),
    float(delta_high.max() + 0.18))
  for position, value in enumerate(delta_values):
    axes[2].text(
      position, value + 0.045, f'{value:+.3f} pp',
      ha='center', va='bottom', fontsize=10, fontweight='bold')

  axes[0].set_ylabel('Conditional masked-token NLL (lower is better)')
  axes[1].set_ylabel('Masked-token top-1 accuracy (%)')
  axes[2].set_ylabel('Accuracy change from first pass (pp)')
  axes[2].set_title('Error bars: paired 95% CI')
  for axis in axes:
    axis.grid(axis='y', alpha=0.25)
  fig.suptitle(
    f'Same {ratio * 100:.0f}%-masked input evaluated twice')
  fig.tight_layout()
  fig.savefig(path, dpi=180, bbox_inches='tight')
  plt.close(fig)


def write_reports(args, output_dir) -> None:
  raw = common.collect_parts(output_dir)
  expected = args.examples * len(args.mask_ratios) * 3
  if len(raw) != expected:
    raise RuntimeError(f'Expected {expected} raw rows, found {len(raw)}')
  raw = raw.sort_values(['mask_ratio', 'condition', 'example_id'])
  common.atomic_write_csv(output_dir / 'per_document_metrics.csv', raw)

  summary = common.aggregate_metrics(raw, ['mask_ratio', 'condition'])
  common.atomic_write_csv(output_dir / 'summary.csv', summary)
  paired_nll = common.paired_condition_differences(
    raw,
    group_columns=['mask_ratio'],
    comparisons=[
      (CORRECT, FIRST),
      (SHUFFLED, FIRST),
      (CORRECT, SHUFFLED),
    ],
    seed=args.seed,
    bootstrap_samples=args.bootstrap_samples)
  common.atomic_write_csv(output_dir / 'paired_nll_differences.csv', paired_nll)
  paired_accuracy = paired_accuracy_differences(
    raw, args.seed, args.bootstrap_samples)
  common.atomic_write_csv(
    output_dir / 'paired_accuracy_differences.csv', paired_accuracy)
  save_plot(
    summary, paired_accuracy, output_dir / 'same_state_recurrence.png')

  print('\nSame-state recurrence summary:', flush=True)
  print(summary.to_string(index=False), flush=True)
  print('\nPaired top-1 changes (percentage points):', flush=True)
  print(paired_accuracy.to_string(index=False), flush=True)
  print(
    '\nNLL deltas use condition - reference; negative means condition is '
    'better. Accuracy deltas use condition - reference; positive means '
    'condition is better.', flush=True)


def main() -> None:
  args = parse_args()
  validate_args(args)
  torch.set_float32_matmul_precision('high')
  device = torch.device(args.device)
  common.print_device_banner(device)

  metadata = {
    'evaluation': 'same_state_dcache_recurrence',
    'protocol': (
      'For each fixed teacher-forced corrupted state x_r: first evaluate '
      'without previous cache and save D=cache(x_r); then evaluate the exact '
      'same x_r with D; then evaluate the same x_r with D batch-rolled across '
      'documents. Score only positions masked in x_r.'),
    'examples': args.examples,
    'batch_size': args.batch_size,
    'seed': args.seed,
    'mask_ratios': args.mask_ratios,
    'bootstrap_samples': args.bootstrap_samples,
    'dcache_checkpoint': common.checkpoint_fingerprint(
      args.dcache_checkpoint),
    'data_dir': str(args.data_dir),
    'weights': 'EMA',
    'dcache_gate_enabled': args.dcache_gate_enabled,
    'shuffle': 'cyclic batch roll by one document',
  }
  output_dir = common.prepare_output(args.output_dir, metadata, args.force)
  config = common.compose_eval_config(
    args.batch_size,
    args.data_dir,
    recurrent=True,
    num_workers=args.num_workers,
    gate_enabled=args.dcache_gate_enabled)
  tokenizer = common.load_tokenizer(config)
  loader = common.load_validation_data(
    config, tokenizer, args.examples, args.batch_size, args.num_workers)
  run_evaluation(args, loader, tokenizer, output_dir, device)
  write_reports(args, output_dir)
  print(f'Wrote same-state recurrence evaluation to {output_dir}', flush=True)


if __name__ == '__main__':
  main()
