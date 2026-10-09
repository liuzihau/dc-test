#!/usr/bin/env python3
"""Plot completed atomic recurrence-audit parts, including partial running jobs.

This command is CPU only. Never reads an aggregate summary as per-document data.
Confidence intervals resample documents, keeping all mask seeds for a document
together. Shuffle intervals resample donor batches. Paired comparisons require
identical seed/document sets and mask counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DESCRIPTORS = ['family', 'protocol', 's_mask_ratio', 't_mask_ratio', 'repeat_step']
GROUPS = ['variant', 'condition'] + DESCRIPTORS
SAMPLE = ['seed', 'example_id']
METRICS = ['nll', 'top1_accuracy', 'top5_accuracy']
REQUIRED = GROUPS + SAMPLE + ['masked_tokens'] + METRICS
VARIANTS = ['vanilla', 'objective', 'dcache-v2', 'five-forward', 'two-forward']
LABELS = {
  'vanilla': 'Vanilla BD3',
  'objective': 'Objective-matched BD3',
  'dcache-v2': 'DCache v2',
  'five-forward': 'DCache + final, 5 forwards',
  'two-forward': 'DCache + final, 2 forwards',
}
COLORS = dict(zip(VARIANTS, ['#777777', '#d69216', '#268653', '#287cc0', '#b04d9a']))
CONDITION_LABELS = {
  'correct': 'Correct memory',
  'absent_dcache': 'D absent',
  'absent_final': 'Final absent',
  'absent_both': 'Both absent',
  'shuffle_dcache': 'D shuffled',
  'shuffle_final': 'Final shuffled',
  'shuffle_both_coherent': 'Both shuffled, same donor',
  'shuffle_both_independent': 'Both shuffled, different donors',
}


def arguments(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input-root', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--bootstrap-samples', type=int, default=2000)
  parser.add_argument('--bootstrap-seed', type=int, default=20260906)
  parser.add_argument('--strict-pairing', action='store_true',
                      help='Fail if any candidate comparison has incompatible samples.')
  parser.add_argument('--expected-variants', nargs='+', default=VARIANTS)
  parser.add_argument('--expected-seeds', nargs='+', type=int,
                      help='Planned mask seeds; missing runs keep the report partial.')
  args = parser.parse_args(argv)
  if args.bootstrap_samples < 2:
    parser.error('--bootstrap-samples must be at least 2')
  return args


def write_csv(frame, path):
  temporary = path.with_suffix(path.suffix + '.tmp')
  frame.to_csv(temporary, index=False)
  temporary.replace(path)


def load_parts(root):
  paths = sorted(root.glob('**/parts/*/batch_*.csv'))
  if not paths:
    raise ValueError(f'No completed parts/*/batch_*.csv under {root}')
  frames, coverage = [], []
  by_run = {}
  for path in paths:
    by_run.setdefault(path.parent.parent.parent, []).append(path)
  for run, run_paths in sorted(by_run.items()):
    run_frames = []
    for path in run_paths:
      frame = pd.read_csv(path)
      missing = set(REQUIRED) - set(frame)
      if missing:
        raise ValueError(f'{path}: missing raw columns {sorted(missing)}')
      if frame.empty:
        raise ValueError(f'Empty published part: {path}')
      run_frames.append(frame[REQUIRED])
    data = pd.concat(run_frames, ignore_index=True)
    if len(data['variant'].unique()) != 1 or len(data['seed'].unique()) != 1:
      raise ValueError(f'Expected one variant and seed in {run}')
    completed = False
    marker = next((run / name for name in ('completion.json', 'complete.json')
                   if (run / name).is_file()), None)
    if marker:
      info = json.loads(marker.read_text())
      completed = bool(info.get('complete', info.get('completed', True)))
      expected_rows = info.get('rows', info.get('raw_rows'))
      if expected_rows is not None and int(expected_rows) != len(data):
        completed = False
      if info.get('completed_parts', len(run_paths)) != len(run_paths):
        completed = False
    elif (run / 'per_document_metrics.csv').is_file():
      # Evaluator publishes the full raw file only after all parts validate.
      completed = len(pd.read_csv(run / 'per_document_metrics.csv',
                                  usecols=['example_id'])) == len(data)
    run_status = None
    if (run / 'STATUS.json').is_file():
      run_status = json.loads((run / 'STATUS.json').read_text()).get('status')
      if run_status != 'complete':
        completed = False
    coverage.append({
      'variant': data['variant'].iloc[0], 'seed': int(data['seed'].iloc[0]),
      'run_dir': str(run), 'parts': len(run_paths), 'rows': len(data),
      'documents': data['example_id'].nunique(), 'complete': completed,
      'run_status': run_status or 'unknown',
      'latest_part_mtime': max(path.stat().st_mtime for path in run_paths),
    })
    frames.append(data)
  raw = pd.concat(frames, ignore_index=True)
  if raw[REQUIRED].isna().any().any():
    raise ValueError('Raw results contain missing descriptor/sample/metric values')
  for name in ['s_mask_ratio', 't_mask_ratio']:
    raw[name] = raw[name].round(8)
  for name in SAMPLE + ['repeat_step', 'masked_tokens']:
    values = pd.to_numeric(raw[name], errors='raise')
    if not np.all(values == np.floor(values)):
      raise ValueError(f'{name} contains noninteger values')
    raw[name] = values.astype(np.int64)
  if (raw['masked_tokens'] <= 0).any():
    raise ValueError('Zero-mask rows cannot enter conditional masked-token metrics')
  if not np.isfinite(raw[METRICS].to_numpy(dtype=float)).all():
    raise ValueError('Nonfinite metrics found')
  if ((raw[['top1_accuracy', 'top5_accuracy']] < 0)
      | (raw[['top1_accuracy', 'top5_accuracy']] > 1)).any().any():
    raise ValueError('Accuracy must be recorded as fractions in [0, 1]')
  if raw.duplicated(GROUPS + SAMPLE).any():
    duplicate = raw[raw.duplicated(GROUPS + SAMPLE, keep=False)].iloc[0]
    raise ValueError(f'Duplicate raw observation across parts: {duplicate.to_dict()}')
  return raw, pd.DataFrame(coverage)


def validate_manifests(coverage):
  """Verify shared data/protocol before pooling and preserve checkpoint provenance."""
  invariants = [
    'evaluation', 'schema_version', 'dataset', 'split', 'data', 'length',
    'examples', 'batch_size', 'families', 'ratios', 'jumps', 'repeat_steps',
    'generated_steps', 'generated_jump', 'temperature', 'phases',
    'mask_protocol', 'protocols', 'sampling_rng', 'shuffles',
    'source_sha256', 'torch_version', 'weights',
  ]
  shared, all_present = None, True
  checkpoints = {}
  for index, row in coverage.iterrows():
    path = Path(row.run_dir) / 'manifest.json'
    coverage.loc[index, 'manifest_present'] = path.is_file()
    if not path.is_file():
      all_present = False
      continue
    manifest = json.loads(path.read_text())
    missing = set(invariants + ['variant', 'seed', 'checkpoint']) - set(manifest)
    if missing:
      raise ValueError(f'{path}: incomplete provenance, missing {sorted(missing)}')
    if manifest['variant'] != row.variant or int(manifest['seed']) != int(row.seed):
      raise ValueError(f'{path}: manifest variant/seed disagree with raw results')
    current = {key: manifest[key] for key in invariants}
    if shared is not None:
      incompatible = [key for key in invariants if current[key] != shared[key]]
      if incompatible:
        raise ValueError(f'{path}: incomparable evaluator manifests for {incompatible}; '
                         'use separate reports instead of combining these runs')
    shared = current
    checkpoint = manifest['checkpoint']
    identity = (checkpoint.get('sha256'), checkpoint.get('global_step'))
    if row.variant in checkpoints and checkpoints[row.variant] != identity:
      raise ValueError(f'Multiple checkpoints for variant {row.variant}; '
                       'mask seeds cannot pool different trained checkpoints')
    checkpoints[row.variant] = identity
    coverage.loc[index, 'manifest_fingerprint'] = manifest.get('fingerprint', '')
    coverage.loc[index, 'manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    coverage.loc[index, 'checkpoint_path'] = checkpoint.get('path', '')
    coverage.loc[index, 'checkpoint_sha256'] = checkpoint.get('sha256', '')
    coverage.loc[index, 'checkpoint_step'] = checkpoint.get('global_step', '')
    coverage.loc[index, 'batch_size'] = int(manifest['batch_size'])
    coverage.loc[index, 'dataset_fingerprint'] = hashlib.sha256(
      json.dumps(manifest['data'], sort_keys=True).encode()).hexdigest()
  batch_size = int(shared['batch_size']) if shared is not None and all_present else None
  return batch_size, all_present


def summarize(raw, group_columns=GROUPS):
  rows = []
  for keys, group in raw.groupby(group_columns, sort=True):
    weights = group.masked_tokens.to_numpy(dtype=float)
    row = dict(zip(group_columns, keys))
    row.update(documents=group.example_id.nunique(),
               seeds=group.seed.nunique(), observations=len(group),
               masked_tokens=int(weights.sum()))
    for metric in METRICS:
      name = 'conditional_nll' if metric == 'nll' else metric
      row[name] = float(np.average(group[metric], weights=weights))
    # A reconstruction diagnostic; not unconditional generative perplexity.
    row['conditional_ppl'] = float(np.exp(row['conditional_nll']))
    rows.append(row)
  return pd.DataFrame(rows)


class PairingError(ValueError):
  pass


def paired_document_bootstrap(condition, reference, samples, seed, cluster_size=1):
  """Return pooled effects; group shuffle donors by their evaluation batch."""
  if cluster_size < 1:
    raise ValueError('cluster_size must be at least 1')
  left = condition.set_index(SAMPLE).sort_index()
  right = reference.set_index(SAMPLE).sort_index()
  if not left.index.equals(right.index):
    only_left = len(left.index.difference(right.index))
    only_right = len(right.index.difference(left.index))
    raise PairingError(
      f'Different (seed, example_id) sets: {only_left} condition-only, '
      f'{only_right} reference-only; comparison omitted (no inner-join filtering).')
  if not np.array_equal(left.masked_tokens, right.masked_tokens):
    raise PairingError('Paired masked-token counts differ; comparison omitted.')
  weights = left.masked_tokens.to_numpy(dtype=np.float64)
  differences = left[METRICS].to_numpy() - right[METRICS].to_numpy()
  effects = pd.DataFrame(differences * weights[:, None], columns=METRICS)
  effects['masked_tokens'] = weights
  document_ids = left.index.get_level_values('example_id').to_numpy()
  effects['cluster_id'] = document_ids // cluster_size
  clustered = effects.groupby('cluster_id', sort=True).sum()
  numerator = clustered[METRICS].to_numpy(dtype=float)
  denominator = clustered.masked_tokens.to_numpy(dtype=float)
  point = numerator.sum(axis=0) / denominator.sum()
  count = len(clustered)
  if count < 2:
    lower = upper = np.full(len(METRICS), np.nan)
  else:
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((samples, len(METRICS)))
    chunk = max(1, min(128, 500_000 // count))
    for start in range(0, samples, chunk):
      end = min(start + chunk, samples)
      indices = rng.integers(0, count, size=(end - start, count))
      bootstrap[start:end] = (numerator[indices].sum(axis=1)
                              / denominator[indices].sum(axis=1)[:, None])
    lower, upper = np.quantile(bootstrap, [0.025, 0.975], axis=0)
  row = {'documents': len(np.unique(document_ids)), 'clusters': count,
         'cluster_unit': 'document' if cluster_size == 1 else 'evaluation_batch',
         'seeds': left.index.get_level_values('seed').nunique(),
         'observations': len(left), 'masked_tokens': int(weights.sum())}
  for index, metric in enumerate(METRICS):
    name, scale = ('nll', 1) if metric == 'nll' else (metric + '_pp', 100)
    row['delta_' + name] = float(point[index] * scale)
    row['ci95_low_' + name] = float(lower[index] * scale)
    row['ci95_high_' + name] = float(upper[index] * scale)
  return row


def canonical(frame):
  return frame[(frame.condition == 'correct')
               | (frame.variant.isin(['vanilla', 'objective'])
                  & (frame.condition == 'absent_both'))]


def comparisons(raw, samples, seed, batch_size=None):
  rows, issues = [], []

  def compare(left, right, metadata):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True,
                                       default=str).encode()).digest()
    pair_seed = (seed + int.from_bytes(digest[:4], 'big')) % (2 ** 32)
    try:
      shuffled = any('shuffle' in str(metadata.get(key, ''))
                     for key in ['condition', 'reference_condition'])
      if shuffled and batch_size is None:
        raise PairingError('Shuffle donor batch size is unknown; cannot compute valid '
                           'batch-cluster intervals. Provide evaluator manifest.')
      estimates = paired_document_bootstrap(
        left, right, samples, pair_seed, cluster_size=batch_size if shuffled else 1)
      rows.append(dict(metadata, **estimates))
    except PairingError as error:
      issues.append(dict(metadata, issue=str(error)))

  for keys, group in raw.groupby(['variant'] + DESCRIPTORS, sort=True):
    metadata = dict(zip(['variant'] + DESCRIPTORS, keys))
    correct = group[group.condition == 'correct']
    if correct.empty:
      continue
    for condition, intervention in group.groupby('condition', sort=True):
      if condition != 'correct':
        compare(correct, intervention, dict(
          metadata, contrast_type='condition', condition='correct',
          reference_variant=metadata['variant'], reference_condition=condition))

  primary = canonical(raw)
  for keys, group in primary.groupby(DESCRIPTORS, sort=True):
    metadata = dict(zip(DESCRIPTORS, keys))
    available = [variant for variant in VARIANTS if variant in set(group.variant)]
    for index, variant in enumerate(available):
      for reference in available[:index]:
        left, right = group[group.variant == variant], group[group.variant == reference]
        compare(left, right, dict(
          metadata, contrast_type='model', variant=variant,
          condition=left.condition.iloc[0], reference_variant=reference,
          reference_condition=right.condition.iloc[0]))

  warm_keys = ['variant', 'condition', 'family', 's_mask_ratio',
               't_mask_ratio', 'repeat_step']
  for keys, group in primary.groupby(warm_keys, sort=True):
    warm = group[group.protocol == 'full_warmup']
    cold = group[group.protocol == 'no_warmup']
    if warm.empty or cold.empty:
      continue
    metadata = dict(zip(warm_keys, keys))
    compare(warm, cold, dict(metadata, contrast_type='warmup',
                            protocol='full_warmup', reference_protocol='no_warmup',
                            reference_variant=metadata['variant'],
                            reference_condition=metadata['condition']))

  same = raw[raw.protocol == 'same_state']
  repeat_keys = ['variant', 'condition', 'family', 'protocol',
                 's_mask_ratio', 't_mask_ratio']
  for keys, group in same.groupby(repeat_keys, sort=True):
    first = group[group.repeat_step == 1]
    if first.empty:
      continue
    metadata = dict(zip(repeat_keys, keys))
    for count, later in group.groupby('repeat_step', sort=True):
      if count > 1:
        compare(later, first, dict(metadata, contrast_type='repeat', repeat_step=count,
                                  reference_repeat_step=1,
                                  reference_variant=metadata['variant'],
                                  reference_condition=metadata['condition']))
  return pd.DataFrame(rows), pd.DataFrame(issues)


def build_plots(summary, paired, output, status):
  # Configure before importing matplotlib; caches stay on the project filesystem.
  cache = output / '.matplotlib'
  cache.mkdir(exist_ok=True)
  os.environ.setdefault('MPLCONFIGDIR', str(cache.resolve()))
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})
  saved = []

  def finish(fig, axes, title, name, xlabel):
    for axis in np.asarray(axes).flat:
      axis.set_xlabel(xlabel)
      axis.grid(alpha=0.22)
    handles, labels = [], []
    for axis in np.asarray(axes).flat:
      h, lab = axis.get_legend_handles_labels()
      for handle, label in zip(h, lab):
        if label not in labels:
          handles.append(handle)
          labels.append(label)
    fig.suptitle(f'{title}\n{status}', fontsize=12)
    if handles:
      fig.legend(handles, labels, loc='lower center',
                 ncol=min(4, len(labels)), fontsize=8)
    legend_rows = max(1, int(np.ceil(len(labels) / 4)))
    bottom = min(0.20, 0.04 + 0.035 * legend_rows)
    fig.tight_layout(rect=(0, bottom, 1, 0.93))
    fig.savefig(output / name, dpi=170, bbox_inches='tight')
    plt.close(fig)
    saved.append(name)

  def lines(axes, data, xkey, factor=100, conditions=False):
    identity = 'condition' if conditions else 'variant'
    for name, group in data.groupby(identity, sort=False):
      group = group.sort_values(xkey)
      label = CONDITION_LABELS.get(name, name) if conditions else LABELS.get(name, name)
      style = {'marker': 'o', 'linewidth': 1.8, 'markersize': 4, 'label': label}
      if not conditions:
        style['color'] = COLORS.get(name, '#222222')
      axes[0].plot(group[xkey] * factor, group.top1_accuracy * 100, **style)
      axes[1].plot(group[xkey] * factor, group.conditional_nll, **style)
    axes[0].set_ylabel('Masked-token top-1 accuracy (%) ↑')
    axes[1].set_ylabel('Conditional masked-token NLL ↓')

  primary = canonical(summary)
  transitions = primary[primary.protocol.isin(['no_warmup', 'full_warmup'])].copy()
  transitions['jump'] = (transitions.s_mask_ratio - transitions.t_mask_ratio).round(8)
  for protocol, data in transitions.groupby('protocol', sort=True):
    jumps = sorted(data.jump.unique())
    fig, axes = plt.subplots(2, len(jumps), figsize=(4.0 * len(jumps), 7), squeeze=False)
    for index, jump in enumerate(jumps):
      lines(axes[:, index], data[data.jump == jump], 't_mask_ratio')
      axes[0, index].set_title(f'Mask-ratio jump = {jump * 100:g} pp')
    finish(fig, axes, f'Teacher-forced transition quality: {protocol}',
           f'transition_quality_{protocol}.png', 'Final mask ratio (%)')

  same = summary[summary.protocol == 'same_state']
  if not same.empty:
    ratios = sorted(same.t_mask_ratio.unique())
    fig, axes = plt.subplots(2, len(ratios), figsize=(4.8 * len(ratios), 7), squeeze=False)
    for index, ratio in enumerate(ratios):
      lines(axes[:, index], canonical(same[same.t_mask_ratio == ratio]), 'repeat_step', 1)
      axes[0, index].set_title(f'Unchanged input: {ratio * 100:g}% masked')
      for axis in axes[:, index]:
        axis.set_xticks(sorted(same.repeat_step.unique()))
    finish(fig, axes, 'Same-input recurrent computation',
           'same_state_recurrence.png', 'Forward count (input stays unchanged)')
    for variant, data in same.groupby('variant', sort=True):
      if data.condition.nunique() < 2:
        continue
      fig, axes = plt.subplots(2, len(ratios), figsize=(5 * len(ratios), 7.5), squeeze=False)
      for index, ratio in enumerate(ratios):
        lines(axes[:, index], data[data.t_mask_ratio == ratio], 'repeat_step', 1, True)
        axes[0, index].set_title(f'{ratio * 100:g}% masked')
        for axis in axes[:, index]:
          axis.set_xticks(sorted(data.repeat_step.unique()))
      finish(fig, axes, f'Same-input memory interventions: {LABELS.get(variant, variant)}',
             f'same_state_interventions_{variant}.png', 'Forward count')

  if not paired.empty:
    channels = paired[(paired.contrast_type == 'condition')
                      & paired.protocol.isin(['no_warmup', 'full_warmup'])].copy()
    channels['jump'] = (channels.s_mask_ratio - channels.t_mask_ratio).round(8)
    for (variant, protocol), data in channels.groupby(['variant', 'protocol'], sort=True):
      jumps = sorted(data.jump.unique())
      fig, axes = plt.subplots(2, len(jumps), figsize=(4.3 * len(jumps), 8), squeeze=False)
      for column, jump in enumerate(jumps):
        for condition, group in data[data.jump == jump].groupby('reference_condition'):
          group = group.sort_values('t_mask_ratio')
          x = group.t_mask_ratio * 100
          label = CONDITION_LABELS.get(condition, condition)
          for row, (metric, sign) in enumerate([('top1_accuracy_pp', 1), ('nll', -1)]):
            y = sign * group['delta_' + metric]
            low = group['ci95_low_' + metric] if sign == 1 else -group['ci95_high_' + metric]
            high = group['ci95_high_' + metric] if sign == 1 else -group['ci95_low_' + metric]
            line = axes[row, column].plot(x, y, marker='o', label=label)[0]
            axes[row, column].fill_between(x, low, high, color=line.get_color(), alpha=0.08)
        axes[0, column].set_title(f'Jump = {jump * 100:g} pp')
        axes[0, column].set_ylabel('Accuracy benefit of correct memory (pp)')
        axes[1, column].set_ylabel('NLL benefit of correct memory')
        for axis in axes[:, column]:
          axis.axhline(0, color='black', linewidth=0.8)
      finish(fig, axes, f'Memory benefit: {LABELS.get(variant, variant)}, {protocol}',
             f'channel_contribution_{variant}_{protocol}.png', 'Final mask ratio (%)')

    warm = paired[paired.contrast_type == 'warmup'].copy()
    if not warm.empty:
      warm['jump'] = (warm.s_mask_ratio - warm.t_mask_ratio).round(8)
      jumps = sorted(warm.jump.unique())
      fig, axes = plt.subplots(2, len(jumps), figsize=(4.3 * len(jumps), 7), squeeze=False)
      for column, jump in enumerate(jumps):
        for variant, group in warm[warm.jump == jump].groupby('variant', sort=False):
          group = group.sort_values('t_mask_ratio')
          x = group.t_mask_ratio * 100
          for row, (metric, sign) in enumerate([('top1_accuracy_pp', 1), ('nll', -1)]):
            axes[row, column].plot(x, sign * group['delta_' + metric], marker='o',
                                   label=LABELS.get(variant, variant), color=COLORS.get(variant))
          axes[0, column].set_title(f'Jump = {jump * 100:g} pp')
        axes[0, column].set_ylabel('Warmup accuracy benefit (pp)')
        axes[1, column].set_ylabel('Warmup NLL benefit')
        for axis in axes[:, column]:
          axis.axhline(0, color='black', linewidth=0.8)
      finish(fig, axes, 'Effect of an extra full-mask warmup (warmup uses more compute)',
             'warmup_sensitivity.png', 'Final mask ratio (%)')

  generated = summary[summary.protocol == 'generated']
  if not generated.empty:
    initial = sorted(generated.s_mask_ratio.unique())
    fig, axes = plt.subplots(2, len(initial), figsize=(5 * len(initial), 7), squeeze=False)
    for column, ratio in enumerate(initial):
      lines(axes[:, column], canonical(generated[generated.s_mask_ratio == ratio]),
            'repeat_step', 1)
      axes[0, column].set_title(f'Initial mask ratio = {ratio * 100:g}%')
      for axis in axes[:, column]:
        axis.set_xticks(sorted(generated.repeat_step.unique()))
    finish(fig, axes, 'Own-token trajectories: remaining-mask reconstruction diagnostic',
           'generated_trajectory.png', 'Generated reveal step (not free-generation PPL)')
    for variant, data in generated.groupby('variant', sort=True):
      if data.condition.nunique() < 2:
        continue
      fig, axes = plt.subplots(2, len(initial), figsize=(5 * len(initial), 7.5), squeeze=False)
      for column, ratio in enumerate(initial):
        lines(axes[:, column], data[data.s_mask_ratio == ratio], 'repeat_step', 1, True)
        axes[0, column].set_title(f'Initial mask ratio = {ratio * 100:g}%')
        for axis in axes[:, column]:
          axis.set_xticks(sorted(data.repeat_step.unique()))
      finish(fig, axes,
             f'Own-token memory interventions: {LABELS.get(variant, variant)}',
             f'generated_interventions_{variant}.png', 'Generated reveal step')
  return saved


def main(argv=None):
  args = arguments(argv)
  root = args.input_root.resolve()
  output = args.output_dir.resolve()
  if not root.is_dir():
    raise ValueError(f'Input root does not exist: {root}')
  output.mkdir(parents=True, exist_ok=True)
  raw, coverage = load_parts(root)
  batch_size, manifests_verified = validate_manifests(coverage)
  summary = summarize(raw)
  per_seed = summarize(raw, ['seed'] + GROUPS)
  paired, issues = comparisons(raw, args.bootstrap_samples, args.bootstrap_seed, batch_size)
  missing_variants = sorted(set(args.expected_variants) - set(raw.variant))
  expected_seeds = (args.expected_seeds if args.expected_seeds is not None
                    else sorted(raw.seed.unique()))
  present_runs = set(zip(coverage.variant, coverage.seed))
  missing_runs = [(variant, seed) for variant in args.expected_variants
                  for seed in expected_seeds if (variant, seed) not in present_runs]
  partial = bool(missing_variants or missing_runs
                 or not coverage.complete.all() or not issues.empty or not manifests_verified)
  status = ('PARTIAL snapshot: completed parts only' if partial
            else 'Completed evaluation; cluster-bootstrap 95% intervals')
  write_csv(summary, output / 'pooled_summary.csv')
  write_csv(per_seed, output / 'per_seed_summary.csv')
  write_csv(coverage, output / 'coverage.csv')
  write_csv(paired, output / 'paired_deltas.csv')
  write_csv(issues if not issues.empty else pd.DataFrame(columns=['issue']),
            output / 'pairing_issues.csv')
  if args.strict_pairing and not issues.empty:
    raise PairingError(f'{len(issues)} incompatible comparisons; see {output / "pairing_issues.csv"}')
  figures = build_plots(summary, paired, output, status)
  report = [
    '# Recurrence audit', '', status, '',
    f'Input: `{root}`', '',
    f'- Loaded {len(raw):,} observations from {int(coverage.parts.sum())} completed parts.',
    f'- Variants present: {", ".join(sorted(raw.variant.unique()))}.',
    f'- Missing expected variants: {", ".join(missing_variants) or "none"}.',
    f'- Missing planned variant/seed runs: {missing_runs or "none"}.',
    f'- Shared dataset/protocol provenance verified: {manifests_verified}.',
    f'- Omitted incompatible paired comparisons: {len(issues)} (see `pairing_issues.csv`).',
    '', '## Interpretation', '',
    'NLL and accuracy are pooled by the number of masked tokens. `conditional_ppl` is '
    'exp(conditional NLL), not unconditional generative perplexity. Generated-trajectory '
    'scores measure remaining-mask reconstruction under model-generated context.', '',
    'Paired deltas use condition minus reference: negative NLL and positive accuracy '
    'indicate improvement. Accuracy deltas are percentage points. Contribution figures '
    'flip the NLL sign so a positive value means correct memory helps.', '',
    'Comparisons require exactly matching (seed, example_id) observations and masked-token '
    'counts; incomplete mismatched sets are omitted, never silently inner-joined. The evaluator '
    'must also use identical document order and mask schedules, which cannot be proved from '
    'aggregate token counts alone. Shared data fingerprints, protocol definitions, evaluation '
    'code hashes and sampling settings are checked when manifests are available; missing '
    'manifests keep the report partial. Dataset Arrow fingerprints use size/mtime, plus '
    'SHA256 for metadata JSON, rather than hashing all token data.', '',
    f'The {args.bootstrap_samples} bootstrap draws resample example_id clusters and keep '
    'all mask seeds of each document together. For shuffled-memory contrasts the resampled '
    'unit is a whole evaluation batch, because its documents share memory donors. '
    '`cluster_unit` and `clusters` record the effective unit/count in the paired table. '
    'Here `example_id` identifies a packed 1024-token validation block, not necessarily '
    'an independent original source document. Adjacent blocks or donor batches can share '
    'a source or topic; this bootstrap cannot account for unknown original-document grouping. '
    'These intervals describe evaluation-document '
    'uncertainty for fixed checkpoints, not training-seed uncertainty. Intervals are exploratory '
    'and are not adjusted for multiple comparisons. No statistical-significance claim is made '
    'from a partial run.', '',
    'Each own-token condition follows its own prediction history. Later trajectory differences '
    'therefore include cumulative effects, rather than a single-step cache intervention. '
    'Repeated cyclic donor shuffles can eventually return information through donor cycles. '
    'The schema name `shuffle_both_independent` means different wrong donors (fixed rolls '
    'one and two), not statistically independent random draws. '
    'Full-mask warmup adds a forward. Same-input indices are nominal recurrence slots: '
    'the evaluator reuses deterministic first-pass predictions for the vanilla and '
    'objective-matched no-memory baselines, so their actual forward count is one. '
    'The repeated rows do not measure extra baseline compute, and these plots alone '
    'do not establish an efficiency improvement.', '',
    '## Figures', '',
  ]
  report.extend(f'- [{name}]({name})' for name in figures)
  report += ['', '## Tables', '',
             '- `pooled_summary.csv`: per-token pooled metrics across mask seeds.',
             '- `per_seed_summary.csv`: each mask seed separately.',
             '- `paired_deltas.csv`: model, condition, warmup, and recurrence contrasts.',
             '- `coverage.csv`: completion, counts, checkpoint paths/hashes, manifest and data fingerprints.',
             '- `pairing_issues.csv`: explicit reasons for omitted contrasts.', '']
  (output / 'REPORT.md').write_text('\n'.join(report))
  print(f'{status}\nWrote {len(figures)} figures and audit tables to {output}')
  if not issues.empty:
    print(f'WARNING: {len(issues)} unpaired comparisons omitted; see pairing_issues.csv')


if __name__ == '__main__':
  main()
