#!/usr/bin/env python3
"""CPU-only paired fixed-probe anchor-schedule report, including partial runs.

Read atomic raw parts, never per-run aggregate summaries. Comparisons require
exact seed/block/token-count matches; a block bootstrap retains every mask seed
of each packed validation block. Positive plotted gains always mean improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
  from .plot_recurrence_audit import (
    COLORS, LABELS, METRICS, PairingError, paired_document_bootstrap, write_csv)
except ImportError:
  from plot_recurrence_audit import (
    COLORS, LABELS, METRICS, PairingError, paired_document_bootstrap, write_csv)


VARIANTS = ['five-forward', 'two-forward', 'dcache-v2', 'objective', 'vanilla']
BASELINES = {'objective', 'vanilla'}
DESCRIPTORS = ['source', 'initial_mask_ratio', 'round']
GROUPS = ['variant', 'source', 'schedule', 'condition', 'initial_mask_ratio', 'round']
SAMPLE = ['seed', 'example_id']
REQUIRED = GROUPS + SAMPLE + ['masked_tokens', 'remaining_mask_tokens'] + METRICS
CONTRAST_COLUMNS = [
  'contrast_type', 'variant', *DESCRIPTORS, 'schedule', 'condition',
  'reference_schedule', 'reference_condition', 'documents', 'clusters',
  'cluster_unit', 'seeds', 'observations', 'masked_tokens',
  *[prefix + metric for metric in ['nll', 'top1_accuracy_pp', 'top5_accuracy_pp']
    for prefix in ['delta_', 'ci95_low_', 'ci95_high_']],
]


def arguments(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input-root', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--bootstrap-samples', type=int, default=2000)
  parser.add_argument('--bootstrap-seed', type=int, default=20260907)
  parser.add_argument('--expected-variants', nargs='+', choices=VARIANTS, default=VARIANTS)
  parser.add_argument('--expected-seeds', nargs='+', type=int)
  parser.add_argument('--strict-pairing', action='store_true')
  args = parser.parse_args(argv)
  if args.bootstrap_samples < 2:
    parser.error('--bootstrap-samples must be at least 2')
  if args.expected_seeds and len(set(args.expected_seeds)) != len(args.expected_seeds):
    parser.error('--expected-seeds must be distinct')
  return args


def load_parts(root):
  """Snapshot only published part paths; subsequent publications await next report."""
  paths = sorted(Path(root).glob('**/parts/*/batch_*.csv'))
  if not paths:
    raise ValueError(f'No completed parts/*/batch_*.csv under {root}')
  by_run = {}
  for path in paths:
    by_run.setdefault(path.parent.parent.parent, []).append(path)
  frames, coverage = [], []
  for run, run_paths in sorted(by_run.items()):
    parts = []
    for path in run_paths:
      data = pd.read_csv(path)
      missing = set(REQUIRED) - set(data)
      if missing or data.empty:
        raise ValueError(f'{path}: empty part or missing raw columns {sorted(missing)}')
      parts.append(data[REQUIRED])
    data = pd.concat(parts, ignore_index=True)
    if data.variant.nunique() != 1 or data.seed.nunique() != 1:
      raise ValueError(f'Expected one variant and seed in {run}')
    status_path, marker = run / 'STATUS.json', run / 'completion.json'
    status = json.loads(status_path.read_text()).get('status') if status_path.is_file() else None
    completed = False
    if marker.is_file():
      info = json.loads(marker.read_text())
      completed = bool(info.get('complete', info.get('completed', info.get('status') == 'complete')))
      completed &= int(info.get('completed_parts', -1)) == len(run_paths)
      completed &= int(info.get('expected_parts', -1)) == len(run_paths)
      expected_rows = info.get('rows', info.get('raw_rows'))
      completed &= expected_rows is None or int(expected_rows) == len(data)
    if status is not None and status != 'complete':
      completed = False
    coverage.append({
      'variant': data.variant.iloc[0], 'seed': int(data.seed.iloc[0]),
      'run_dir': str(run), 'parts': len(run_paths), 'rows': len(data),
      'documents': data.example_id.nunique(), 'complete': completed,
      'run_status': status or 'unknown',
      'latest_part_mtime': max(path.stat().st_mtime for path in run_paths),
    })
    frames.append(data)
  raw = pd.concat(frames, ignore_index=True)
  if raw[REQUIRED].isna().any().any():
    raise ValueError('Raw data contain missing descriptors, samples, or metrics')
  raw['initial_mask_ratio'] = pd.to_numeric(raw.initial_mask_ratio).round(8)
  if not raw.initial_mask_ratio.between(0, 1, inclusive='right').all():
    raise ValueError('Initial mask ratios must be in (0, 1]')
  for name in SAMPLE + ['round', 'masked_tokens', 'remaining_mask_tokens']:
    values = pd.to_numeric(raw[name], errors='raise')
    if not np.isfinite(values).all() or not np.all(values == np.floor(values)):
      raise ValueError(f'{name} contains noninteger/nonfinite values')
    raw[name] = values.astype(np.int64)
  if (raw.masked_tokens <= 0).any() or (raw.remaining_mask_tokens < raw.masked_tokens).any():
    raise ValueError('Probe counts must be positive and no larger than remaining mask counts')
  if (raw['round'] < 0).any() or (raw.example_id < 0).any():
    raise ValueError('Round and example_id must be nonnegative')
  if not np.isfinite(raw[METRICS].to_numpy(dtype=float)).all():
    raise ValueError('Nonfinite metrics found')
  if (raw.nll < 0).any() or not raw.top1_accuracy.between(0, 1).all() or not raw.top5_accuracy.between(0, 1).all():
    raise ValueError('NLL must be nonnegative; accuracies must be fractions in [0, 1]')
  for name, allowed in [('variant', VARIANTS), ('source', ['teacher', 'model']),
                        ('schedule', ['random', 'strided']),
                        ('condition', ['correct', 'absent_dcache'])]:
    if not raw[name].isin(allowed).all():
      raise ValueError(f'Unknown {name}: {sorted(set(raw[name]) - set(allowed))}')
  if ((raw.variant.isin(BASELINES)) & (raw.condition != 'absent_dcache')).any():
    raise ValueError('Baselines must use only the absent_dcache condition')
  if raw.duplicated(GROUPS + SAMPLE).any():
    raise ValueError('Duplicate raw observation across published parts')
  if pd.DataFrame(coverage).duplicated(['variant', 'seed']).any():
    raise ValueError('Multiple run directories for one variant/seed')
  return raw, pd.DataFrame(coverage)


def validate_manifests(coverage):
  """Reject incompatible protocols/checkpoints; missing provenance stays partial."""
  invariants = [
    'evaluation', 'schema_version', 'data', 'examples', 'batch_size',
    'initial_mask_ratios', 'sources', 'probe_count', 'stride', 'score_rounds',
    'source_sha256', 'mask_protocol', 'probe_protocol', 'schedule_protocol', 'sampling_rng',
    'sigma_protocol', 'teacher_protocol', 'model_protocol', 'memory_protocol',
    'baseline_protocol', 'interpretation', 'forwards_per_trajectory', 'schedules',
    'expected_step', 'weights', 'dataset', 'split', 'length', 'torch_version', 'phases',
  ]
  shared, all_present, checkpoints = None, True, {}
  for index, row in coverage.iterrows():
    path = Path(row.run_dir) / 'manifest.json'
    coverage.loc[index, 'manifest_present'] = path.is_file()
    if not path.is_file():
      all_present = False
      continue
    manifest = json.loads(path.read_text())
    missing = set(invariants + ['variant', 'seed', 'checkpoint', 'conditions']) - set(manifest)
    if missing:
      raise ValueError(f'{path}: missing provenance fields {sorted(missing)}')
    if manifest['evaluation'] != 'anchor_schedule' or manifest['schema_version'] != 1:
      raise ValueError(f'{path}: unsupported anchor schedule protocol/schema')
    if manifest['variant'] != row.variant or int(manifest['seed']) != row.seed:
      raise ValueError(f'{path}: manifest variant/seed disagree with raw data')
    expected_conditions = {'absent_dcache'} if row.variant in BASELINES else {'correct', 'absent_dcache'}
    if set(manifest['conditions']) != expected_conditions:
      raise ValueError(f'{path}: unexpected condition arms')
    current = {key: manifest[key] for key in invariants}
    if shared is not None:
      different = [key for key in current if current[key] != shared[key]]
      if different:
        raise ValueError(f'{path}: incomparable manifests for {different}; use separate reports')
    shared = current
    checkpoint = manifest['checkpoint']
    identity = (checkpoint.get('sha256'), checkpoint.get('global_step'))
    if not identity[0] or identity[1] != 5000 or manifest['expected_step'] != 5000:
      raise ValueError(f'{path}: expected a hashed global-step-5000 checkpoint')
    if row.variant in checkpoints and checkpoints[row.variant] != identity:
      raise ValueError(f'Multiple source checkpoints for variant {row.variant}')
    checkpoints[row.variant] = identity
    coverage.loc[index, 'manifest_fingerprint'] = manifest.get('fingerprint', '')
    coverage.loc[index, 'manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    coverage.loc[index, 'checkpoint_path'] = checkpoint.get('path', '')
    coverage.loc[index, 'checkpoint_sha256'] = checkpoint['sha256']
    coverage.loc[index, 'checkpoint_step'] = checkpoint['global_step']
    coverage.loc[index, 'dataset_fingerprint'] = hashlib.sha256(
      json.dumps(manifest['data'], sort_keys=True).encode()).hexdigest()
    expected_parts = ((int(manifest['examples']) + int(manifest['batch_size']) - 1)
                      // int(manifest['batch_size'])
                      * len(manifest['sources']) * len(manifest['initial_mask_ratios']))
    expected_rows = (int(manifest['examples']) * len(manifest['sources'])
                     * len(manifest['initial_mask_ratios']) * len(manifest['score_rounds'])
                     * 2 * len(expected_conditions))
    coverage.loc[index, 'expected_parts'] = expected_parts
    coverage.loc[index, 'expected_rows'] = expected_rows
    if row.parts != expected_parts or row.rows != expected_rows or row.documents != manifest['examples']:
      coverage.loc[index, 'complete'] = False
  return shared, all_present


def summarize(raw, group_columns=GROUPS):
  rows = []
  for keys, group in raw.groupby(group_columns, sort=True):
    weights = group.masked_tokens.to_numpy(dtype=float)
    row = dict(zip(group_columns, keys))
    row.update(documents=group.example_id.nunique(), seeds=group.seed.nunique(),
               observations=len(group), masked_tokens=int(weights.sum()),
               mean_remaining_mask_tokens=float(group.remaining_mask_tokens.mean()))
    for metric in METRICS:
      row['conditional_nll' if metric == 'nll' else metric] = float(np.average(group[metric], weights=weights))
    rows.append(row)
  return pd.DataFrame(rows)


def paired_linear_contrast(arms, coefficients, samples, seed, expected_seeds=None):
  """Bootstrap any paired linear contrast jointly, including four-arm interactions."""
  if len(arms) != len(coefficients) or not arms or not np.isclose(sum(coefficients), 0):
    raise ValueError('A contrast needs matching arms/coefficients whose sum is zero')
  indexed = []
  for arm in arms:
    if arm.empty:
      raise PairingError('A required comparison arm is missing; no inner-join filtering')
    if arm.duplicated(SAMPLE).any():
      raise PairingError('Duplicate (seed, example_id) in a comparison arm')
    indexed.append(arm.set_index(SAMPLE).sort_index())
  reference = indexed[0]
  for arm in indexed[1:]:
    if not arm.index.equals(reference.index):
      raise PairingError('Different (seed, example_id) sets; comparison omitted (no inner join)')
    for name in ['masked_tokens', 'remaining_mask_tokens']:
      if not np.array_equal(arm[name], reference[name]):
        raise PairingError(f'Paired {name} counts differ; comparison omitted')
  seeds = sorted(reference.index.get_level_values('seed').unique())
  if expected_seeds is not None and seeds != sorted(expected_seeds):
    raise PairingError(f'Incomplete planned seed vector: observed {seeds}, expected {sorted(expected_seeds)}')
  ids = sorted(reference.index.get_level_values('example_id').unique())
  complete_index = pd.MultiIndex.from_product([seeds, ids], names=SAMPLE)
  if not reference.index.equals(complete_index):
    raise PairingError('Incomplete seed/block grid; every resampled block must retain all seeds')
  # The existing bootstrap resamples example_id, aggregating all seeds first.
  # Constructing the four-arm difference before resampling preserves covariance.
  left = reference.reset_index().copy()
  right = left.copy()
  left[METRICS] = sum(coefficient * arm[METRICS].to_numpy(dtype=float)
                      for coefficient, arm in zip(coefficients, indexed))
  right[METRICS] = 0.0
  result = paired_document_bootstrap(left, right, samples, seed)
  result['cluster_unit'] = 'packed_validation_block_all_mask_seeds'
  return result


def comparisons(raw, samples, seed, expected_seeds=None):
  rows, issues = [], []

  def compare(arms, coefficients, metadata):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, default=str).encode()).digest()
    pair_seed = (seed + int.from_bytes(digest[:4], 'big')) % (2 ** 32)
    try:
      estimates = paired_linear_contrast(arms, coefficients, samples, pair_seed, expected_seeds)
      rows.append(dict(metadata, **estimates))
    except PairingError as error:
      issues.append(dict(metadata, issue=str(error)))

  keys = ['variant'] + DESCRIPTORS
  for values, group in raw.groupby(keys, sort=True):
    metadata = dict(zip(keys, values))
    conditions = ['absent_dcache'] if metadata['variant'] in BASELINES else ['correct', 'absent_dcache']

    def arm(schedule, condition):
      return group[(group.schedule == schedule) & (group.condition == condition)]

    for condition in conditions:
      compare([arm('strided', condition), arm('random', condition)], [1, -1], dict(
        metadata, contrast_type='schedule', condition=condition, reference_condition=condition,
        schedule='strided', reference_schedule='random'))
    if metadata['variant'] not in BASELINES:
      for schedule in ['strided', 'random']:
        compare([arm(schedule, 'correct'), arm(schedule, 'absent_dcache')], [1, -1], dict(
          metadata, contrast_type='dcache', schedule=schedule, reference_schedule=schedule,
          condition='correct', reference_condition='absent_dcache'))
      compare([arm('strided', 'correct'), arm('strided', 'absent_dcache'),
               arm('random', 'correct'), arm('random', 'absent_dcache')], [1, -1, -1, 1], dict(
        metadata, contrast_type='interaction', schedule='strided', reference_schedule='random',
        condition='correct', reference_condition='absent_dcache'))
  return pd.DataFrame(rows, columns=CONTRAST_COLUMNS), pd.DataFrame(issues)


def validate_raw_protocol(raw, shared):
  """Verify observed cells against manifest protocol, including fixed probe counts."""
  if shared is None:
    return
  for name, allowed in [('source', shared['sources']), ('round', shared['score_rounds']),
                        ('initial_mask_ratio', np.round(shared['initial_mask_ratios'], 8))]:
    if not raw[name].isin(allowed).all():
      raise ValueError(f'Raw {name} values disagree with evaluator manifests')
  if not (raw.masked_tokens == shared['probe_count']).all():
    raise ValueError('Raw probe counts disagree with fixed-probe manifest')
  if (raw.example_id >= int(shared['examples'])).any():
    raise ValueError('Raw example_id exceeds planned validation-block range')
  final = raw[raw['round'] == int(shared['stride'])]
  if not final.remaining_mask_tokens.eq(shared['probe_count']).all():
    raise ValueError('Final-round remaining mask does not equal the fixed probes')
  probe_variability = raw.groupby(SAMPLE + ['initial_mask_ratio']).masked_tokens.nunique()
  if (probe_variability != 1).any():
    raise ValueError('Probe count changes across schedules, conditions, sources or rounds')
  remaining_variability = raw.groupby(SAMPLE + ['initial_mask_ratio', 'round']).remaining_mask_tokens.nunique()
  if (remaining_variability != 1).any():
    raise ValueError('Remaining mask counts differ between matched reveal schedules/arms')


def endpoint_table(summary, final_round=32):
  endpoint = summary[summary['round'] == final_round].copy()
  if endpoint.empty:
    return endpoint
  endpoint['descriptive_nll_rank'] = endpoint.groupby(
    ['source', 'initial_mask_ratio', 'schedule', 'condition'])['conditional_nll'].rank(method='min')
  return endpoint.sort_values(['source', 'initial_mask_ratio', 'schedule', 'condition', 'conditional_nll'])


def build_plots(summary, paired, output, status):
  cache = output / '.matplotlib'
  cache.mkdir(exist_ok=True)
  os.environ.setdefault('MPLCONFIGDIR', str(cache.resolve()))
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})
  saved = []

  def finish(fig, axes, title, name):
    handles, labels = [], []
    for axis in np.asarray(axes).flat:
      axis.grid(alpha=0.2)
      axis.set_xlabel('Reveal round (round r uses r + 1 forwards)')
      axis.set_xticks([0, 4, 8, 16, 24, 32])
      for handle, label in zip(*axis.get_legend_handles_labels()):
        if label not in labels:
          handles.append(handle)
          labels.append(label)
    fig.suptitle(f'{title}\n{status}', fontsize=12)
    if handles:
      fig.legend(handles, labels, loc='lower center', ncol=min(4, len(labels)), fontsize=8)
    bottom = min(.23, .035 + .028 * max(1, int(np.ceil(len(labels) / 4))))
    fig.tight_layout(rect=(0, bottom, 1, .93))
    fig.savefig(output / name, dpi=160, bbox_inches='tight')
    plt.close(fig)
    saved.append(name)

  for (source, ratio), data in summary.groupby(['source', 'initial_mask_ratio'], sort=True):
    variants = [variant for variant in VARIANTS if variant in set(data.variant)]
    fig, axes = plt.subplots(2, len(variants), figsize=(4 * len(variants), 7), squeeze=False)
    for column, variant in enumerate(variants):
      for (schedule, condition), group in data[data.variant == variant].groupby(['schedule', 'condition']):
        group = group.sort_values('round')
        style = dict(color='#287cc0' if schedule == 'strided' else '#d69216',
                     linestyle='-' if condition == 'correct' else '--', marker='o', markersize=3,
                     label=f'{schedule}; ' + ('D retained' if condition == 'correct' else 'D absent'))
        axes[0, column].plot(group['round'], group.top1_accuracy * 100, **style)
        axes[1, column].plot(group['round'], group.conditional_nll, **style)
      axes[0, column].set_title(LABELS.get(variant, variant))
      axes[0, column].set_ylabel('Fixed-probe top-1 accuracy (%) ↑')
      axes[1, column].set_ylabel('Conditional fixed-probe NLL ↓')
    finish(fig, axes, f'{source.title()} reveals; initial mask ratio {ratio:g}',
           f'curves_{source}_ratio{ratio:g}.png')

  for contrast, filename, title in [
    ('schedule', 'schedule_gain.png', 'Strided − random schedule benefit'),
    ('dcache', 'dcache_contribution.png', 'D retained − D absent benefit'),
    ('interaction', 'dcache_schedule_interaction.png', 'D-benefit interaction: strided − random'),
  ]:
    data = paired[paired.contrast_type == contrast]
    if data.empty:
      continue
    panels = sorted(set(zip(data.source, data.initial_mask_ratio)))
    fig, axes = plt.subplots(2, len(panels), figsize=(5 * len(panels), 7.6), squeeze=False)
    for column, (source, ratio) in enumerate(panels):
      panel = data[(data.source == source) & (data.initial_mask_ratio == ratio)]
      group_keys = ['variant', 'condition'] if contrast == 'schedule' else ['variant', 'schedule']
      if contrast == 'interaction':
        group_keys = ['variant']
      for keys, group in panel.groupby(group_keys, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        variant = keys[0]
        secondary = keys[1] if len(keys) > 1 else ''
        label = LABELS.get(variant, variant) + (f'; {secondary}' if secondary else '')
        group = group.sort_values('round')
        x = group['round'].to_numpy()
        for row, (metric, sign) in enumerate([('top1_accuracy_pp', 1), ('nll', -1)]):
          low = group['ci95_low_' + metric].to_numpy()
          high = group['ci95_high_' + metric].to_numpy()
          if sign < 0:
            low, high = -high, -low
          axes[row, column].plot(x, sign * group['delta_' + metric].to_numpy(),
            color=COLORS.get(variant), linestyle='--' if secondary in ['absent_dcache', 'random'] else '-',
            marker='o', markersize=3, label=label)
          axes[row, column].fill_between(x, low, high, color=COLORS.get(variant), alpha=.08)
      axes[0, column].set_title(f'{source.title()}, initial ratio {ratio:g}')
      axes[0, column].set_ylabel('Top-1 benefit (percentage points) ↑')
      axes[1, column].set_ylabel('NLL reduction ↑')
      for axis in axes[:, column]:
        axis.axhline(0, color='black', linewidth=.7)
    finish(fig, axes, title, filename)
  return saved


def main(argv=None):
  args = arguments(argv)
  root, output = args.input_root.resolve(), args.output_dir.resolve()
  output.mkdir(parents=True, exist_ok=True)
  raw, coverage = load_parts(root)
  shared, manifests_verified = validate_manifests(coverage)
  validate_raw_protocol(raw, shared)
  summary = summarize(raw)
  per_seed = summarize(raw, ['seed'] + GROUPS)
  paired, issues = comparisons(raw, args.bootstrap_samples, args.bootstrap_seed, args.expected_seeds)
  expected_seeds = args.expected_seeds or sorted(raw.seed.unique())
  present_runs = set(zip(coverage.variant, coverage.seed))
  missing_runs = [(variant, seed) for variant in args.expected_variants for seed in expected_seeds
                  if (variant, seed) not in present_runs]
  partial = bool(missing_runs or not coverage.complete.all() or not issues.empty or not manifests_verified)
  status = ('PARTIAL snapshot: completed parts only' if partial
            else 'Complete evaluation; paired block-bootstrap 95% intervals')
  endpoint = endpoint_table(summary, max(shared['score_rounds']) if shared else 32)
  for name, data in [('pooled_summary.csv', summary), ('per_seed_summary.csv', per_seed),
                     ('paired_deltas.csv', paired), ('coverage.csv', coverage),
                     ('endpoint_table.csv', endpoint),
                     ('pairing_issues.csv', issues if not issues.empty else pd.DataFrame(columns=['issue']))]:
    write_csv(data, output / name)
  if args.strict_pairing and not issues.empty:
    raise PairingError(f'{len(issues)} incompatible comparisons; see {output / "pairing_issues.csv"}')
  figures = build_plots(summary, paired, output, status)
  probe_count = shared['probe_count'] if shared else int(raw.masked_tokens.iloc[0])
  stride = int(shared['stride']) if shared else 32
  score_rounds = shared['score_rounds'] if shared else sorted(raw['round'].unique())
  report = [
    '# Fixed-probe anchor-schedule comparison', '', status, '',
    f'Input: `{root}`', '',
    f'- Loaded {len(raw):,} raw observations from {int(coverage.parts.sum()):,} published parts.',
    f'- Planned mask seeds: {expected_seeds}. Missing variant/seed runs: {missing_runs or "none"}.',
    f'- Shared protocol/dataset/source provenance verified: {manifests_verified}.',
    f'- Omitted incompatible or incomplete paired comparisons: {len(issues)}; see `pairing_issues.csv`.',
    '', '## What is measured', '',
    'These are conditional fixed-probe token NLL and top-1/top-5 accuracy, not generative '
    'perplexity. Lower raw NLL and higher accuracy are better. Scores pool by probe-token '
    f'count; the {probe_count} target probes remain masked and are never revealed. The first token '
    'stays visible. Probe positions are fixed within each seed/block/initial-ratio trajectory '
    'and shared across models, reveal sources, schedules, conditions, and scored rounds.', '',
    'Strided and random schedules reveal the same number of non-probe candidates at each '
    f'round. The {stride} strided candidate groups have variable counts; random groups match those '
    f'counts rather than imposing equal-size steps. Scored rounds are {score_rounds}; '
    f'the full trajectory uses {stride + 1} actual forwards in every arm.', '',
    f'Teacher reveals insert true candidate tokens. By the INPUT to teacher round {stride}, both '
    'schedules have the same visible token context and only probes remain masked: a final '
    'schedule difference therefore reflects recurrent history. Model reveals insert that '
    'arm\'s own sampled tokens, so final contexts differ and scores include accumulated '
    'generation errors and history effects. No target probe is sampled or fed back.', '',
    'D absent removes only previous-forward DCache K/V, retaining the current DCache '
    'attention computation. It is an intervention on the same checkpoint, not a separately '
    'trained no-cache control. Final-state memory remains enabled in both D conditions '
    'where the architecture supports it, and each arm evolves its own final-state history. '
    'Consequently the D contrast includes indirect changes to that history. Vanilla and '
    'objective-matched baselines have only the absent_dcache arm.', '',
    '## Paired comparisons and uncertainty', '',
    '`paired_deltas.csv` records raw metric deltas: schedule = strided − random; '
    'D contribution = correct − absent_dcache; interaction = '
    '(correct_strided − absent_strided) − (correct_random − absent_random). '
    'Negative NLL deltas and positive accuracy deltas mean improvement. Accuracy '
    'deltas are percentage points. Benefit plots reverse the NLL sign so higher '
    'always means better; positive interaction means D provides more benefit for strided.', '',
    'Every contrast requires exactly matching (seed, example_id) sets, probe counts, and '
    'remaining-mask counts. Four-arm interactions are formed per observation before '
    'resampling, preserving paired covariance. Nothing is silently inner-joined. '
    'Every block must have the complete seed vector; with --expected-seeds, contrasts '
    'wait until all planned seeds are present. Partial curves can have unequal coverage '
    'and must not be treated as a balanced comparison.', '',
    f'{args.bootstrap_samples} bootstrap draws resample packed validation blocks, keeping '
    'all mask seeds of each block together. These seeds vary evaluation masks, not model '
    'training. Intervals describe block-level uncertainty for fixed global-step-5000 '
    'checkpoints and are exploratory, with no multiple-comparison correction. Packed '
    '1024-token blocks are not necessarily independent original documents; unknown '
    'original-source correlations are not modeled. No significance claim is made from '
    'partial runs. Exact target identities rely on the verified shared deterministic '
    'protocol and source hashes; raw CSV token counts alone cannot prove mask identity.', '',
    'Manifest compatibility validates data fingerprints, evaluator source hashes, and '
    'protocol settings. Each variant must retain one hashed step-5000 source checkpoint '
    'across seeds. Missing manifests keep the report partial. Dataset Arrow fingerprints '
    'use file metadata plus metadata-JSON hashes, not a full token-data content hash.', '',
    '## Figures', '',
  ]
  report += [f'- [{name}]({name})' for name in figures]
  if not figures:
    report.append('No figures are available yet.')
  report += ['', '## Tables', '',
    '- `pooled_summary.csv`: probe-token-weighted metrics per model/source/schedule/condition/round.',
    '- `per_seed_summary.csv`: the same cells separately by evaluation mask seed.',
    '- `paired_deltas.csv`: paired schedule, D contribution, and four-arm interaction estimates and 95% CIs.',
    '- `endpoint_table.csv`: final scored-round metrics; ranks are descriptive within each condition, not a model significance test.',
    '- `coverage.csv`: run completion, counts, manifest/data fingerprints, checkpoint path/hash/step.',
    '- `pairing_issues.csv`: exact reasons comparisons were omitted.', '']
  temporary = output / 'REPORT.md.tmp'
  temporary.write_text('\n'.join(report))
  temporary.replace(output / 'REPORT.md')
  print(f'{status}\nWrote {len(figures)} figures and tables to {output}')


if __name__ == '__main__':
  main()
