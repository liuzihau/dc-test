"""CPU-only checks for registry-driven training plots and status summaries."""

import csv
from importlib import import_module
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(REPO_ROOT / '.cache/matplotlib'))


@pytest.fixture
def workspace():
  """Keep synthetic CSVs, manifests and outputs off system /tmp."""
  directory = REPO_ROOT / '.cache/test-canonical-results'
  directory.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix='case-', dir=str(directory)) as path:
    yield Path(path)


@pytest.fixture
def refresh():
  return import_module('scripts.results.refresh_canonical_results')


@pytest.fixture
def saved_figures(monkeypatch):
  """Inspect plotted data without writing any production or test image."""
  import matplotlib
  matplotlib.use('Agg')
  from matplotlib.figure import Figure
  import matplotlib.pyplot as plt

  figures = []

  def capture(figure, filename, *args, **kwargs):
    del args, kwargs
    figures.append((Path(filename), figure))

  monkeypatch.setattr(Figure, 'savefig', capture)
  yield figures
  for _, figure in figures:
    plt.close(figure)


def write_metrics(path, rows, columns=None, mtime=100):
  path.parent.mkdir(parents=True, exist_ok=True)
  pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
  os.utime(path, (mtime, mtime))
  return path


def make_run(workspace, run_id, ordinal=0, validation='present'):
  vanilla = run_id == 'bd3_vanilla'
  run = {
    'id': run_id,
    'label': f'Label {run_id}',
    'role': 'synthetic_test',
    'path': str(workspace / run_id),
    'train_metric': 'trainer/loss' if vanilla else 'trainer/loss_t2',
    'validation_metric': 'val/nll' if vanilla else 'val/loss_t2',
    'status': 'running' if run_id == 'future_adjacent' else 'complete',
  }
  rows = []
  for index, step in enumerate([399, 400, 1899, 1900, 4999, 5000, 5001]):
    row = {'step': step, run['train_metric']: 10.0 - index + ordinal / 10}
    if validation != 'missing':
      row[run['validation_metric']] = (
        7.0 - index / 10 + ordinal / 10
        if validation == 'present' else None)
    rows.append(row)
  write_metrics(Path(run['path']) / 'version_0/metrics.csv', rows)
  return run


def make_registry(workspace, validation='present'):
  ids = ['bd3_vanilla', 'objective_aligned', 'dcache_v2',
         'dcache_final_state', 'future_adjacent']
  runs = {
    run_id: make_run(workspace, run_id, index,
                     validation if run_id == 'future_adjacent' else 'present')
    for index, run_id in enumerate(ids)
  }
  return {'comparison_step_limit': 5000, 'runs': list(runs.values())}, runs


def matching_lines(axis, label):
  return [line for line in axis.lines if label in line.get_label()]


def read_status(path):
  with path.open(newline='') as handle:
    return {row['id']: row for row in csv.DictReader(handle)}


def test_manifest_accepts_arbitrary_fifth_run(refresh, workspace):
  manifest, expected = make_registry(workspace)
  path = workspace / 'manifest.json'
  path.write_text(json.dumps(manifest))
  actual_manifest, runs = refresh.load_manifest(path)
  assert actual_manifest == manifest
  assert list(runs) == list(expected)
  assert runs['future_adjacent'] == expected['future_adjacent']


def test_manifest_still_requires_historical_four(refresh, workspace):
  manifest, _ = make_registry(workspace)
  manifest['runs'] = [run for run in manifest['runs']
                      if run['id'] != 'objective_aligned']
  path = workspace / 'incomplete.json'
  path.write_text(json.dumps(manifest))
  with pytest.raises(ValueError, match='objective_aligned'):
    refresh.load_manifest(path)


def test_cloud_selection_keeps_only_present_runs_without_mutating_registry(
    refresh, workspace):
  manifest, _ = make_registry(workspace)
  original = json.loads(json.dumps(manifest))
  overrides = [(run['id'], str(workspace / 'not-transferred' / run['id']))
               for run in manifest['runs'] if run['id'] != 'future_adjacent']
  selected, runs = refresh.select_runs(manifest, True, overrides)
  assert list(runs) == ['future_adjacent']
  assert len(selected['runs']) == 1
  assert manifest == original


def test_cloud_missing_supplement_is_optional_only_in_available_mode(
    refresh, workspace):
  manifest, _ = make_registry(workspace)
  manifest['runs'][0]['validation_supplement'] = {
    'path': str(workspace / 'not-transferred.csv'),
    'step_column': 'step', 'metric_column': 'nll'}
  _, strict = refresh.select_runs(manifest)
  assert 'validation_supplement' in strict['bd3_vanilla']
  with pytest.raises(FileNotFoundError):
    refresh.append_validation_supplement(
      refresh.load_metrics(strict['bd3_vanilla']['path']), strict['bd3_vanilla'])
  _, partial = refresh.select_runs(manifest, True)
  assert 'validation_supplement' not in partial['bd3_vanilla']
  assert 'validation_supplement' in manifest['runs'][0]


def test_cloud_selection_fails_for_no_data_or_unknown_id(refresh, workspace):
  manifest, _ = make_registry(workspace)
  overrides = [(run['id'], str(workspace / 'missing' / run['id']))
               for run in manifest['runs']]
  with pytest.raises(FileNotFoundError, match='No registered runs'):
    refresh.select_runs(manifest, True, overrides)
  with pytest.raises(ValueError, match='Unknown run ID'):
    refresh.select_runs(manifest, True, [('typo', '/unused')])


def test_main_plot_includes_all_runs_and_caps_display(
    refresh, workspace, saved_figures):
  _, runs = make_registry(workspace)
  output = workspace / 'plots/main.png'
  refresh.run_plot(runs, output, min_step=400, max_step=5000, x_max=5200)
  assert len(saved_figures) == 1
  assert saved_figures[0][0] == output
  training, validation = saved_figures[0][1].axes
  expected_steps = [400, 1899, 1900, 4999, 5000]
  for run in runs.values():
    assert matching_lines(training, run['label'])
    assert matching_lines(validation, run['label'])
  for axis in [training, validation]:
    assert axis.get_xlim() == pytest.approx((400, 5200))
    for line in axis.lines:
      np.testing.assert_array_equal(line.get_xdata(), expected_steps)
  # Raw plus smoothed traces should both be retained for every run.
  assert len(training.lines) == 2 * len(runs)
  assert len(validation.lines) == len(runs)
  assert not output.exists()


@pytest.mark.parametrize('validation', ['present', 'missing'])
def test_sixth_merged_no_aux_run_uses_registered_metrics_and_color(
    refresh, workspace, saved_figures, validation):
  manifest, runs = make_registry(workspace)
  run = make_run(workspace, 'dcache_merged_final_state_adjacent_no_aux',
                 ordinal=5, validation=validation)
  run['color'] = '#008b8b'
  runs[run['id']] = run
  manifest['runs'].append(run)
  refresh.run_plot(runs, workspace / 'six-trials.png', min_step=400)
  train_axis, val_axis = saved_figures[0][1].axes
  assert len(train_axis.lines) == 12
  assert matching_lines(train_axis, run['label'])[0].get_color() == '#008b8b'
  lines = matching_lines(val_axis, run['label'])
  assert bool(lines) == (validation == 'present')
  if lines:
    assert lines[0].get_color() == '#008b8b'
  refresh.write_status_table(manifest, runs, workspace / 'status.csv')
  row = read_status(workspace / 'status.csv')[run['id']]
  assert row['train_metric'] == 'trainer/loss_t2'
  assert row['validation_metric'] == 'val/loss_t2'
  assert bool(row['validation_value']) == (validation == 'present')


def test_rolling_window_is_60_records_and_uses_pre_zoom_history(
    refresh, workspace, saved_figures):
  run = make_run(workspace, 'future_adjacent')
  values = np.arange(1, 66, dtype=float)
  rows = [{'step': step, run['train_metric']: value}
          for step, value in enumerate(values, start=1)]
  write_metrics(Path(run['path']) / 'version_0/metrics.csv', rows)
  refresh.run_plot({run['id']: run}, workspace / 'smooth.png',
                   min_step=60, max_step=65, x_max=70)
  training = saved_figures[0][1].axes[0]
  labeled = matching_lines(training, run['label'])
  assert len(labeled) == 1
  np.testing.assert_array_equal(labeled[0].get_xdata(), np.arange(60, 66))
  expected = pd.Series(values).rolling(60, min_periods=1).mean().iloc[59:]
  np.testing.assert_allclose(labeled[0].get_ydata(), expected)
  raw = [line for line in training.lines if line is not labeled[0]]
  assert len(raw) == 1
  np.testing.assert_array_equal(raw[0].get_ydata(), values[59:])


@pytest.mark.parametrize('validation', ['missing', 'empty'])
def test_unvalidated_run_is_train_only_without_failure(
    refresh, workspace, saved_figures, validation):
  manifest, runs = make_registry(workspace, validation)
  extra = runs['future_adjacent']
  refresh.run_plot(runs, workspace / 'main.png', min_step=400)
  training, valid = saved_figures[0][1].axes
  assert matching_lines(training, extra['label'])
  assert not matching_lines(valid, extra['label'])
  assert len(valid.lines) == 4

  refresh.plot_validation_zoom(manifest, workspace / 'zoom.png')
  zoom = saved_figures[1][1].axes[0]
  assert not matching_lines(zoom, extra['label'])
  assert len(zoom.lines) == 4

  status_path = workspace / 'status.csv'
  refresh.write_status_table(manifest, runs, status_path)
  status = read_status(status_path)[extra['id']]
  assert status['train_step'] == '5000'
  assert status['train_value'] != ''
  assert status['validation_step'] == ''
  assert status['validation_value'] == ''
  assert status['status'] == 'running'


def test_zoom_unknown_id_and_inclusive_step_boundaries(
    refresh, workspace, saved_figures):
  manifest, runs = make_registry(workspace)
  refresh.plot_validation_zoom(manifest, workspace / 'zoom.png',
                                min_step=1900, x_max=5100)
  axis = saved_figures[0][1].axes[0]
  assert len(axis.lines) == 5
  assert matching_lines(axis, runs['future_adjacent']['label'])
  assert axis.get_xlim() == pytest.approx((1900, 5100))
  for line in axis.lines:
    np.testing.assert_array_equal(line.get_xdata(), [1900, 4999, 5000])


def test_resume_dedup_prefers_latest_file_then_latest_nonempty_row(
    refresh, workspace):
  run = make_run(workspace, 'future_adjacent')
  train, valid = run['train_metric'], run['validation_metric']
  # Deliberately reverse name order versus mtime to test actual resume ordering.
  write_metrics(Path(run['path']) / 'version_z/metrics.csv', [
    {'step': 4999, train: 3.0, valid: 2.9},
    {'step': 5000, train: 2.8, valid: 2.7},
  ], mtime=200)
  write_metrics(Path(run['path']) / 'version_a/metrics.csv', [
    {'step': 5000, train: 2.6, valid: 2.5},
    {'step': 5000, train: 2.4, valid: None},
    {'step': 5000, train: None, valid: 2.3},
    {'step': 'not-a-step', train: 1000, valid: 1000},
    {'step': 5001, train: 1.0, valid: 0.9},
  ], mtime=300)
  frame = refresh.load_metrics(run['path'])
  training = refresh.metric_series(frame, train)
  validation = refresh.metric_series(frame, valid)
  assert training[training.step == 5000][train].item() == pytest.approx(2.4)
  assert validation[validation.step == 5000][valid].item() == pytest.approx(2.3)
  assert training.step.is_unique and validation.step.is_unique
  assert training.step.is_monotonic_increasing

  manifest = {'comparison_step_limit': 5000, 'runs': [run]}
  output = workspace / 'status.csv'
  refresh.write_status_table(manifest, {run['id']: run}, output)
  status = read_status(output)[run['id']]
  assert status['train_step'] == status['validation_step'] == '5000'
  assert float(status['train_value']) == pytest.approx(2.4)
  assert float(status['validation_value']) == pytest.approx(2.3)


def test_validation_supplement_updates_plot_zoom_and_capped_status(
    refresh, workspace, saved_figures):
  run = make_run(workspace, 'future_adjacent')
  supplement = workspace / 'standalone.csv'
  write_metrics(supplement, [
    {'checkpoint_step': 5000, 'val_loss_t2': 2.123},
    {'checkpoint_step': 5500, 'val_loss_t2': 1.234},
  ])
  run['validation_supplement'] = {
    'path': str(supplement), 'step_column': 'checkpoint_step',
    'metric_column': 'val_loss_t2',
  }
  runs = {run['id']: run}
  manifest = {'comparison_step_limit': 5000, 'runs': [run]}
  refresh.run_plot(runs, workspace / 'main.png', min_step=400)
  refresh.plot_validation_zoom(manifest, workspace / 'zoom.png')
  for axis in [saved_figures[0][1].axes[1], saved_figures[1][1].axes[0]]:
    line = matching_lines(axis, run['label'])[0]
    assert line.get_xdata()[-1] == 5000
    assert line.get_ydata()[-1] == pytest.approx(2.123)

  status_path = workspace / 'status.csv'
  refresh.write_status_table(manifest, runs, status_path)
  status = read_status(status_path)[run['id']]
  assert status['validation_step'] == '5000'
  assert float(status['validation_value']) == pytest.approx(2.123)
  assert status['train_step'] == '5000'
