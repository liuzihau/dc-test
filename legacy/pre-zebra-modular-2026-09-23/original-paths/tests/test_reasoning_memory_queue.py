"""Memory queue orchestration, using fake GPU checks and fake training children."""
import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

from reasoning import baseline_queue as queue
from reasoning.model import DEFAULTS


ROOT = Path(__file__).resolve().parents[1]
MEMORY_VARIANTS = ('both', 'both_aux')


def option(command, name):
    return command[command.index(name) + 1]


@pytest.fixture
def memory_job(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(('REASONING_', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK')):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    args = queue.parser().parse_args([
        'run', '--suite', 'memory', '--data-root', str(tmp_path / 'data'),
        '--output-root', str(tmp_path / 'runs'), '--queue-dir', str(tmp_path / 'queue')])
    return queue.BaselineQueue(args)


def fake_data(job, task):
    directory = job.data_dir(task)
    directory.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name, count in (('train', job.args.train_examples),
                        ('validation', job.args.valid_examples),
                        ('test', job.args.test_examples)):
        path = directory / (name + '.jsonl')
        path.write_text('{}\n')  # Only checks manifest/checksum orchestration.
        splits[name] = dict(filename=path.name, records=count, sha256=queue.digest(path))
    queue.atomic_json(directory / 'manifest.json', dict(
        schema_version=1, task=task, seed=17,
        source=dict(kind='synthetic_pilot'), splits=splits))


def fake_checkpoint(directory, step):
    directory = Path(directory)
    target = directory / 'checkpoints/step.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'fake checkpoint; never load into torch')
    queue.atomic_json(target.with_suffix('.pt.json'), dict(
        step=step, size=target.stat().st_size,
        sha256=queue.digest(target), file=target.name))
    last = target.parent / 'last.pt'
    if not last.exists():
        last.symlink_to(target.name)
    return target


@pytest.fixture
def memory_children(memory_job, monkeypatch):
    job = memory_job
    calls = []
    monkeypatch.setattr(job, 'gpu_check', lambda: None)

    def run(command, stage):
        calls.append((stage, command))
        action = command[2]
        if action == 'prepare':
            fake_data(job, option(command, '--task'))
        elif action == 'train':
            directory = Path(option(command, '--run-dir'))
            fake_checkpoint(directory, int(option(command, '--max-steps')))
            logs = directory / 'logs/attempt-1'
            logs.mkdir(parents=True, exist_ok=True)
            (logs / 'metrics.csv').write_text(
                'step,seconds_per_update,peak_cuda_allocated_gib\n'
                + option(command, '--max-steps') + ',0.1,24.5\n')
        elif action == 'evaluate':
            checkpoint = Path(option(command, '--checkpoint'))
            variant = stage.split('-')[-1]
            queue.atomic_json(option(command, '--output'), dict(
                step=job.args.max_steps, checkpoint=str(checkpoint),
                contract=dict(variant=variant,
                              data_sha256=queue.digest(Path(option(command, '--data-dir')) / 'manifest.json')),
                arguments=dict(examples=int(option(command, '--examples')),
                               protocol='generate', split='test', policy='top_prob',
                               seed=2026, batch_size=8),
                metrics=dict(num_examples=job.args.eval_examples)))
        elif action == 'plot':
            assert option(command, '--train-metric') == 'train/base_loss'
        else:
            raise AssertionError(command)

    monkeypatch.setattr(job, 'subprocess', run)
    return calls


def test_memory_recipe_is_adjacent_current_preserving_128(memory_job):
    job = memory_job
    assert tuple(job.variants) == MEMORY_VARIANTS
    assert job.args.micro_batch == job.args.global_batch == 128
    assert job.args.max_steps == 5000
    assert job.spec['gradient_mode'] == 'adjacent'
    assert job.spec['no_robustness'] is False
    for task in queue.TASKS:
        for variant in MEMORY_VARIANTS:
            command = job.train_command(task, variant, job.run_dir(task, variant))
            assert option(command, '--variant') == variant
            assert option(command, '--gradient-mode') == 'adjacent'
            assert option(command, '--merged-policy') == 'current_preserving'
            assert '--no-robustness' not in command
            assert '--stress-memory-routes' not in command
            assert option(command, '--size') == 'mini'
            assert option(command, '--precision') == 'bf16'
            assert option(command, '--micro-batch') == option(command, '--global-batch') == '128'
            assert '--fresh' not in command  # Trainer validates its own resume contract.


def test_memory_uses_shared_data_but_isolated_recipe_paths(memory_job):
    memory = memory_job
    args = queue.parser().parse_args([
        'plan', '--data-root', str(memory.args.data_root),
        '--output-root', str(memory.args.output_root)])
    baseline = queue.BaselineQueue(args)
    assert baseline.args.suite == 'baselines'
    assert tuple(baseline.variants) == ('vanilla', 'mdm', 'mdm_aux')
    assert baseline.spec['gradient_mode'] == 'detached'
    assert baseline.spec['no_robustness'] is True
    for task in queue.TASKS:
        assert memory.data_dir(task) == baseline.data_dir(task)
        for variant in MEMORY_VARIANTS:
            directory = memory.run_dir(task, variant)
            assert 'current-preserving' in directory.name
            assert 'mb128-gb128-seed1' in directory.name
            assert directory not in [baseline.run_dir(task, name) for name in baseline.variants]


def test_four_memory_trials_run_in_order_and_resume_with_no_duplicate_evaluation(
        memory_job, memory_children):
    job = memory_job
    job.run()
    stages = [stage for stage, _ in memory_children]
    assert stages[:4] == ['prepare-sudoku', 'prepare-zebra', 'smoke-sudoku', 'smoke-zebra']
    assert [stage for stage in stages if stage.startswith('train-')] == [
        f'train-{task}-{variant}' for task in queue.TASKS for variant in MEMORY_VARIANTS]
    assert [stage for stage in stages if stage.startswith('evaluate-')] == [
        f'evaluate-{task}-{variant}' for task in queue.TASKS for variant in MEMORY_VARIANTS]
    assert json.loads((job.directory / 'status.json').read_text())['experiments_completed'] == 4
    for stage, command in memory_children:
        if stage.startswith('evaluate-'):
            assert option(command, '--examples') == '1000'
            assert option(command, '--protocol') == 'generate'
            assert option(command, '--policy') == 'top_prob'
            assert option(command, '--seed') == '2026'
            assert option(command, '--split') == 'test'
            assert option(command, '--batch-size') == '8'
    memory_children.clear()
    job.run()
    assert not any(stage.startswith(('prepare-', 'evaluate-')) for stage, _ in memory_children)
    assert len([stage for stage, _ in memory_children if stage.startswith('train-')]) == 4
    smoke = json.loads((job.directory / 'full_model_smoke.json').read_text())
    assert smoke['sudoku']['start_step'] == 2 and smoke['sudoku']['end_step'] == 4
    assert len(list((job.directory / 'smoke').iterdir())) == 2


def test_memory_smoke_uses_heaviest_variant_full_models_and_does_not_train_production(
        memory_job, memory_children):
    memory_job.run(smoke_only=True)
    assert not any(stage.startswith('train-') for stage, _ in memory_children)
    commands = [command for stage, command in memory_children if stage.startswith('smoke-')]
    assert len(commands) == 2
    for command in commands:
        assert option(command, '--variant') == 'both_aux'
        assert option(command, '--size') == 'mini'
        assert option(command, '--micro-batch') == '128'
        assert option(command, '--max-steps') == '2'
        assert '--stress-memory-routes' in command
        assert '--fresh' not in command
        assert Path(option(command, '--run-dir')).is_relative_to(memory_job.directory / 'smoke')
    assert json.loads((memory_job.directory / 'status.json').read_text())['experiments_started'] == 0


def test_memory_oom_stops_without_batch_reduction_or_production_work(
        memory_job, memory_children, monkeypatch):
    run = memory_job.subprocess

    def fail(command, stage):
        if stage == 'smoke-zebra':
            raise RuntimeError('CUDA out of memory')
        return run(command, stage)

    monkeypatch.setattr(memory_job, 'subprocess', fail)
    with pytest.raises(RuntimeError, match='out of memory'):
        memory_job.run()
    assert memory_job.args.micro_batch == memory_job.args.global_batch == 128
    assert not any(stage.startswith('train-') for stage, _ in memory_children)
    assert json.loads((memory_job.directory / 'status.json').read_text())['status'] == 'stopped'


def test_memory_queue_rejects_changed_recipe_on_restart(memory_job, memory_children):
    memory_job.run(smoke_only=True)
    spec_path = memory_job.directory / 'queue_config.json'
    spec = json.loads(spec_path.read_text())
    spec['gradient_mode'] = 'detached'
    queue.atomic_json(spec_path, spec)
    memory_children.clear()
    with pytest.raises(ValueError, match='settings changed'):
        memory_job.run()
    assert not memory_children


def test_memory_tmux_preserves_suite_and_batch_geometry(memory_job, monkeypatch):
    calls = []

    def launch(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if 'has-session' in command else 0)

    monkeypatch.setattr(queue.subprocess, 'run', launch)
    with tempfile.TemporaryDirectory(prefix='mq-', dir=ROOT / '.cache') as root:
        monkeypatch.setattr(queue, 'ROOT', Path(root))
        memory_job.tmux()
    actual = shlex.split(calls[-1][-1])
    assert option(actual, '--suite') == 'memory'
    assert option(actual, '--micro-batch') == option(actual, '--global-batch') == '128'
    assert option(actual, '--queue-dir') == str(memory_job.directory)
    assert 'WORLD_SIZE=1' in actual and 'LOCAL_RANK=0' in actual


def test_memory_wrapper_delegates_suite_with_user_arguments(tmp_path):
    scripts = tmp_path / 'scripts'
    (scripts / 'train').mkdir(parents=True)
    (scripts / 'reasoning').mkdir()
    target = scripts / 'train/train_reasoning_memory_h100.sh'
    shutil.copyfile(ROOT / 'scripts/train/train_reasoning_memory_h100.sh', target)
    # Either delegate through the thin common wrapper or directly to its entry.
    shutil.copyfile(ROOT / 'scripts/train/train_reasoning_baselines_h100.sh',
                    scripts / 'train/train_reasoning_baselines_h100.sh')
    (scripts / 'reasoning/run_baseline_queue.py').write_text(
        'import json, sys\nprint(json.dumps(sys.argv[1:]))\n')
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('DCACHE_', 'REASONING_'))}
    env['DCACHE_PYTHON'] = sys.executable
    result = subprocess.run([
        '/bin/bash', str(target), 'plan', '--micro-batch', '128', '--global-batch', '128'],
        env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    arguments = json.loads(result.stdout.splitlines()[-1])
    assert 'plan' in arguments
    assert option(arguments, '--suite') == 'memory'
    assert option(arguments, '--micro-batch') == option(arguments, '--global-batch') == '128'


def write_plot_contract(job, task, variant, baseline=False):
    """Construct faithful scientific metadata without instantiating a large model."""
    if baseline:
        directory = (job.args.output_root / task /
                     f'{variant}-h100-{job.label}-{job.batch_label}')
    else:
        directory = job.run_dir(task, variant)
    model = copy.deepcopy(DEFAULTS)
    model.update(vocab_size=32, hidden_size=512, n_heads=8, n_layers=6,
                 max_length=192 if task == 'sudoku' else 384,
                 memory_mode='none' if baseline else 'both',
                 attention_mode='vanilla' if variant == 'vanilla' else 'merged',
                 neighbors=variant.endswith('_aux'),
                 gradient_mode='detached' if baseline else 'adjacent',
                 trajectory='single' if variant == 'vanilla' else 'five',
                 gate_enabled=baseline, cache_only_probability=0.0,
                 current_only_probability=0.0 if baseline else 0.05,
                 final_dropout=0.0 if baseline else 0.1,
                 identity_probability=0.0 if baseline else 0.25)
    if not baseline:
        model['merged_policy'] = 'current_preserving'
    contract = dict(task=task, variant=variant, model_config=model,
                    data_sha256=queue.digest(job.data_dir(task) / 'manifest.json'),
                    global_batch=128, micro_batch=128, world_size=1, seed=1,
                    precision='bf16', lr=3e-4, warmup_steps=1000,
                    weight_decay=0.0, grad_clip=1.0, device_type='cuda')
    queue.atomic_json(directory / 'contract.json', contract)
    queue.atomic_json(directory / 'launch.json', dict(
        eval_seed=2026, eval_batch_size=8, validation_examples=128))
    log = directory / 'logs/attempt-1/metrics.csv'
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text('step,train/base_loss\n500,1.0\n')
    return directory


@pytest.mark.parametrize('variant', ('vanilla', 'mdm', 'mdm_aux'))
def test_compatible_baseline_overlay_requires_contract_and_preserves_files(memory_job, variant):
    job = memory_job
    fake_data(job, 'sudoku')
    assert job.compatible_baseline_runs('sudoku') == []
    write_plot_contract(job, 'sudoku', 'both')
    assert job.compatible_baseline_runs('sudoku') == []
    directory = write_plot_contract(job, 'sudoku', variant, baseline=True)
    before = {path: queue.digest(path) for path in directory.rglob('*') if path.is_file()}
    assert job.compatible_baseline_runs('sudoku') == [str(directory)]
    after = {path: queue.digest(path) for path in directory.rglob('*') if path.is_file()}
    assert after == before  # Existing controls are read-only, never migrated or retrained.


@pytest.mark.parametrize('mismatch', ('data', 'model', 'batch', 'auxiliary', 'validation', 'metrics'))
def test_incompatible_baseline_is_not_silently_plotted(memory_job, mismatch, capsys):
    job = memory_job
    fake_data(job, 'sudoku')
    write_plot_contract(job, 'sudoku', 'both')
    directory = write_plot_contract(job, 'sudoku', 'mdm_aux', baseline=True)
    contract = json.loads((directory / 'contract.json').read_text())
    if mismatch == 'data':
        contract['data_sha256'] = 'another dataset'
    elif mismatch == 'model':
        contract['model_config']['hidden_size'] = 256
    elif mismatch == 'batch':
        contract['micro_batch'] = 8
    elif mismatch == 'auxiliary':
        contract['model_config']['neighbor_weight'] = 0.2
    elif mismatch == 'validation':
        queue.atomic_json(directory / 'launch.json', dict(
            eval_seed=42, eval_batch_size=8, validation_examples=128))
    elif mismatch == 'metrics':
        (directory / 'logs/attempt-1/metrics.csv').unlink()
    queue.atomic_json(directory / 'contract.json', contract)
    assert job.compatible_baseline_runs('sudoku') == []
    assert 'Skipping incompatible baseline plot overlay' in capsys.readouterr().out


def test_memory_queue_plots_verified_controls_and_excludes_wrong_controls(
        memory_job, memory_children):
    job = memory_job
    fake_data(job, 'sudoku')
    write_plot_contract(job, 'sudoku', 'both')
    valid = write_plot_contract(job, 'sudoku', 'mdm', baseline=True)
    invalid = write_plot_contract(job, 'sudoku', 'mdm_aux', baseline=True)
    contract = json.loads((invalid / 'contract.json').read_text())
    contract['seed'] = 9
    queue.atomic_json(invalid / 'contract.json', contract)
    job.run()
    for stage, command in memory_children:
        if stage.startswith('plot-sudoku-'):
            runs = command[command.index('--runs') + 1:command.index('--output')]
            assert str(valid) in runs
            assert str(invalid) not in runs
            assert str(job.run_dir('sudoku', 'both')) in runs


@pytest.mark.parametrize('condition', ('none', 'shuffle_dcache', 'shuffle_final', 'shuffle_both'))
def test_wrong_memory_evaluation_cannot_be_reused_as_correct(memory_job, memory_children, condition):
    job = memory_job
    job.run()
    directory = job.run_dir('sudoku', 'both')
    path = next(directory.glob('evaluation-*.json'))
    result = json.loads(path.read_text())
    result['arguments']['memory_condition'] = condition
    queue.atomic_json(path, result)
    with pytest.raises(ValueError, match='metadata'):
        job.evaluate('sudoku', 'both', directory)
