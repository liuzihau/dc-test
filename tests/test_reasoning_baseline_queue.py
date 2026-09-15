"""Queue orchestration contracts. Fake GPU/children; no actual GPU or tmux use."""
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace

import pytest

from reasoning import baseline_queue as queue


@pytest.fixture
def job(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(('REASONING_', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK')):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    args = queue.parser().parse_args(['run', '--data-root', str(tmp_path / 'data'),
                                     '--output-root', str(tmp_path / 'runs'), '--queue-dir', str(tmp_path / 'queue')])
    return queue.BaselineQueue(args)


def option(command, name):
    return command[command.index(name) + 1]


def fake_data(job, task):
    directory = job.data_dir(task)
    directory.mkdir(parents=True)
    splits = {}
    for name, count in (('train', job.args.train_examples), ('validation', job.args.valid_examples),
                        ('test', job.args.test_examples)):
        file = directory / (name + '.jsonl')
        file.write_text('{}\n')  # Only manifest/checksum orchestration, not Dataset decoding.
        splits[name] = dict(filename=file.name, records=count, sha256=queue.digest(file))
    queue.atomic_json(directory / 'manifest.json', dict(schema_version=1, task=task, seed=17,
                      source=dict(kind='synthetic_pilot'), splits=splits))


def fake_checkpoint(directory, step):
    directory = Path(directory)
    target = directory / 'checkpoints/step.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'fake checkpoint for orchestration tests only')
    queue.atomic_json(target.with_suffix('.pt.json'), dict(step=step, size=target.stat().st_size,
                      sha256=queue.digest(target), file=target.name))
    last = target.parent / 'last.pt'
    if not last.exists():
        last.symlink_to(target.name)
    return target


@pytest.fixture
def fake_children(job, monkeypatch):
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
            (logs / 'metrics.csv').write_text('step,seconds_per_update,peak_cuda_allocated_gib\n'
                                            + option(command, '--max-steps') + ',0.1,12.5\n')
        elif action == 'evaluate':
            checkpoint = Path(option(command, '--checkpoint'))
            variant = stage.split('-')[-1]
            queue.atomic_json(option(command, '--output'), dict(
                step=job.args.max_steps, checkpoint=str(checkpoint),
                contract=dict(variant=variant, data_sha256=queue.digest(Path(option(command, '--data-dir')) / 'manifest.json')),
                arguments=dict(examples=int(option(command, '--examples')), protocol='generate', split='test',
                               policy='top_prob', seed=2026, batch_size=8),
                metrics=dict(num_examples=job.args.eval_examples)))
        elif action == 'plot':
            assert option(command, '--train-metric') == 'train/base_loss'
        else:
            raise AssertionError(command)

    monkeypatch.setattr(job, 'subprocess', run)
    return calls


def test_128_batch_six_no_memory_variants_and_isolated_full_smoke(job):
    assert job.args.micro_batch == job.args.global_batch == 128
    assert job.args.max_steps == 5000
    for task in queue.TASKS:
        for variant in queue.VARIANTS:
            directory = job.run_dir(task, variant)
            assert 'mb128-gb128-seed1' in directory.name
            command = job.train_command(task, variant, directory)
            assert option(command, '--variant') == variant
            assert option(command, '--size') == 'mini'
            assert option(command, '--micro-batch') == option(command, '--global-batch') == '128'
            assert '--no-robustness' in command
            assert option(command, '--precision') == 'bf16'
    command = job.train_command('zebra', 'mdm_aux', Path('/isolated/smoke'), smoke=True)
    assert option(command, '--size') == 'mini'  # Must NOT inherit the old debug-size smoke.
    assert option(command, '--micro-batch') == '128'
    assert option(command, '--max-steps') == '2'
    assert '--fresh' in command


def test_queue_runs_sequentially_and_reuses_valid_evaluation_on_restart(job, fake_children):
    job.run()
    stages = [stage for stage, _ in fake_children]
    assert stages[:4] == ['prepare-sudoku', 'prepare-zebra', 'smoke-sudoku', 'smoke-zebra']
    expected = [f'{stage}-{task}-{variant}' for task in queue.TASKS for variant in queue.VARIANTS
                for stage in ('train', 'plot', 'evaluate')]
    assert stages[4:] == expected
    assert json.loads((job.directory / 'status.json').read_text())['experiments_completed'] == 6
    fake_children.clear()
    job.run()
    # Train is still invoked to validate its own checkpoint and resume contract;
    # its real implementation exits early if already complete. Evaluation reused.
    assert not any(stage.startswith(('prepare-', 'evaluate-')) for stage, _ in fake_children)
    assert len([stage for stage, _ in fake_children if stage.startswith('train-')]) == 6
    summary = json.loads((job.directory / 'full_model_smoke.json').read_text())
    assert summary['sudoku']['start_step'] == 2 and summary['sudoku']['end_step'] == 4
    assert summary['sudoku']['run'] == str(job.directory / 'smoke/sudoku')
    assert len(list((job.directory / 'smoke').iterdir())) == 2
    for stage, command in fake_children:
        if stage.startswith('smoke-'):
            assert '--fresh' not in command
            assert option(command, '--max-steps') == '4'


def test_smoke_only_does_not_start_any_research_run(job, fake_children):
    job.run(smoke_only=True)
    assert not any(stage.startswith('train-') for stage, _ in fake_children)
    assert json.loads((job.directory / 'status.json').read_text())['experiments_started'] == 0
    summary = json.loads((job.directory / 'full_model_smoke.json').read_text())
    assert set(summary) == {'sudoku', 'zebra'}
    assert summary['zebra']['micro_batch'] == 128


def test_smoke_retry_after_contract_written_before_first_checkpoint(job, fake_children):
    directory = job.directory / 'smoke/sudoku'
    queue.atomic_json(directory / 'contract.json', {'owned_smoke_fixture': True})
    assert not (directory / 'checkpoints/last.pt').exists()
    job.run(smoke_only=True)
    command = next(command for stage, command in fake_children if stage == 'smoke-sudoku')
    assert '--fresh' not in command
    assert option(command, '--max-steps') == '2'


def test_failure_stops_queue_without_batch_fallback(job, fake_children, monkeypatch):
    delegate = job.subprocess
    def fail(command, stage):
        if stage == 'smoke-zebra':
            raise RuntimeError('CUDA out of memory')
        delegate(command, stage)
    monkeypatch.setattr(job, 'subprocess', fail)
    with pytest.raises(RuntimeError, match='out of memory'):
        job.run()
    assert not any(stage.startswith('train-') for stage, _ in fake_children)
    assert job.args.micro_batch == 128
    assert json.loads((job.directory / 'status.json').read_text())['status'] == 'stopped'


def test_wrong_pilot_size_or_corrupted_data_not_overwritten(job):
    fake_data(job, 'sudoku')
    path = job.data_dir('sudoku') / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['splits']['train']['records'] = 1000
    queue.atomic_json(path, manifest)
    with pytest.raises(ValueError, match='count/file'):
        job.verify_data('sudoku')
    manifest['splits']['train']['records'] = 20000
    queue.atomic_json(path, manifest)
    (path.parent / 'train.jsonl').write_text('corrupted')
    with pytest.raises(ValueError, match='checksum'):
        job.verify_data('sudoku')


def test_wrong_checkpoint_or_evaluation_cannot_be_marked_complete(job, fake_children):
    job.run()
    directory = job.run_dir('sudoku', 'mdm')
    result_file = next(directory.glob('evaluation-*.json'))
    result = json.loads(result_file.read_text())
    result['arguments']['examples'] = 100
    queue.atomic_json(result_file, result)
    with pytest.raises(ValueError, match='metadata'):
        job.evaluate('sudoku', 'mdm', directory)
    result['arguments']['examples'] = job.args.eval_examples
    result['arguments']['split'] = 'validation'
    queue.atomic_json(result_file, result)
    with pytest.raises(ValueError, match='metadata'):
        job.evaluate('sudoku', 'mdm', directory)
    target = (directory / 'checkpoints/last.pt').resolve()
    target.write_bytes(b'truncated')
    with pytest.raises(ValueError, match='checkpoint'):
        job.checkpoint(directory, job.args.max_steps)


def test_changed_queue_contract_is_rejected(job, fake_children):
    job.run(smoke_only=True)
    job.spec['micro_batch'] = 64
    with pytest.raises(ValueError, match='settings changed'):
        job.run()


def test_gpu_busy_is_explicit_stop_not_kill(job, monkeypatch):
    responses = iter(['NVIDIA H100 NVL, 95830 MiB\n', '1234\n'])
    calls = []
    def answer(command, **kwargs):
        calls.append(command)
        return next(responses)
    monkeypatch.setattr(queue.subprocess, 'check_output', answer)
    with pytest.raises(RuntimeError, match='already has compute processes'):
        job.gpu_check()
    assert all(command[:3] == ['nvidia-smi', '-i', '0'] for command in calls)


def test_real_cpu_child_error_is_logged_and_propagated(job):
    with pytest.raises(RuntimeError, match='exit 7'):
        job.subprocess([sys.executable, '-c', 'print("expected test failure"); raise SystemExit(7)'], 'test-failure')
    log = next((job.directory / 'console').glob('*-test-failure.log'))
    assert 'expected test failure' in log.read_text()


def test_gpu_is_rechecked_immediately_before_gpu_child(job, monkeypatch):
    def busy():
        raise RuntimeError('new GPU job detected after preparation')
    monkeypatch.setattr(job, 'gpu_check', busy)
    with pytest.raises(RuntimeError, match='after preparation'):
        job.subprocess(job.train_command('sudoku', 'mdm', job.run_dir('sudoku', 'mdm')), 'train')
    assert not job.run_dir('sudoku', 'mdm').exists()


def test_scheduler_visibility_is_not_silently_widened(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
    with pytest.raises(ValueError, match='inherited CUDA_VISIBLE_DEVICES'):
        queue.BaselineQueue(queue.parser().parse_args(['plan', '--gpu', '0']))


def test_active_queue_lock_prevents_another_queue(job, fake_children):
    lock_path = queue.ROOT / '.cache/runtime/reasoning/queue-gpu0.lock'
    lock_path.parent.mkdir(parents=True)
    with lock_path.open('a') as lock:
        queue.fcntl.flock(lock, queue.fcntl.LOCK_EX | queue.fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='owns this GPU lock'):
            job.run()
    assert not fake_children


def test_normalized_child_env_and_tmux_pin_the_requested_geometry(job, monkeypatch):
    assert job.env['WORLD_SIZE'] == '1' and job.env['LOCAL_RANK'] == '0'
    assert job.env['CUDA_VISIBLE_DEVICES'] == '0'
    assert job.env['TMPDIR'].startswith(str(queue.ROOT / '.cache/runtime/reasoning'))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if 'has-session' in command else 0)
    monkeypatch.setattr(queue.subprocess, 'run', run)
    # A short scratch checkout keeps the real UNIX socket length guard active.
    with tempfile.TemporaryDirectory(prefix='queue-', dir=Path(__file__).resolve().parents[1] / '.cache') as root:
        monkeypatch.setattr(queue, 'ROOT', Path(root))
        job.tmux()
    actual = shlex.split(calls[-1][-1])
    assert option(actual, '--micro-batch') == option(actual, '--global-batch') == '128'
    assert 'WORLD_SIZE=1' in actual and 'LOCAL_RANK=0' in actual
    assert option(actual, '--queue-dir') == str(job.directory)


@pytest.mark.parametrize('flag,value', [('--micro-batch', '0'), ('--micro-batch', '256'),
                                      ('--gpu', '0,1'), ('--eval-examples', '1001')])
def test_invalid_settings_fail_before_gpu_work(flag, value):
    with pytest.raises(ValueError):
        queue.BaselineQueue(queue.parser().parse_args(['plan', flag, value]))
