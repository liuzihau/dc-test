"""CPU/fake-process checks for the non-invasive overnight coordinator."""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from reasoning import baseline_queue as base
from reasoning import overnight_queue as overnight


@pytest.fixture
def job(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith('REASONING_') or name in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK'):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(overnight, 'ROOT', tmp_path)
    monkeypatch.setattr(base, 'ROOT', tmp_path)
    args = overnight.parser().parse_args([
        'run', '--data-root', str(tmp_path / 'data'), '--output-root', str(tmp_path / 'runs'),
        '--queue-dir', str(tmp_path / 'queue'), '--idle-seconds', '2', '--poll-seconds', '1'])
    return overnight.OvernightQueue(args)


def fake_clock(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(overnight.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(overnight.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    return clock


def test_recipe_matches_original_memory_suite_exactly(job):
    reference = base.BaselineQueue(job.args)
    assert job.spec == reference.spec
    assert job.variants == ('both', 'both_aux')
    assert job.spec['merged_policy'] == 'current_preserving'
    assert job.spec['micro_batch'] == job.spec['global_batch'] == 128
    assert job.spec['gradient_mode'] == 'adjacent'


def test_parse_proc_with_spaces_and_parentheses(monkeypatch):
    fields = ['S'] + ['0'] * 18 + ['98765'] + ['0'] * 3
    monkeypatch.setattr(Path, 'read_text', lambda self: '123 (my (training) job) ' + ' '.join(fields))
    assert overnight.process_identity(123) == ('S', 98765)


def test_missing_zombie_and_reused_pid_are_not_predecessor(job, monkeypatch):
    job.args.wait_pid, job.args.wait_pid_starttime = 123, 456
    for value, expected in ((None, False), (('Z', 456), False), (('S', 999), False), (('S', 456), True)):
        monkeypatch.setattr(overnight, 'process_identity', lambda pid, value=value: value)
        assert job.predecessor_running() is expected


@pytest.mark.parametrize('extra', [
    ['--wait-pid', '4'], ['--wait-pid-starttime', '90'], ['--poll-seconds', '61'],
    ['--max-wait-seconds', '0'], ['--idle-seconds', '-1'], ['--report-every-seconds', '0'],
    ['--suite', 'baselines'], ['--wait-pid', '0', '--wait-pid-starttime', '1']])
def test_rejects_ambiguous_or_invalid_wait_settings(job, extra):
    with pytest.raises(ValueError):
        overnight.OvernightQueue(overnight.parser().parse_args(['run', *extra]))


def test_waits_for_exact_parent_not_momentary_gpu_idle(job, monkeypatch):
    clock = fake_clock(monkeypatch)
    reports, gpu_times = [], []
    monkeypatch.setattr(job, 'refresh_report', lambda **kwargs: reports.append(clock[0]))
    monkeypatch.setattr(job, 'predecessor_running', lambda: clock[0] < 3)
    monkeypatch.setattr(job, 'all_complete', lambda: False)
    monkeypatch.setattr(job, 'gpu_busy', lambda: gpu_times.append(clock[0]) or False)
    assert job.wait_for_handoff() is False
    assert clock[0] == 5
    assert gpu_times == [3, 4, 5]
    assert reports == [0, 1, 2, 3, 4, 5]


def test_idle_timer_resets_when_gpu_becomes_busy(job, monkeypatch):
    clock = fake_clock(monkeypatch)
    monkeypatch.setattr(job, 'refresh_report', lambda **kwargs: None)
    monkeypatch.setattr(job, 'predecessor_running', lambda: False)
    monkeypatch.setattr(job, 'all_complete', lambda: False)
    monkeypatch.setattr(job, 'gpu_busy', lambda: clock[0] == 1)
    assert job.wait_for_handoff() is False
    assert clock[0] == 4


def test_wait_timeout_never_signals_any_process(job, monkeypatch):
    fake_clock(monkeypatch)
    job.args.max_wait_seconds = 2
    monkeypatch.setattr(job, 'refresh_report', lambda **kwargs: None)
    monkeypatch.setattr(job, 'predecessor_running', lambda: True)
    monkeypatch.setattr(os, 'kill', lambda *args: pytest.fail('Coordinator must never kill jobs'))
    monkeypatch.setattr(job, 'gpu_busy', lambda: pytest.fail('Do not check idle between predecessor children'))
    with pytest.raises(TimeoutError):
        job.wait_for_handoff()


def test_completed_suite_skips_smoke_training_and_gpu_queries(job, monkeypatch):
    monkeypatch.setattr(job, 'refresh_report', lambda **kwargs: None)
    monkeypatch.setattr(job, 'predecessor_running', lambda: False)
    monkeypatch.setattr(job, 'all_complete', lambda: True)
    monkeypatch.setattr(job, 'gpu_busy', lambda: pytest.fail('Completed suite needs no GPU'))
    monkeypatch.setattr(base.BaselineQueue, 'run', lambda self: pytest.fail('Do not rerun completed suite'))
    job.run_after_wait()
    status = json.loads((job.directory / 'overnight_status.json').read_text())
    assert status['status'] == 'finished'
    assert status['resumed_training'] is False


def test_resumes_only_via_existing_memory_queue_and_refreshes_on_error(job, monkeypatch):
    reports = []
    monkeypatch.setattr(job, 'wait_for_handoff', lambda: False)
    monkeypatch.setattr(job, 'refresh_report', lambda **kwargs: reports.append(kwargs))
    def fail(self):
        assert self.spec['merged_policy'] == 'current_preserving'
        raise RuntimeError('Synthetic OOM')
    monkeypatch.setattr(base.BaselineQueue, 'run', fail)
    with pytest.raises(RuntimeError, match='Synthetic OOM'):
        job.run_after_wait()
    assert reports == [{'force': True}]
    assert json.loads((job.directory / 'overnight_status.json').read_text())['status'] == 'stopped'


def test_report_is_cpu_only_rate_limited_and_contains_existing_controls(job, monkeypatch):
    clock = fake_clock(monkeypatch)
    calls = []
    for task in base.TASKS:
        for variant in base.VARIANTS:
            directory = (job.args.output_root / task /
                         f'{variant}-h100-{job.label}-{job.batch_label}')
            directory.mkdir(parents=True)
            (directory / 'contract.json').write_text('{}')
        memory = job.run_dir(task, 'both')
        memory.mkdir(parents=True)
        (memory / 'contract.json').write_text('{}')
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert command[2] == '--runs'
        assert command[-2:] == ['--step', '5000']
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(overnight.subprocess, 'run', run)
    job.refresh_report()
    clock[0] = 100
    job.refresh_report()
    clock[0] = 301
    job.refresh_report()
    assert len(calls) == 2
    assert len(calls[0][3:calls[0].index('--output-dir')]) == 8


def test_report_failure_is_recorded_without_stopping_existing_training(job, monkeypatch):
    monkeypatch.setattr(job, 'report_runs', lambda: ['one-run'])
    monkeypatch.setattr(overnight.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=9))
    job.refresh_report(force=True)
    assert 'exited 9' in json.loads((job.directory / 'report_error.json').read_text())['error']


def test_all_complete_cannot_spawn_missing_evaluation(job, monkeypatch):
    base.atomic_json(job.directory / 'queue_config.json', job.spec)
    def needs_evaluation(self, task, variant, directory):
        self.subprocess(['python', 'runner', 'evaluate'], 'missing-eval')
    monkeypatch.setattr(base.BaselineQueue, 'evaluate', needs_evaluation)
    monkeypatch.setattr(base.BaselineQueue, 'subprocess', lambda *args: pytest.fail('Must not launch GPU evaluation'))
    assert job.all_complete() is False
    assert job.verification_only is False


def completed_metadata(job, task, variant):
    directory = job.run_dir(task, variant)
    contract = dict(task=task, variant=variant, micro_batch=128, global_batch=128,
                    world_size=1, seed=1, precision='bf16', device_type='cuda',
                    model_config=dict(memory_mode='both', attention_mode='merged',
                        merged_policy='current_preserving', gradient_mode='adjacent', trajectory='five',
                        neighbors=variant == 'both_aux', neighbor_weight=0.5, gate_enabled=False,
                        cache_only_probability=0.0, current_only_probability=0.05, final_dropout=0.10,
                        identity_probability=0.25, identity_weight=0.10, identity_margin=0.05,
                        identity_final_probability=0.50, hidden_size=512, n_heads=8, n_layers=6))
    base.atomic_json(directory / 'contract.json', contract)
    target = directory / 'checkpoints/step-5000.pt'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'fake fixture; never load')
    (target.parent / 'last.pt').symlink_to(target.name)
    base.atomic_json(target.with_suffix('.pt.json'), dict(step=5000, sha256='test-digest'))
    output = directory / 'evaluation-last-step5000-test-digest-n1000.json'
    base.atomic_json(output, dict(contract=contract))
    return directory, contract, output


def test_completed_metadata_still_invokes_existing_hash_protocol_verifier(job, monkeypatch):
    base.atomic_json(job.directory / 'queue_config.json', job.spec)
    calls = []
    for task in base.TASKS:
        for variant in job.variants:
            completed_metadata(job, task, variant)
    monkeypatch.setattr(base.BaselineQueue, 'evaluate', lambda self, task, variant, directory:
                        calls.append((task, variant)))
    assert job.all_complete() is True
    assert calls == [(task, variant) for task in base.TASKS for variant in job.variants]


@pytest.mark.parametrize('change', ['legacy', 'wrong_neighbors', 'changed_evaluation_contract'])
def test_complete_skip_rejects_wrong_model_or_different_evaluation_contract(job, change):
    base.atomic_json(job.directory / 'queue_config.json', job.spec)
    directory, contract, output = completed_metadata(job, 'sudoku', 'both')
    if change == 'changed_evaluation_contract':
        base.atomic_json(output, dict(contract={**contract, 'seed': 2}))
    else:
        if change == 'legacy':
            contract['model_config']['merged_policy'] = 'legacy'
        else:
            contract['model_config']['neighbors'] = True
        base.atomic_json(directory / 'contract.json', contract)
        base.atomic_json(output, dict(contract=contract))
    with pytest.raises(ValueError, match='Completed-run contract'):
        job.all_complete()


def test_mismatched_existing_spec_stops_instead_of_mixing_runs(job):
    base.atomic_json(job.directory / 'queue_config.json', {**job.spec, 'micro_batch': 8})
    with pytest.raises(ValueError, match='settings differ'):
        job.all_complete()


def test_reports_after_each_verified_evaluation(job, monkeypatch):
    order = []
    monkeypatch.setattr(base.BaselineQueue, 'evaluate', lambda *args: order.append('evaluate'))
    monkeypatch.setattr(job, 'refresh_report', lambda **kwargs: order.append(('report', kwargs)))
    job.evaluate('sudoku', 'both', job.run_dir('sudoku', 'both'))
    assert order == ['evaluate', ('report', {'force': True})]
