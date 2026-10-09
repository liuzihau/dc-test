"""Follow-up sequencing must honor the live queue lock and successful completion."""
import fcntl
import json
import sys
import threading
import time

import pytest

from owt import schedule


def complete(root, variant, step=5000):
    path = root/variant
    path.mkdir(parents=True, exist_ok=True)
    (path/'complete.json').write_text(json.dumps({'optimizer_step':step}))


def test_followup_waits_for_original_lock_and_launches_only_fresh_zero_variant(tmp_path, monkeypatch):
    complete(tmp_path, 'mdm'); complete(tmp_path, 'mdm_np')
    original_state = tmp_path/'current.json'
    original_state.write_text('{"stage":"train","variant":"mdm_np"}')
    calls, failures = [], []
    def fake_run(command, **kwargs):
        calls.append(command)
        variant = command[command.index('--variant')+1]
        assert variant == 'mdm_np_zero_init'
        assert '--resume' not in command
        assert command[command.index('--steps')+1] == '5000'
        complete(tmp_path, variant)
    monkeypatch.setattr(schedule.subprocess, 'run', fake_run)
    monkeypatch.setattr(sys, 'argv', ['schedule', 'followup-zero', '--root', str(tmp_path)])
    def worker():
        try: schedule.main()
        except Exception as exc: failures.append(exc)
    with (tmp_path/'.queue.lock').open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        deadline = time.monotonic()+5
        while not (tmp_path/'zero_init_queue.json').exists() and time.monotonic()<deadline:
            time.sleep(.01)
        assert json.loads((tmp_path/'zero_init_queue.json').read_text())['stage'] == 'waiting_for_original_queue'
        assert thread.is_alive() and not calls
        fcntl.flock(held, fcntl.LOCK_UN)
    thread.join(5)
    assert not thread.is_alive() and not failures and len(calls) == 1
    assert json.loads((tmp_path/'zero_init_queue.json').read_text())['stage'] == 'complete'
    assert json.loads(original_state.read_text())['stage'] == 'train'


def test_failed_or_incomplete_original_run_never_launches_training(tmp_path, monkeypatch):
    complete(tmp_path, 'mdm'); complete(tmp_path, 'mdm_np', 4999)
    def unexpected(*args, **kwargs):
        pytest.fail('Training was launched before prerequisites completed')
    monkeypatch.setattr(schedule.subprocess, 'run', unexpected)
    monkeypatch.setattr(sys, 'argv', ['schedule', 'followup-zero', '--root', str(tmp_path)])
    with pytest.raises(RuntimeError, match='mdm_np has not successfully completed'):
        schedule.main()
    assert json.loads((tmp_path/'zero_init_queue.json').read_text())['stage'] == 'failed_prerequisites'


def test_user_required_diagnostic_blocks_training_relaunch(tmp_path, monkeypatch):
    (tmp_path/'reveal_sweep_required.json').write_text('{"status":"waiting_for_zero_completion"}')
    monkeypatch.setattr(sys,'argv',['owt.schedule','followup-low-weight','--root',str(tmp_path)])
    with pytest.raises(RuntimeError,match='sweep and analysis must precede'):
        schedule.main()
    assert not (tmp_path/'low_weight_queue.json').exists()


def final_validation(root, variant, value):
    folder = root/variant/'local_metrics'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'validation.csv').write_text(f'optimizer_step,val_nll\n5000,{value}\n')


def test_low_weight_policy_requires_successful_zero_completion(tmp_path):
    complete(tmp_path, 'mdm'); complete(tmp_path, 'mdm_np')
    with pytest.raises(RuntimeError, match='Zero-init NP'):
        schedule.low_weight_decision(tmp_path)


@pytest.mark.parametrize('zero,launch', [(4.12,True), (4.04,False), (3.98,False)])
def test_low_weight_policy_uses_final_matched_validation_only(tmp_path, zero, launch):
    for variant in ('mdm','mdm_np','mdm_np_zero_init'):
        complete(tmp_path, variant)
    final_validation(tmp_path, 'mdm', 4.02)
    final_validation(tmp_path, 'mdm_np_zero_init', zero)
    decision = schedule.low_weight_decision(tmp_path)
    assert decision['launch'] is launch
    assert decision['delta_nats'] == pytest.approx(zero-4.02)


def test_low_weight_policy_rejects_nonfinite_final_validation(tmp_path):
    for variant in ('mdm','mdm_np','mdm_np_zero_init'):
        complete(tmp_path, variant)
    final_validation(tmp_path, 'mdm', 4.02)
    final_validation(tmp_path, 'mdm_np_zero_init', 'nan')
    with pytest.raises(RuntimeError, match='Nonfinite'):
        schedule.low_weight_decision(tmp_path)


def test_conditional_followup_honors_shared_lock_and_runs_only_low_weight(tmp_path, monkeypatch):
    for variant in ('mdm','mdm_np','mdm_np_zero_init'):
        complete(tmp_path, variant)
    final_validation(tmp_path, 'mdm', 4.02)
    final_validation(tmp_path, 'mdm_np_zero_init', 4.12)
    original = tmp_path/'zero_init_queue.json'
    original.write_text('{"stage":"train","variant":"mdm_np_zero_init"}')
    calls, failures = [], []
    def fake_run(command, **kwargs):
        calls.append(command)
        assert command[command.index('--variant')+1] == 'mdm_np_zero_init_low_weight'
        assert '--resume' not in command
        complete(tmp_path, 'mdm_np_zero_init_low_weight')
    from owt import research
    monkeypatch.setattr(research, 'record_event', lambda *args, **kwargs: True)
    monkeypatch.setattr(schedule.subprocess, 'run', fake_run)
    monkeypatch.setattr(sys, 'argv', ['schedule','followup-low-weight','--root',str(tmp_path)])
    def worker():
        try: schedule.main()
        except Exception as exc: failures.append(exc)
    with (tmp_path/'.queue.lock').open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        deadline = time.monotonic()+5
        while not (tmp_path/'low_weight_queue.json').exists() and time.monotonic()<deadline:
            time.sleep(.01)
        assert not calls and thread.is_alive()
        fcntl.flock(held, fcntl.LOCK_UN)
    thread.join(5)
    assert not failures and len(calls) == 1
    assert json.loads((tmp_path/'low_weight_queue.json').read_text())['stage'] == 'complete'
    assert json.loads(original.read_text())['stage'] == 'train'


def test_conditional_followup_defers_when_zero_init_closes_deficit(tmp_path, monkeypatch):
    for variant in ('mdm','mdm_np','mdm_np_zero_init'):
        complete(tmp_path, variant)
    final_validation(tmp_path, 'mdm', 4.02)
    final_validation(tmp_path, 'mdm_np_zero_init', 4.03)
    from owt import research
    monkeypatch.setattr(research, 'record_event', lambda *args, **kwargs: True)
    monkeypatch.setattr(schedule.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Must defer'))
    monkeypatch.setattr(sys, 'argv', ['schedule','followup-low-weight','--root',str(tmp_path)])
    schedule.main()
    assert json.loads((tmp_path/'low_weight_queue.json').read_text())['stage'] == 'not_launched'
