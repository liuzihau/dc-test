"""CPU-only launch gates: unresolved settings, evidence, pins and shared lock."""
import fcntl
import json
import sys
from unittest.mock import patch

import pytest

from owt import transformer_np_schedule as schedule


def selection(tmp_path, monkeypatch):
    monkeypatch.setattr(schedule, 'ROOT', tmp_path)
    # Keep the production list intact; create isolated fixtures for all its pins.
    protocol = tmp_path / 'outputs/research-notes/source_pairing_diagnostic_protocol_20261002.json'
    protocol.parent.mkdir(parents=True)
    protocol.write_text(json.dumps({'source_sha256': {'dependency.py': 'original'}}))
    pins = {}
    for name in (*schedule.IMPLEMENTATION_FILES, 'dependency.py'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('immutable fixture')
        pins[name] = schedule.digest(path)
    evidence = {}
    for name in schedule.REQUIRED_EVIDENCE:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'optimizer_step': 5000} if name.endswith('complete.json')
            else {'preflight': False, 'cells': list(range(5))}))
        evidence[name] = schedule.digest(path)
    lock = tmp_path / schedule.LOCK
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.touch()
    data = dict(execution_ready=True, scientific_review_complete=True,
        selected_variant=schedule.VARIANTS[0], np_weight_per_direction=.05,
        run_root=str(schedule.RUN_ROOT), source_sha256=pins, evidence_sha256=evidence)
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(data))
    return path, data


def test_complete_selected_trial_can_be_verified(tmp_path, monkeypatch):
    path, data = selection(tmp_path, monkeypatch)
    assert schedule.verify_selection(path, schedule.VARIANTS[0]) == data


@pytest.mark.parametrize('change', ['pending', 'weight', 'pin', 'missing_evidence', 'preflight'])
def test_incomplete_or_changed_trial_is_rejected(tmp_path, monkeypatch, change):
    path, data = selection(tmp_path, monkeypatch)
    if change == 'pending':
        data['execution_ready'] = False
    elif change == 'weight':
        data['np_weight_per_direction'] = float('nan')
    elif change == 'pin':
        (tmp_path / schedule.IMPLEMENTATION_FILES[0]).write_text('changed')
    elif change == 'missing_evidence':
        data['evidence_sha256'].clear()
    else:
        report = schedule.REQUIRED_EVIDENCE[1]
        artifact = tmp_path / report
        artifact.write_text(json.dumps({'preflight': True, 'cells': list(range(5))}))
        data['evidence_sha256'][report] = schedule.digest(artifact)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        schedule.verify_selection(path, schedule.VARIANTS[0])


def test_busy_shared_gpu_lock_never_starts_a_trainer(tmp_path, monkeypatch):
    path, _ = selection(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['schedule', '--selection', str(path), '--variant', schedule.VARIANTS[0]])
    with (tmp_path / schedule.LOCK).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with patch.object(schedule.subprocess, 'Popen') as launch:
            with pytest.raises(BlockingIOError):
                schedule.main()
            launch.assert_not_called()


def test_direct_entrypoint_requires_selection_environment(monkeypatch):
    monkeypatch.delenv('NP_TRANSFORMER_SELECTION', raising=False)
    monkeypatch.delenv('NP_TRANSFORMER_VARIANT', raising=False)
    with pytest.raises(RuntimeError, match='final loss-condition'):
        schedule.entrypoint_authorization()


def transformer_first_selection(tmp_path, monkeypatch):
    path, data = selection(tmp_path, monkeypatch)
    # The deferred arm has no completion or report; do not invent its evidence.
    for name in schedule.REQUIRED_EVIDENCE[2:]:
        (tmp_path / name).unlink()
        del data['evidence_sha256'][name]
    review = tmp_path / schedule.MASKED_REVIEW
    review.write_text(json.dumps(dict(scientific_review_complete=True,
        variant='mdm_np_zero_init_masked_source',
        report_sha256=schedule.digest(tmp_path / schedule.REQUIRED_EVIDENCE[1]),
        primary_validation={'mdm_np_zero_init_masked_source': {'optimizer_step': 5000}})))
    data['evidence_sha256'][schedule.MASKED_REVIEW] = schedule.digest(review)
    data.update(experiment_order='transformer_first', count_control_deferred_by_user=True,
        user_instruction="I'd like to do2 first, and then see if1 is needed",
        selected_variant=schedule.VARIANTS[1], np_weight_per_direction=.25,
        linear_reference='mdm_np_zero_init_masked_source')
    path.write_text(json.dumps(data))
    return path, data


def test_explicit_transformer_first_order_does_not_require_deferred_control(tmp_path, monkeypatch):
    path, data = transformer_first_selection(tmp_path, monkeypatch)
    assert schedule.verify_selection(path, schedule.VARIANTS[1]) == data


@pytest.mark.parametrize('change', ['no_user_instruction', 'different_reference', 'unreviewed', 'changed_report'])
def test_transformer_first_still_requires_user_order_and_reviewed_matching_reference(tmp_path, monkeypatch, change):
    path, data = transformer_first_selection(tmp_path, monkeypatch)
    if change == 'no_user_instruction':
        data.pop('user_instruction')
    elif change == 'different_reference':
        data['np_weight_per_direction'] = .05
    elif change == 'unreviewed':
        review = tmp_path / schedule.MASKED_REVIEW
        value = json.loads(review.read_text())
        value['scientific_review_complete'] = False
        review.write_text(json.dumps(value))
        data['evidence_sha256'][schedule.MASKED_REVIEW] = schedule.digest(review)
    else:
        report = tmp_path / schedule.REQUIRED_EVIDENCE[1]
        value = json.loads(report.read_text())
        value['changed'] = True
        report.write_text(json.dumps(value))
        data['evidence_sha256'][schedule.REQUIRED_EVIDENCE[1]] = schedule.digest(report)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        schedule.verify_selection(path, schedule.VARIANTS[1])
