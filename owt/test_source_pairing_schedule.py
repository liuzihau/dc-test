"""The next trial cannot bypass the current final review or its shared lock."""
import json
import subprocess
import sys
from pathlib import Path
import pytest
from owt import source_pairing_schedule as schedule


@pytest.fixture
def registered(tmp_path, monkeypatch):
    monkeypatch.setattr(schedule, 'ROOT', tmp_path)
    def write(name, data):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path
    source = write('source.json', {'version': 1})
    low = write('low.json', {'row_ids': [0, 1], 'fixed_fusion': {'lambda_weight': .2, 'scoring_row_ids': [1]}})
    complete = write('run/complete.json', {'optimizer_step': 5000})
    common = dict(protocol_sha256=schedule.digest(low), row_ids=[0, 1], cells=[{}]*5, preflight=False)
    collection = write('collection.json', common)
    report = write('report.json', dict(common, fitting_performed=False, training_evidence={'steps': 5000}, fixed_lambda=.2, scoring_row_ids=[1]))
    memo = write('review-memo.json', {'interpretation': 'Final outcomes reviewed; select the registered trial.'})
    protocol = write('protocol.json', dict(variants=list(schedule.VARIANTS), run_root='runs',
        source_sha256={'source.json': schedule.digest(source)}, low_weight_protocol='low.json',
        required_final_evidence=['run/complete.json', 'collection.json', 'report.json']))
    review = write('review.json', dict(scientific_review_complete=True, source_protocol_sha256=schedule.digest(protocol),
        selected_variants=[schedule.VARIANTS[0]], scientific_memo='review-memo.json',
        evidence_sha256={str(p.relative_to(tmp_path)):schedule.digest(p) for p in (complete, collection, report, memo)}))
    (tmp_path/'runs').mkdir()
    return protocol, review


def test_missing_review_cannot_authorize(registered):
    protocol, review = registered
    with pytest.raises(FileNotFoundError):
        schedule.verify_authorization(protocol, review.with_name('missing.json'), schedule.VARIANTS[0])
    with pytest.raises(ValueError, match='has not selected'):
        schedule.verify_authorization(protocol, review, schedule.VARIANTS[1])


def test_review_pins_evidence_and_sources(registered):
    protocol, review = registered
    schedule.verify_authorization(protocol, review, schedule.VARIANTS[0])
    (protocol.parent/'report.json').write_text('{}')
    with pytest.raises(ValueError, match='changed reviewed'):
        schedule.verify_authorization(protocol, review, schedule.VARIANTS[0])
    (protocol.parent/'source.json').write_text('{}')
    with pytest.raises(ValueError, match='source changed'):
        schedule.verify_authorization(protocol, review, schedule.VARIANTS[0])


def test_review_cannot_label_preflight_as_final(registered):
    protocol, review = registered
    collection = protocol.parent/'collection.json'
    artifact = json.loads(collection.read_text()); artifact['preflight'] = True
    collection.write_text(json.dumps(artifact))
    receipt = json.loads(review.read_text()); receipt['evidence_sha256']['collection.json'] = schedule.digest(collection)
    review.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='real final'):
        schedule.verify_authorization(protocol, review, schedule.VARIANTS[0])


def test_busy_training_lock_never_launches(registered, monkeypatch):
    protocol, review = registered
    monkeypatch.setattr(sys, 'argv', ['schedule', '--protocol', str(protocol), '--review', str(review), '--variant', schedule.VARIANTS[0]])
    import fcntl
    with (protocol.parent/'runs/.queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            schedule.main()
    assert not (protocol.parent/'runs'/schedule.VARIANTS[0]).exists()


def test_controller_has_no_torch_import():
    code = "import sys; import owt.source_pairing_schedule; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, '-c', code], check=True, cwd=Path(__file__).resolve().parents[1])
