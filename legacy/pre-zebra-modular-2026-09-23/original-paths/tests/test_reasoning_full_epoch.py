"""Full-source packing, exact one-epoch accounting, recovery and queue contracts."""
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from reasoning.benchmark import convert_sudoku, convert_zebra
from reasoning.data import ReasoningDataset, _sha256, prepare_dataset, write_prepared_dataset
from reasoning.full_data import prepare_full
from reasoning.full_epoch_queue import full_training_command, verify_subset_completion
from reasoning import runner
from reasoning.zebra_official import load_source, convert_source_record
from test_reasoning_benchmark import sudoku_row
from test_reasoning_official_zebra import raw_record
from test_reasoning_runner import train_args, assert_tree_equal, cpu_only


@pytest.fixture(params=['sudoku-benchmark', 'zebra-benchmark'])
def full_source(tmp_path, request):
    task = request.param
    if task.startswith('sudoku'):
        train, test = tmp_path / 'train.npy', tmp_path / 'test.npy'
        rows = [sudoku_row(i + 100) for i in range(6)]
        np.save(train, np.array(rows + [rows[0]]))
        np.save(test, np.array([rows[4], rows[5]]))
        converted = [convert_sudoku(r, 'train', i) for i, r in enumerate(rows)]
    else:
        train, test = tmp_path / 'train.pkl', tmp_path / 'test.pkl'
        rows = [raw_record(i) for i in range(6)]
        train.write_bytes(pickle.dumps(rows + [rows[0]], protocol=4))
        test.write_bytes(pickle.dumps([rows[4], rows[5]], protocol=4))
        converted = [convert_zebra(convert_source_record(r, 'train', i))
                     for i, r in enumerate(load_source(train)[:6])]
    frozen = tmp_path / 'frozen'
    source = dict(source_files={s: dict(sha256=_sha256(p)) for s, p in (('train', train), ('test', test))})
    write_prepared_dataset(frozen, task, dict(train=[converted[0]],
                           validation=[converted[2]], test=[converted[4]]), 17, source)
    return task, train, test, frozen, converted


def test_full_sources_exclude_all_test_and_validation_dedup_and_preserve_tensors(full_source, tmp_path):
    task, train, test, frozen, records = full_source
    output = tmp_path / 'full'
    result = prepare_full(task, train, test, frozen, output)
    assert result['source']['counters'] == dict(raw_train=7, excluded_test=2,
           excluded_validation=1, duplicate_train=1, retained_train=3)
    for split in ('validation', 'test'):
        assert (output / (split + '.jsonl')).read_bytes() == (frozen / (split + '.jsonl')).read_bytes()
        assert ReasoningDataset(output, split).records == ReasoningDataset(frozen, split).records
    packed = ReasoningDataset(output)
    assert len(packed) == 3
    assert packed.source_indices.tolist() == [0, 1, 3]
    reference = tmp_path / 'reference'
    write_prepared_dataset(reference, task, dict(train=[records[i] for i in (0, 1, 3)],
                           validation=[records[2]], test=[records[4]]), 17, {})
    original = ReasoningDataset(reference)
    for i in range(3):
        assert_tree_equal(packed[i], original[i])
    with pytest.raises(FileExistsError):
        prepare_full(task, train, test, frozen, output)
    tokens = output / result['splits']['train']['files']['tokens']['filename']
    with tokens.open('r+b') as stream:
        stream.write(b'\xff')
    with pytest.raises(ValueError, match='checksum'):
        ReasoningDataset(output)


def test_full_source_hash_must_match_frozen_provenance(full_source, tmp_path):
    task, train, test, frozen, _ = full_source
    with train.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='Raw source differs'):
        prepare_full(task, train, test, frozen, tmp_path / 'full')


@pytest.mark.parametrize('variant', ['vanilla', 'mdm', 'mdm_aux', 'both', 'both_aux'])
def test_one_epoch_partial_tail_resume_and_single_exposure(tmp_path, monkeypatch, variant):
    data = tmp_path / 'data'
    prepare_dataset(data, 'countdown', train_size=7, valid_size=2, test_size=2, seed=47)
    seen = []
    class TrackedDataset(ReasoningDataset):
        def __init__(self, directory, split='train', **kwargs):
            super().__init__(directory, split, **kwargs)
            self.track = split == 'train'
        def __getitem__(self, index):
            if self.track:
                seen.append(index)
            return super().__getitem__(index)
    monkeypatch.setattr('reasoning.data.ReasoningDataset', TrackedDataset)
    def args(run, pause=None):
        a = train_args(data, run, max_steps=5000, variant=variant)
        a.global_batch, a.micro_batch = 4, 2
        a.one_epoch, a.stop_after_steps = True, pause
        a.validation_protocol = 'both'
        return a
    direct, resumed = tmp_path / 'direct', tmp_path / 'resumed'
    runner.train(args(direct))
    direct_order = seen.copy(); seen.clear()
    assert len(direct_order) == 7 and len(set(direct_order)) == 7
    assert direct_order == runner.GlobalExampleStream(7, 14).indices(0, 7)
    runner.train(args(resumed, 1))
    assert json.loads((resumed / 'status.json').read_text())['status'] == 'paused'
    runner.train(args(resumed))
    assert seen == direct_order
    expected = runner.load_checkpoint(direct / 'checkpoints/last.pt')
    actual = runner.load_checkpoint(resumed / 'checkpoints/last.pt')
    assert actual['step'] == 2 and actual['examples_seen'] == 7
    assert actual['contract']['epochs'] == 1 and actual['contract']['epoch_examples'] == 7
    assert_tree_equal(actual['model'], expected['model'])
    assert_tree_equal(actual['optimizer'], expected['optimizer'])
    assert_tree_equal(actual['rng_by_rank'], expected['rng_by_rank'])
    validation = json.loads((resumed / 'validation/step-000000002.json').read_text())
    assert validation['protocol'] == 'cold_independent'
    assert validation['nested_teacher_forced']['protocol'] == 'teacher_forced_nested'
    if variant in ('vanilla', 'mdm', 'mdm_aux'):
        assert validation['ratios'] == validation['nested_teacher_forced']['ratios']
    seen.clear()
    runner.train(args(resumed))
    assert seen == []  # Completed epoch cannot silently become a second epoch.


def test_one_epoch_distributed_tail_weights_cover_each_example_once():
    parts = [runner.distributed_microbatch(3, 2, 2, rank, 0) for rank in range(2)]
    assert parts == [(0, 2, 4/3), (2, 1, 2/3)]
    # DDP mean of rank-weighted losses is the three-example mean.
    assert sum(weight for _, _, weight in parts) / 2 == 1


def test_full_queue_commands_keep_all_five_controls_and_exact_epoch():
    from reasoning.benchmark_queue import TASKS, VARIANTS
    for task in TASKS:
        for variant in VARIANTS:
            command = full_training_command(task, variant, Path('data'), Path('run'), 1498933)
            a = runner.parser().parse_args(command[3:])
            assert a.one_epoch and a.max_steps == math.ceil(1498933 / 128)
            assert a.global_batch == a.micro_batch == 128
            assert a.validation_protocol == 'both' and a.validation_examples == 1000
            assert a.val_every == a.save_every == 500 and a.save_seconds == 1200
            assert a.neighbor_weight == .5
            assert a.gradient_mode == ('detached' if variant == 'vanilla' else 'adjacent')
            assert a.merged_policy == ('legacy' if variant == 'vanilla' else 'current_preserving')


def test_handoff_does_not_accept_a_checkpoint_without_finished_evaluation(tmp_path):
    assert verify_subset_completion(tmp_path, tmp_path) is None


def test_handoff_waits_then_retires_only_old_queue(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from reasoning import full_epoch_queue as q
    args = SimpleNamespace(action='run', output=tmp_path / 'new', gpu='0',
                           after_subset=tmp_path / 'old', data_root=tmp_path / 'data')
    queue = q.FullEpochQueue(args)
    ready, running, calls = [False], [True], []
    monkeypatch.setattr(q, 'verify_subset_completion', lambda *a: {'verified': True} if ready[0] else None)
    def supervisor(command, **kwargs):
        calls.append(command)
        assert command[0] == 'supervisorctl'
        if command[1] in ('status', 'stop', 'update'):
            assert command[2] == 'dcache_benchmark_v2'
        if command[1] == 'stop':
            running[0] = False
        return SimpleNamespace(stdout='RUNNING' if running[0] else 'STOPPED', returncode=0)
    monkeypatch.setattr(q.subprocess, 'run', supervisor)
    read_text = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda self, *a, **k:
                        'autostart=false' if str(self) == '/etc/supervisor/conf.d/dcache_benchmark_v2.conf'
                        else read_text(self, *a, **k))
    monkeypatch.setattr(q, 'report', lambda *a, **k: None)
    assert not queue.handoff()
    assert calls == [['supervisorctl', 'status', 'dcache_benchmark_v2']]
    ready[0] = True
    assert queue.handoff()
    assert ['supervisorctl', 'stop', 'dcache_benchmark_v2'] in calls
    assert ['supervisorctl', 'update', 'dcache_benchmark_v2'] in calls
    assert json.loads(queue.handoff_path.read_text())['verified']
    previous = len(calls)
    assert queue.handoff() and len(calls) == previous


def test_report_mode_cannot_stop_training_service(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from reasoning.full_epoch_queue import FullEpochQueue
    args = SimpleNamespace(action='report', output=tmp_path, gpu='0',
                           after_subset=tmp_path / 'old', data_root=tmp_path / 'data')
    queue = FullEpochQueue(args)
    def forbidden(*args, **kwargs):
        raise AssertionError('Plot refresh must never manage the training service')
    monkeypatch.setattr('reasoning.full_epoch_queue.subprocess.run', forbidden)
    assert queue.handoff()


def test_queue_records_failure_without_discarding_stage_context(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from reasoning.full_epoch_queue import FullEpochQueue
    args = SimpleNamespace(action='run', output=tmp_path / 'run', gpu='0',
                           after_subset=None, data_root=tmp_path / 'data', raw_root=tmp_path / 'raw')
    queue = FullEpochQueue(args)
    def failed(*args, **kwargs):
        runner.atomic_json(queue.output / 'status.json', dict(status='running', stage='prepare', console='example.log'))
        raise RuntimeError('preparation failed')
    monkeypatch.setattr(queue, 'execute', failed)
    with pytest.raises(RuntimeError, match='preparation failed'):
        queue.run()
    status = json.loads((queue.output / 'status.json').read_text())
    assert status['status'] == 'stopped' and status['stage'] == 'prepare'
    assert status['console'] == 'example.log'
    assert json.loads((queue.output / 'failure.json').read_text())['error'] == 'preparation failed'
