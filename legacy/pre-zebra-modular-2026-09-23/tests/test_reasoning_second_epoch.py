"""Exact epoch-tail continuation, unchanged state and safe queue handoff."""
import json
import math
from pathlib import Path

import pytest

from reasoning import runner
from reasoning.data import ReasoningDataset, prepare_dataset
from reasoning.second_epoch_queue import (
    acquire_finished_predecessor, continuation_command, fork_second_epoch, fork_next_epoch, paired_changes,
    split_continuation_command, validate_split_source)
from reasoning.benchmark_queue import TASKS, VARIANTS
from reasoning.variants import SPLIT_VARIANTS, SPLIT_LABELS
from test_reasoning_runner import train_args, assert_tree_equal, cpu_only


@pytest.mark.parametrize('variant', VARIANTS)
def test_fork_epoch2_matches_uninterrupted_and_resumed_training(tmp_path, monkeypatch, variant):
    data = tmp_path/'data'
    prepare_dataset(data, 'countdown', train_size=7, valid_size=2, test_size=2, seed=47)
    seen = []
    class Tracked(ReasoningDataset):
        def __init__(self, directory, split='train', **kwargs):
            super().__init__(directory, split, **kwargs)
            self.track = split == 'train'
        def __getitem__(self, index):
            if self.track:
                seen.append(index)
            return super().__getitem__(index)
    monkeypatch.setattr('reasoning.data.ReasoningDataset', Tracked)
    def args(run, epochs=None, pause=None):
        a = train_args(data, run, variant=variant)
        a.global_batch, a.micro_batch = 4, 2
        a.one_epoch, a.epochs = epochs is None, epochs
        a.validation_protocol, a.stop_after_steps = 'both', pause
        return a
    first, second, direct = [tmp_path/x for x in ('first', 'second', 'direct')]
    runner.train(args(first))
    source = (first/'checkpoints/last.pt').resolve()
    source_digest = runner.digest(source)
    original = runner.load_checkpoint(source)
    seen.clear()
    fork_second_epoch(first, second)
    forked = runner.load_checkpoint(second/'checkpoints/last.pt')
    for key in ('model', 'optimizer', 'rng_by_rank', 'step', 'examples_seen', 'model_config'):
        assert_tree_equal(forked[key], original[key])
    assert forked['contract'] == dict(original['contract'], epochs=2)
    runner.train(args(second, 2, 3))
    assert json.loads((second/'status.json').read_text())['status'] == 'paused'
    fork_second_epoch(first, second)  # Idempotent after additional updates too.
    runner.train(args(second, 2))
    assert seen == runner.GlobalExampleStream(7, 14).indices(7, 7)
    assert len(set(seen)) == 7
    seen.clear()
    runner.train(args(direct, 2))
    assert seen == runner.GlobalExampleStream(7, 14).indices(0, 14)
    actual = runner.load_checkpoint(second/'checkpoints/last.pt')
    expected = runner.load_checkpoint(direct/'checkpoints/last.pt')
    assert actual['step'] == 4 and actual['examples_seen'] == 14
    for key in ('model', 'optimizer', 'rng_by_rank', 'contract'):
        assert_tree_equal(actual[key], expected[key])
    assert runner.digest(source) == source_digest
    assert json.loads((first/'contract.json').read_text())['epochs'] == 1
    seen.clear()
    runner.train(args(second, 2))
    assert seen == []
    # Plain one-epoch launch must not silently continue the extended contract.
    with pytest.raises(ValueError, match='Resume contract differs'):
        runner.train(args(second))


def test_production_zebra_partial_tail_offsets():
    contract = dict(global_batch=128, epoch_examples=853792, epochs=2)
    assert runner.examples_at_step(6670, contract) == 853760
    assert runner.examples_at_step(6671, contract) == 853792
    assert runner.examples_at_step(6672, contract) == 853920
    assert runner.examples_at_step(13342, contract) == 1707584


def test_third_epoch_production_partial_tail_offsets():
    for count, per_epoch in [(853792, 6671), (1803463, 14090)]:
        contract = dict(global_batch=128, epoch_examples=count, epochs=3)
        assert runner.examples_at_step(2*per_epoch, contract) == 2*count
        assert runner.examples_at_step(2*per_epoch+1, contract) == 2*count+128
        assert runner.examples_at_step(3*per_epoch, contract) == 3*count


@pytest.mark.parametrize('variant', SPLIT_VARIANTS)
def test_fork_third_epoch_exact_resume_and_next_data_permutation(tmp_path, monkeypatch, variant):
    data = tmp_path/'data'
    prepare_dataset(data, 'countdown', train_size=7, valid_size=2, test_size=2, seed=47)
    seen = []
    class Tracked(ReasoningDataset):
        def __init__(self, directory, split='train', **kwargs):
            super().__init__(directory, split, **kwargs)
            self.track = split == 'train'
        def __getitem__(self, index):
            if self.track:
                seen.append(index)
            return super().__getitem__(index)
    monkeypatch.setattr('reasoning.data.ReasoningDataset', Tracked)
    def args(run, epochs, pause=None):
        a = train_args(data, run, variant=variant)
        a.suite = 'split'
        a.global_batch, a.micro_batch = 4, 2
        a.one_epoch, a.epochs = False, epochs
        a.validation_protocol, a.stop_after_steps = 'both', pause
        return a
    second, third, direct = [tmp_path/name for name in ('second','third','direct')]
    runner.train(args(second, 2))
    source_path = (second/'checkpoints/last.pt').resolve()
    source_hash = runner.digest(source_path)
    original = runner.load_checkpoint(source_path)
    seen.clear()
    fork_next_epoch(second, third, target_epochs=3)
    forked = runner.load_checkpoint(third/'checkpoints/last.pt')
    for key in ('model', 'optimizer', 'rng_by_rank', 'step', 'examples_seen', 'model_config'):
        assert_tree_equal(forked[key], original[key])
    assert forked['contract'] == dict(original['contract'], epochs=3)
    provenance = json.loads((third/'continuation_source.json').read_text())
    assert (provenance['source_step'],provenance['target_step']) == (4,6)
    assert (provenance['source_examples'],provenance['target_examples']) == (14,21)
    runner.train(args(third, 3, 5))
    fork_next_epoch(second, third, target_epochs=3)
    runner.train(args(third, 3))
    assert seen == runner.GlobalExampleStream(7, 14).indices(14,7)
    assert len(set(seen)) == 7
    runner.train(args(direct, 3))
    actual = runner.load_checkpoint(third/'checkpoints/last.pt')
    expected = runner.load_checkpoint(direct/'checkpoints/last.pt')
    for key in ('model', 'optimizer', 'rng_by_rank', 'contract', 'examples_seen', 'step'):
        assert_tree_equal(actual[key], expected[key])
    assert runner.digest(source_path) == source_hash
    assert actual['step'] == 6 and actual['examples_seen'] == 21
    with pytest.raises(ValueError, match='preceding-epoch'):
        fork_next_epoch(second, tmp_path/'skip', target_epochs=4)
    seen.clear()
    runner.train(args(third, 3))
    assert not seen


def test_all_third_epoch_commands_preserve_recipe():
    for task in TASKS:
        for variant in SPLIT_VARIANTS:
            old = split_continuation_command(task, variant, 'data', 'run')
            new = split_continuation_command(task, variant, 'data', 'run', target_epochs=3)
            expected = old.copy()
            expected[expected.index('--epochs')+1] = '3'
            assert new == expected


def test_all_ten_commands_and_protocols():
    for task in TASKS:
        for variant in VARIANTS:
            command = continuation_command(task, variant, Path('data'), Path('run'))
            a = runner.parser().parse_args(command[3:])
            assert a.epochs == 2 and not a.one_epoch
            assert a.task == task and a.variant == variant
            assert a.global_batch == a.micro_batch == 128
            assert a.validation_protocol == 'both' and a.validation_examples == 1000
            assert a.val_every == a.save_every == 500 and a.save_seconds == 1200
            assert a.neighbor_weight == .5


@pytest.mark.parametrize('variant', SPLIT_VARIANTS)
def test_split_continuation_command_and_exact_state(tmp_path, variant):
    for task in TASKS:
        command = split_continuation_command(task, variant, 'data', 'run')
        assert command[command.index('--nproc_per_node')+1] == '2'
        a = runner.parser().parse_args(command[command.index('train'):])
        assert a.suite == 'split' and a.variant == variant
        assert a.epochs == 2 and not a.one_epoch
        assert a.micro_batch == a.eval_batch_size == 32 and a.global_batch == 128
        assert a.lr == .0003 and a.warmup_steps == 1000
        assert a.validation_protocol == 'both' and a.val_every == 500
        assert a.save_every == 500 and a.save_seconds == 1200
    data = tmp_path/'data'
    prepare_dataset(data, 'countdown', train_size=7, valid_size=2, test_size=2, seed=47)
    def args(run, epochs=None):
        a = train_args(data, run, variant=variant)
        a.suite = 'split'
        a.global_batch, a.micro_batch = 4, 2
        a.one_epoch, a.epochs = epochs is None, epochs
        a.validation_protocol = 'both'
        return a
    first, second, direct = [tmp_path/name for name in ('first', 'second', 'direct')]
    runner.train(args(first))
    before = runner.load_checkpoint(first/'checkpoints/last.pt')
    source_digest = runner.digest((first/'checkpoints/last.pt').resolve())
    fork_second_epoch(first, second)
    forked = runner.load_checkpoint(second/'checkpoints/last.pt')
    for key in ('model', 'optimizer', 'rng_by_rank', 'step', 'examples_seen', 'model_config'):
        assert_tree_equal(forked[key], before[key])
    runner.train(args(second, 2))
    runner.train(args(direct, 2))
    actual = runner.load_checkpoint(second/'checkpoints/last.pt')
    expected = runner.load_checkpoint(direct/'checkpoints/last.pt')
    for key in ('model', 'optimizer', 'rng_by_rank', 'contract', 'examples_seen', 'step'):
        assert_tree_equal(actual[key], expected[key])
    assert runner.digest((first/'checkpoints/last.pt').resolve()) == source_digest


def test_split_predecessor_waits_for_all_twelve_and_lock(tmp_path):
    runner.atomic_json(tmp_path/'status.json', {'status': 'running'})
    assert acquire_finished_predecessor(tmp_path, TASKS, SPLIT_VARIANTS) is None
    runner.atomic_json(tmp_path/'status.json', {'status': 'finished'})
    for task in TASKS:
        for variant in SPLIT_VARIANTS:
            run = tmp_path/task/variant
            runner.atomic_json(run/'status.json', dict(status='finished', max_steps=2))
            runner.atomic_json(run/'generation.json', dict(step=2, metrics={'num_examples': 1000},
                              contract=dict(task=task, variant=variant)))
    last = tmp_path/TASKS[-1]/SPLIT_VARIANTS[-1]/'generation.json'
    saved = json.loads(last.read_text())
    runner.atomic_json(last, dict(saved, step=1))
    with pytest.raises(ValueError, match='All 12'):
        acquire_finished_predecessor(tmp_path, TASKS, SPLIT_VARIANTS)
    runner.atomic_json(last, saved)
    lock = acquire_finished_predecessor(tmp_path, TASKS, SPLIT_VARIANTS)
    assert lock is not None
    try:
        assert acquire_finished_predecessor(tmp_path, TASKS, SPLIT_VARIANTS) is None
    finally:
        lock.close()


def test_split_source_rejects_changed_geometry_and_recipe(tmp_path):
    contract = dict(suite='split', epochs=1, world_size=2, micro_batch=32,
                    global_batch=128, lr=.0003, warmup_steps=1000, precision='bf16',
                    seed=1, validation_protocol='both', variant='tt_ea_rm_np',
                    model_config=dict(gradient_mode='adjacent', identity_probability=.25))
    runner.atomic_json(tmp_path/'contract.json', contract)
    assert validate_split_source(tmp_path, 2, 32) == contract
    with pytest.raises(ValueError, match='batch/optimizer'):
        validate_split_source(tmp_path, 2, 8)
    with pytest.raises(ValueError, match='robustness'):
        validate_split_source(tmp_path, 2, 32, False)
    with pytest.raises(ValueError, match='batch/optimizer'):
        validate_split_source(tmp_path, 2, 32, source_epochs=2)
    runner.atomic_json(tmp_path/'contract.json', dict(contract, epochs=2))
    assert validate_split_source(tmp_path, 2, 32, source_epochs=2)['epochs'] == 2


@pytest.mark.parametrize('target_epochs', [2, 3])
def test_split_two_rank_full_state_continuation(tmp_path, target_epochs):
    import os
    import socket
    import subprocess
    import sys
    try:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
    except PermissionError:
        pytest.skip('Local DDP rendezvous requires unsandboxed CPU integration test')
    data = tmp_path/'data'
    prepare_dataset(data, 'countdown', train_size=9, valid_size=1, test_size=1, seed=48)
    first, second, direct = [tmp_path/name for name in ('first', 'second', 'direct')]
    def train(run, epochs):
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2',
            'scripts/reasoning/run_reasoning.py', 'train', '--suite', 'split', '--variant', 'tt_ea_rm_np',
            '--task', 'countdown', '--data-dir', str(data), '--run-dir', str(run), '--size', 'debug',
            '--device', 'cpu', '--precision', 'fp32', '--global-batch', '8', '--micro-batch', '2',
            '--epochs', str(epochs), '--warmup-steps', '2', '--val-every', '2', '--validation-examples', '1',
            '--eval-batch-size', '1', '--save-every', '1', '--log-every', '1', '--cpu-threads', '1',
            '--validation-protocol', 'both', '--stress-memory-routes']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
        for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'):
            env.pop(name, None)
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env,
                                capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout+result.stderr
    train(first, target_epochs-1)
    before = runner.load_checkpoint(first/'checkpoints/last.pt')
    assert len(before['rng_by_rank']) == 2
    source_hash = runner.digest((first/'checkpoints/last.pt').resolve())
    fork_next_epoch(first, second, target_epochs)
    forked = runner.load_checkpoint(second/'checkpoints/last.pt')
    for key in ('model', 'optimizer', 'rng_by_rank', 'step', 'examples_seen'):
        assert_tree_equal(forked[key], before[key])
    train(second, target_epochs)
    train(direct, target_epochs)
    actual = runner.load_checkpoint(second/'checkpoints/last.pt')
    expected = runner.load_checkpoint(direct/'checkpoints/last.pt')
    assert actual['step'] == 2*target_epochs and actual['examples_seen'] == 9*target_epochs
    for key in ('model', 'optimizer', 'rng_by_rank', 'contract', 'step', 'examples_seen'):
        assert_tree_equal(actual[key], expected[key])
    assert runner.digest((first/'checkpoints/last.pt').resolve()) == source_hash


def test_predecessor_waits_for_all_ten_evaluations(tmp_path):
    runner.atomic_json(tmp_path/'status.json', {'status': 'running'})
    assert acquire_finished_predecessor(tmp_path) is None
    runner.atomic_json(tmp_path/'status.json', {'status': 'stopped'})
    with pytest.raises(RuntimeError, match='stopped'):
        acquire_finished_predecessor(tmp_path)
    runner.atomic_json(tmp_path/'status.json', {'status': 'finished'})
    for task in TASKS:
        for variant in VARIANTS:
            run = tmp_path/task/variant
            runner.atomic_json(run/'status.json', dict(status='finished', max_steps=2))
            runner.atomic_json(run/'generation.json', dict(step=2, metrics={'num_examples': 1000},
                               contract=dict(task=task, variant=variant)))
    last = tmp_path/TASKS[-1]/VARIANTS[-1]/'generation.json'
    saved = json.loads(last.read_text())
    runner.atomic_json(last, dict(saved, step=1))
    with pytest.raises(ValueError, match='All ten'):
        acquire_finished_predecessor(tmp_path)
    runner.atomic_json(last, saved)
    lock = acquire_finished_predecessor(tmp_path)
    assert lock is not None
    try:
        assert acquire_finished_predecessor(tmp_path) is None
    finally:
        lock.close()


def test_pairing_rejects_different_ids_and_counts_discordant_examples():
    def result(values):
        return {'examples': [dict(id=str(i), scores={'valid_solution': v}) for i, v in enumerate(values)]}
    change = paired_changes(result([1, 0, 0, 1]), result([1, 1, 1, 0]))
    assert change == dict(gained=2, lost=1, solved_both=1, delta_accuracy_pp=25., paired_exact_p=1.)
    with pytest.raises(ValueError, match='identical'):
        paired_changes(result([1]), result([1, 0]))


@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('target_epochs', [2,3])
def test_comparison_report_writes_epoch_bars_and_paired_tables(tmp_path, monkeypatch, split, target_epochs):
    from types import SimpleNamespace
    from reasoning import second_epoch_queue as q
    data = tmp_path/'data'; data.mkdir()
    (data/'test.jsonl').write_text('frozen-test')
    source, output = tmp_path/'source', tmp_path/'output'
    task = 'zebra-benchmark'
    variants, labels = (SPLIT_VARIANTS, SPLIT_LABELS) if split else (VARIANTS, q.DISPLAY_NAMES)
    monkeypatch.setattr(q, 'ReasoningDataset', lambda *a: SimpleNamespace(task=task,
                        records=[dict(id='a', answer=['1']), dict(id='b', answer=['2'])]))
    def generation(run, data, variant, step):
        second = Path(run).is_relative_to(output)
        return dict(step=step, examples=[
            dict(id='a', scores={'valid_solution': True}, predicted_answer_slots=['1']),
            dict(id='b', scores={'valid_solution': second}, predicted_answer_slots=['2' if second else '1'])])
    monkeypatch.setattr(q, 'validated_generation', generation)
    for variant in variants:
        runner.atomic_json(source/task/variant/'contract.json', dict(epoch_examples=7, global_batch=4,
                                                                   epochs=target_epochs-1))
        runner.atomic_json(output/task/variant/'generation.json', {})
    q.comparison_report(source, output, data, variants, labels, target_epochs)
    report = output/'report'/task
    result = json.loads((report/'comparison.json').read_text())
    assert len(result['rows']) == 2*len(variants) and len(result['paired']) == len(variants)
    assert all(r['gained'] == 1 and r['lost'] == 0 for r in result['paired'])
    assert {r['epoch'] for r in result['rows']} == {target_epochs-1,target_epochs}
    assert {r['step'] for r in result['rows']} == {2*(target_epochs-1),2*target_epochs}
    assert (report/'generation_comparison.png').stat().st_size > 1000
    assert (report/'generation_comparison.csv').exists()
