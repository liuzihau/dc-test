import fcntl
import json

import pytest
import torch

from reasoning.runner import atomic_json, digest, load_checkpoint
from reasoning.single_gpu_resume import fork_single_gpu
from test_reasoning_runner import assert_tree_equal


def source_run(tmp_path):
    source = tmp_path / 'source'
    directory = source / 'checkpoints'
    directory.mkdir(parents=True)
    contract = dict(suite='split', variant='mdm', world_size=2, micro_batch=32,
                    global_batch=128, epoch_examples=853792, epochs=3,
                    model_config=dict(memory_mode='none', neighbors=False,
                                      trajectory='single', attention_mode='vanilla'))
    payload = dict(format_version=1, step=1000, examples_seen=128000,
                   model={'weight': torch.tensor([1.2])},
                   optimizer={'state': {'moment': torch.tensor([2.3])}},
                   model_config=contract['model_config'], contract=contract,
                   rng_by_rank=[{'torch': torch.tensor([1])}, {'torch': torch.tensor([2])}])
    path = directory / 'step-000001000-test.pt'
    torch.save(payload, path)
    atomic_json(path.with_suffix('.pt.json'), dict(file=path.name, step=1000,
                sha256=digest(path), size=path.stat().st_size, created_ns=1))
    (directory / 'last.pt').symlink_to(path.name)
    atomic_json(source / 'contract.json', contract)
    return source, path, payload


def test_full_state_fork_only_changes_world_and_rng_list(tmp_path):
    source, path, original = source_run(tmp_path)
    before = digest(path)
    destination = tmp_path / 'one_gpu'
    provenance = fork_single_gpu(source, destination)
    forked = load_checkpoint(destination / 'checkpoints/last.pt')
    for key in ('model', 'optimizer', 'step', 'examples_seen', 'model_config'):
        assert_tree_equal(original[key], forked[key])
    assert forked['contract'] == dict(original['contract'], world_size=1)
    assert_tree_equal(forked['rng_by_rank'], original['rng_by_rank'][:1])
    assert digest(path) == before == provenance['source_sha256']
    assert provenance['bitwise_replay'] is False
    assert json.loads((source / 'contract.json').read_text())['world_size'] == 2
    with pytest.raises(FileExistsError):
        fork_single_gpu(source, destination)


def test_refuses_running_source(tmp_path):
    source, _, _ = source_run(tmp_path)
    with (source / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            fork_single_gpu(source, tmp_path / 'one_gpu')


def test_refuses_changed_contract_and_nested_destination(tmp_path):
    source, _, original = source_run(tmp_path)
    with pytest.raises(ValueError, match='separate'):
        fork_single_gpu(source, source / 'nested')
    atomic_json(source / 'contract.json', dict(original['contract'], global_batch=256))
    with pytest.raises(ValueError, match='consistent'):
        fork_single_gpu(source, tmp_path / 'one_gpu')
