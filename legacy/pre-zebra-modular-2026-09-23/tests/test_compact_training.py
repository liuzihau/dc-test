import json
import pickle

import numpy as np
import pytest
import torch

from compact_training import CompactTrainingDataset, digest, order_indices, verify_compact
from dataloader import FaultTolerantDistributedSampler


@pytest.fixture(params=[0, 1])
def compact(tmp_path, request):
    seed = request.param
    order = order_indices(100, 20, 60, seed=seed)
    indices = np.sort(order)
    np.save(tmp_path / 'training_order.npy', order)
    np.save(tmp_path / 'original_indices.npy', indices)
    np.save(tmp_path / 'tokens.npy', np.repeat(indices[:, None], 1024, axis=1).astype(np.int32))
    meta = dict(version=1, original_num_rows=100, physical_rows=40, sequence_length=1024,
                start_step=2, end_step=6, global_batch=10, epoch=0, sampler_seed=seed,
                export_stop_offset=60, files={})
    for name in ('tokens.npy', 'original_indices.npy', 'training_order.npy'):
        meta['files'][name] = dict(size_bytes=(tmp_path/name).stat().st_size, sha256=digest(tmp_path/name))
    (tmp_path / 'compact_train.json').write_text(json.dumps(meta))
    return tmp_path, meta, order


@pytest.mark.parametrize('world', [1, 2, 5])
def test_original_sampler_order_and_tokens_preserved(compact, world):
    path, meta, order = compact
    data = CompactTrainingDataset(path)
    assert len(data) == 100
    for rank in range(world):
        sampler = FaultTolerantDistributedSampler(data, num_replicas=world, rank=rank,
                                                  seed=meta['sampler_seed'])
        sampler.load_state_dict({'epoch': 0, 'counter': 20 // world})
        iterator = iter(sampler)
        actual = [next(iterator) for _ in range(40 // world)]
        assert actual == order[rank::world].tolist()
        for index in actual:
            row = data[index]
            assert row['input_ids'].dtype == torch.int64
            assert torch.all(row['input_ids'] == index)
            assert torch.all(row['attention_mask'] == 1)
    assert verify_compact(path, meta) == meta


def test_pickle_and_absent_rows(compact):
    path, _, order = compact
    data = pickle.loads(pickle.dumps(CompactTrainingDataset(path)))
    assert data[int(order[0])]['input_ids'][0] == int(order[0])
    missing = next(i for i in range(100) if i not in order)
    with pytest.raises(IndexError):
        data[missing]


@pytest.mark.parametrize('override', [dict(epoch=1), dict(seed=2), dict(counter=0),
                                     dict(global_batch=20), dict(max_steps=7)])
def test_resume_guards(compact, override):
    path, _, _ = compact
    data = CompactTrainingDataset(path)
    args = dict(epoch=0, counter=20, world_size=1, global_batch=10, max_steps=6,
                seed=data.meta['sampler_seed'])
    data.validate_resume(**args)
    args.update(override)
    with pytest.raises(ValueError):
        data.validate_resume(**args)


def test_corrupt_payload_rejected(compact):
    path, meta, _ = compact
    with open(path / 'tokens.npy', 'r+b') as f:
        f.seek(-1, 2)
        f.write(b'\xff')
    with pytest.raises(ValueError, match='integrity'):
        verify_compact(path, meta)


def test_real_model_hook_replaces_live_lightning_iterator(compact, monkeypatch):
    from functools import partial
    from types import SimpleNamespace as NS
    from lightning.pytorch.loops.fetchers import _PrefetchDataFetcher
    from lightning.pytorch.utilities.combined_loader import CombinedLoader
    import dataloader
    from diffusion import Diffusion

    path, _, order = compact
    data = CompactTrainingDataset(path)
    # Establish the real Lightning iterator BEFORE calling the model hook.
    initial = torch.utils.data.DataLoader(data, batch_size=2, num_workers=0,
        sampler=torch.utils.data.DistributedSampler(data, num_replicas=1, rank=0,
                                                   seed=data.meta['sampler_seed']))
    combined = CombinedLoader(initial, 'max_size_cycle')
    fetcher = _PrefetchDataFetcher()
    fetcher.setup(combined)
    iter(fetcher)
    # Test the actual hook/fetcher ordering without OS tensor-sharing sockets.
    # Multi-worker file access is covered by the dataset's pickle contract;
    # the destination smoke exercises actual worker IPC.
    base_loader = torch.utils.data.DataLoader
    class SocketFreeLoader(base_loader):
        def __init__(self, *args, **kwargs):
            kwargs['num_workers'] = 0
            kwargs['persistent_workers'] = False
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(torch.utils.data, 'DataLoader', SocketFreeLoader)
    monkeypatch.setattr(dataloader, 'FaultTolerantDistributedSampler',
                       partial(FaultTolerantDistributedSampler, num_replicas=1, rank=0))
    trainer = NS(_accelerator_connector=NS(use_distributed_sampler=True, is_distributed=True),
                 world_size=1, max_steps=6,
                 fit_loop=NS(_combined_loader=combined, _data_fetcher=fetcher))
    model = NS(ema=None, trainer=trainer, fast_forward_epochs=0, fast_forward_batches=10,
               config=NS(loader=NS(batch_size=2, num_workers=1, pin_memory=False,
                                   global_batch_size=10)))
    try:
        Diffusion.on_train_start(model)
        for offset in range(0, 8, 2):
            batch, _, _ = next(fetcher)
            assert batch['input_ids'][:, 0].tolist() == order[offset:offset+2].tolist()
    finally:
        fetcher.teardown()


def test_legacy_flattened_replacement_does_not_change_live_iterator():
    from lightning.pytorch.loops.fetchers import _PrefetchDataFetcher
    from lightning.pytorch.utilities.combined_loader import CombinedLoader
    original = torch.utils.data.DataLoader(torch.arange(10), batch_size=2)
    replacement = torch.utils.data.DataLoader(torch.arange(20, 30), batch_size=2)
    combined = CombinedLoader(original, 'max_size_cycle')
    fetcher = _PrefetchDataFetcher()
    fetcher.setup(combined)
    iter(fetcher)
    combined.flattened = [replacement]
    assert next(fetcher)[0].tolist() == [0, 1]
    from compact_training import reset_compact_fetcher
    reset_compact_fetcher(fetcher)
    assert next(fetcher)[0].tolist() == [20, 21]
    fetcher.teardown()
