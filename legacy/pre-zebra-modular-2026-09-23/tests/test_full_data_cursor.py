"""Opt-in full-data cursor restoration through a real Lightning live fetcher.

Explicit sampler rank geometry avoids process-group sockets; these are not
multi-process or CUDA tests. Both consumed rows and the actual hook are tested.
"""
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace as NS

import pytest
import torch
from lightning.pytorch.loops.fetchers import _PrefetchDataFetcher
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from omegaconf import OmegaConf

import dataloader
from diffusion import Diffusion


@contextmanager
def live_model(*, world=1, rank=0, epoch=0, completed=None, shuffle=True,
               sampler_drop_last=False, loader_drop_last=False, size=81):
    dataset = torch.arange(size)
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world, rank=rank, seed=1,
        shuffle=shuffle, drop_last=sampler_drop_last)
    sampler.set_epoch(epoch)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=0, sampler=sampler,
        drop_last=loader_drop_last)
    combined = CombinedLoader(loader, 'max_size_cycle')
    fetcher = _PrefetchDataFetcher()
    fetcher.setup(combined)
    iter(fetcher)
    config = OmegaConf.create({
        'loader': {'batch_size': 2, 'num_workers': 0, 'pin_memory': False,
                   'global_batch_size': world * 2 * 4},
        'checkpointing': {'restore_data_cursor': True},
    })
    trainer = NS(
        _accelerator_connector=NS(use_distributed_sampler=True, is_distributed=True),
        world_size=world, global_rank=rank, accumulate_grad_batches=4,
        current_epoch=epoch, max_steps=50,
        fit_loop=NS(_combined_loader=combined, _data_fetcher=fetcher))
    model = NS(
        ema=None, trainer=trainer, config=config,
        fast_forward_epochs=None if completed is None else epoch,
        fast_forward_batches=completed)
    try:
        yield model, fetcher, sampler
    finally:
        fetcher.teardown()


@pytest.mark.parametrize('world,rank', [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize('epoch', [0, 2])
@pytest.mark.parametrize('completed', [None, 8])
@pytest.mark.parametrize('shuffle', [False, True])
def test_first_consumed_rows_follow_original_sampler_at_completed_cursor(
        world, rank, epoch, completed, shuffle):
    with live_model(world=world, rank=rank, epoch=epoch,
                    completed=completed, shuffle=shuffle) as (model, fetcher, original):
        expected = list(iter(original))
        cursor = (completed or 0) * 2
        # Emulate an unrelated prefetched sampler position. Only the completed
        # batch count is an authoritative training cursor.
        original.counter = 999_999
        if completed is not None:
            next(fetcher)
        Diffusion.on_train_start(model)
        replacement = model.trainer.fit_loop._combined_loader.flattened[0]
        assert isinstance(replacement.sampler, dataloader.FaultTolerantDistributedSampler)
        assert replacement.sampler.seed == original.seed == 1
        assert replacement.sampler.epoch == epoch
        assert replacement.sampler.shuffle == shuffle
        assert replacement.sampler.rank == rank
        assert replacement.sampler.num_replicas == world
        assert replacement.num_workers == 0
        assert not replacement.persistent_workers
        consumed = []
        for _ in range(3):
            consumed.extend(next(fetcher)[0].tolist())
        assert consumed == expected[cursor:cursor + 6]


@pytest.mark.parametrize('sampler_drop_last', [False, True])
@pytest.mark.parametrize('loader_drop_last', [False, True])
def test_sampler_and_loader_drop_last_are_preserved(sampler_drop_last, loader_drop_last):
    with live_model(world=2, rank=1, sampler_drop_last=sampler_drop_last,
                    loader_drop_last=loader_drop_last) as (model, fetcher, original):
        expected = list(iter(original))
        if loader_drop_last:
            expected = expected[:len(expected) // 2 * 2]
        Diffusion.on_train_start(model)
        loader = model.trainer.fit_loop._combined_loader.flattened[0]
        assert loader.sampler.drop_last == sampler_drop_last
        assert loader.drop_last == loader_drop_last
        consumed = []
        for batch, _, _ in fetcher:
            consumed.extend(batch.tolist())
        assert consumed == expected


def test_next_epoch_uses_original_seed_plus_new_epoch_after_resumed_tail():
    with live_model(world=2, rank=0, epoch=2, completed=8) as (model, fetcher, original):
        expected_tail = list(iter(original))[16:]
        Diffusion.on_train_start(model)
        consumed = []
        for batch, _, _ in fetcher:
            consumed.extend(batch.tolist())
        assert consumed == expected_tail
        sampler = model.trainer.fit_loop._combined_loader.flattened[0].sampler
        sampler.set_epoch(3)
        original.set_epoch(3)
        expected_next_epoch = list(iter(original))
        consumed = []
        for batch, _, _ in iter(fetcher):
            consumed.extend(batch.tolist())
        assert consumed == expected_next_epoch


def test_opt_out_keeps_legacy_live_iterator_unchanged(monkeypatch):
    # Existing legacy branch requires workers > 0. Keep this compatibility test
    # socket-free without changing its original default behavior.
    base_loader = torch.utils.data.DataLoader
    class SocketFreeLoader(base_loader):
        def __init__(self, *args, **kwargs):
            kwargs['num_workers'] = 0
            kwargs['persistent_workers'] = False
            super().__init__(*args, **kwargs)
    with live_model(completed=8) as (model, fetcher, original):
        expected = list(iter(original))[:2]
        model.config.checkpointing.restore_data_cursor = False
        monkeypatch.setattr(torch.utils.data, 'DataLoader', SocketFreeLoader)
        monkeypatch.setattr(dataloader, 'FaultTolerantDistributedSampler',
            partial(dataloader.FaultTolerantDistributedSampler, num_replicas=1, rank=0))
        Diffusion.on_train_start(model)
        assert next(fetcher)[0].tolist() == expected
        assert model.trainer.fit_loop._combined_loader.flattened[0].sampler.seed == 0


@pytest.mark.parametrize('flag', ['use_distributed_sampler', 'is_distributed'])
def test_restore_requires_explicit_ddp(flag):
    with live_model() as (model, _, _):
        setattr(model.trainer._accelerator_connector, flag, False)
        with pytest.raises(ValueError, match='explicit DDP'):
            Diffusion.on_train_start(model)


def test_rejects_active_sampler_with_wrong_geometry():
    with live_model(world=2, rank=1) as (model, _, _):
        model.trainer.global_rank = 0
        with pytest.raises(ValueError, match='rank/world geometry'):
            Diffusion.on_train_start(model)


def test_rejects_loader_microbatch_mismatch():
    with live_model() as (model, _, _):
        model.config.loader.batch_size = 4
        with pytest.raises(ValueError, match='microbatch'):
            Diffusion.on_train_start(model)


@pytest.mark.parametrize('epoch,completed,match', [
    (None, 8, 'both saved'),
    (0, None, 'both saved'),
    (-1, 8, 'nonnegative integers'),
    (0, -4, 'nonnegative integers'),
    (0, 1.5, 'nonnegative integers'),
    (False, 8, 'nonnegative integers'),
    (0, True, 'nonnegative integers'),
    (0, 7, 'complete optimizer boundary'),
    (0, 20, 'epoch boundary'),
    (0, 24, 'epoch boundary'),
])
def test_invalid_saved_cursors_fail_before_replacing_loader(epoch, completed, match):
    with live_model(world=2, rank=0, size=80) as (model, _, _):
        original_loader = model.trainer.fit_loop._combined_loader.flattened[0]
        model.fast_forward_epochs = epoch
        model.fast_forward_batches = completed
        with pytest.raises(ValueError, match=match):
            Diffusion.on_train_start(model)
        assert model.trainer.fit_loop._combined_loader.flattened[0] is original_loader


def test_rejects_active_non_distributed_sampler():
    with live_model() as (model, _, _):
        model.trainer.fit_loop._combined_loader.flattened = [
            torch.utils.data.DataLoader(torch.arange(80), batch_size=2)]
        with pytest.raises(ValueError, match='active DistributedSampler'):
            Diffusion.on_train_start(model)


@pytest.mark.parametrize('size,first_steps,epoch,cursor', [(128, 2, 0, 16), (16, 3, 1, 8)])
def test_real_cpu_fit_resume_consumes_next_rows_and_keeps_validation(
        tmp_path, size, first_steps, epoch, cursor):
    import lightning as L
    from lightning.pytorch.strategies import SingleDeviceStrategy
    from test_checkpoint_resume import TinyResumeModel

    class SocketFreeSamplerStrategy(SingleDeviceStrategy):
        # Exercise actual fit/restore/iterator ordering with explicit rank-one
        # DistributedSamplers, but no process group or network initialization.
        is_distributed = True

    class CursorModel(TinyResumeModel):
        def __init__(self):
            super().__init__(micro=4)
            self.config.loader.num_workers = 0
            self.config.checkpointing.restore_data_cursor = True
            self.config.checkpointing.allow_batch_geometry_change = False
            self.seen = []

        def training_step(self, batch, batch_idx):
            self.seen.extend(batch[0].long().tolist())
            return super().training_step(batch, batch_idx)

    dataset = torch.utils.data.TensorDataset(torch.arange(size, dtype=torch.float32))
    def loader(shuffle):
        return torch.utils.data.DataLoader(
            dataset, batch_size=4, num_workers=0,
            sampler=torch.utils.data.DistributedSampler(
                dataset, num_replicas=1, rank=0, shuffle=shuffle, seed=1))

    path = tmp_path / 'full-data.ckpt'
    class SaveDuringFit(L.Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if trainer.global_step == first_steps and not path.exists():
                # Match timer/step recovery checkpoints, before fit shutdown
                # increments Lightning's epoch-completion bookkeeping.
                trainer.save_checkpoint(path)

    def trainer(steps, saving=False):
        return L.Trainer(
            accelerator='cpu', devices=1, strategy=SocketFreeSamplerStrategy(),
            max_steps=steps, accumulate_grad_batches=2, val_check_interval=2,
            limit_val_batches=1, num_sanity_val_steps=0,
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, default_root_dir=tmp_path,
            callbacks=[SaveDuringFit()] if saving else [])

    first = CursorModel()
    first_trainer = trainer(first_steps, saving=True)
    first_trainer.fit(first, loader(True), loader(False))
    assert path.exists()

    resumed = CursorModel()
    second_trainer = trainer(first_steps + 1)
    second_trainer.fit(resumed, loader(True), loader(False), ckpt_path=path)
    expected = torch.randperm(size, generator=torch.Generator().manual_seed(1 + epoch))
    assert resumed.seen == expected[cursor:cursor + 8].tolist()
    assert resumed.loaded_state['step'] == first_steps
    assert resumed._batch_geometry_migration is None
    assert second_trainer.global_step == first_steps + 1
    assert first_steps + 1 in resumed.valid_steps
