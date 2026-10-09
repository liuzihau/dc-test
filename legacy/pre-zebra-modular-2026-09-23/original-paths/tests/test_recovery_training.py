import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd
import pytest
import torch

import recovery_training as recovery
from metrics_history import load_history
from scripts.cloud import select_recovery as selector
from scripts.cloud.supervise_training import retryable


class FakeTrainer:
    def __init__(self, directory):
        self.global_step, self.max_steps, self.world_size = 10, 1000, 1
        self.loggers = []
        self.ckpt_path = 'source.ckpt'
        self.directory = directory
    def save_checkpoint(self, path, weights_only=False):
        assert not weights_only
        Path(path).write_text(json.dumps({'global_step': self.global_step}))


def test_timer_only_saves_completed_updates(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(recovery.time, 'monotonic', lambda: clock[0])
    trainer = FakeTrainer(tmp_path)
    callback = recovery.RecoveryCheckpoint(tmp_path, every_seconds=1200)
    callback.on_train_start(trainer, None)
    clock[0] = 1201
    for _ in range(3):
        callback.on_train_batch_end(trainer, None, None, None, 0)
    assert not recovery.committed_receipts(tmp_path)
    trainer.global_step = 11
    callback.on_train_batch_end(trainer, None, None, None, 3)
    assert [x[0]['step'] for x in recovery.committed_receipts(tmp_path)] == [11]
    trainer.global_step = 500
    callback.on_train_batch_end(trainer, None, None, None, 4)
    assert len(recovery.committed_receipts(tmp_path)) == 2


def test_commit_retention_and_partial_failure(tmp_path):
    trainer = FakeTrainer(tmp_path)
    callback = recovery.RecoveryCheckpoint(tmp_path)
    (tmp_path / '0-500.ckpt').write_text('legacy must remain')
    for step in (11, 12, 13, 14):
        trainer.global_step = step
        callback.save(trainer)
    records = recovery.committed_receipts(tmp_path)
    assert [x[0]['step'] for x in records] == [12, 13, 14]
    assert (tmp_path / 'last.ckpt').resolve().name == records[-1][0]['file']
    assert (tmp_path / '0-500.ckpt').is_file()
    previous = (tmp_path / 'last.ckpt').resolve()
    def broken(path, **kwargs):
        Path(path).write_bytes(b'incomplete')
        raise OSError('simulated power loss')
    trainer.save_checkpoint = broken
    with pytest.raises(OSError):
        callback.save(trainer)
    assert (tmp_path / 'last.ckpt').resolve() == previous
    assert len(recovery.committed_receipts(tmp_path)) == 3


def test_legacy_full_last_is_preserved(tmp_path):
    (tmp_path / 'last.ckpt').write_bytes(b'previous full checkpoint')
    recovery.RecoveryCheckpoint(tmp_path).save(FakeTrainer(tmp_path))
    copies = list(tmp_path.glob('legacy-last-*.ckpt'))
    assert len(copies) == 1
    assert copies[0].read_bytes() == b'previous full checkpoint'
    assert (tmp_path / 'last.ckpt').is_symlink()


def test_corrupt_latest_falls_back_and_finished_stops(tmp_path, monkeypatch):
    trainer = FakeTrainer(tmp_path)
    callback = recovery.RecoveryCheckpoint(tmp_path)
    for step in (11, 12, 13):
        trainer.global_step = step
        callback.save(trainer)
    records = recovery.committed_receipts(tmp_path)
    (tmp_path / records[-1][0]['file']).write_bytes(b'corrupt')
    monkeypatch.setattr(selector, 'checkpoint_snapshot',
                        lambda path, *a, **kw: json.loads(Path(path).read_text()))
    selected = selector.select(tmp_path, tmp_path / 'missing-source.ckpt', {'checkpoint': {}}, 100)
    assert Path(selected).name == records[-2][0]['file']
    trainer.global_step = 100
    callback.save(trainer, completed=True)
    assert selector.select(tmp_path, tmp_path / 'missing', {'checkpoint': {}}, 100) == 'DONE'


def test_uncommitted_recovery_cannot_be_used_through_last(tmp_path, monkeypatch):
    path = tmp_path / ('recovery-000000123-' + 'a'*12 + '.ckpt')
    path.write_text('{}')
    (tmp_path / 'last.ckpt').symlink_to(path.name)
    monkeypatch.setattr(selector, 'checkpoint_snapshot', lambda *a, **kw: pytest.fail('orphan loaded'))
    with pytest.raises(ValueError, match='No valid'):
        selector.select(tmp_path, tmp_path / 'missing', {'checkpoint': {}}, 5000)


def test_final_checkpoint_waits_for_training_end(tmp_path):
    trainer = FakeTrainer(tmp_path)
    trainer.max_steps = 11
    callback = recovery.RecoveryCheckpoint(tmp_path, every_seconds=0.000001)
    callback.on_train_start(trainer, None)
    trainer.global_step = 11
    callback.on_train_batch_end(trainer, None, None, None, 1)
    assert not recovery.committed_receipts(tmp_path)
    callback.on_train_end(trainer, None)
    assert recovery.committed_receipts(tmp_path)[0][0]['completed'] is True


def make_attempt(root, version, cutoff, steps, losses, started):
    directory = root / f'version_{version}'
    directory.mkdir()
    recovery.atomic_json(directory / 'resume_attempt.json', {
        'version': 1, 'attempt_id': str(version), 'started_ns': started, 'resume_step': cutoff})
    if steps is not None:
        pd.DataFrame({'step': steps, 'trainer/loss': losses}).to_csv(directory / 'metrics.csv', index=False)
    return directory


def test_merge_removes_abandoned_future_and_ignores_mtime(tmp_path):
    first = make_attempt(tmp_path, 0, 0, [9, 19, 29, 39], [1, 2, 3, 4], 100)
    second = make_attempt(tmp_path, 1, 20, [29], [30], 200)
    os.utime(first / 'metrics.csv', (9999, 9999))
    result = load_history(tmp_path)
    assert result['step'].tolist() == [9, 19, 29]
    assert result['trainer/loss'].tolist() == [1, 2, 30]
    make_attempt(tmp_path, 2, 10, None, None, 300)
    assert load_history(tmp_path)['step'].tolist() == [9]
    assert pd.read_csv(first / 'metrics.csv')['step'].tolist() == [9, 19, 29, 39]
    assert second.exists()


def test_legacy_csv_is_trimmed_by_first_explicit_resume(tmp_path):
    directory = tmp_path / 'old'
    directory.mkdir()
    pd.DataFrame({'step': [1499, 1599, 1699], 'trainer/loss': [1, 2, 3]}).to_csv(directory/'metrics.csv', index=False)
    make_attempt(tmp_path, 1, 1600, [1609], [4], 100)
    assert load_history(tmp_path)['step'].tolist() == [1499, 1599, 1609]


@pytest.mark.parametrize('code,text,expected', [
    (1, 'CUDA unavailable', True), (1, 'NCCL timeout', True),
    (137, 'Killed', True), (1, 'CUDA out of memory', False),
    (1, 'No space left on device', False), (2, 'Permission denied', False),
    (1, 'Traceback (most recent call last): bug', False), (0, '', False),
])
def test_retry_policy(code, text, expected):
    assert retryable(code, text) == expected


def test_real_cpu_checkpoint_is_after_accumulation_and_final_validation(tmp_path, monkeypatch):
    import lightning as L
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader, TensorDataset
    from test_checkpoint_resume import TinyResumeModel

    monkeypatch.setattr(TinyResumeModel, 'on_train_start', lambda self: None)
    original_training_step = TinyResumeModel.training_step
    def logged_step(self, batch, batch_idx):
        loss = original_training_step(self, batch, batch_idx)
        self.log('trainer/loss', loss, on_step=True, on_epoch=False)
        return loss
    monkeypatch.setattr(TinyResumeModel, 'training_step', logged_step)
    original_validation_step = TinyResumeModel.validation_step
    def logged_validation(self, batch, batch_idx):
        original_validation_step(self, batch, batch_idx)
        self.log('val/nll', self.weight.square(), on_step=False, on_epoch=True)
    monkeypatch.setattr(TinyResumeModel, 'validation_step', logged_validation)
    model = TinyResumeModel(micro=2)
    logger = CSVLogger(str(tmp_path), name='logs')
    callback = recovery.RecoveryCheckpoint(tmp_path / 'checkpoints', every_seconds=0.000001)
    trainer = L.Trainer(accelerator='cpu', devices=1, max_steps=2,
        accumulate_grad_batches=4, val_check_interval=4, limit_val_batches=1,
        num_sanity_val_steps=0, callbacks=[callback], logger=logger,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        log_every_n_steps=1)
    data = TensorDataset(torch.arange(1, 33, dtype=torch.float32))
    trainer.fit(model, DataLoader(data, batch_size=2), DataLoader(data, batch_size=2))
    records = recovery.committed_receipts(tmp_path / 'checkpoints')
    assert [x[0]['step'] for x in records] == [1, 2]
    assert 2 in model.valid_steps
    for entry, _ in records:
        c = torch.load(tmp_path / 'checkpoints' / entry['file'], weights_only=False)
        assert c['global_step'] == entry['step']
        assert c['optimizer_states'] and c['lr_schedulers']
        counts = c['loops']['fit_loop']['epoch_loop.batch_progress']['current']
        assert all(counts[k] == entry['step'] * 4 for k in ('ready', 'started', 'processed', 'completed'))
    assert records[-1][0]['completed']
    assert (Path(logger.log_dir) / 'resume_attempt.json').exists()
    # Simulate losing step 2 and resuming the first recovery checkpoint.
    restored = TinyResumeModel(micro=2)
    new_logger = CSVLogger(str(tmp_path), name='logs')
    new_callback = recovery.RecoveryCheckpoint(tmp_path / 'restarted', every_seconds=1200)
    second = L.Trainer(accelerator='cpu', devices=1, max_steps=2,
        accumulate_grad_batches=4, val_check_interval=4, limit_val_batches=1,
        num_sanity_val_steps=0, callbacks=[new_callback], logger=new_logger,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        log_every_n_steps=1)
    second.fit(restored, DataLoader(data, batch_size=2), DataLoader(data, batch_size=2),
               ckpt_path=str(tmp_path / 'checkpoints' / records[0][0]['file']))
    meta = json.loads((Path(new_logger.log_dir) / 'resume_attempt.json').read_text())
    assert meta['resume_step'] == 1
    assert restored.loaded_state['step'] == 1
    frame = load_history(tmp_path / 'logs')
    old = frame[frame['_source_csv'] == str(Path(logger.log_dir) / 'metrics.csv')]
    assert (old['step'] < 1).all()
    current = frame[frame['_source_csv'] == str(Path(new_logger.log_dir) / 'metrics.csv')]
    assert 1 in current['step'].tolist()
    assert current.loc[current['trainer/loss'].notna(), 'step'].tolist() == [1]
    assert 1 in current.loc[current['val/nll'].notna(), 'step'].tolist()


@pytest.mark.parametrize('codes,texts,expected', [
    ([1, 0], ['NCCL timeout', 'Training target already complete; nothing to restart.'], 'COMPLETE'),
    ([1], ['CUDA out of memory'], 'STOPPED_ERROR'),
    ([0], ['returned without completion'], 'STOPPED_EARLY'),
    ([1, 1, 1], ['NCCL timeout'] * 3, 'STOPPED_ERROR'),
])
def test_supervisor_retries_bounded_and_honors_saved_settings(tmp_path, monkeypatch, codes, texts, expected):
    from scripts.cloud import supervise_training as supervisor
    output = tmp_path / 'run'
    config = tmp_path / 'launch.json'
    config.write_text(json.dumps({'version': 1, 'repo': str(tmp_path), 'env': {
        'DCACHE_PYTHON': '/saved/python', 'DCACHE_RUN_DIR': str(output),
        'DCACHE_MAX_STEPS': '5000', 'DCACHE_MICRO_BATCH': '8'}}))
    calls = []
    def fake_run(command, **kwargs):
        index = len(calls)
        calls.append(kwargs['env'])
        kwargs['stdout'].write(texts[index].encode())
        return NS(returncode=codes[index])
    monkeypatch.setattr(supervisor.subprocess, 'run', fake_run)
    monkeypatch.setattr(supervisor.time, 'sleep', lambda seconds: None)
    supervisor.run(config, delay=1, retries=2)
    assert len(calls) == len(codes)
    assert all(c['DCACHE_MICRO_BATCH'] == '8' and c['DCACHE_RECOVERY_ENABLED'] == '1' for c in calls)
    assert json.loads((output / 'supervisor_status.json').read_text())['state'] == expected


def test_actual_plot_loaders_share_restart_lineage(tmp_path):
    from scripts.results.refresh_canonical_results import load_metrics as canonical
    from scripts.plot_pretrain_losses import load_metrics as legacy_plot
    make_attempt(tmp_path, 0, 0, [9, 19, 29], [1, 2, 3], 100)
    make_attempt(tmp_path, 1, 20, [29], [30], 200)
    pd.testing.assert_frame_equal(canonical(str(tmp_path)), legacy_plot(str(tmp_path)))
    assert canonical(str(tmp_path))['trainer/loss'].tolist() == [1, 2, 30]
