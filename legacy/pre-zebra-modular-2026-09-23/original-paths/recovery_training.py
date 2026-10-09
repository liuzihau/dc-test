"""Single-GPU recovery checkpoints, published only after a complete write."""
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger

NAME = re.compile(r'recovery-\d{9}-[a-f0-9]{12}\.ckpt$')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def sync_directory(directory):
    fd = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x') as f:
        json.dump(content, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    sync_directory(path.parent)


def committed_receipts(directory):
    results = []
    for path in Path(directory).glob('recovery-*.ckpt.json'):
        try:
            entry = json.loads(path.read_text())
            name = entry['file']
            if not NAME.fullmatch(name) or path.name != name + '.json':
                continue
            if (entry['version'] != 1 or type(entry['step']) is not int or entry['step'] < 0
                    or type(entry['created_ns']) is not int
                    or type(entry['size_bytes']) is not int or entry['size_bytes'] <= 0
                    or type(entry['completed']) is not bool
                    or not re.fullmatch(r'[a-f0-9]{64}', entry['sha256'])):
                continue
            results.append((entry, path))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(results, key=lambda item: (item[0]['step'], item[0]['created_ns']))


class RecoveryCheckpoint(Callback):
    """One GPU only. No checkpoint of partially accumulated gradients."""
    def __init__(self, directory, every_seconds=1200, keep=3):
        if every_seconds <= 0 or keep < 2:
            raise ValueError('Recovery interval must be positive; keep at least two files')
        self.directory = Path(directory)
        self.every_seconds, self.keep = every_seconds, keep
        self.resume_step = 0
        self.last_step = 0
        self.last_time = time.monotonic()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.resume_step = int(checkpoint['global_step'])

    def on_train_start(self, trainer, pl_module):
        if trainer.world_size != 1:
            raise ValueError('Wall-clock recovery currently supports exactly one GPU/rank')
        self.last_step = int(trainer.global_step)
        self.last_time = time.monotonic()
        self.directory.mkdir(parents=True, exist_ok=True)
        for logger in trainer.loggers:
            if isinstance(logger, CSVLogger):
                atomic_json(Path(logger.log_dir) / 'resume_attempt.json', {
                    'version': 1, 'attempt_id': uuid.uuid4().hex,
                    'started_ns': time.time_ns(), 'resume_step': self.resume_step,
                    'checkpoint': str(trainer.ckpt_path),
                    'step_convention': 'CSV train step S is optimizer update S+1; keep old steps < resume_step',
                })

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step <= self.last_step:
            return
        self.last_step = step
        # Final checkpoint is published by on_train_end AFTER final validation.
        if step >= trainer.max_steps:
            return
        if step % 500 == 0 or time.monotonic() - self.last_time >= self.every_seconds:
            self.save(trainer, completed=False)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Our model stores a normalized, completed-update counter. Lightning
        # may increment its logging counter again while replaying the pending
        # validation/end-of-batch phase on resume. Anchor logging to the actual
        # optimizer counter, not that replay-dependent counter. No optimizer or
        # data-progress counters are changed.
        trainer.fit_loop.epoch_loop._batches_that_stepped = int(trainer.global_step)

    def on_validation_start(self, trainer, pl_module):
        if not trainer.sanity_checking:
            trainer.fit_loop.epoch_loop._batches_that_stepped = max(0, int(trainer.global_step) - 1)

    def on_train_end(self, trainer, pl_module):
        if trainer.global_step >= trainer.max_steps:
            self.save(trainer, completed=True)

    def save(self, trainer, completed=False):
        import shutil
        if shutil.disk_usage(self.directory).free < 4 * 1024**3:
            raise OSError('Insufficient recovery checkpoint space: need at least 4 GiB free')
        step = int(trainer.global_step)
        name = f'recovery-{step:09d}-{uuid.uuid4().hex[:12]}.ckpt'
        final = self.directory / name
        temp = self.directory / ('.' + name + '.tmp')
        trainer.save_checkpoint(str(temp), weights_only=False)
        with temp.open('rb') as f:
            os.fsync(f.fileno())
        receipt = {'version': 1, 'file': name, 'step': step,
                   'created_ns': time.time_ns(), 'size_bytes': temp.stat().st_size,
                   'sha256': sha256(temp), 'completed': bool(completed)}
        os.replace(temp, final)
        sync_directory(self.directory)
        # Receipt is the commit record. A crash before this leaves an ignored orphan.
        atomic_json(final.with_name(name + '.json'), receipt)
        last = self.directory / 'last.ckpt'
        if last.is_file() and not last.is_symlink():
            # Preserve a legacy full last.ckpt (not merely a pointer) before
            # replacing it. Hard linking does not duplicate its disk payload.
            os.link(last, self.directory / f'legacy-last-{time.time_ns()}.ckpt')
        link = self.directory / ('.last-' + uuid.uuid4().hex)
        link.symlink_to(name)
        os.replace(link, last)
        sync_directory(self.directory)
        # Only files owned by this callback; old periodic/source checkpoints remain.
        for old, record in committed_receipts(self.directory)[:-self.keep]:
            (self.directory / old['file']).unlink(missing_ok=True)
            record.unlink(missing_ok=True)
        sync_directory(self.directory)
        self.last_time = time.monotonic()
        print(f'RECOVERY COMMITTED: step={step}, complete={completed}, file={final}', flush=True)
