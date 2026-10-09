"""Per-optimizer-update CSV metrics and reproducible validation corruption."""
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from owt.gradient_metrics import GRADIENT_COLUMNS, gradient_norms

TRAIN_COLUMNS = ['optimizer_step', 'main_elbo', 'objective', 'np_prev', 'np_next',
                 'learning_rate', 'elapsed_seconds', 'peak_allocated_gib']
VAL_COLUMNS = ['optimizer_step', 'val_nll', 'val_ppl', 'val_bpd', 'elapsed_seconds']


def append(path, columns, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if fresh:
            writer.writeheader()
        writer.writerow(values)


def truncate_after(path, step, columns):
    if not path.exists():
        return
    with path.open() as stream:
        rows = [row for row in csv.DictReader(stream) if int(row['optimizer_step']) <= step]
    temporary = path.with_suffix('.partial')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class PreclipGradientMetrics(Callback):
    """Future trials only: log synchronized, accumulated gradients before clipping.

    Lightning's installed precision plugin invokes this hook after the closure
    and AMP unscaling, before clipping and the optimizer update. These are joint
    gradients, not separate estimates of main/auxiliary trunk contributions.
    """
    def __init__(self, run):
        self.path = Path(run)/'local_metrics/gradient_norms.csv'
        self.start = time.monotonic()

    def on_train_start(self, trainer, pl_module):
        algorithm = getattr(trainer.gradient_clip_algorithm, 'value', trainer.gradient_clip_algorithm)
        # Installed Lightning represents the default norm algorithm as None.
        if algorithm is not None and str(algorithm) != 'norm':
            raise ValueError('Preclip diagnostics require L2 norm clipping')
        if trainer.is_global_zero:
            truncate_after(self.path, trainer.global_step, GRADIENT_COLUMNS)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if not trainer.is_global_zero:
            return
        record = gradient_norms(pl_module, trainer.gradient_clip_val)
        record.update(optimizer_step=trainer.global_step+1,
                      elapsed_seconds=time.monotonic()-self.start)
        append(self.path, GRADIENT_COLUMNS, record)


class LocalMetrics(Callback):
    def __init__(self, run):
        self.run = Path(run)
        self.start = time.monotonic()
        self.last_step = 0
        self.pending = []
        self.rng = None

    def on_train_start(self, trainer, pl_module):
        self.last_step = trainer.global_step
        if trainer.is_global_zero:
            truncate_after(self.run/'local_metrics/train.csv', self.last_step, TRAIN_COLUMNS)
            truncate_after(self.run/'local_metrics/validation.csv', self.last_step, VAL_COLUMNS)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        names = ['main_elbo', 'objective', 'np_prev', 'np_next']
        values = torch.stack([pl_module._last_components[name] for name in names])
        self.pending.append(values)
        step = trainer.global_step
        if step == self.last_step:
            return
        mean = torch.stack(self.pending).mean(0)
        self.pending.clear()
        self.last_step = step
        mean = trainer.strategy.reduce(mean, reduce_op='mean')
        if not torch.isfinite(mean).all():
            raise FloatingPointError(f'Nonfinite loss at optimizer update {step}: {mean}')
        if trainer.is_global_zero:
            numbers = mean.cpu().tolist()
            record = dict(zip(names, numbers))
            record.update(optimizer_step=step,
                          learning_rate=trainer.optimizers[0].param_groups[0]['lr'],
                          elapsed_seconds=time.monotonic()-self.start,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            append(self.run/'local_metrics/train.csv', TRAIN_COLUMNS, record)
            if step <= 3 or step % 10 == 0:
                print(f'OWT update {step}/{trainer.max_steps}: main_ELBO={numbers[0]:.5f} '
                      f'objective={numbers[1]:.5f} lr={record["learning_rate"]:.3g}', flush=True)

    def on_validation_start(self, trainer, pl_module):
        self.rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                    torch.cuda.get_rng_state_all())
        # Same validation rows AND corruption on every model/checkpoint.
        seed = int(pl_module.config.seed) + 90001 + trainer.global_rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def on_validation_end(self, trainer, pl_module):
        if trainer.is_global_zero and not trainer.sanity_checking:
            record = dict(optimizer_step=trainer.global_step,
                          elapsed_seconds=time.monotonic()-self.start)
            for name in ('nll', 'ppl', 'bpd'):
                record['val_'+name] = float(trainer.callback_metrics['val/'+name])
            append(self.run/'local_metrics/validation.csv', VAL_COLUMNS, record)
            print(f'OWT validation update {trainer.global_step}: NLL={record["val_nll"]:.6f}, '
                  f'PPL bound={record["val_ppl"]:.3f}', flush=True)
            from owt.report import refresh
            refresh(self.run.parent)
        if self.rng is not None:
            random.setstate(self.rng[0])
            np.random.set_state(self.rng[1])
            torch.set_rng_state(self.rng[2])
            torch.cuda.set_rng_state_all(self.rng[3])
            self.rng = None
