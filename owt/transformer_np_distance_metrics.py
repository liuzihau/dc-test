"""Record every distance-two loss and source count without changing A/B callbacks."""
import time
from pathlib import Path
import torch

from owt.metrics import append, truncate_after, VAL_COLUMNS
from owt.source_pairing_metrics import SourcePairMetrics
from owt.source_pairing import STAT_NAMES
from owt.transformer_np_metrics import TransformerLocalMetrics

OFFSETS = (-1, 1, -2, 2)
DIRECTIONS = ('prev', 'next', 'prev2', 'next2')
LOSS_NAMES = ('main_elbo', 'objective', 'np_prev', 'np_next', 'np_prev2', 'np_next2')
TRAIN_COLUMNS = [ 'optimizer_step', *LOSS_NAMES, 'learning_rate', 'elapsed_seconds', 'peak_allocated_gib']
PAIR_COLUMNS = ['optimizer_step'] + [f'{direction}_maskbin{maskbin}_{name}'
    for direction in DIRECTIONS for maskbin in range(5) for name in STAT_NAMES] + ['elapsed_seconds']


class DistanceLocalMetrics(TransformerLocalMetrics):
    def on_train_start(self, trainer, pl_module):
        self.last_step = trainer.global_step
        if trainer.is_global_zero:
            truncate_after(self.run / 'local_metrics/train.csv', self.last_step, TRAIN_COLUMNS)
            truncate_after(self.run / 'local_metrics/validation.csv', self.last_step, VAL_COLUMNS)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.pending.append(torch.stack([pl_module._last_components[name] for name in LOSS_NAMES]))
        step = trainer.global_step
        if step == self.last_step: return
        mean = torch.stack(self.pending).mean(0); self.pending.clear(); self.last_step = step
        mean = trainer.strategy.reduce(mean, reduce_op='mean')
        if not torch.isfinite(mean).all(): raise FloatingPointError('Nonfinite distance NP objective')
        if trainer.is_global_zero:
            values = mean.cpu().tolist(); record = dict(zip(LOSS_NAMES, values))
            record.update(optimizer_step=step, learning_rate=trainer.optimizers[0].param_groups[0]['lr'],
                elapsed_seconds=time.monotonic()-self.start, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            append(self.run / 'local_metrics/train.csv', TRAIN_COLUMNS, record)
            if step <= 3 or step % 10 == 0:
                print(f'OWT distance NP update {step}/{trainer.max_steps}: main_ELBO={values[0]:.5f} '
                      f'objective={values[1]:.5f}', flush=True)


class DistancePairMetrics(SourcePairMetrics):
    def on_train_start(self, trainer, pl_module):
        self.last_step = trainer.global_step
        if trainer.is_global_zero: truncate_after(self.path, self.last_step, PAIR_COLUMNS)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.pending.append(torch.cat([pl_module._last_pair_statistics[o].flatten() for o in OFFSETS]))
        if trainer.global_step == self.last_step: return
        total = torch.stack(self.pending).sum(0); self.pending.clear(); self.last_step = trainer.global_step
        total = trainer.strategy.reduce(total, reduce_op='sum')
        if not torch.isfinite(total).all(): raise FloatingPointError('Nonfinite distance pair statistics')
        if trainer.is_global_zero:
            record = dict(zip(PAIR_COLUMNS[1:-1], total.cpu().tolist()))
            record.update(optimizer_step=self.last_step, elapsed_seconds=time.monotonic()-self.start)
            append(self.path, PAIR_COLUMNS, record)
