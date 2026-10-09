"""Per-update pair counts and weight mass, stratified by empirical mask fraction."""
import time
from pathlib import Path
import torch
from lightning.pytorch.callbacks import Callback
from owt.metrics import append, truncate_after
from owt.source_pairing import STAT_NAMES

PAIR_COLUMNS = ['optimizer_step'] + [f'{direction}_maskbin{maskbin}_{name}'
    for direction in ('prev', 'next') for maskbin in range(5) for name in STAT_NAMES] + ['elapsed_seconds']


class SourcePairMetrics(Callback):
    def __init__(self, run):
        self.path = Path(run)/'local_metrics/source_pairs.csv'
        self.pending = []
        self.last_step = 0
        self.start = time.monotonic()

    def on_train_start(self, trainer, pl_module):
        self.last_step = trainer.global_step
        if trainer.is_global_zero:
            truncate_after(self.path, self.last_step, PAIR_COLUMNS)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.pending.append(torch.cat([pl_module._last_pair_statistics[offset].flatten() for offset in (-1, 1)]))
        if trainer.global_step == self.last_step:
            return
        total = torch.stack(self.pending).sum(0)
        self.pending.clear()
        self.last_step = trainer.global_step
        total = trainer.strategy.reduce(total, reduce_op='sum')
        if not torch.isfinite(total).all():
            raise FloatingPointError('Nonfinite source pair statistics')
        if trainer.is_global_zero:
            record = dict(zip(PAIR_COLUMNS[1:-1], total.cpu().tolist()))
            record.update(optimizer_step=self.last_step, elapsed_seconds=time.monotonic()-self.start)
            append(self.path, PAIR_COLUMNS, record)
