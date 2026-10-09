"""Joint gradient norms with NP processing separate from the shared trunk."""
import random
import time

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from owt.metrics import LocalMetrics, VAL_COLUMNS, append, truncate_after

COLUMNS = ['optimizer_step', 'joint_l2', 'shared_trunk_l2', 'main_readout_l2',
           'neighbor_readouts_l2', 'neighbor_processing_l2', 'clip_limit',
           'estimated_clip_multiplier', 'elapsed_seconds']


@torch.no_grad()
def transformer_gradient_norms(module, clip_limit):
    groups = {key: [] for key in COLUMNS[2:6]}
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith('backbone.neighbor_branches.'):
            group = 'neighbor_processing_l2'
        elif name.startswith('backbone.neighbor_heads.'):
            group = 'neighbor_readouts_l2'
        elif name.startswith('backbone.output_layer.linear.'):
            group = 'main_readout_l2'
        else:
            group = 'shared_trunk_l2'
        groups[group].append(torch.linalg.vector_norm(parameter.grad.detach(), 2))
    squared = {key: float(torch.stack(values).double().square().sum()) if values else 0.
               for key, values in groups.items()}
    result = {key: value**.5 for key, value in squared.items()}
    result['joint_l2'] = sum(squared.values())**.5
    result['clip_limit'] = float(clip_limit or 0.)
    result['estimated_clip_multiplier'] = min(1., result['clip_limit'] / (result['joint_l2'] + 1e-6)) if result['clip_limit'] > 0 else 1.
    return result


class TransformerGradientMetrics(Callback):
    def __init__(self, run):
        from pathlib import Path
        self.path = Path(run) / 'local_metrics/gradient_norms.csv'
        self.start = time.monotonic()

    def on_train_start(self, trainer, pl_module):
        algorithm = getattr(trainer.gradient_clip_algorithm, 'value', trainer.gradient_clip_algorithm)
        if algorithm is not None and str(algorithm) != 'norm':
            raise ValueError('Transformer NP diagnostics require L2 clipping')
        if trainer.is_global_zero:
            truncate_after(self.path, trainer.global_step, COLUMNS)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.is_global_zero:
            record = transformer_gradient_norms(pl_module, trainer.gradient_clip_val)
            record.update(optimizer_step=trainer.global_step + 1, elapsed_seconds=time.monotonic() - self.start)
            if not all(torch.isfinite(torch.tensor(value)) for value in record.values()):
                raise FloatingPointError('Nonfinite transformer NP gradients')
            append(self.path, COLUMNS, record)


class TransformerLocalMetrics(LocalMetrics):
    """Same fixed validation/RNG protocol without the legacy variant plot list."""
    def on_validation_end(self, trainer, pl_module):
        try:
            if trainer.is_global_zero and not trainer.sanity_checking:
                record = dict(optimizer_step=trainer.global_step,
                              elapsed_seconds=time.monotonic() - self.start)
                for name in ('nll', 'ppl', 'bpd'):
                    record['val_' + name] = float(trainer.callback_metrics['val/' + name])
                append(self.run / 'local_metrics/validation.csv', VAL_COLUMNS, record)
                print(f'OWT transformer NP validation {trainer.global_step}: NLL={record["val_nll"]:.6f}', flush=True)
        finally:
            if self.rng is not None:
                random.setstate(self.rng[0])
                np.random.set_state(self.rng[1])
                torch.set_rng_state(self.rng[2])
                torch.cuda.set_rng_state_all(self.rng[3])
                self.rng = None
