"""Audit restored state and pair the new continuation RNG streams."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from owt.continuation import tensor_fingerprint, optimizer_fingerprint
from owt.research import atomic_write


def batch_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class ContinuationAudit(Callback):
    def __init__(self,run,expected,continuation_seed=750001):
        self.run=Path(run);self.expected=expected;self.seed=continuation_seed
        self.initialized=False;self.seen=set();self.traces=[];self.hook=None

    def on_train_start(self,trainer,module):
        if trainer.global_step!=5000 or trainer.max_steps!=7500:
            raise ValueError('Continuation must start5000 and stop7500')
        observed=dict(step=trainer.global_step,
            model=tensor_fingerprint(sorted(module.state_dict().items())),
            optimizer=optimizer_fingerprint(trainer.optimizers[0].state_dict()),
            ema=tensor_fingerprint([(str(i),x) for i,x in enumerate(module.ema.shadow_params)]),
            ema_updates=module.ema.num_updates,
            scheduler_last_epoch=trainer.lr_scheduler_configs[0].scheduler.last_epoch,
            learning_rates=[g['lr'] for g in trainer.optimizers[0].param_groups],
            fast_forward_batches=module.fast_forward_batches,
            fast_forward_epochs=module.fast_forward_epochs,
            branch_calls=getattr(module,'branch_calls',None))
        if observed!=self.expected:
            raise ValueError('Restored training state differs: '+str({k:(self.expected.get(k),v) for k,v in observed.items() if self.expected.get(k)!=v}))
        self.restored=observed

    def on_train_batch_start(self,trainer,module,batch,batch_idx):
        if not self.initialized:
            seed=self.seed+trainer.global_rank
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.initialized=True
            atomic_write(self.run/f'resume-rank{trainer.global_rank}.json',json.dumps(
                dict(restored=self.restored,continuation_seed=seed,
                     data_cursor_after_model_hook=module.fast_forward_batches,
                     rng_policy='matched new stream because original checkpoint has no per-rank RNG'),indent=2)+'\n')
        step=trainer.global_step
        if step not in self.seen and step in (5000,5001,5002):
            self.seen.add(step)
            row=dict(optimizer_step=step+1,clean_sha256=batch_digest(batch['input_ids']),
                     attention_sha256=batch_digest(batch['attention_mask']))
            self.traces.append(row)
            def capture(backbone,inputs):
                row['noisy_sha256']=batch_digest(inputs[0])
                self.hook.remove();self.hook=None
                atomic_write(self.run/f'first-batches-rank{trainer.global_rank}.json',
                             json.dumps(self.traces,indent=2)+'\n')
            self.hook=module.backbone.register_forward_pre_hook(capture)
