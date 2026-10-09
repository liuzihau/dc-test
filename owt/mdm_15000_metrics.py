"""Verify full7500-step restoration before the baseline-only continuation."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from owt.continuation import tensor_fingerprint,optimizer_fingerprint
from owt.continuation_metrics import batch_digest
from owt.research import atomic_write


class ResumeAudit(Callback):
    def __init__(self,run,expected):
        self.run=Path(run);self.expected=expected;self.initialized=False;self.seen=set();self.traces=[]

    def on_train_start(self,trainer,module):
        if trainer.global_step!=7500 or trainer.max_steps!=15000:raise ValueError('Require7500→15000')
        observed=dict(step=trainer.global_step,
            model=tensor_fingerprint(sorted(module.state_dict().items())),
            optimizer=optimizer_fingerprint(trainer.optimizers[0].state_dict()),
            ema=tensor_fingerprint([(str(i),x) for i,x in enumerate(module.ema.shadow_params)]),
            ema_updates=module.ema.num_updates,scheduler_last_epoch=trainer.lr_scheduler_configs[0].scheduler.last_epoch,
            learning_rates=[g['lr'] for g in trainer.optimizers[0].param_groups],
            fast_forward_batches=module.fast_forward_batches,fast_forward_epochs=module.fast_forward_epochs)
        if observed!=self.expected:raise ValueError('Full restored state differs: '+str({k:v for k,v in observed.items() if v!=self.expected.get(k)}))
        self.restored=observed

    def on_train_batch_start(self,trainer,module,batch,batch_idx):
        if not self.initialized:
            seed=1500001+trainer.global_rank
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            self.initialized=True
            atomic_write(self.run/f'resume-rank{trainer.global_rank}.json',json.dumps(dict(restored=self.restored,
                seed=seed,rng_policy='new reproducible per-rank stream; previous checkpoint has no per-rank RNG'),indent=2)+'\n')
        step=trainer.global_step
        if step in (7500,7501,7502) and step not in self.seen:
            self.seen.add(step)
            row=dict(optimizer_step=step+1,clean_sha256=batch_digest(batch['input_ids']))
            self.traces.append(row)
            def capture(backbone,inputs):
                row['noisy_sha256']=batch_digest(inputs[0]);hook.remove()
                atomic_write(self.run/f'first-batches-rank{trainer.global_rank}.json',json.dumps(self.traces,indent=2)+'\n')
            hook=module.backbone.register_forward_pre_hook(capture)
