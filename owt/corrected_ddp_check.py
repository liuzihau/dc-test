"""Actual two-GPU resume check, with per-rank RNG and a partial last batch."""
import argparse
import json
import hashlib
from pathlib import Path
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader
import torch
from owt.test_corrected_training import corrected,Rows,Trace

def tensor_fingerprint(tensors):
    result=hashlib.sha256()
    for name,tensor in tensors:
        result.update(str((name,tuple(tensor.shape),tensor.dtype)).encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()

def optimizer_fingerprint(state):
    return tensor_fingerprint([(f'{index}:{key}',value) for index,values in sorted(state['state'].items())
        for key,value in sorted(values.items())])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--phase',choices=['full','stop','resume'],required=True)
    parser.add_argument('--variant',default='mdm')
    args=parser.parse_args();args.run.mkdir(parents=True,exist_ok=True)
    L.seed_everything(999 if args.phase=='resume' else 21,workers=True)
    model=corrected(args.variant)
    model.config.trainer.devices=2;model.config.trainer.accumulate_grad_batches=2
    model.config.loader.global_batch_size=8
    trace=Trace(str(args.run/'middle.ckpt'),3 if args.phase=='full' else None)
    trainer=L.Trainer(accelerator='cuda',devices=2,precision='32-true',
        strategy=DDPStrategy(find_unused_parameters=False),max_steps=3 if args.phase=='stop' else 6,max_epochs=-1,
        accumulate_grad_batches=2,logger=False,enable_checkpointing=False,
        enable_progress_bar=False,enable_model_summary=False,limit_val_batches=0,
        num_sanity_val_steps=0,callbacks=[trace])
    trainer.fit(model,DataLoader(Rows(),batch_size=2,num_workers=2),
        ckpt_path=str(args.run/'middle.ckpt') if args.phase=='resume' else None)
    if args.phase=='stop':trainer.save_checkpoint(args.run/'middle.ckpt')
    result=dict(rows=trace.rows,losses=trace.losses,
        weights=tensor_fingerprint(sorted(model.state_dict().items())),
        ema=tensor_fingerprint([(str(i),x) for i,x in enumerate(model.ema.shadow_params)]),
        optimizer=optimizer_fingerprint(trainer.optimizers[0].state_dict()),
        scheduler=trainer.lr_scheduler_configs[0].scheduler.state_dict(),branch_calls=getattr(model,'branch_calls',None))
    (args.run/f'{args.phase}-rank{trainer.global_rank}.json').write_text(json.dumps(result,indent=2)+'\n')
    if args.phase=='resume':
        original=json.loads((args.run/f'full-rank{trainer.global_rank}.json').read_text())
        saved=torch.load(args.run/'middle.ckpt',map_location='cpu',weights_only=False)
        assert len(saved['rng_by_rank'])==2
        consumed=saved['corrected_training']['cursor']['total_batches']
        # 5 rows per rank/epoch: batches [2,2,1]. Step3 consumes [2,2] in epoch1.
        consumed_rows=(consumed//3)*5+sum([2,2,1][:consumed%3])
        assert result['rows']==original['rows'][consumed_rows:]
        assert result['losses']==original['losses'][consumed:]
        for key in ('weights','ema','optimizer','scheduler','branch_calls'):assert result[key]==original[key],key
        print(f'PASS: {args.variant} exact DDP resume, rank{trainer.global_rank}',flush=True)

if __name__=='__main__':main()
