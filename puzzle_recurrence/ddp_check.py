"""Small CPU-DDP correctness check; no benchmark data or GPU allocation."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from puzzle_recurrence.entrypoint import ROOT
from puzzle_recurrence.test_models import make,Rows,Trace,full_mask,author_dit
import torch
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader


def fingerprint(items):
    h=hashlib.sha256()
    for name,tensor in items:
        h.update(str((name,tuple(tensor.shape),str(tensor.dtype))).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['sudoku','zebra'],required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--phase',choices=['full','stop','resume'],required=True)
    args=parser.parse_args();args.run=args.run.resolve();args.run.mkdir(parents=True,exist_ok=True)
    L.seed_everything(999 if args.phase=='resume' else 21)
    with patch.object(author_dit,'FORCE_NAIVE_ATTENTION',True),patch.object(author_dit,'_get_full_mask',side_effect=full_mask),\
            patch('metrics.transformers.AutoTokenizer.from_pretrained',return_value=SimpleNamespace(pad_token='[PAD]',pad_token_id=0)):
        model=make(args.task)
        model.config.trainer.devices=2;model.config.trainer.accumulate_grad_batches=2;model.config.loader.global_batch_size=8
        trace=Trace()
        trainer=L.Trainer(accelerator='cpu',devices=2,strategy=DDPStrategy(process_group_backend='gloo',find_unused_parameters=False),
            max_steps=3 if args.phase=='stop' else 6,max_epochs=-1,accumulate_grad_batches=2,
            logger=False,enable_checkpointing=False,enable_model_summary=False,enable_progress_bar=False,
            num_sanity_val_steps=0,limit_val_batches=0,callbacks=[trace])
        trainer.fit(model,DataLoader(Rows(model),batch_size=2,num_workers=2),
            ckpt_path=str(args.run/'stop.ckpt') if args.phase=='resume' else None)
        if args.phase=='stop':trainer.save_checkpoint(args.run/'stop.ckpt')
        optimizer=trainer.optimizers[0].state_dict()
        record=dict(losses=trace.losses,weights=fingerprint(sorted(model.state_dict().items())),
            ema=fingerprint([(str(i),v) for i,v in enumerate(model.ema.shadow_params)]),
            optimizer=fingerprint([(str((i,k)),v) for i,state in optimizer['state'].items() for k,v in state.items()]),
            scheduler=trainer.lr_scheduler_configs[0].scheduler.state_dict(),global_step=trainer.global_step)
        (args.run/f'{args.phase}-rank{trainer.global_rank}.json').write_text(json.dumps(record,indent=2)+'\n')
        if args.phase=='resume':
            full=json.loads((args.run/f'full-rank{trainer.global_rank}.json').read_text())
            prefix=json.loads((args.run/f'stop-rank{trainer.global_rank}.json').read_text())
            assert prefix['losses']+record['losses']==full['losses']
            for key in ('weights','ema','optimizer','scheduler','global_step'):assert record[key]==full[key],key
            checkpoint=torch.load(args.run/'stop.ckpt',map_location='cpu',weights_only=False)
            assert len(checkpoint['puzzle_rng_by_rank'])==2
            print(f'PASS: {args.task} exact DDP resume, rank{trainer.global_rank}',flush=True)

if __name__=='__main__':main()
