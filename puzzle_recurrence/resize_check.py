"""Tiny CPU-DDP test of full training-state migration from two to four ranks."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES']=''
from puzzle_recurrence.entrypoint import ROOT
from puzzle_recurrence.test_models import make,Rows,full_mask,author_dit
from puzzle_recurrence.ddp_check import fingerprint
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader


class TwentyRows(Rows):
    def __len__(self):return 20


class VerifyRestoration(Callback):
    def __init__(self,path):self.path=path
    def on_train_start(self,tr,model):
        saved=torch.load(self.path,map_location='cpu',weights_only=False)
        assert tr.global_step==3
        for name,value in model.state_dict().items():torch.testing.assert_close(value,saved['state_dict'][name],rtol=0,atol=0)
        for a,b in zip(model.ema.shadow_params,saved['ema']['shadow_params']):torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert model.ema.num_updates==saved['ema']['num_updates']==3
        assert tr.lr_scheduler_configs[0].scheduler.state_dict()==saved['lr_schedulers'][0]
        assert fingerprint([(str((i,k)),v) for i,s in tr.optimizers[0].state_dict()['state'].items() for k,v in s.items()])==\
               fingerprint([(str((i,k)),v) for i,s in saved['optimizer_states'][0]['state'].items() for k,v in s.items()])
        assert model._resume_cursor['rows']==0 and model._resume_cursor['epoch']==1
        if tr.global_rank<2:torch.testing.assert_close(model._resume_rng['torch'],saved['puzzle_rng_by_rank'][tr.global_rank]['torch'],rtol=0,atol=0)
        streams=[None]*4
        torch.distributed.all_gather_object(streams,bytes(model._resume_rng['torch'].tolist()))
        assert len(set(streams))==4
        print(f'PASS full-state 2->4 restore rank{tr.global_rank}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['save','resume'],required=True)
    p.add_argument('--task',choices=['sudoku','zebra'],required=True)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();args.run=args.run.resolve();args.run.mkdir(parents=True,exist_ok=True)
    resume=args.phase=='resume';devices=4 if resume else 2;accumulation=1 if resume else 2
    L.seed_everything(21)
    with patch.object(author_dit,'FORCE_NAIVE_ATTENTION',True),patch.object(author_dit,'_get_full_mask',side_effect=full_mask),\
        patch('metrics.transformers.AutoTokenizer.from_pretrained',return_value=SimpleNamespace(pad_token='[PAD]',pad_token_id=0)):
        model=make(args.task);model.config.trainer.devices=devices;model.config.trainer.accumulate_grad_batches=accumulation
        model.config.loader.global_batch_size=8;model.config.puzzle_allow_device_change=resume
        trainer=L.Trainer(accelerator='cpu',devices=devices,strategy=DDPStrategy(process_group_backend='gloo',find_unused_parameters=False),
            max_steps=6 if resume else 3,max_epochs=-1,accumulate_grad_batches=accumulation,logger=False,
            enable_checkpointing=False,enable_model_summary=False,enable_progress_bar=False,num_sanity_val_steps=0,
            limit_val_batches=0,callbacks=[VerifyRestoration(args.run/'two.ckpt')] if resume else [])
        trainer.fit(model,DataLoader(TwentyRows(model),batch_size=2,num_workers=0),ckpt_path=str(args.run/'two.ckpt') if resume else None)
        trainer.save_checkpoint(args.run/('four.ckpt' if resume else 'two.ckpt'))
        if trainer.is_global_zero and resume:
            saved=torch.load(args.run/'four.ckpt',map_location='cpu',weights_only=False)
            assert saved['global_step']==6 and len(saved['puzzle_rng_by_rank'])==4
            assert saved['puzzle_data_cursor']['batch_policy']['devices']==4
            (args.run/'verification.json').write_text(json.dumps(dict(task=args.task,old_devices=2,new_devices=4,
                restored_optimizer_ema_scheduler=True,unique_rng_streams=4,completed_step=6),indent=2)+'\n')


if __name__=='__main__':main()
