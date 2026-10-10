"""Consumed-data cursor and per-rank RNG, including partial epoch endings."""
import itertools
import random
import numpy as np
import torch
from torch.utils.data import DataLoader,DistributedSampler

SCHEMA='puzzle_consumed_cursor_v1'


def added_rank_rng(seed,step,rank):
    """Independent new-rank streams without changing any global RNG state."""
    seed=(int(seed)+910003+1000003*int(rank)+9176*int(step))%(2**32)
    return dict(python=random.Random(seed).getstate(),numpy=np.random.RandomState(seed).get_state(),
        torch=torch.Generator().manual_seed(seed).get_state(),
        cuda=torch.Generator(device='cuda').manual_seed(seed).get_state() if torch.cuda.is_available() else None,
        loader=torch.Generator().manual_seed(seed+701001).get_state())

class ResumeSampler(DistributedSampler):
    def __init__(self,*args,resume_epoch=0,resume_rows=0,**kwargs):
        super().__init__(*args,**kwargs)
        if not 0<=resume_rows<self.num_samples:raise ValueError('Use a normalized consumed-data cursor')
        self.resume_epoch=resume_epoch;self.resume_rows=resume_rows
    def __iter__(self):return itertools.islice(super().__iter__(),self.resume_rows if self.epoch==self.resume_epoch else 0,None)

class PuzzleDataCursor:
    def __init__(self,config,tokenizer):
        # Lightning filters saved hyperparameters against this signature.
        # TrainerBase also saves vocab_size; our author adapters derive it
        # from the tokenizer and do not accept that constructor argument.
        super().__init__(config,tokenizer)
        self._data_epoch=0;self._data_rows=0;self._data_batches=0;self._total_data_batches=0
        self._resume_cursor=None;self._resume_rng=None;self._loader_generator=None;self._rng_restored=False;self._batch_change=None

    def training_step(self,batch,batch_idx):
        loss=super().training_step(batch,batch_idx)
        self._data_rows+=int(batch['input_ids'].shape[0]);self._data_batches+=1;self._total_data_batches+=1
        return loss

    def on_train_epoch_start(self):
        if self._data_epoch!=self.trainer.current_epoch:
            self._data_epoch=int(self.trainer.current_epoch);self._data_rows=0;self._data_batches=0
        return super().on_train_epoch_start()

    def on_load_checkpoint(self,checkpoint):
        metadata=checkpoint.get('puzzle_data_cursor')
        if not metadata or metadata['schema']!=SCHEMA:raise ValueError('Resume this experiment from its own checkpoint')
        if metadata['variant']!=self.config.puzzle_recurrence_variant:raise ValueError('Resume requires the same ablation variant')
        if metadata['task']!=self.config.data.name:raise ValueError('Resume requires the same puzzle task')
        current=dict(batch=int(self.config.loader.batch_size),global_batch=int(self.config.loader.global_batch_size),
            devices=int(self.config.trainer.devices),accumulation=int(self.config.trainer.accumulate_grad_batches))
        # Evaluation may use a different batch size; model signature still checks architecture.
        if self.config.mode=='train':
            from puzzle_recurrence.batch_change import prepare_batch_change
            checkpoint,self._batch_change=prepare_batch_change(checkpoint,current,bool(self.config.get('puzzle_allow_microbatch_change',False)),
                bool(self.config.get('puzzle_allow_device_change',False)))
        super().on_load_checkpoint(checkpoint)
        self._resume_cursor=metadata['cursor'];self._sampler_seed=metadata['sampler_seed']
        states=checkpoint['puzzle_rng_by_rank'];rank=int(self.global_rank)
        self._resume_rng=states[rank] if rank<len(states) else added_rank_rng(self.config.seed,checkpoint['global_step'],rank)
        for field,name in [('epoch','_data_epoch'),('rows','_data_rows'),('batches','_data_batches'),('total_batches','_total_data_batches')]:
            setattr(self,name,int(self._resume_cursor[field]))

    def on_train_start(self):
        if self.ema:self.ema.move_shadow_params_to_device(self.device)
        if self._batch_change and self.trainer.is_global_zero:
            from puzzle_recurrence.results import atomic_json
            from pathlib import Path
            atomic_json(Path(self.config.checkpointing.save_dir)/('batch-change-step'+str(self._batch_change['step'])+'.json'),self._batch_change)
        loaders=[]
        for old in self.trainer.fit_loop._combined_loader.flattened:
            cursor=self._resume_cursor or dict(epoch=int(self.trainer.current_epoch),rows=0)
            seed=self._sampler_seed if self._resume_cursor else int(getattr(old.sampler,'seed',self.config.seed))
            sampler=ResumeSampler(old.dataset,num_replicas=self.trainer.world_size,rank=self.global_rank,
                shuffle=getattr(old.sampler,'shuffle',True),seed=seed,drop_last=False,
                resume_epoch=cursor['epoch'],resume_rows=cursor['rows'])
            sampler.set_epoch(cursor['epoch']);self._sampler_seed=seed;self._sampler_rows=sampler.num_samples
            generator=torch.Generator().manual_seed(int(self.config.seed)+701001+int(self.global_rank))
            if self._resume_rng:generator.set_state(self._resume_rng['loader'].cpu())
            self._loader_generator=generator
            loaders.append(DataLoader(old.dataset,batch_size=old.batch_size,sampler=sampler,
                num_workers=old.num_workers,pin_memory=old.pin_memory,persistent_workers=old.num_workers>0,
                collate_fn=old.collate_fn,drop_last=old.drop_last,worker_init_fn=old.worker_init_fn,generator=generator))
        self.trainer.fit_loop._combined_loader.flattened=loaders
        fetcher=self.trainer.fit_loop._data_fetcher;fetcher.teardown();iter(fetcher)

    def on_train_batch_start(self,batch,batch_idx):
        if self._rng_restored:return
        if self._resume_rng:
            state=self._resume_rng;random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
            if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'].cpu())
        else:
            seed=int(self.config.seed)+int(self.global_rank)
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        self._rng_restored=True

    def on_save_checkpoint(self,checkpoint):
        super().on_save_checkpoint(checkpoint)
        if self._data_rows>self._sampler_rows:raise ValueError('Consumed rows exceed sampler length')
        boundary=self._data_rows==self._sampler_rows
        cursor=dict(epoch=self._data_epoch+int(boundary),rows=0 if boundary else self._data_rows,
            batches=0 if boundary else self._data_batches,total_batches=self._total_data_batches)
        fit=checkpoint['loops']['fit_loop']
        for field in ('ready','started','processed','completed'):
            fit['epoch_loop.batch_progress']['total'][field]=self._total_data_batches
            fit['epoch_loop.batch_progress']['current'][field]=cursor['batches']
        fit['epoch_loop.batch_progress']['is_last_batch']=False
        for scope in ('current','total'):
            fit['epoch_progress'][scope].update(ready=cursor['epoch']+int(not boundary),started=cursor['epoch']+int(not boundary),
                processed=cursor['epoch'],completed=cursor['epoch'])
        if boundary:
            step=fit['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']['current']
            for key in step:step[key]=0
        fit['epoch_loop.state_dict']['_batches_that_stepped']=int(checkpoint['global_step'])
        policy=dict(batch=int(self.config.loader.batch_size),global_batch=int(self.config.loader.global_batch_size),
            devices=int(self.config.trainer.devices),accumulation=int(self.config.trainer.accumulate_grad_batches))
        checkpoint['puzzle_data_cursor']=dict(schema=SCHEMA,cursor=cursor,sampler_seed=self._sampler_seed,batch_policy=policy,
            variant=self.config.puzzle_recurrence_variant,task=self.config.data.name)
        local=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,loader=self._loader_generator.get_state())
        if torch.distributed.is_initialized():
            states=[None]*torch.distributed.get_world_size();torch.distributed.all_gather_object(states,local)
        else:states=[local]
        checkpoint['puzzle_rng_by_rank']=states
