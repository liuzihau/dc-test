"""Separate PAD/MASK and checkpoint consumed data and RNG across epochs."""
import copy
import itertools
import random
import numpy as np
import torch
from torch.utils.data import DataLoader,DistributedSampler
from owt.model import OWTMDM
from owt.transformer_np_model import TransformerNPMDM

SCHEMA='separate_pad_mask_consumed_cursor_v1'


class TokenizerAdapter:
    """HF vocab_size excludes added tokens; the model needs the full vocabulary."""
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
        self.vocab_size=len(tokenizer)
        self.mask_token=tokenizer.mask_token
        self.mask_token_id=tokenizer.mask_token_id
        self.pad_token_id=tokenizer.pad_token_id
        self.all_special_ids=tokenizer.all_special_ids

    def __getattr__(self,name):
        tokenizer=self.__dict__.get('tokenizer')
        if tokenizer is None:raise AttributeError(name)
        return getattr(tokenizer,name)


class EpochResumeSampler(DistributedSampler):
    def __init__(self,*args,resume_epoch=None,resume_rows=0,**kwargs):
        super().__init__(*args,**kwargs)
        if resume_rows<0 or resume_rows>=self.num_samples:raise ValueError('Normalize end-of-epoch cursor before loading')
        self.resume_epoch=resume_epoch;self.resume_rows=resume_rows

    def __iter__(self):
        indices=super().__iter__()
        skip=self.resume_rows if self.epoch==self.resume_epoch else 0
        return itertools.islice(indices,skip,None)


def capture_rng(generator=None):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        loader=generator.get_state() if generator is not None else None)


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None:torch.cuda.set_rng_state(state['cuda'].cpu())


class CorrectedTraining:
    def __init__(self,config,tokenizer):
        self.corrected=dict(config.corrected_training)
        if self.corrected['schema']!=SCHEMA or tokenizer.pad_token_id==tokenizer.mask_token_id:
            raise ValueError('Distinct PAD and MASK IDs are required')
        super().__init__(config,tokenizer)
        if self.vocab_size!=self.corrected['vocab_size'] or self.mask_index!=self.corrected['mask_id']:
            raise ValueError('Tokenizer/model vocabulary mismatch')
        self._data_epoch=0;self._data_rows=0;self._data_batches=0;self._total_data_batches=0
        self._resume_cursor=None;self._resume_rng=None;self._rng_restored=False;self._loader_generator=None

    def resume_signature(self):
        signature=super().resume_signature()
        signature['corrected_vocabulary']=self.corrected
        return signature

    def _loss(self,x0,attention_mask,**kwargs):
        if (x0==self.mask_index).any():raise ValueError('Clean labels contain reserved MASK')
        # Literal [PAD] tokens in wrapped text remain valid if the original
        # attention mask marks them valid. Actual padded targets remain excluded.
        return super()._loss(x0,attention_mask,**kwargs)

    def training_step(self,batch,batch_idx):
        loss=super().training_step(batch,batch_idx)
        self._data_rows+=int(batch['input_ids'].shape[0]);self._data_batches+=1;self._total_data_batches+=1
        return loss

    def on_train_epoch_start(self):
        if self._data_epoch!=self.trainer.current_epoch:
            self._data_epoch=int(self.trainer.current_epoch);self._data_rows=0;self._data_batches=0
        return super().on_train_epoch_start()

    def on_load_checkpoint(self,checkpoint):
        metadata=checkpoint.get('corrected_training')
        if not metadata or metadata['schema']!=SCHEMA or metadata['vocabulary']!=self.corrected:
            raise ValueError('Use a migrated or corrected checkpoint')
        self._resume_cursor=metadata['cursor']
        self._resume_sampler_seed=int(metadata['sampler_seed'])
        states=checkpoint.get('rng_by_rank')
        self._resume_rng=states[int(self.global_rank)] if states else None
        super().on_load_checkpoint(checkpoint)
        cursor=self._resume_cursor
        self._data_epoch=int(cursor['epoch']);self._data_rows=int(cursor['rows'])
        self._data_batches=int(cursor['batches']);self._total_data_batches=int(cursor['total_batches'])

    def on_train_start(self):
        if self.ema:self.ema.move_shadow_params_to_device(self.device)
        old_loaders=self.trainer.fit_loop._combined_loader.flattened
        loaders=[]
        for old in old_loaders:
            cursor=self._resume_cursor or dict(epoch=int(self.trainer.current_epoch),rows=0,batches=0,total_batches=0)
            replicas=int(self.trainer.world_size);rank=int(self.global_rank)
            seed=int(self._resume_sampler_seed if self._resume_cursor else getattr(old.sampler,'seed',self.config.seed))
            sampler=EpochResumeSampler(old.dataset,num_replicas=replicas,rank=rank,
                shuffle=getattr(old.sampler,'shuffle',True),seed=seed,drop_last=getattr(old.sampler,'drop_last',False),
                resume_epoch=int(cursor['epoch']),resume_rows=int(cursor['rows']))
            sampler.set_epoch(int(cursor['epoch']))
            generator=torch.Generator().manual_seed(int(self.config.seed)+701001+rank)
            if self._resume_rng and self._resume_rng.get('loader') is not None:
                generator.set_state(self._resume_rng['loader'].cpu())
            self._loader_generator=generator;self._sampler_rows=sampler.num_samples;self._sampler_seed=seed
            loaders.append(DataLoader(old.dataset,batch_size=old.batch_size,sampler=sampler,
                num_workers=old.num_workers,pin_memory=old.pin_memory,persistent_workers=old.num_workers>0,
                collate_fn=old.collate_fn,drop_last=old.drop_last,worker_init_fn=old.worker_init_fn,generator=generator))
            print(f'Corrected data rank{rank}: epoch={cursor["epoch"]}, consumed rows={cursor["rows"]}/{sampler.num_samples}',flush=True)
        self.trainer.fit_loop._combined_loader.flattened=loaders
        fetcher=self.trainer.fit_loop._data_fetcher;fetcher.teardown();iter(fetcher)

    def on_train_batch_start(self,batch,batch_idx):
        if not self._rng_restored:
            if self._resume_rng:restore_rng(self._resume_rng)
            elif self._resume_cursor:
                seed=int(self.corrected['legacy_resume_seed'])+int(self.global_rank)
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            self._rng_restored=True

    def on_save_checkpoint(self,checkpoint):
        # Base BD3 assumes full accumulation groups. Use consumed batches/rows,
        # including partial final batches, instead of optimizer_steps*accumulation.
        super().on_save_checkpoint(checkpoint)
        if self._data_rows>self._sampler_rows:raise ValueError('Consumed beyond epoch length')
        boundary=self._data_rows==self._sampler_rows
        cursor=dict(epoch=self._data_epoch+int(boundary),rows=0 if boundary else self._data_rows,
            batches=0 if boundary else self._data_batches,total_batches=self._total_data_batches)
        fit=checkpoint['loops']['fit_loop']
        for key in ('ready','started','processed','completed'):
            fit['epoch_loop.batch_progress']['total'][key]=self._total_data_batches
            fit['epoch_loop.batch_progress']['current'][key]=cursor['batches']
        fit['epoch_loop.batch_progress']['is_last_batch']=False
        # Trainer.fit increments its epoch counter even when max_steps stops
        # mid-epoch. Restore the epoch recorded by the consumed-data cursor.
        for scope in ('current','total'):
            fit['epoch_progress'][scope].update(ready=cursor['epoch']+int(not boundary),
                started=cursor['epoch']+int(not boundary),processed=cursor['epoch'],completed=cursor['epoch'])
        if boundary:
            fit['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']['current']={
                k:0 for k in fit['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']['current']}
        fit['epoch_loop.state_dict']['_batches_that_stepped']=int(checkpoint['global_step'])
        checkpoint['corrected_training']=dict(schema=SCHEMA,vocabulary=self.corrected,cursor=cursor,
            global_step=int(checkpoint['global_step']),sampler_seed=self._sampler_seed,consumed_cursor=True)
        local=capture_rng(self._loader_generator)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            states=[None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(states,local)
        else:states=[local]
        checkpoint['rng_by_rank']=states


class CorrectedMDM(CorrectedTraining,OWTMDM):pass
class CorrectedTransformerNP(CorrectedTraining,TransformerNPMDM):pass
