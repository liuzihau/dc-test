"""Preserve learned rows and Adam/EMA state while assigning a distinct MASK ID."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/runtime/corrected-mpl'))
import torch
from omegaconf import OmegaConf
from owt.continuation import ROOT,digest,continuation_config
from owt.research import atomic_write,timestamp
from owt.corrected_training import SCHEMA,CorrectedMDM,CorrectedTransformerNP
from owt.model import OWTMDM
from owt.transformer_np_model import TransformerNPMDM


def model_for_parameter_names(config,vocabulary):
    c=OmegaConf.create(OmegaConf.to_container(config,resolve=True));c.training.ema=0
    tokenizer=SimpleNamespace(vocab_size=vocabulary-1,mask_token=None,all_special_ids=[vocabulary-2])
    with patch('diffusion.metrics.Metrics',return_value=torch.nn.Module()):
        model=(TransformerNPMDM if c.mechanisms.np.enabled else OWTMDM)(c,tokenizer)
    by_id={id(p):name for name,p in model.named_parameters()}
    names=[by_id[id(p)] for p in model._get_parameters()]
    shapes={name:tuple(p.shape) for name,p in model.named_parameters()}
    del model;gc.collect()
    return names,shapes


def vocabulary_parameter(name):
    return name=='backbone.vocab_embed.embedding' or name.startswith('backbone.output_layer.linear.') or name.startswith('backbone.neighbor_heads.')


def expand(tensor,name,old_vocab,eos_id,moment=False):
    if tensor.shape[0]!=old_vocab:raise ValueError('Unexpected vocabulary row count')
    result=tensor.new_zeros((old_vocab+1,*tensor.shape[1:]))
    result[:old_vocab-1]=tensor[:old_vocab-1]
    result[old_vocab]=tensor[old_vocab-1] # old MASK row moves to new MASK ID
    if not moment:
        if name=='backbone.vocab_embed.embedding':result[old_vocab-1]=tensor[eos_id]
        elif name.endswith('bias'):result[old_vocab-1]=-30.0
    return result


def migrate_payload(payload,config,parameter_names,shapes,eos_id):
    p=dict(payload);state=dict(payload['state_dict'])
    old_vocab=state['backbone.vocab_embed.embedding'].shape[0]
    pad_id=old_vocab-1;mask_id=old_vocab
    vocab=dict(schema=SCHEMA,pad_id=pad_id,mask_id=mask_id,vocab_size=old_vocab+1,
        legacy_resume_seed=20261008)
    config=OmegaConf.create(OmegaConf.to_container(config,resolve=True));OmegaConf.set_struct(config,False)
    config.corrected_training=vocab
    config.trainer.max_steps=85100
    migrated=[]
    for name,tensor in state.items():
        if vocabulary_parameter(name):
            if name not in shapes or tensor.shape!=torch.Size(shapes[name]):raise ValueError('Parameter mapping differs')
            state[name]=expand(tensor,name,old_vocab,eos_id);migrated.append(name)
    p['state_dict']=state
    p['hyper_parameters']=dict(payload['hyper_parameters'],config=config)
    ema=dict(payload['ema']);shadow=[]
    if len(parameter_names)!=len(ema['shadow_params']):raise ValueError('EMA order length differs')
    for name,tensor in zip(parameter_names,ema['shadow_params']):
        if tensor.shape!=torch.Size(shapes[name]):raise ValueError('EMA parameter order differs')
        shadow.append(expand(tensor,name,old_vocab,eos_id) if name in migrated else tensor)
    ema['shadow_params']=shadow;p['ema']=ema
    optimizers=[]
    for optimizer in payload['optimizer_states']:
        optimizer=dict(optimizer);optimizer['state']={key:dict(value) for key,value in optimizer['state'].items()}
        ids=[index for group in optimizer['param_groups'] for index in group['params']]
        if len(ids)!=len(parameter_names):raise ValueError('Optimizer parameter order length differs')
        for index,name in zip(ids,parameter_names):
            for key,tensor in optimizer['state'][index].items():
                if torch.is_tensor(tensor) and tensor.ndim and name in migrated:
                    if tensor.shape!=torch.Size(shapes[name]):raise ValueError('Adam moment shape differs')
                    optimizer['state'][index][key]=expand(tensor,name,old_vocab,eos_id,moment=True)
        optimizers.append(optimizer)
    p['optimizer_states']=optimizers
    loops=payload['loops']['fit_loop'];batch=loops['epoch_loop.batch_progress']['current']['completed']
    epoch=loops['epoch_progress']['current']['completed']
    cursor=dict(epoch=epoch,rows=batch*config.loader.batch_size,batches=batch,
        total_batches=loops['epoch_loop.batch_progress']['total']['completed'])
    p['corrected_training']=dict(schema=SCHEMA,vocabulary=vocab,cursor=cursor,
        global_step=payload['global_step'],sampler_seed=int(config.seed),consumed_cursor=True,
        migrated_from_legacy=True,legacy_rng_missing=True)
    p['rng_by_rank']=None
    if 'transformer_np' in payload:
        p['transformer_np']=copy.deepcopy(payload['transformer_np'])
        p['transformer_np']['signature']['corrected_vocabulary']=vocab
    return p,migrated


def audit_preservation(old,new,names):
    old_vocab=old['state_dict']['backbone.vocab_embed.embedding'].shape[0]
    for name,value in old['state_dict'].items():
        actual=new['state_dict'][name]
        if vocabulary_parameter(name):
            if not torch.equal(value[:old_vocab-1],actual[:old_vocab-1]) or not torch.equal(value[-1],actual[old_vocab]):
                raise ValueError('A learned vocabulary row changed')
        elif not torch.equal(value,actual):raise ValueError('A backbone/branch parameter changed')
    for name,value,actual in zip(names,old['ema']['shadow_params'],new['ema']['shadow_params']):
        if vocabulary_parameter(name):
            if not torch.equal(value[:-1],actual[:old_vocab-1]) or not torch.equal(value[-1],actual[old_vocab]):raise ValueError('EMA row changed')
        elif not torch.equal(value,actual):raise ValueError('EMA state changed')
    for original,migrated in zip(old['optimizer_states'],new['optimizer_states']):
        ids=[i for group in original['param_groups'] for i in group['params']]
        for index,name in zip(ids,names):
            for key,value in original['state'][index].items():
                actual=migrated['state'][index][key]
                if torch.is_tensor(value):
                    if value.ndim and vocabulary_parameter(name):
                        if not torch.equal(value[:-1],actual[:old_vocab-1]) or not torch.equal(value[-1],actual[old_vocab]) or actual[old_vocab-1].any():
                            raise ValueError('Adam row migration differs')
                    elif not torch.equal(value,actual):raise ValueError('Adam state changed')
                elif value!=actual:raise ValueError('Optimizer scalar changed')


def migrate(source,destination):
    if destination.exists():raise FileExistsError('Preserve existing migration')
    payload=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    raw=payload['hyper_parameters']['config']
    config=continuation_config(raw,destination.parent,source)
    names,shapes=model_for_parameter_names(config,payload['state_dict']['backbone.vocab_embed.embedding'].shape[0])
    result,migrated=migrate_payload(payload,config,names,shapes,50256)
    audit_preservation(payload,result,names)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.partial.ckpt');torch.save(result,temporary);temporary.replace(destination)
    record=dict(created_at=timestamp(),source=str(source.relative_to(ROOT)),source_sha256=digest(source),
        checkpoint=str(destination.relative_to(ROOT)),checkpoint_sha256=digest(destination),step=result['global_step'],
        old_vocab=50258,new_vocab=50259,pad_id=50257,mask_id=50258,
        learned_rows_preserved=True,mask_embedding_and_moments_moved=True,backbone_and_branch_weights_preserved=True,
        optimizer_and_ema_preserved=True,new_pad_moments_zero=True,new_pad_embedding='copy EOS',new_pad_readout_bias=-30,
        original_checkpoints_modified=False,migrated_parameters=migrated,cursor=result['corrected_training']['cursor'])
    atomic_write(destination.with_suffix('.json'),json.dumps(record,indent=2)+'\n')
    print('Migrated',source,'→',destination,flush=True)
    return record


def verify_migrated(source,destination):
    """Reload the written checkpoint and exercise strict model/Adam/EMA loading."""
    original=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    payload=torch.load(destination,map_location='cpu',weights_only=False,mmap=True)
    config=payload['hyper_parameters']['config']
    old_vocab=original['state_dict']['backbone.vocab_embed.embedding'].shape[0]
    names,_=model_for_parameter_names(config,old_vocab)
    audit_preservation(original,payload,names)
    vocabulary=dict(config.corrected_training)
    tokenizer=SimpleNamespace(vocab_size=vocabulary['vocab_size'],mask_token='[MASK]',
        mask_token_id=vocabulary['mask_id'],pad_token_id=vocabulary['pad_id'],all_special_ids=[50256,50257,50258])
    with patch('diffusion.metrics.Metrics',return_value=torch.nn.Module()):
        model=(CorrectedTransformerNP if config.mechanisms.np.enabled else CorrectedMDM)(config,tokenizer)
    model.load_state_dict(payload['state_dict'],strict=True)
    model.on_load_checkpoint(payload)
    optimizers,schedulers=model.configure_optimizers()
    optimizers[0].load_state_dict(payload['optimizer_states'][0])
    schedulers[0]['scheduler'].load_state_dict(payload['lr_schedulers'][0])
    assert model.ema.num_updates==payload['global_step']
    assert schedulers[0]['scheduler'].last_epoch==payload['global_step']
    assert {int(s['step']) for s in optimizers[0].state.values()}=={payload['global_step']}
    for parameter in model._get_parameters():
        for key in ('exp_avg','exp_avg_sq'):
            assert optimizers[0].state[parameter][key].shape==parameter.shape
    record=dict(checkpoint=str(destination.relative_to(ROOT)),step=payload['global_step'],
        strict_model_load=True,adam_and_ema_load=True,scheduler_load=True,
        written_checkpoint_preserves_learned_rows=True,legacy_rng_missing=True)
    atomic_write(destination.with_suffix('.verified.json'),json.dumps(record,indent=2)+'\n')
    print('Verified',destination,flush=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    (verify_migrated if args.verify else migrate)(args.source.resolve(),args.output.resolve())
