"""Continuation contracts, real Adam/EMA restoration, and matched-input audits."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from owt.continuation import A, continuation_config, inspect_checkpoint, tensor_fingerprint, optimizer_fingerprint
from owt.continuation_schedule import verify_startup, prepare_run
from owt.continuation_metrics import ContinuationAudit
from owt.test_initialization import config, model
from owt.test_transformer_np import make_model, batch

torch.set_num_threads(2)


def test_checkpoint_interpolations_resolve_in_fresh_process(tmp_path):
    import subprocess
    import sys
    code = '''
from omegaconf import OmegaConf
from owt.continuation import ROOT,continuation_config
assert not OmegaConf.has_resolver('cwd')
saved=OmegaConf.create({'trainer':{'max_steps':5000},'checkpointing':{},
    'eval':{'checkpoint_path':'${cwd:}/outputs/model.ckpt'},
    'devices':'${device_count:}', 'value':'${eval:2+3}', 'ceil':'${div_up:9,2}'})
actual=continuation_config(saved,ROOT/'run',ROOT/'original.ckpt')
assert actual.eval.checkpoint_path==str(ROOT/'outputs/model.ckpt')
assert actual.value==actual.ceil==5
assert isinstance(actual.devices,int) and actual.trainer.max_steps==7500
'''
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)


@pytest.mark.parametrize('variant',['mdm',A])
def test_continuation_preserves_training_configuration_except_destination_and_stop(tmp_path,variant):
    saved=config(variant)
    actual=continuation_config(saved,tmp_path/'run',tmp_path/'original.ckpt')
    assert actual.trainer.max_steps==7500 and actual.checkpointing.resume_from_ckpt
    assert actual.checkpointing.resume_ckpt_path==str(tmp_path/'original.ckpt')
    actual.trainer.max_steps=saved.trainer.max_steps
    actual.checkpointing=copy.deepcopy(saved.checkpointing)
    assert OmegaConf.to_container(actual,resolve=True)==OmegaConf.to_container(saved,resolve=True)


def checkpoint_header(variant='mdm'):
    cfg=OmegaConf.to_container(config(variant),resolve=True)
    cfg['model']['length']=1024
    return dict(global_step=5000,hyper_parameters={'config':cfg},
        optimizer_states=[{'state':{0:{'step':torch.tensor(5000.),'exp_avg':torch.ones(2),
            'exp_avg_sq':torch.ones(2)}},'param_groups':[{'lr':.0003}]}],
        lr_schedulers=[{'last_epoch':5000}],ema={'num_updates':5000},
        loops={'fit_loop':{'epoch_loop.batch_progress':{'current':{'completed':160000}},
            'epoch_progress':{'current':{'completed':0}}}},
        **({'transformer_np':{'calls':160000}} if variant==A else {}))


@pytest.mark.parametrize('variant',['mdm',A])
def test_production_checkpoint_header(variant):
    result=inspect_checkpoint(checkpoint_header(variant),variant)
    assert result['sampler_rows_per_rank']==1280000


@pytest.mark.parametrize('change',['step','optimizer','ema','lr','cursor','weight'])
def test_reject_incomplete_or_changed_production_resume(change):
    p=checkpoint_header(A)
    if change=='step': p['global_step']=4999
    elif change=='optimizer': del p['optimizer_states'][0]['state'][0]['exp_avg']
    elif change=='ema': p['ema']['num_updates']=4999
    elif change=='lr': p['optimizer_states'][0]['param_groups'][0]['lr']=.0001
    elif change=='cursor': p['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']=159999
    else: p['hyper_parameters']['config']['mechanisms']['np']['weights']=[.05,.05]
    with pytest.raises(ValueError): inspect_checkpoint(p,A)


@pytest.mark.parametrize('variant',['mdm',A])
def test_real_model_adam_scheduler_ema_and_private_counter_resume_matches_next_update(tmp_path,variant):
    original=make_model(dropout=.2) if variant==A else model(config('mdm'))
    params=list(original._get_parameters())
    optimizer=torch.optim.AdamW(params,lr=.0003,weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/2))
    x,valid=batch()
    for i in range(3):
        torch.manual_seed(31+i)
        optimizer.zero_grad()
        original._loss(x,valid.clone()).loss.backward()
        optimizer.step();scheduler.step();original.ema.update(params)
    payload=dict(global_step=3,state_dict=original.state_dict(),
        hyper_parameters={'config':original.config},ema=original.ema.state_dict(),
        loops={'fit_loop':{'epoch_progress':{'current':{'completed':0}},
            'epoch_loop.batch_progress':{'current':{'completed':96}}}},
        optimizer_states=[optimizer.state_dict()],lr_schedulers=[scheduler.state_dict()])
    if variant==A:
        payload['transformer_np']=dict(calls=original.branch_calls,signature=original.resume_signature())
    path=tmp_path/'resume.ckpt';torch.save(payload,path)
    loaded=torch.load(path,weights_only=False)
    cfg=continuation_config(original.config,tmp_path/'run',path)
    # The test model has a small vocabulary/backbone, while retaining real BD3 hooks.
    from unittest.mock import patch
    from owt.transformer_np_model import TransformerNPMDM
    with patch('diffusion.metrics.Metrics',return_value=torch.nn.Module()):
        restored=(TransformerNPMDM(cfg,SimpleNamespace(vocab_size=7,mask_token=None,all_special_ids=[]))
                  if variant==A else model(cfg)).train()
    restored.on_load_checkpoint(loaded);restored.load_state_dict(loaded['state_dict'])
    other=torch.optim.AdamW(restored._get_parameters(),lr=.0003,weight_decay=0.)
    other_scheduler=torch.optim.lr_scheduler.LambdaLR(other,lambda step:min(1.,(step+1)/2))
    other.load_state_dict(loaded['optimizer_states'][0]);other_scheduler.load_state_dict(loaded['lr_schedulers'][0])
    assert restored.fast_forward_batches==96
    for mdl,opt,sch in [(original,optimizer,scheduler),(restored,other,other_scheduler)]:
        torch.manual_seed(750001);opt.zero_grad()
        mdl._loss(x,valid.clone()).loss.backward()
        opt.step();sch.step();mdl.ema.update(mdl._get_parameters())
    for key,tensor in original.state_dict().items():
        torch.testing.assert_close(tensor,restored.state_dict()[key],rtol=0,atol=0)
    assert optimizer_fingerprint(optimizer.state_dict())==optimizer_fingerprint(other.state_dict())
    assert scheduler.state_dict()==other_scheduler.state_dict()
    for expected,actual in zip(original.ema.shadow_params,restored.ema.shadow_params):
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    assert original.ema.num_updates==restored.ema.num_updates==4
    assert getattr(original,'branch_calls',None)==getattr(restored,'branch_calls',None)


def test_callback_records_noisy_input_and_rejects_wrong_initial_step(tmp_path):
    mdl=SimpleNamespace(backbone=torch.nn.Identity(),fast_forward_batches=160000)
    trainer=SimpleNamespace(global_step=5000,max_steps=7500,global_rank=1)
    callback=ContinuationAudit(tmp_path,{})
    callback.restored={'step':5000}
    x=torch.arange(4).reshape(2,2)
    for step in (5000,5001,5002):
        trainer.global_step=step
        callback.on_train_batch_start(trainer,mdl,{'input_ids':x,'attention_mask':torch.ones_like(x)},0)
        mdl.backbone(x+1)
    trace=json.loads((tmp_path/'first-batches-rank1.json').read_text())
    assert len(trace)==3 and trace[0]['clean_sha256']!=trace[0]['noisy_sha256']
    assert not mdl.backbone._forward_pre_hooks
    trainer.global_step=0
    with pytest.raises(ValueError,match='start5000'): callback.on_train_start(trainer,mdl)


def test_partial_output_cannot_silently_restart(tmp_path,monkeypatch):
    import owt.continuation_schedule as controller
    monkeypatch.setattr(controller,'ROOT',tmp_path)
    run=tmp_path/controller.RUN_ROOT/'mdm';run.mkdir(parents=True)
    (run/'train.log').write_text('partial')
    with pytest.raises(RuntimeError,match='already exists'): prepare_run('mdm')


def test_startup_detects_wrong_weight_before_accepting_job(tmp_path):
    from owt.metrics import append,TRAIN_COLUMNS
    for step in (5001,5002,5003):
        append(tmp_path/'local_metrics/train.csv',TRAIN_COLUMNS,
            dict(optimizer_step=step,main_elbo=4.,objective=4.1,np_prev=1.,np_next=1.,
                learning_rate=.0003,elapsed_seconds=1.,peak_allocated_gib=1.))
    with pytest.raises(ValueError,match='coefficient'): verify_startup(tmp_path,A)
