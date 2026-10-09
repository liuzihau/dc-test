"""Baseline continuation settings and restoration failures."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from owt.mdm_15000 import audit_checkpoint,config_for_resume
from owt.mdm_15000_metrics import ResumeAudit
from owt.test_continuation import checkpoint_header


def payload():
    p=checkpoint_header('mdm');p['global_step']=7500;p['ema']['num_updates']=7500
    p['hyper_parameters']['config']['trainer']['max_steps']=7500
    p['optimizer_states'][0]['state'][0]['step']=torch.tensor(7500.)
    p['lr_schedulers'][0]['last_epoch']=7500
    p['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']=240000
    p['state_dict']={'p':torch.ones(2)};p['ema']['shadow_params']=[torch.ones(2)]
    return p


def test_config_preserves_original_recipe_and_warmup(tmp_path):
    original=OmegaConf.create(payload()['hyper_parameters']['config'])
    actual=config_for_resume(original,tmp_path/'new',tmp_path/'step7500.ckpt')
    assert actual.trainer.max_steps==15000 and actual.checkpointing.resume_from_ckpt
    assert actual.lr_scheduler.num_warmup_steps==2500 and not actual.mechanisms.np.enabled
    actual.trainer.max_steps=original.trainer.max_steps;actual.checkpointing=copy.deepcopy(original.checkpointing)
    assert OmegaConf.to_container(actual,resolve=True)==OmegaConf.to_container(original,resolve=True)


def test_real_resume_header_contract():
    expected=audit_checkpoint(payload())
    assert expected['step']==7500 and expected['fast_forward_batches']==240000


@pytest.mark.parametrize('bad',['step','ema','np','adam','moments','scheduler','cursor'])
def test_reject_misaligned_resume(bad):
    p=payload()
    if bad=='step':p['global_step']=5000
    elif bad=='ema':p['ema']['num_updates']=7400
    elif bad=='np':p['hyper_parameters']['config']['mechanisms']['np']['enabled']=True
    elif bad=='adam':p['optimizer_states'][0]['state'][0]['step']=torch.tensor(7400.)
    elif bad=='moments':del p['optimizer_states'][0]['state'][0]['exp_avg']
    elif bad=='scheduler':p['lr_schedulers'][0]['last_epoch']=0
    else:p['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']=239999
    with pytest.raises(ValueError):audit_checkpoint(p)


def test_restoration_audit_and_correct_first_three_steps(tmp_path):
    p=payload();expected=audit_checkpoint(p)
    module=SimpleNamespace(state_dict=lambda:p['state_dict'],ema=SimpleNamespace(shadow_params=p['ema']['shadow_params'],num_updates=7500),
        fast_forward_batches=240000,fast_forward_epochs=0,backbone=torch.nn.Identity())
    optimizer=SimpleNamespace(state_dict=lambda:p['optimizer_states'][0],param_groups=p['optimizer_states'][0]['param_groups'])
    trainer=SimpleNamespace(global_step=7500,max_steps=15000,global_rank=0,optimizers=[optimizer],
        lr_scheduler_configs=[SimpleNamespace(scheduler=SimpleNamespace(last_epoch=7500))])
    callback=ResumeAudit(tmp_path,expected);callback.on_train_start(trainer,module)
    x=torch.ones((2,4),dtype=torch.long)
    for step in (7500,7501,7502):
        trainer.global_step=step
        callback.on_train_batch_start(trainer,module,{'input_ids':x},0);module.backbone(x+1)
    traces=json.loads((tmp_path/'first-batches-rank0.json').read_text())
    assert [r['optimizer_step'] for r in traces]==[7501,7502,7503]
    assert not module.backbone._forward_pre_hooks
    assert json.loads((tmp_path/'resume-rank0.json').read_text())['seed']==1500001


def test_wrong_optimizer_state_is_rejected_before_updates(tmp_path):
    p=payload();expected=audit_checkpoint(p)
    p['optimizer_states'][0]['state'][0]['exp_avg'].zero_()
    module=SimpleNamespace(state_dict=lambda:p['state_dict'],ema=SimpleNamespace(shadow_params=p['ema']['shadow_params'],num_updates=7500),
        fast_forward_batches=240000,fast_forward_epochs=0)
    optimizer=SimpleNamespace(state_dict=lambda:p['optimizer_states'][0],param_groups=p['optimizer_states'][0]['param_groups'])
    trainer=SimpleNamespace(global_step=7500,max_steps=15000,global_rank=0,optimizers=[optimizer],
        lr_scheduler_configs=[SimpleNamespace(scheduler=SimpleNamespace(last_epoch=7500))])
    with pytest.raises(ValueError,match='restored state'):ResumeAudit(tmp_path,expected).on_train_start(trainer,module)
