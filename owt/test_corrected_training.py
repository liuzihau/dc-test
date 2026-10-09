"""Check real BD3 migration and Lightning resume across partial epoch ends."""
import copy
from types import SimpleNamespace
from unittest.mock import patch

import lightning as L
from lightning.pytorch.callbacks import Callback
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

from owt.test_initialization import config, model
from owt.test_transformer_np import make_model, batch
from owt.corrected_training import SCHEMA, CorrectedMDM, CorrectedTransformerNP
from owt.migrate_corrected_checkpoint import migrate_payload, audit_preservation
from owt.corrected_run import use_optimizer_validation_interval

torch.set_num_threads(2)

class Metrics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.train_nlls=SimpleNamespace(update=lambda *args:None,
            nll=SimpleNamespace(mean_value=0,weight=0))
    def reset(self):pass

def corrected(variant='mdm'):
    cfg=config(variant)
    OmegaConf.set_struct(cfg,False)
    cfg.loader.batch_size=2;cfg.loader.global_batch_size=6
    cfg.trainer.devices=1;cfg.trainer.accumulate_grad_batches=3
    cfg.corrected_training=dict(schema=SCHEMA,pad_id=7,mask_id=8,vocab_size=9,legacy_resume_seed=20261008)
    tokenizer=SimpleNamespace(vocab_size=9,mask_token='[MASK]',mask_token_id=8,pad_token_id=7,all_special_ids=[0,7,8])
    with patch('diffusion.metrics.Metrics',return_value=Metrics()):
        return (CorrectedMDM if variant=='mdm' else CorrectedTransformerNP)(cfg,tokenizer)

@pytest.mark.parametrize('variant',['mdm','mdm_np_zero_init_transformer_masked_source'])
def test_full_state_migration_and_valid_pad_loss(variant):
    old=model(config('mdm')) if variant=='mdm' else make_model(dropout=0.)
    opt=torch.optim.AdamW(old._get_parameters(),lr=3e-4)
    x,valid=batch()
    for _ in range(3):
        opt.zero_grad();old._loss(x,valid.clone()).loss.backward();opt.step();old.ema.update(old._get_parameters())
    params=list(old._get_parameters());by_id={id(p):name for name,p in old.named_parameters()}
    names=[by_id[id(p)] for p in params];shapes={name:tuple(p.shape) for name,p in old.named_parameters()}
    cfg=old.config
    payload=dict(state_dict=copy.deepcopy(old.state_dict()),ema=copy.deepcopy(old.ema.state_dict()),
        optimizer_states=[copy.deepcopy(opt.state_dict())],global_step=3,
        hyper_parameters=dict(config=cfg),lr_schedulers=[dict(last_epoch=3)],
        loops={'fit_loop':{'epoch_loop.batch_progress':{'current':{'completed':9},'total':{'completed':9}},
                         'epoch_progress':{'current':{'completed':0}}}})
    if variant!='mdm':payload['transformer_np']=dict(calls=old.branch_calls,signature=old.resume_signature())
    new_payload,_=migrate_payload(payload,cfg,names,shapes,eos_id=6)
    audit_preservation(payload,new_payload,names)
    assert new_payload['lr_schedulers']==payload['lr_schedulers']
    with patch('diffusion.metrics.Metrics',return_value=Metrics()):
        tokenizer=SimpleNamespace(vocab_size=9,mask_token='[MASK]',mask_token_id=8,pad_token_id=7,all_special_ids=[7,8])
        new=(CorrectedMDM if variant=='mdm' else CorrectedTransformerNP)(new_payload['hyper_parameters']['config'],tokenizer)
    new.load_state_dict(new_payload['state_dict'],strict=True)
    new.on_load_checkpoint(new_payload)
    newopt=torch.optim.AdamW(new._get_parameters(),lr=3e-4)
    newopt.load_state_dict(new_payload['optimizer_states'][0])
    # Same normal-token main distribution after moving MASK: added PAD has negligible mass.
    old.eval();new.eval();old.backbone.force_fp32_eval=True;new.backbone.force_fp32_eval=True
    noisy=x.clone();noisy[:,1:4]=7;fresh=noisy.clone();fresh[fresh==7]=8
    with torch.no_grad():
        a=old.forward(noisy,torch.ones(2,1));b=new.forward(fresh,torch.ones(2,1))
    torch.testing.assert_close(a[...,:7],b[...,:7],rtol=2e-6,atol=2e-6)
    new.train();x[:,5]=7
    with patch.object(new,'q_xt',side_effect=lambda clean,p,**kw:torch.full_like(clean,8)):
        loss=new._loss(x,valid.clone()).loss
    assert torch.isfinite(loss)
    loss.backward();newopt.step()
    assert int(next(iter(newopt.state.values()))['step'])==4
    x[:,5]=8
    with pytest.raises(ValueError,match='reserved MASK'):new._loss(x,valid)

class Rows(Dataset):
    def __len__(self):return 10
    def __getitem__(self,index):
        x=(torch.arange(16)+index)%6+1
        return dict(input_ids=x,attention_mask=torch.ones(16,dtype=torch.long),row=index)

class Trace(Callback):
    def __init__(self,path=None,save_step=None):self.rows=[];self.losses=[];self.path=path;self.save_step=save_step
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        self.rows.extend(batch['row'].tolist());self.losses.append(float(outputs['loss']))
        if self.save_step==trainer.global_step and not trainer.fit_loop.epoch_loop._should_accumulate():
            trainer.save_checkpoint(self.path);self.save_step=None

def fit(mdl,trace,steps,checkpoint=None,workers=0):
    trainer=L.Trainer(accelerator='cpu',devices=1,max_steps=steps,max_epochs=-1,
        accumulate_grad_batches=3,logger=False,enable_checkpointing=False,
        enable_progress_bar=False,enable_model_summary=False,limit_val_batches=0,
        num_sanity_val_steps=0,callbacks=[trace])
    trainer.fit(mdl,DataLoader(Rows(),batch_size=2,num_workers=workers),ckpt_path=checkpoint)
    return trainer

@pytest.mark.parametrize('save_step',[1,2,3,4])
def test_lightning_exact_resume_at_middle_or_epoch_boundary(tmp_path,save_step):
    checkpoint=str(tmp_path/'resume.ckpt')
    L.seed_everything(21,workers=True)
    original=corrected();trace=Trace(checkpoint,save_step);fit(original,trace,6,workers=2)
    payload=torch.load(checkpoint,weights_only=False)
    assert payload['corrected_training']['cursor']['rows'] in (0,6)
    consumed=payload['corrected_training']['cursor']['total_batches']
    L.seed_everything(999,workers=True)
    restored=corrected();tail=Trace();fit(restored,tail,6,checkpoint,workers=2)
    assert tail.rows==trace.rows[consumed*2:]
    assert tail.losses==trace.losses[consumed:]
    for name,value in original.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[name],rtol=0,atol=0)
    for a,b in zip(original.ema.shadow_params,restored.ema.shadow_params):
        torch.testing.assert_close(a,b,rtol=0,atol=0)

def test_validation_stays_on_optimizer_steps_after_partial_epoch():
    mdl=corrected();seen=[]
    # Exercise the real training loop; isolate the validation scheduling from metrics.
    mdl.on_validation_epoch_start=lambda:None
    mdl.on_validation_epoch_end=lambda:None
    mdl.validation_step=lambda batch,batch_idx:seen.append(mdl.global_step)
    trainer=L.Trainer(accelerator='cpu',devices=1,max_steps=6,max_epochs=-1,
        accumulate_grad_batches=3,logger=False,enable_checkpointing=False,
        enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,
        val_check_interval=100,check_val_every_n_epoch=None,limit_val_batches=1)
    use_optimizer_validation_interval(trainer,2)
    trainer.fit(mdl,DataLoader(Rows(),batch_size=2),DataLoader(Rows(),batch_size=2))
    assert seen==[2,4,6]

@pytest.mark.parametrize('stop_step',[1,2,3,4])
def test_checkpoint_saved_after_fit_preserves_data_epoch(tmp_path,stop_step):
    L.seed_everything(21,workers=True)
    full=corrected();all_rows=Trace();fit(full,all_rows,6)
    L.seed_everything(21,workers=True)
    partial=corrected();prefix=Trace();trainer=fit(partial,prefix,stop_step)
    path=str(tmp_path/'after-fit.ckpt');trainer.save_checkpoint(path)
    payload=torch.load(path,weights_only=False)
    cursor=payload['corrected_training']['cursor']
    assert payload['loops']['fit_loop']['epoch_progress']['current']['completed']==cursor['epoch']
    L.seed_everything(999,workers=True)
    resumed=corrected();tail=Trace();fit(resumed,tail,6,path)
    assert prefix.rows+tail.rows==all_rows.rows
    assert prefix.losses+tail.losses==all_rows.losses
    for name,value in full.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[name],rtol=0,atol=0)
