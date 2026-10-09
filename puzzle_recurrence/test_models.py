"""Author-backbone tests: matched attention, memory routing, one-hop credit, resume."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
import pytest
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader,Dataset
from puzzle_recurrence.entrypoint import ROOT,build_config
from puzzle_recurrence.factory import model_class,PuzzleBaselineMDM
from puzzle_recurrence.adjacent import AdjacentCacheGradients
from puzzle_recurrence.trajectory import sample_trajectory
from puzzle_recurrence.attention import MemoryAttention,gather_bank,rope
from zebra.model import ZebraMDM
import models.dit as author_dit

torch.set_num_threads(2)

def full_mask(seq_len,mask_cutoffs):
    return (torch.arange(seq_len,device=mask_cutoffs.device)[None,None,None,:]<mask_cutoffs[:,None,None,None]).expand(-1,1,seq_len,-1)

@pytest.fixture(autouse=True)
def cpu_author_attention():
    with patch.object(author_dit,'FORCE_NAIVE_ATTENTION',True),patch.object(author_dit,'_get_full_mask',side_effect=full_mask),patch('metrics.transformers.AutoTokenizer.from_pretrained',return_value=SimpleNamespace(pad_token='[PAD]',pad_token_id=0)):
        yield


def config(task='sudoku',variant='trajectory_recurrent'):
    args=SimpleNamespace(task=task,variant=variant,run=ROOT/'.cache/runtime/puzzle-recurrence/tests',
        recipe=ROOT/'puzzle_recurrence/configs'/task/(variant+'.yaml'),seed=1,resume=None,workers=0,
        microbatch=2,devices=1,target_steps=6,checkpoint_interval=2,stage='check',eval_batches=1,
        eval_batch_size=2,candidate_window=0,generation_layout='author',smoke=False)
    cfg,upstream=build_config(args)
    cfg.model.hidden_size=32;cfg.model.cond_dim=16;cfg.model.n_blocks=2;cfg.model.n_heads=4
    cfg.model.length=16;cfg.model.dropout=.0;cfg.training.ema=.9
    cfg.loader.global_batch_size=6;cfg.trainer.accumulate_grad_batches=3
    cfg.puzzle_recurrence.source_dropout.enabled=False
    return cfg,upstream.dataloader.get_tokenizer(cfg)


def make(task='sudoku',variant='trajectory_recurrent',learned=True):
    cfg,tok=config(task,variant)
    m=model_class(cfg)(cfg,tok)
    if learned:
        with torch.no_grad():m.backbone.output_layer.linear.weight.normal_(std=.1)
    return m


def batch(model):
    ids=[i for i in range(model.vocab_size) if i not in set(model.tokenizer.all_special_ids)|{model.mask_index,model.pad_index}]
    x=torch.tensor([[ids[j%len(ids)] for j in range(16)],[ids[(j+2)%len(ids)] for j in range(16)]])
    valid=torch.ones_like(x);loss_mask=torch.ones_like(x);loss_mask[:,:4]=0
    return dict(input_ids=x,attention_mask=valid,loss_mask=loss_mask)


def test_three_masks_are_nested_nonfull_and_preserve_ineligible_positions():
    x=torch.arange(16).expand(100,16);eligible=torch.ones_like(x,dtype=torch.bool);eligible[:,:4]=False
    t=sample_trajectory(x,eligible,99)
    assert (t.counts[:,0]>t.counts[:,1]).all() and (t.counts[:,1]>t.counts[:,2]).all()
    assert (t.counts[:,0]<12).all() and (t.counts[:,2]>=1).all()
    for j in range(3):
        assert torch.equal(t.states[j][:,:4],x[:,:4])
        if j:assert not (t.masks[j]&~t.masks[j-1]).any()
    assert torch.allclose(t.requested_ratios[:,0]-t.requested_ratios[:,1],t.spacing)
    assert torch.allclose(t.requested_ratios[:,1]-t.requested_ratios[:,2],t.spacing)
    with pytest.raises(ValueError):sample_trajectory(x[:1],eligible[:1]&False,99)


@pytest.mark.parametrize('task',['sudoku','zebra'])
def test_pure_and_np_models_preserve_existing_author_adapter(task):
    for variant in ('mdm','mdm_np'):
        cfg,tok=config(task,variant)
        torch.manual_seed(7);old=ZebraMDM(cfg,tok)
        torch.manual_seed(7);new=PuzzleBaselineMDM(cfg,tok)
        assert not hasattr(new.backbone,'memory_attention')
        for key,value in old.state_dict().items():torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
        b=batch(new);mask=b['loss_mask'] if not cfg.training.train_on_all_tokens else None
        torch.manual_seed(19);a=old._loss(b['input_ids'],b['attention_mask'],train_mode=True,loss_mask=mask)
        torch.manual_seed(19);bval=new._loss(b['input_ids'],b['attention_mask'],train_mode=True,loss_mask=mask)
        torch.testing.assert_close(a.loss,bval.loss,rtol=0,atol=0)


@pytest.mark.parametrize('task',['sudoku','zebra'])
def test_attention_and_recurrent_arms_have_identical_parameters_and_first_pass(task):
    torch.manual_seed(13);control=make(task,'trajectory_attention',False);state=torch.get_rng_state().clone()
    torch.manual_seed(13);recurrent=make(task,'trajectory_recurrent',False)
    torch.testing.assert_close(state,torch.get_rng_state(),rtol=0,atol=0)
    assert control.state_dict().keys()==recurrent.state_dict().keys()
    for key,value in control.state_dict().items():torch.testing.assert_close(value,recurrent.state_dict()[key],rtol=0,atol=0)
    for m in (control,recurrent):
        assert len(list(m._get_parameters()))==len(m.ema.shadow_params)
        assert not any('final_hidden' in name for name,_ in m.named_parameters())
    b=batch(control);x=b['input_ids'].clone();x[:,5:]=control.mask_index
    with torch.no_grad():
        a=control.forward(x,torch.ones(2,1));other=recurrent.forward(x,torch.ones(2,1))
    torch.testing.assert_close(a,other,rtol=0,atol=0)


@pytest.mark.parametrize('task',['sudoku','zebra'])
@pytest.mark.parametrize('variant',['trajectory_attention','trajectory_recurrent'])
def test_three_state_loss_weighting_backward_and_no_extra_reference(task,variant):
    model=make(task,variant);b=batch(model)
    mask=b['loss_mask'] if not model.config.training.train_on_all_tokens else None
    result=model._loss(b['input_ids'],b['attention_mask'],train_mode=True,loss_mask=mask)
    trace=model._last_trajectory
    expected=(.25*trace['losses'][0]+trace['losses'][1]+.25*trace['losses'][2])/1.5
    torch.testing.assert_close(result.loss.detach(),expected,rtol=0,atol=0)
    assert trace['main_forwards']==3 and trace['identity_forwards']==0
    assert trace['gradient_edges']==(2 if variant=='trajectory_recurrent' else 0)
    result.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    if variant=='trajectory_recurrent':assert model.backbone.memory_final_writer.kv.weight.grad.abs().sum()>0
    else:assert model.backbone.memory_final_writer.kv.weight.grad.abs().sum()==0


def test_last_loss_reaches_only_immediate_previous_writer_in_actual_model():
    model=make();b=batch(model);t=sample_trajectory(b['input_ids'],b['loss_mask'].bool(),model.mask_index)
    bridge=AdjacentCacheGradients();previous=None;written=[];positions=torch.arange(16).expand(2,-1)
    for state in t.states:
        with model.memory.run(previous,positions,b['attention_mask'].bool()):
            scores=model.forward(state,torch.ones(2,1));raw=model.memory.banks()
        for tensor in raw:tensor.retain_grad()
        written.append(raw);previous=bridge.consume(raw)
    eligible=t.masks[-1]
    last=-scores.gather(-1,b['input_ids'][:,:,None]).squeeze(-1)[eligible].mean()
    bridge.attach(last).backward()
    assert all(w.grad is None or not w.grad.any() for w in written[0])
    assert any(w.grad is not None and w.grad.abs().sum()>0 for w in written[1])


def test_rope_uses_token_positions_and_age_without_touching_values():
    x=torch.randn(2,4,16,8);pos=torch.arange(16).expand(2,-1)
    current=rope(x,pos,1,6);previous=rope(x,pos,0,6)
    torch.testing.assert_close(current[...,:6],previous[...,:6],rtol=0,atol=0)
    assert not torch.equal(current[...,6:],previous[...,6:])
    torch.testing.assert_close(x.square().sum(-1),current.square().sum(-1),rtol=1e-5,atol=1e-5)


def test_invalid_padding_and_source_dropout_are_applied_before_softmax():
    a=MemoryAttention(32,4,0.)
    hidden=torch.randn(2,16,32);pos=torch.arange(16).expand(2,-1);valid=torch.ones(2,16,dtype=torch.bool);valid[:,-3:]=False
    past=torch.randn(2,16,2,4,8);bad=past.clone();bad[:,-3:]=100000
    modes=torch.zeros(2,16,dtype=torch.long)
    first,_=a(hidden,past,pos,valid,valid,modes);second,_=a(hidden,bad,pos,valid,valid,modes)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    # Cache-only control queries have no available source and yield zero extra residual.
    empty,_=a(hidden,None,pos,valid,valid,torch.ones_like(modes))
    torch.testing.assert_close(empty,hidden,rtol=0,atol=0)


@pytest.mark.parametrize('task',['sudoku','zebra'])
def test_author_generation_keeps_clues_and_clears_recurrent_memory(task):
    model=make(task).eval();b=batch(model)
    output=model.generate_completions(b,num_steps=8)
    assert output.shape==b['input_ids'].shape
    assert torch.equal(output[:,:4],b['input_ids'][:,:4])
    assert model.memory.generation_calls>1
    assert model.memory.generation_previous is None and model.memory.generation_valid is None


def test_canonical_bank_reindexing_matches_sampler_permutations():
    bank=torch.randn(2,16,2,4,8);p=torch.stack((torch.randperm(16),torch.randperm(16)))
    torch.testing.assert_close(gather_bank(gather_bank(bank,p),p.argsort(1)),bank,rtol=0,atol=0)


class Rows(Dataset):
    def __init__(self,mdl):self.b=batch(mdl)
    def __len__(self):return 10
    def __getitem__(self,i):return {k:v[i%2].clone() for k,v in self.b.items()}

class Trace(Callback):
    def __init__(self):self.losses=[]
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):self.losses.append(float(outputs['loss']))

def fit(model,steps,trace,path=None):
    trainer=L.Trainer(accelerator='cpu',devices=1,max_steps=steps,max_epochs=-1,
        accumulate_grad_batches=3,logger=False,enable_checkpointing=False,enable_model_summary=False,
        enable_progress_bar=False,num_sanity_val_steps=0,limit_val_batches=0,callbacks=[trace])
    trainer.fit(model,DataLoader(Rows(model),batch_size=2),ckpt_path=path);return trainer

@pytest.mark.parametrize('stop_step',[1,2,3])
def test_exact_full_state_resume_through_partial_epochs(tmp_path,stop_step):
    L.seed_everything(21);full=make();all_trace=Trace();fit(full,6,all_trace)
    L.seed_everything(21);part=make();prefix=Trace();trainer=fit(part,stop_step,prefix)
    path=str(tmp_path/'resume.ckpt');trainer.save_checkpoint(path)
    L.seed_everything(999);restored=make();tail=Trace();fit(restored,6,tail,path)
    assert prefix.losses+tail.losses==all_trace.losses
    for key,value in full.state_dict().items():torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
    for a,b in zip(full.ema.shadow_params,restored.ema.shadow_params):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_control_checkpoint_cannot_resume_as_recurrent(tmp_path):
    control=make(variant='trajectory_attention');trainer=fit(control,1,Trace())
    path=str(tmp_path/'control.ckpt');trainer.save_checkpoint(path)
    payload=torch.load(path,weights_only=False)
    with pytest.raises(ValueError,match='variant|memory policy'):make().on_load_checkpoint(payload)


@pytest.mark.parametrize('task',['sudoku','zebra'])
def test_validation_metrics_and_rng_isolation(task):
    model=make(task).eval();b=batch(model)
    model._trainer=SimpleNamespace(sanity_checking=True,global_rank=0,global_step=0)
    before=torch.get_rng_state().clone()
    with patch.object(model,'log'):
        model.on_validation_epoch_start()
        model.validation_step(b,0)
        expected=model._last_trajectory['correct']/model._last_trajectory['targets']
        calls=[]
        with patch.object(model,'log',side_effect=lambda name,value,**kw:calls.append((name,value))):
            model.on_validation_epoch_end()
    metrics=dict(calls)
    for j,name in enumerate(('high','center','low')):
        torch.testing.assert_close(metrics['val/trajectory_'+name+'_accuracy'].float(),expected[j].float())
    assert torch.isfinite(metrics['val/nll'])
    assert model._last_trajectory['identity_forwards']==0


def test_source_dropout_choices_match_between_attention_and_memory():
    control=make(variant='trajectory_attention');recurrent=make()
    for model in (control,recurrent):model.config.puzzle_recurrence.source_dropout.enabled=True
    masked=torch.ones(50,16,dtype=torch.bool)
    with patch.object(type(control),'global_step',new=property(lambda self:1000)):
        torch.manual_seed(71);a=control._source_modes(masked,1)
        torch.manual_seed(71);b=recurrent._source_modes(masked,1)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert set(a.unique().tolist())=={0,1,2}
        assert not control._source_modes(masked,0).any()


def test_recurrent_sampling_bank_uses_original_position_ids():
    model=make().eval();b=batch(model)
    with torch.no_grad():
        model.memory.generation_valid=b['attention_mask'].bool()
        canonical=torch.arange(16).expand(2,-1)
        perm=torch.stack((torch.randperm(16),torch.randperm(16)))
        state=b['input_ids'].clone();state[:,5:]=model.mask_index
        first=model.backbone.forward_sample(state,canonical,attn_mode='full')
        bank=model.memory.generation_previous
        model.memory.generation_previous=bank
        plain=model.backbone.forward_sample(state,canonical,attn_mode='full')
        model.memory.generation_previous=bank
        shuffled=model.backbone.forward_sample(state.gather(1,perm),perm,attn_mode='full')
        torch.testing.assert_close(shuffled.gather(1,perm.argsort(1)[:,:,None].expand_as(shuffled)),plain,atol=1e-5,rtol=1e-5)
        model.memory.generation_previous=None;model.memory.generation_valid=None


def test_clean_targets_cannot_use_reserved_mask_class():
    model=make();b=batch(model);b['input_ids'][:,7]=model.mask_index
    with pytest.raises(ValueError,match='reserved MASK'):
        model._loss(b['input_ids'],b['attention_mask'],train_mode=True,loss_mask=b['loss_mask'])
