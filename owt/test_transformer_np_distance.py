"""Check distance labels, A alignment, auxiliary gradients, logging and resume."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F
from owt.model import OWTMDM
from owt.neighbor import NeighborHeads
from owt.test_initialization import config
from owt.test_transformer_np import make_model as make_a, batch
from owt.transformer_np import transformer_neighbor_terms
from owt.transformer_np_distance import DistanceTransformerNPMDM, distance_neighbor_terms, OFFSETS
from owt.transformer_np_distance_metrics import DistanceLocalMetrics, DistancePairMetrics, LOSS_NAMES

torch.set_num_threads(2)


def make_model(dropout=.2):
    cfg = config('mdm_np_zero_init_transformer_distance2_masked_source'); cfg.model.dropout = dropout
    with patch('diffusion.metrics.Metrics', return_value=torch.nn.Module()):
        return DistanceTransformerNPMDM(cfg, SimpleNamespace(vocab_size=7, mask_token=None, all_special_ids=[])).train()


def test_four_offsets_match_explicit_masked_pair_ce_and_gradients():
    torch.manual_seed(21)
    heads = NeighborHeads(4, 8, OFFSETS).double(); ref = copy.deepcopy(heads)
    features = {o:torch.randn(2, 6, 4, dtype=torch.float64, requires_grad=True) for o in OFFSETS}
    other = {o:x.detach().clone().requires_grad_() for o,x in features.items()}
    clean = torch.tensor([[1,2,3,4,5,6],[1,0,3,4,5,6]])
    noisy = torch.tensor([[7,2,7,7,5,7],[7,0,7,4,7,7]])
    valid = torch.ones_like(clean); valid[1,5] = 0
    weight = torch.arange(1,13,dtype=torch.float64).reshape(2,6)
    actual, stats = distance_neighbor_terms(heads, features, clean, noisy, valid, 7, [0], weight,
        valid.sum(), 'masked_source', chunk_size=1)
    expected = {}
    for offset, head in zip(OFFSETS, ref.heads):
        loss = other[offset].sum()*0 + sum(p.reshape(-1)[0]*0 for p in head.parameters())
        eligible = selected = 0
        for row in range(2):
            for i in range(6):
                j = i+offset
                if not 0 <= j < 6 or j == 0 or noisy[row,j] != 7: continue
                lo,hi = sorted((i,j))
                if not all(valid[row,k] and clean[row,k] not in (0,7) for k in range(lo,hi+1)): continue
                eligible += 1
                if noisy[row,i] != 7: continue
                selected += 1
                logits = head(other[offset][row,i]).clone(); logits[7] = -torch.inf
                loss = loss + F.cross_entropy(logits[None,:], clean[row,j:j+1])*weight[row,j]
        expected[offset] = loss/valid.sum()
        torch.testing.assert_close(actual[offset],expected[offset],rtol=0,atol=1e-12)
        assert stats[offset][:,0].sum() == eligible
        assert stats[offset][:,2].sum() == stats[offset][:,1].sum() == stats[offset][:,3].sum() == selected
    sum(actual.values()).backward(); sum(expected.values()).backward()
    for o in OFFSETS: torch.testing.assert_close(features[o].grad,other[o].grad,rtol=0,atol=1e-12)
    for a,b in zip(heads.parameters(),ref.parameters()): torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=1e-12)


def test_distance_two_allows_revealed_middle_but_never_crosses_boundary():
    heads = NeighborHeads(3,7,OFFSETS)
    clean = torch.tensor([[1,2,3,4,5]]); noisy = torch.tensor([[6,2,6,4,5]])
    features = {o:torch.randn(1,5,3) for o in OFFSETS}; valid = torch.ones_like(clean)
    _,stats = distance_neighbor_terms(heads,features,clean,noisy,valid,6,[0],torch.ones(1,1),valid.sum(),'masked_source')
    assert stats[2][:,2].sum() == 1
    clean[0,1] = noisy[0,1] = 0
    _,stats = distance_neighbor_terms(heads,features,clean,noisy,valid,6,[0],torch.ones(1,1),valid.sum(),'masked_source')
    assert stats[2][:,2].sum() == 0


def test_near_terms_and_counts_are_exactly_a_rules():
    heads = NeighborHeads(4,8,OFFSETS).double()
    clean = torch.tensor([[1,2,3,4,5,6]]); noisy = torch.tensor([[7,7,3,7,7,7]])
    features = {o:torch.randn(1,6,4,dtype=torch.float64) for o in OFFSETS}; valid = torch.ones_like(clean)
    value,stats = distance_neighbor_terms(heads,features,clean,noisy,valid,7,[],torch.ones(1,1),valid.sum(),'masked_source')
    proxy = SimpleNamespace(offsets=(-1,1),heads=heads.heads[:2])
    old,old_stats = transformer_neighbor_terms(proxy,features,clean,noisy,valid,7,[],torch.ones(1,1),valid.sum(),'masked_source')
    for o in (-1,1):
        torch.testing.assert_close(value[o],old[o],rtol=0,atol=0)
        torch.testing.assert_close(stats[o],old_stats[o],rtol=0,atol=0)


def test_a_parameters_ema_main_initial_loss_and_rng_preserved():
    torch.manual_seed(71); a=make_a(); rng=torch.get_rng_state().clone()
    torch.manual_seed(71); new=make_model()
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    for name,value in a.state_dict().items(): torch.testing.assert_close(value,new.state_dict()[name],rtol=0,atol=0)
    assert len(new.backbone.neighbor_branches) == len(new.backbone.neighbor_heads.heads) == 4
    assert all(torch.equal(p,s) for p,s in zip(new._get_parameters(),new.ema.shadow_params))
    assert all(not head[-1].weight.any() for head in new.backbone.neighbor_heads.heads)
    x,valid=batch()
    torch.manual_seed(42); a._loss(x,valid.clone()).loss.backward(); after=torch.get_rng_state().clone()
    torch.manual_seed(42); loss=new._loss(x,valid.clone()).loss; loss.backward()
    torch.testing.assert_close(a._last_components['main_elbo'],new._last_components['main_elbo'],rtol=0,atol=0)
    torch.testing.assert_close(torch.get_rng_state(),after,rtol=0,atol=0)
    for o in (-1,1): torch.testing.assert_close(a._last_pair_statistics[o],new._last_pair_statistics[o],rtol=0,atol=0)
    c=new._last_components
    torch.testing.assert_close(c['objective'],c['main_elbo']+.25*(c['np_prev']+c['np_next'])+.1*(c['np_prev2']+c['np_next2']),rtol=1e-6,atol=1e-6)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())


def test_all_four_branches_learn_and_np_bypasses_main_final_block():
    new=make_model(dropout=0); new.config.objective.current_weight=0
    opt=torch.optim.AdamW(new._get_parameters(),lr=1e-3,weight_decay=0); x,valid=batch()
    with patch.object(new,'q_xt',side_effect=lambda clean,p,**kw:torch.full_like(clean,new.mask_index)):
        for step in range(3):
            opt.zero_grad(); torch.manual_seed(42); new._loss(x,valid.clone()).loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())
            assert all(not p.grad.any() for p in new.backbone.blocks[-1].parameters())
            if step == 2:
                assert all(b.block.attn_qkv.weight.grad.abs().sum()>0 for b in new.backbone.neighbor_branches)
                assert new.backbone.vocab_embed.embedding.grad.abs().sum()>0
            opt.step()


def test_empty_pairs_and_main_only_validation():
    new=make_model(); x,valid=batch()
    def isolated(clean,p,**kw):
        noisy=clean.clone(); noisy[:,4]=new.mask_index; return noisy
    with patch.object(new,'q_xt',side_effect=isolated): loss=new._loss(x,valid.clone()).loss
    assert all(s[:,2].sum()==0 for s in new._last_pair_statistics.values())
    loss.backward(); assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())
    new.eval()
    with patch.object(new.backbone.neighbor_branches[2],'forward',side_effect=AssertionError('auxiliary in validation')):
        assert torch.isfinite(new._loss(x,valid.clone()).loss)
    assert new.branch_calls==1


def test_checkpoint_counter_optimizer_roundtrip_and_changed_weight_rejected():
    new=make_model(); opt=torch.optim.AdamW(new._get_parameters(),lr=1e-3); x,valid=batch()
    for _ in range(2): opt.zero_grad(); new._loss(x,valid.clone()).loss.backward(); opt.step()
    state,optim=copy.deepcopy(new.state_dict()),copy.deepcopy(opt.state_dict()); receipt={}
    with patch.object(OWTMDM,'on_save_checkpoint'): new.on_save_checkpoint(receipt)
    rng=torch.get_rng_state().clone(); other=make_model(); other.load_state_dict(state)
    restored=torch.optim.AdamW(other._get_parameters(),lr=1e-3); restored.load_state_dict(optim)
    with patch.object(OWTMDM,'on_load_checkpoint'): other.on_load_checkpoint(receipt)
    for model,optimizer in [(new,opt),(other,restored)]:
        torch.set_rng_state(rng); optimizer.zero_grad(); model._loss(x,valid.clone()).loss.backward(); optimizer.step()
    assert new.branch_calls==other.branch_calls==3
    for name,value in new.state_dict().items(): torch.testing.assert_close(value,other.state_dict()[name],rtol=0,atol=0)
    other.np_config.weights[-1]=.05
    with pytest.raises(ValueError,match='identical transformer'): other.on_load_checkpoint(receipt)


def test_callbacks_record_all_four_losses_and_pair_directions(tmp_path):
    from owt.research import read_csv
    new=make_model(); x,valid=batch(); new._loss(x,valid)
    trainer=SimpleNamespace(global_step=0,is_global_zero=True,strategy=SimpleNamespace(reduce=lambda t,reduce_op:t),
        optimizers=[SimpleNamespace(param_groups=[{'lr':.001}])],max_steps=5000)
    losses=DistanceLocalMetrics(tmp_path); pairs=DistancePairMetrics(tmp_path)
    losses.on_train_start(trainer,new); pairs.on_train_start(trainer,new); trainer.global_step=1
    losses.on_train_batch_end(trainer,new,None,None,0); pairs.on_train_batch_end(trainer,new,None,None,0)
    row=read_csv(tmp_path/'local_metrics/train.csv')[0]
    assert all(name in row for name in LOSS_NAMES)
    row=read_csv(tmp_path/'local_metrics/source_pairs.csv')[0]
    for direction,o in zip(('prev','next','prev2','next2'),OFFSETS):
        assert sum(row[f'{direction}_maskbin{b}_selected'] for b in range(5))==float(new._last_pair_statistics[o][:,2].sum())


def test_zero_far_weights_reproduce_a_gradients():
    torch.manual_seed(71); a=make_a(dropout=0)
    torch.manual_seed(71); new=make_model(dropout=0)
    new.np_config.weights[2]=new.np_config.weights[3]=0.
    x,valid=batch()
    torch.manual_seed(42); first=a._loss(x,valid.clone()).loss; first.backward()
    torch.manual_seed(42); second=new._loss(x,valid.clone()).loss; second.backward()
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    named=dict(new.named_parameters())
    for name,parameter in a.named_parameters():
        torch.testing.assert_close(parameter.grad,named[name].grad,rtol=1e-6,atol=1e-8)


def test_schedule_rejects_changing_near_weight(tmp_path):
    import json
    from owt.transformer_np_distance_schedule import VARIANT,verify_selection,RUN_ROOT
    selection=dict(selected_variant=VARIANT,execution_ready=True,optimizer_steps=5000,
        near_weight_per_direction=.1,far_weight_per_direction=.1,offsets=[-1,1,-2,2],
        source_policy='masked_source',run_root=str(RUN_ROOT),wait_for_B_final_diagnostics=True,
        user_instruction='enqueue A plus two farther heads')
    path=tmp_path/'selection.json';path.write_text(json.dumps(selection))
    with pytest.raises(ValueError,match='user-selected'): verify_selection(path)


def test_waiting_controller_leaves_gpu_lock_free(tmp_path,monkeypatch):
    import fcntl
    import owt.transformer_np_distance_schedule as schedule
    class StopWait(Exception): pass
    monkeypatch.setattr(schedule,'ROOT',tmp_path)
    monkeypatch.setattr(schedule,'verify_selection',lambda *args:{})
    monkeypatch.setattr(schedule,'b_ready',lambda:False)
    monkeypatch.setattr(schedule.time,'sleep',lambda seconds:(_ for _ in ()).throw(StopWait()))
    monkeypatch.setattr(schedule.sys,'argv',['schedule','--selection','unused','--variant',schedule.VARIANT])
    with patch.object(schedule.subprocess,'Popen') as worker:
        with pytest.raises(StopWait): schedule.main()
        worker.assert_not_called()
    lock=tmp_path/schedule.LOCK;lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (tmp_path/schedule.RUN_ROOT/schedule.VARIANT).exists()


def test_busy_gpu_lock_never_starts_extension(tmp_path,monkeypatch):
    import fcntl
    import owt.transformer_np_distance_schedule as schedule
    class StopWait(Exception): pass
    monkeypatch.setattr(schedule,'ROOT',tmp_path)
    monkeypatch.setattr(schedule,'verify_selection',lambda *args:{})
    monkeypatch.setattr(schedule,'b_ready',lambda:True)
    monkeypatch.setattr(schedule.time,'sleep',lambda seconds:(_ for _ in ()).throw(StopWait()))
    monkeypatch.setattr(schedule.sys,'argv',['schedule','--selection','unused','--variant',schedule.VARIANT])
    lock=tmp_path/schedule.LOCK;lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with patch.object(schedule.subprocess,'Popen') as worker:
            with pytest.raises(StopWait): schedule.main()
            worker.assert_not_called()
