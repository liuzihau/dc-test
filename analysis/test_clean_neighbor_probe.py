import numpy as np
import pytest
import torch
from analysis.clean_neighbor_probe import pairs,new_heads,logits,batch_indices,shared_features,evaluate,macro_loss


def example():
    clean=np.array([[99,10,11,12,13,14,99]])
    masked=np.array([[False,True,True,False,True,False,False]])
    return clean,masked


def test_clean_source_and_independent_masked_targets():
    c,m=example();row,pos,target=pairs(c,m,block_size=7,special_ids=(99,))
    ix=np.flatnonzero(pos==3)[0]
    assert target[ix].tolist()==[10,11,13,-1]
    assert not m[row,pos].any()
    for j,offset in enumerate((-2,-1,1,2)):
        valid=target[:,j]>=0
        assert m[row[valid],pos[valid]+offset].all()
        assert np.array_equal(target[valid,j],c[row[valid],pos[valid]+offset])


def test_repeated_labels_are_allowed_in_probe_population():
    c,m=example();c[0,4]=c[0,3]
    _,pos,target=pairs(c,m,block_size=7,special_ids=(99,))
    assert target[np.flatnonzero(pos==3)[0],2]==12


def test_special_intermediate_token_and_block_boundary_exclude_pair():
    c,m=example();c[0,2]=99
    _,pos,t=pairs(c,m,block_size=7,special_ids=(99,))
    assert t[np.flatnonzero(pos==3)[0],0]==-1
    c,m=example();_,pos,t=pairs(c,m,block_size=3,special_ids=(99,))
    assert t[np.flatnonzero(pos==3)[0],0]==t[np.flatnonzero(pos==3)[0],1]==-1


def test_fully_revealed_input_has_no_probe_pairs():
    c,m=example();row,pos,target=pairs(c,np.zeros_like(m),block_size=7,special_ids=(99,))
    assert len(pos)==0 and target.shape==(0,4)


def test_four_heads_are_independent_and_initialization_is_paired():
    a=new_heads(4,8,19);b=new_heads(4,8,19)
    assert len(a)==4 and a[0].weight.data_ptr()!=a[1].weight.data_ptr()
    for x,y in zip(a.parameters(),b.parameters()):assert torch.equal(x,y)


def test_mask_logit_is_excluded_and_gradient_is_finite():
    head=new_heads(4,8,1)[0];h=torch.randn(3,4)
    value=logits(head,h,mask_id=7)
    assert torch.isneginf(value[:,7]).all()
    torch.nn.functional.cross_entropy(value,torch.tensor([1,2,3])).backward()
    assert torch.isfinite(head.weight.grad).all() and not head.weight.grad[7].any()


def test_sampler_is_identical_across_models():
    eligible=[np.arange(6),np.arange(8),np.arange(9),np.arange(7)]
    a=list(batch_indices(eligible,2,1,3));b=list(batch_indices(eligible,2,1,3))
    assert all(np.array_equal(x,y) for aa,bb in zip(a,b) for x,y in zip(aa,bb))


@pytest.mark.parametrize('variant',['mdm','A'])
def test_actual_shared_layer_matches_full_forward_and_stays_frozen(variant):
    from owt.test_initialization import config,model
    from owt.test_transformer_np import make_model,batch
    torch.set_num_threads(2)
    net=(model(config('mdm')) if variant=='mdm' else make_model()).eval().requires_grad_(False)
    net.backbone.force_fp32_eval=True;x,_=batch();x=x.clone();x[:,2:4]=net.mask_index
    before={k:v.clone() for k,v in net.state_dict().items()}
    full=shared_features(net,x,full_forward=True);early=shared_features(net,x)
    assert full.dtype==torch.float32 and not full.requires_grad and torch.equal(full,early)
    assert all(torch.equal(before[k],v) for k,v in net.state_dict().items())
    assert not net.backbone.blocks[-1]._forward_pre_hooks
    assert getattr(net,'branch_calls',0)==0


def test_probe_optimizer_cannot_update_frozen_feature_source():
    backbone=torch.nn.Linear(4,4).requires_grad_(False)
    frozen=[p.clone() for p in backbone.parameters()]
    h=backbone(torch.randn(5,4)).detach();heads=new_heads(4,8,1)
    optimizer=torch.optim.AdamW(heads.parameters(),lr=.01)
    initial=[p.clone() for p in heads.parameters()]
    sum(torch.nn.functional.cross_entropy(logits(head,h,7),torch.tensor([1,2,3,4,5]))/4 for head in heads).backward()
    optimizer.step()
    assert all(torch.equal(p,q) for p,q in zip(frozen,backbone.parameters()))
    assert any(not torch.equal(p,q) for p,q in zip(initial,heads.parameters()))


def test_evaluation_uses_per_head_eligible_denominators():
    features=np.zeros((4,4),dtype=np.float32)
    target=np.array([[1,1,1,1],[2,-1,2,-1],[-1,3,-1,3],[4,-1,-1,-1]])
    heads=new_heads(4,8,1)
    for head in heads:
        torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
    result=evaluate(heads,features,target,np.array([0,0,1,1]),np.array([0,0,0,0]),
        np.array([1,1,1,1]),2,'cpu',2,1)
    assert result[:,:,:,0].sum((0,1)).tolist()==[3,2,2,2]
    assert macro_loss(result)==pytest.approx(np.log(8),rel=1e-6)
