import math
import numpy as np
import pytest
import torch

from owt.reveal_sweep import score_batch, summarize, exposed_target_mask, MASK_RATIOS, CORRECT_FRACTIONS


def test_scoring_ignores_wrong_reveals_and_the_first_target():
    class Fake(torch.nn.Module):
        def forward(self,x,sigma):
            result=torch.full((*x.shape,3),-math.log(2))
            result[:,:,-1]=-torch.inf
            # SUBS copies a revealed token, making its original clean target impossible.
            result[0,1]=torch.tensor([0.,-torch.inf,-torch.inf])
            return result
    clean=np.array([[0,1,0,1]])
    xt=np.array([[2,0,2,2]])
    mask=xt==2
    wrong=np.array([[False,True,False,False]])
    exposed=np.array([[False,False,False,True]])
    reference=np.array([[True,True,False,True]])
    row=score_batch(Fake(),clean,xt,mask,wrong,'cpu',exposed,reference_exposures={.8:reference})[0]
    assert row['masked_targets']==2
    assert row['masked_ce_sum']==pytest.approx(2*math.log(2))
    assert row['masked_correct_count']==1
    assert row['copied_targets']==1
    assert row['reference_copied_targets_c080']==1
    assert row['reference_copied_ce_sum_c080']==pytest.approx(math.log(2))


def observations():
    rows=[]
    for m in MASK_RATIOS:
        for r in CORRECT_FRACTIONS:
            for variant in ['mdm','mdm_np']:
                for index in [0,1]:
                    penalty=(1-m)*(1-r)
                    ce=2.+.1*index+penalty
                    if variant=='mdm_np':
                        ce+=.2+penalty
                    count=10-index
                    copied=int(m<1 and r<1)
                    rows.append(dict(variant=variant,mask_ratio=m,correct_fraction=r,row_id=index,
                        masked_ce_sum=ce*count,masked_targets=count,masked_correct_count=2,
                        adjacent_wrong_ce_sum=ce*copied,adjacent_wrong_targets=copied,
                        copied_target_ce_sum=ce*copied,copied_targets=copied,
                        input_sha256=f'{index}-{m}-{r if m<1 else 1.}',
                        clean_sha256=f'clean{index}',mask_sha256=f'{index}-{m}'))
    return rows


def test_gap_and_excess_unreliability_penalty_are_separated():
    result=summarize(observations(),['mdm','mdm_np'],bootstraps=20)
    row=next(v for v in result if v['variant']=='mdm_np' and v['mask_ratio']==.6 and v['correct_fraction']==.8)
    assert row['delta_from_mdm']==pytest.approx(.28)
    assert row['excess_reliability_penalty_vs_mdm']==pytest.approx(.08)
    assert row['degradation_from_correct_reveals']==pytest.approx(.16)
    for cell in result:
        if cell['mask_ratio']==1.:
            assert cell['excess_reliability_penalty_vs_mdm']==pytest.approx(0.)


def test_pairing_rejects_different_context_inputs():
    rows=observations()
    rows[-1]['input_sha256']='different context'
    with pytest.raises(AssertionError):
        summarize(rows,['mdm','mdm_np'],bootstraps=20)


def referenced_observations():
    rows=observations()
    for row in rows:
        row['copied_target_mask_sha256']=(f"copied-{row['row_id']}-{row['mask_ratio']}"
            if row['mask_ratio']<1 and row['correct_fraction']<1 else 'empty')
        if row['correct_fraction']==1.:
            count=int(row['mask_ratio']<1)
            for c in [80,60]:
                row[f'reference_copied_targets_c{c:03d}']=count
                row[f'reference_copied_ce_sum_c{c:03d}']=(row['masked_ce_sum']/row['masked_targets'])*count
                row[f'reference_copied_mask_sha256_c{c:03d}']=(f"copied-{row['row_id']}-{row['mask_ratio']}"
                    if count else 'empty')
    return rows


def test_same_target_degradation_partitions_reconstruct_the_primary_penalty():
    rows=referenced_observations()
    for row in rows:
        if row['mask_ratio']==.6 and row['correct_fraction']==.8:
            clean_ce=2.+.1*row['row_id']+(.2 if row['variant']=='mdm_np' else 0)
            row['copied_target_ce_sum']=clean_ce+(.5 if row['variant']=='mdm_np' else .1)
    summary=summarize(rows,['mdm','mdm_np'],bootstraps=20)
    cell=next(r for r in summary if r['variant']=='mdm_np' and r['mask_ratio']==.6 and r['correct_fraction']==.8)
    groups=cell['matched_exposure_degradation']
    assert groups['copied']['masked_targets']==2
    assert groups['copied']['degradation_from_correct_reveals']==pytest.approx(.5)
    assert groups['copied']['excess_reliability_penalty_vs_mdm']==pytest.approx(.4)
    assert groups['not_copied']['degradation_from_correct_reveals']==pytest.approx(.12)
    assert sum(g['fraction_of_masked_targets']*g['excess_reliability_penalty_vs_mdm']
               for g in groups.values())==pytest.approx(cell['excess_reliability_penalty_vs_mdm'])
    fully_masked=next(r for r in summary if r['variant']=='mdm_np' and r['mask_ratio']==1 and r['correct_fraction']==.6)
    assert fully_masked['matched_exposure_degradation']['copied']['corrupted_ce'] is None
    assert fully_masked['matched_exposure_degradation']['not_copied']['degradation_from_correct_reveals']==pytest.approx(0)


def test_same_target_reference_rejects_changed_population_counts():
    rows=referenced_observations()
    target=next(r for r in rows if r['variant']=='mdm_np' and r['mask_ratio']==.6 and r['correct_fraction']==1.)
    target['reference_copied_targets_c080']=2
    with pytest.raises(AssertionError):
        summarize(rows,['mdm','mdm_np'],bootstraps=20)


def test_same_target_reference_rejects_equal_counts_on_different_positions():
    rows=referenced_observations()
    target=next(r for r in rows if r['variant']=='mdm_np' and r['mask_ratio']==.6 and r['correct_fraction']==1.)
    target['reference_copied_mask_sha256_c080']='different target subset with the same count'
    with pytest.raises(AssertionError):
        summarize(rows,['mdm','mdm_np'],bootstraps=20)


def test_exposure_counts_unique_masked_sources_and_rejects_invalid_donors():
    masked=np.array([[True,False,False,False]])
    wrong=np.array([[False,True,True,True]])
    sources=np.array([[1,0,0,1]])
    assert exposed_target_mask(masked,wrong,sources).tolist()==[[True,False,False,False]]
    sources[0,1]=-1
    with pytest.raises(ValueError,match='no valid'):
        exposed_target_mask(masked,wrong,sources)


def test_block_cosine_hooks_measure_updates_without_persisting():
    class Rotate(torch.nn.Module):
        def forward(self,h):
            return torch.stack([-h[...,1],h[...,0]],dim=-1)
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=torch.nn.Module()
            self.backbone.blocks=torch.nn.ModuleList([Rotate(),torch.nn.Identity()])
        def forward(self,x,sigma):
            h=torch.zeros((*x.shape,2));h[...,0]=1
            for block in self.backbone.blocks:
                h=block(h)
            return torch.full((*x.shape,3),-math.log(3))
    model=Fake()
    clean=np.array([[0,1,0,1]])
    xt=np.array([[2,0,2,2]]);mask=xt==2
    wrong=np.array([[False,True,False,False]])
    row=score_batch(model,clean,xt,mask,wrong,'cpu',capture_cosines=True)[0]
    assert row['cosine_masked_count']==2
    assert row['cosine_masked_layer01_sum']==pytest.approx(0.)
    assert row['cosine_masked_layer02_sum']==pytest.approx(2.)
    assert row['cosine_wrong_revealed_count']==1
    assert all(not block._forward_hooks for block in model.backbone.blocks)


def test_frozen_fp32_guard_rejects_lower_precision_outputs():
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=torch.nn.Module()
            self.backbone.force_fp32_eval=True
        def forward(self,x,sigma):
            return torch.full((*x.shape,3),-math.log(3),dtype=torch.bfloat16)
    clean=np.array([[0,1,0,1]])
    xt=np.full_like(clean,2)
    with pytest.raises(RuntimeError,match='lower-precision log probabilities'):
        score_batch(Fake(),clean,xt,xt==2,np.zeros_like(xt,dtype=bool),'cpu')
