"""Paired diagnostic comparisons require identical data, masks, and coverage."""
import pytest
import csv
import json
import torch
from analysis import owt_mask_profile as profile
from analysis.owt_mask_profile import summarize


def observations(comparison='mdm_np_zero_init'):
    rows=[]
    for variant in ['mdm',comparison]:
        for index in [1,2]:
            rows.append(dict(variant=variant,row_id=index,noise_level=.5,
                clean_sha256=f'clean{index}',noisy_sha256=f'noisy{index}',
                paired_source_count=0,main_elbo=2.+index+(variant!='mdm')*.1))
    return rows


def test_selected_comparison_uses_identical_row_population():
    summary=summarize(observations(),'mdm_np_zero_init')
    assert summary[0]['delta_main_elbo']==pytest.approx(.1)
    assert summary[0]['rows']==2
    assert 'random_np_main_elbo' not in summary[0]


def test_unmatched_coverage_cannot_silently_reduce_the_population():
    with pytest.raises(AssertionError,match='exactly the same rows'):
        summarize(observations()[:-1],'mdm_np_zero_init')


def test_different_corruption_cannot_be_treated_as_paired():
    rows=observations()
    rows[-1]['noisy_sha256']='different'
    with pytest.raises(AssertionError):
        summarize(rows,'mdm_np_zero_init')


def test_reference_reuse_checks_checkpoint_and_validation_data(tmp_path, monkeypatch):
    data={1:dict(input_ids=[1,2,3])}
    clean=torch.tensor([[1,2,3]])
    provenance=dict(checkpoint='immutable.ckpt')
    monkeypatch.setattr(profile,'checkpoint_provenance',lambda run:provenance.copy())
    summary=tmp_path/'summary.json'
    summary.write_text(json.dumps(dict(row_ids=[1],noise_levels=[.5],provenance={'mdm':provenance})))
    row=dict(variant='mdm',row_id=1,noise_level=.5,seed=20271001,
             clean_sha256=profile.checksum(clean),noisy_sha256='noisy',paired_source_count=0,main_elbo=2.)
    with (tmp_path/'observations.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row))
        writer.writeheader();writer.writerow(row)
    reused,_=profile.reuse_baseline(summary,tmp_path,[1],[.5],data)
    assert reused[0]['main_elbo']==2.
    data[1]['input_ids']=[1,2,4]
    with pytest.raises(AssertionError,match='validation data changed'):
        profile.reuse_baseline(summary,tmp_path,[1],[.5],data)
    provenance['checkpoint']='changed.ckpt'
    with pytest.raises(AssertionError,match='checkpoint or configuration changed'):
        profile.reuse_baseline(summary,tmp_path,[1],[.5],data)
