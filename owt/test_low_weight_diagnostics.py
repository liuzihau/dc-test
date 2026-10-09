import json
import numpy as np
import pytest

from owt.low_weight_diagnostics import reference_canvases, verify_saved_arrays, verify_completed_trial, digest
from owt.reveal_corruption import make_canvas, nearest_different_sources
from owt.low_weight_diagnostic_schedule import require_terminal_training, require_final_report
from analysis.owt_low_weight_report import compare_cell, final_training_evidence


def reference():
    clean=np.tile(np.array([2,3,4,5]*4,dtype=np.int64),(2,1)); ids=np.array([3,7])
    observations={}
    masks=[]
    for row_id,x in zip(ids,clean):
        source=nearest_different_sources(x,[50256,50257])
        canvas,masked,_,_=make_canvas(x,.6,1.,20261004,int(row_id),None,50257,[50256],nearest_sources=source)
        observations[int(row_id)]=dict(clean_sha256=digest(x),input_sha256=digest(canvas),mask_sha256=digest(masked))
        masks.append(masked)
    return clean,ids,observations,np.stack(masks)


def test_reconstruction_matches_nearest_source_protocol_without_changing_rng():
    clean,ids,observations,masks=reference()
    before=np.random.get_state()
    canvases,masked,wrong=reference_canvases(clean,ids,.6,20261004,observations)
    after=np.random.get_state()
    assert np.array_equal(before[1],after[1])
    assert np.array_equal(masked,masks) and not wrong.any()
    assert np.array_equal(canvases[~masked],clean[~masked])


def test_same_mask_count_different_input_rejected():
    clean,ids,observations,_=reference()
    observations[int(ids[0])]['input_sha256']='different'
    with pytest.raises(ValueError,match='corruption hash'):
        reference_canvases(clean,ids,.6,20261004,observations)


def test_saved_scalar_scores_and_populations_must_reconstruct():
    clean,ids,observations,masked=reference(); scored=masked.copy(); scored[:,0]=False
    arrays=dict(clean=clean.astype(np.int32),row_ids=ids,scored=scored,main_prediction=clean.copy(),
        main_true_logp=np.where(scored,-.5,np.nan))
    for k,i in enumerate(ids):
        observations[int(i)].update(masked_targets=int(scored[k].sum()),masked_correct_count=int(scored[k].sum()),
            masked_ce_sum=float(scored[k].sum())*.5)
    verify_saved_arrays(arrays,clean,ids,masked,np.zeros_like(masked),observations)
    arrays['main_true_logp'][0,np.flatnonzero(scored[0])[0]]=-.6
    with pytest.raises(ValueError,match='main statistics'):
        verify_saved_arrays(arrays,clean,ids,masked,np.zeros_like(masked),observations)


def contracts(root,step=5000):
    original=dict(variant='mdm_np_zero_init',seed=1,mechanisms=dict(np=dict(weights=[.25,.25])))
    low=dict(variant='mdm_np_zero_init_low_weight',seed=1,mechanisms=dict(np=dict(weights=[.05,.05])),diagnostics='accumulated_joint_preclip_l2_v1')
    for data in (original,low):
        path=root/data['variant']; path.mkdir()
        (path/'contract.json').write_text(json.dumps(data))
        (path/'complete.json').write_text(json.dumps(dict(optimizer_step=step)))


def test_trial_requires_terminal5000_and_matched_contract(tmp_path):
    contracts(tmp_path,4999)
    with pytest.raises(RuntimeError,match='exactly5000'):
        verify_completed_trial(tmp_path)
    run=tmp_path/'mdm_np_zero_init_low_weight'; (run/'complete.json').write_text('{"optimizer_step":5000}')
    verify_completed_trial(tmp_path)
    path=run/'contract.json'; value=json.loads(path.read_text()); value['seed']=2;path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError,match='differs beyond'):
        verify_completed_trial(tmp_path)


def test_follower_requires_successful_controller_not_only_checkpoint(tmp_path):
    contracts(tmp_path)
    state=tmp_path/'low_weight_queue.json';state.write_text('{"stage":"train"}')
    with pytest.raises(RuntimeError,match='controller did not complete'):
        require_terminal_training(tmp_path)
    state.write_text(json.dumps(dict(stage='complete',variants=['mdm_np_zero_init_low_weight'],target_steps=5000)))
    require_terminal_training(tmp_path)


def test_follower_rejects_preflight_or_refitted_result(tmp_path):
    from owt.low_weight_diagnostics import sha256
    protocol_path=tmp_path/'protocol.json';protocol_path.write_text('{}')
    protocol=dict(row_ids=[0,1],fixed_fusion=dict(scoring_row_ids=[1],lambda_weight=.25))
    report=dict(protocol_sha256=sha256(protocol_path),preflight=False,fitting_performed=False,
        row_ids=[0,1],scoring_row_ids=[1],fixed_lambda=.25,cells=[{}]*5,training_evidence={'final':4.})
    path=tmp_path/'summary.json';path.write_text(json.dumps(report))
    require_final_report(tmp_path,protocol_path,protocol)
    report['preflight']=True;path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError,match='incomplete'):
        require_final_report(tmp_path,protocol_path,protocol)
    report['preflight']=False;report['fitting_performed']=True;path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError,match='incomplete'):
        require_final_report(tmp_path,protocol_path,protocol)


def test_fixed_fusion_report_uses_registered_weight_without_fitting(monkeypatch):
    from analysis import owt_frozen_readout
    monkeypatch.setattr(owt_frozen_readout,'fit_weight',lambda *a:pytest.fail('A fixed rule was refitted'))
    clean=np.array([[0,1,2,3],[0,1,2,3]])
    selected=np.array([[False,True,True,False]]*2)
    values=dict(clean=clean,row_ids=np.array([0,1]),scored=selected)
    for head in ('main','left','right'):
        values[head+'_prediction']=np.where(selected,clean,-1)
        values[head+'_true_logp']=np.where(selected,-.5,np.nan)
        values[head+'_top1_probability']=np.where(selected,np.exp(-.5),np.nan)
        if head!='main':values[head+'_source_state']=np.where(selected,0,255)
    low={k:v.copy() for k,v in values.items()}
    low['left_true_logp'][selected]=-.7; low['right_true_logp'][selected]=-.7
    weights=np.array([[1.,1.],[2.,0.],[0.,2.]])
    reports,_,fixed,rows=compare_cell(values,values,low,weights,np.array([0,1]),weights,.25)
    expected=-np.log(.75*np.exp(-.5)+.25*np.exp(-.7))
    assert fixed['low']['metrics']['mixture_ce']['value']==pytest.approx(expected)
    assert reports['standard_vs_low_main']['metrics']['ce_difference']['value']==0
    assert np.array_equal(rows['low']['targets'],rows['standard']['targets'])


def test_final_primary_validation_and_complete_joint_trajectory(tmp_path):
    for variant,nll in [('mdm',4.),('mdm_np_zero_init',4.1),('mdm_np_zero_init_low_weight',4.05)]:
        folder=tmp_path/variant/'local_metrics'; folder.mkdir(parents=True)
        (folder/'validation.csv').write_text(f'optimizer_step,val_nll,val_ppl\n5000,{nll},60\n')
    folder=tmp_path/'mdm_np_zero_init_low_weight/local_metrics'
    (folder/'train.csv').write_text('optimizer_step,main_elbo,objective,np_prev,np_next\n'+
        ''.join(f'{i},4.05,4.475,4.2,4.3\n' for i in range(1,5001)))
    path=folder/'gradient_norms.csv'
    header='optimizer_step,joint_l2,shared_trunk_l2,main_readout_l2,neighbor_readouts_l2,estimated_clip_multiplier\n'
    path.write_text(header+''.join(f'{i},5,3,4,0,.2\n' for i in range(1,5001)))
    result=final_training_evidence(tmp_path)
    assert result['low_minus_mdm_validation_nll']==pytest.approx(.05)
    assert result['low_minus_standard_validation_nll']==pytest.approx(-.05)
    assert result['joint_gradient_summary']['all']['clipped_fraction']==1
    path.write_text(header+''.join(f'{i},5,3,4,0,.2\n' for i in range(1,5001) if i!=4999))
    with pytest.raises(ValueError,match='Complete fresh5000'):
        final_training_evidence(tmp_path)
