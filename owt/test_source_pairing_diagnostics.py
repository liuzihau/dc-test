"""Completion, checkpoint compatibility and paired source-policy analysis."""
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from owt.source_pairing_diagnostics import VARIANTS, verify_trial_completed, POLICY
from analysis import owt_source_pairing_report as report
from owt.source_pairing_diagnostic_schedule import require_terminal_training, require_report
from owt.low_weight_diagnostics import sha256


def contracts(root,step=5000):
    original=dict(variant='mdm_np_zero_init',seed=1,
                  mechanisms=dict(np=dict(weights=[.25,.25],initialization='zero')))
    path=root/'mdm_np_zero_init';path.mkdir()
    (path/'contract.json').write_text(json.dumps(original))
    for variant in VARIANTS:
        actual=copy.deepcopy(original);actual['variant']=variant
        actual['diagnostics']='accumulated_joint_preclip_l2_v1'
        actual['source_diagnostics']='per_direction_maskbin_pair_counts_and_weight_mass_v1'
        actual['mechanisms']['np'].update(source_policy=POLICY[variant],pair_selection_seed=271828)
        path=root/variant;path.mkdir()
        (path/'contract.json').write_text(json.dumps(actual))
        (path/'complete.json').write_text(json.dumps(dict(optimizer_step=step)))


def test_completion_requires_exact_step_and_registered_source_policy(tmp_path):
    contracts(tmp_path,4999)
    for variant in VARIANTS:
        with pytest.raises(RuntimeError,match='exactly5000'):
            verify_trial_completed(tmp_path,variant)
        (tmp_path/variant/'complete.json').write_text('{"optimizer_step":5000}')
        verify_trial_completed(tmp_path,variant)
    path=tmp_path/VARIANTS[0]/'contract.json';actual=json.loads(path.read_text())
    actual['mechanisms']['np']['source_policy']='matched_pair_count'
    path.write_text(json.dumps(actual))
    with pytest.raises(RuntimeError,match='beyond registered'):
        verify_trial_completed(tmp_path,VARIANTS[0])


def test_completed_marker_cannot_bypass_running_controller(tmp_path):
    contracts(tmp_path)
    p=tmp_path/'source_pairing_queue.json'
    p.write_text(json.dumps(dict(stage='training',variant=VARIANTS[0])))
    with pytest.raises(RuntimeError,match='controller has not completed'):
        require_terminal_training(tmp_path,VARIANTS[0])
    p.write_text(json.dumps(dict(stage='complete_waiting_for_scientific_review',variant=VARIANTS[0])))
    require_terminal_training(tmp_path,VARIANTS[0])
    with pytest.raises(RuntimeError,match='controller has not completed'):
        require_terminal_training(tmp_path,VARIANTS[1])


def collection(tmp_path,preflight=False):
    protocol_path=tmp_path/'protocol.json';protocol_path.write_text('{}')
    protocol=dict(row_ids=[0,1],conditions=['a','b','c','d','e'],
                  precision='FP32, TF32 off',run_root='runs',
                  fixed_fusion=dict(scoring_row_ids=[1],lambda_weight=.25))
    ids=protocol['row_ids'][:1] if preflight else protocol['row_ids']
    result=dict(preflight=preflight,protocol_sha256=sha256(protocol_path),requested_variant=VARIANTS[0],
        evaluated_variant='mdm_np_zero_init' if preflight else VARIANTS[0],row_ids=ids,
        optimizer_step=5000,parameter_state='EMA',observations=5*len(ids),
        reference_model_forwards=0,new_sequence_evaluations=0 if preflight else 5*len(ids),
        precision=protocol['precision'],cells=[dict(cell=c) for c in protocol['conditions']],provenance={})
    path=tmp_path/'summary.json';path.write_text(json.dumps(result))
    return protocol_path,protocol,path,result


def test_reference_replay_never_counts_as_final_source_outcome(tmp_path,monkeypatch):
    p,protocol,path,result=collection(tmp_path,True)
    report.require_collection(tmp_path,p,protocol,VARIANTS[0],True)
    monkeypatch.setattr(report,'verify_trial_completed',lambda *a:pytest.fail('Model/trial accessed for rejected preflight'))
    with pytest.raises(ValueError,match='real final source'):
        report.require_collection(tmp_path,p,protocol,VARIANTS[0],False)


def test_collection_rejects_wrong_arm_rows_and_precision(tmp_path,monkeypatch):
    p,protocol,path,result=collection(tmp_path)
    monkeypatch.setattr(report,'verify_trial_completed',lambda *a:None)
    monkeypatch.setattr(report,'verified_checkpoint',lambda *a:None)
    report.require_collection(tmp_path,p,protocol,VARIANTS[0])
    for field,value in [('row_ids',[1,0]),('evaluated_variant',VARIANTS[1]),('precision','bf16'),('new_sequence_evaluations',0)]:
        changed=dict(result);changed[field]=value;path.write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='incomplete'):
            report.require_collection(tmp_path,p,protocol,VARIANTS[0])


def arrays(main=-.5,aux=-.7):
    clean=np.tile(np.array([1,2,3,4]),(2,1));selected=np.tile([False,True,True,False],(2,1))
    result=dict(clean=clean,row_ids=np.array([0,1]),scored=selected,input_masked=selected.copy())
    for head,value in [('main',main),('left',aux),('right',aux)]:
        result[head+'_prediction']=np.where(selected,clean,-1)
        result[head+'_true_logp']=np.where(selected,value,np.nan)
        result[head+'_top1_probability']=np.where(selected,np.exp(value),np.nan)
    result['left_source_state']=np.tile([255,1,0,255],(2,1))
    result['right_source_state']=np.tile([255,0,1,255],(2,1))
    return result


def test_comparison_is_paired_and_fusion_is_fixed_without_fit(monkeypatch):
    from analysis import owt_frozen_readout
    monkeypatch.setattr(owt_frozen_readout,'fit_weight',lambda *a:pytest.fail('Refitted a frozen rule'))
    saved=dict(mdm=arrays(-.3),mdm_np_zero_init=arrays(-.5),mdm_np_zero_init_low_weight=arrays(-.45))
    trial=arrays(-.4,-.9);weights=np.array([[1.,1.],[2.,0.],[0.,2.]])
    reports,stats,fixed,frozen=report.compare_cell(saved,trial,weights,np.array([0,1]),weights,.25)
    assert reports['mdm_vs_trial_main_all']['metrics']['ce_difference']['value']==pytest.approx(.1)
    assert reports['mdm_np_zero_init_vs_trial_main_all']['metrics']['ce_difference']['value']==pytest.approx(-.1)
    expected=-np.log(.75*np.exp(-.4)+.25*np.exp(-.9))
    assert fixed['trial']['metrics']['mixture_ce']['value']==pytest.approx(expected)
    assert reports['trial_main_left_masked']['targets']==2
    assert reports['trial_main_left_correct_revealed']['targets']==2
    assert reports['mdm_vs_trial_main_no_adjacent_visible']['targets']==0
    assert np.array_equal(frozen['trial']['targets'],frozen['mdm_np_zero_init']['targets'])
    other=arrays(-.35,-.8)
    reports,_,_,_=report.compare_cell(saved,trial,weights,np.array([0,1]),weights,.25,other)
    assert reports['count_vs_masked_main_all']['metrics']['ce_difference']['value']==pytest.approx(-.05)


def test_matching_masks_but_changed_native_source_flags_rejected():
    saved=dict(mdm=arrays(),mdm_np_zero_init=arrays(),mdm_np_zero_init_low_weight=arrays())
    trial=arrays();trial['left_source_state'][0,1]=0
    with pytest.raises(ValueError,match='eligibility'):
        report.compare_cell(saved,trial,np.ones((1,2)),np.array([0,1]),np.ones((1,2)),.25)


def pair_log():
    row={}
    for direction in ('prev','next'):
        for b in range(5):
            for name,value in [('eligible',10),('masked_source',4),('selected',4),
                               ('eligible_weight_mass',20),('selected_weight_mass',8),('masked_weight_mass',8)]:
                row[f'{direction}_maskbin{b}_{name}']=value
    return [dict(row,optimizer_step=i) for i in range(1,5001)]


def test_count_and_weight_mass_controls_checked_across_both_trajectories():
    first=pair_log();second=copy.deepcopy(first)
    assert report.pair_rows_match(first,second)['counts_exactly_match']
    second[4700]['prev_maskbin2_selected_weight_mass']+=.0001
    assert report.pair_rows_match(first,second)['weight_mass_matches_with_rounding']
    second[4700]['prev_maskbin2_selected_weight_mass']+=1
    with pytest.raises(ValueError,match='weighted supervision'):
        report.pair_rows_match(first,second)
    second=copy.deepcopy(first);second[3000]['next_maskbin3_selected']+=1
    with pytest.raises(ValueError,match='auxiliary count'):
        report.pair_rows_match(first,second)


def test_count_report_requires_actual_masked_source_contrast(tmp_path):
    p,protocol,collection_path,_=collection(tmp_path)
    folder=tmp_path/'report';folder.mkdir()
    required={'trial_union':{},'mdm_vs_trial_main_all':{},'mdm_np_zero_init_vs_trial_main_all':{},
              'mdm_np_zero_init_low_weight_vs_trial_main_all':{},'count_vs_masked_main_all':{}}
    result=dict(preflight=False,variant=VARIANTS[1],protocol_sha256=sha256(p),
        collection_sha256=sha256(collection_path),row_ids=protocol['row_ids'],
        scoring_row_ids=[1],fixed_lambda=.25,fitting_performed=False,model_forward_passes=0,
        cells=[dict(cell=c,comparisons=required) for c in protocol['conditions']],
        training_evidence={'matched_two_arm_supervision_audit':{'updates':5000}},paired_source_arm_available=True)
    path=folder/'summary.json';path.write_text(json.dumps(result))
    require_report(folder,tmp_path,p,protocol,VARIANTS[1])
    result['paired_source_arm_available']=False;path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='masked-source contrast'):
        require_report(folder,tmp_path,p,protocol,VARIANTS[1])


def test_source_checkpoint_parameter_state_loads_strictly_in_original_inference_model(tmp_path):
    import torch
    from owt.model import OWTMDM
    from owt.source_pairing_model import SourcePairingMDM
    from owt.test_initialization import config,model
    from owt.test_source_pairing import source_model
    torch.set_num_threads(2)
    for variant in VARIANTS:
        new=source_model(variant)
        with torch.no_grad():
            for parameter in new.parameters():
                parameter.add_(torch.linspace(-.1,.1,parameter.numel()).reshape_as(parameter))
        new.ema.update(new._get_parameters())
        # The private sampler counter belongs to metadata, never state_dict.
        new.pair_calls=17
        payload=dict(state_dict=new.state_dict(),ema=new.ema.state_dict(),
                     source_pairing=dict(calls=17,policy=POLICY[variant]))
        path=tmp_path/(variant+'.pt');torch.save(payload,path)
        loaded=torch.load(path,weights_only=False)
        base=model(config(variant));base.load_state_dict(loaded['state_dict'],strict=True)
        base.ema.load_state_dict(loaded['ema']);base.ema.copy_to(base._get_parameters())
        new.ema.copy_to(new._get_parameters());new.ema=base.ema=None
        new.eval();base.eval()
        x=torch.tensor([[1,7,7,4,5,7,1,2,3,4,5,7,1,2,3,4]])
        with torch.no_grad():
            torch.testing.assert_close(new(x,torch.zeros(1,1)),base(x,torch.zeros(1,1)),rtol=0,atol=0)
        assert 'pair_calls' not in loaded['state_dict']


def test_diagnostic_follower_does_not_import_torch():
    code='import sys; import owt.source_pairing_diagnostic_schedule; assert "torch" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],check=True,cwd=Path(__file__).resolve().parents[1])
