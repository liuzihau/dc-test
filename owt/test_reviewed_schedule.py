import hashlib
import json
import sys
from pathlib import Path

import pytest
from owt import schedule


def fixture(tmp_path):
    for variant in ('mdm', 'mdm_np_zero_init'):
        run=tmp_path/variant; run.mkdir()
        (run/'complete.json').write_text(json.dumps(dict(optimizer_step=5000)))
    artifacts={}
    values=dict(diagnostic_decision=dict(status='all_registered_diagnostics_completed_and_scientifically_reviewed',
        coverage={str(i):'completed_and_reviewed' for i in range(6)}),
        frozen_readout_summary=dict(scoring_rows=512,cells=[{}]*19),
        training_decision_pdf='reviewed PDF',training_recipe='weights .05 each')
    selected=dict(status='selected_after_all_diagnostics_and_frozen_readout_review',
        variant='mdm_np_zero_init_low_weight',target_steps=5000,microbatch=8,physical_gpus=[2,3],
        run_root=str(tmp_path),rationale='reviewed diagnostic result',final_validation={})
    for key,value in values.items():
        path=tmp_path/key; path.write_text(json.dumps(value)); selected[key]=str(path)
        artifacts[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    selected['artifact_sha256']=artifacts
    path=tmp_path/'decision.json'; path.write_text(json.dumps(selected))
    return path,selected


def test_reviewed_selection_bypasses_old_threshold_but_retains_marker(tmp_path,monkeypatch):
    path,_=fixture(tmp_path)
    marker=tmp_path/'reveal_sweep_required.json'; marker.write_text('{}')
    calls=[]
    def fake_run(command,**kwargs):
        calls.append(command)
        assert command[command.index('--variant')+1]=='mdm_np_zero_init_low_weight'
        assert '--resume' not in command
        run=tmp_path/'mdm_np_zero_init_low_weight'
        (run/'complete.json').write_text(json.dumps(dict(optimizer_step=5000)))
    monkeypatch.setattr(schedule.subprocess,'run',fake_run)
    monkeypatch.setattr(schedule,'low_weight_decision',lambda *a:pytest.fail('Old automatic threshold used'))
    from owt import research
    monkeypatch.setattr(research,'record_event',lambda *a,**k:True)
    monkeypatch.setattr(sys,'argv',['schedule','followup-low-weight','--root',str(tmp_path),'--decision',str(path)])
    schedule.main()
    assert len(calls)==1 and marker.exists()
    assert json.loads((tmp_path/'low_weight_queue.json').read_text())['stage']=='complete'


def test_changed_review_evidence_rejected(tmp_path):
    path,selected=fixture(tmp_path)
    Path(selected['training_decision_pdf']).write_text('changed')
    with pytest.raises(RuntimeError,match='artifact changed'):
        schedule.reviewed_low_weight_decision(tmp_path,path)


@pytest.mark.parametrize('change',['unreviewed','incomplete_frozen','baseline_incomplete','wrong_variant'])
def test_incomplete_or_wrong_selection_cannot_launch(tmp_path,change):
    path,selected=fixture(tmp_path)
    if change=='unreviewed':
        target=Path(selected['diagnostic_decision']); value=json.loads(target.read_text())
        value['status']='pending'; target.write_text(json.dumps(value))
    elif change=='incomplete_frozen':
        target=Path(selected['frozen_readout_summary']); value=json.loads(target.read_text())
        value['scoring_rows']=511; target.write_text(json.dumps(value))
    elif change=='baseline_incomplete':
        target=tmp_path/'mdm_np_zero_init/complete.json'; target.write_text('{"optimizer_step":4999}')
    else:
        selected['variant']='mdm_np'; target=None
    if change in ('unreviewed','incomplete_frozen'):
        selected['artifact_sha256'][str(target)]=hashlib.sha256(target.read_bytes()).hexdigest()
    path.write_text(json.dumps(selected))
    with pytest.raises(RuntimeError):
        schedule.reviewed_low_weight_decision(tmp_path,path)
