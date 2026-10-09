"""Portable launcher checks; no CUDA allocation or experiment launch."""
import json
import subprocess
import sys
from pathlib import Path
import pytest
from puzzle_recurrence.devices import selected_gpu_ids,gpu_locks
from puzzle_recurrence.run_pair import commands
from puzzle_recurrence.data_setup import verify

@pytest.mark.parametrize('value',['','0,0','-1,2','abc,1'])
def test_reject_invalid_selection(value):
    with pytest.raises(ValueError):selected_gpu_ids(value)

def test_parse_server_pairs():assert selected_gpu_ids('0, 1')==['0','1']

def test_disjoint_pairs_can_lock_but_overlap_cannot(tmp_path):
    with gpu_locks(tmp_path,['0','1']):
        with gpu_locks(tmp_path,['2','3']):pass
        with pytest.raises(RuntimeError):
            with gpu_locks(tmp_path,['1','2']):pass
    with gpu_locks(tmp_path,['0','1']):pass

def test_fresh_checkout_creates_lock_parents(tmp_path):
    with gpu_locks(tmp_path,['2','3']):
        assert (tmp_path/'outputs/owt/mdm-np-5k/.queue.lock').exists()

def test_launch_commands_preserve_pair_batch_and_variant(tmp_path):
    plans=commands('zebra',[['0','1'],['2','3']],32,4,tmp_path)
    assert [p['variant'] for p in plans]==['trajectory_attention','trajectory_recurrent']
    assert [p['gpus'] for p in plans]==['0,1','2,3']
    for plan in plans:
        assert '--devices' in plan['command'] and '--stage' in plan['command']
        assert plan['command'][plan['command'].index('--devices')+1]=='2'

def test_dry_run_never_creates_experiment_outputs(tmp_path):
    subprocess.run([sys.executable,'-m','puzzle_recurrence.run_pair','--tasks','sudoku','zebra',
        '--output-root',str(tmp_path/'not-created'),'--dry-run'],check=True,capture_output=True)
    assert not (tmp_path/'not-created').exists()

def test_data_verifier_checks_expected_files(tmp_path):
    (tmp_path/'data').write_bytes(b'123')
    assert verify(tmp_path,{'files':[{'name':'data','bytes':3}]})==[str(tmp_path/'data')]
    with pytest.raises(ValueError):verify(tmp_path,{'files':[{'name':'data','bytes':2}]})
    with pytest.raises(FileNotFoundError):verify(tmp_path,{'files':[{'name':'missing','bytes':3}]})
