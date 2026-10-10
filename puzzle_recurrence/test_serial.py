"""Four-GPU serial planning and epoch-boundary migration without GPU allocation."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

from puzzle_recurrence import run_serial
from puzzle_recurrence.batch_change import prepare_batch_change
from puzzle_recurrence.cursor import added_rank_rng
from puzzle_recurrence.schedule import TASKS,latest_checkpoint
from puzzle_recurrence.test_resume_evaluation import saved_runs,save_checkpoint
from puzzle_recurrence.test_metrics_schedule import checkpoint


def test_resize_preserves_state_and_requires_epoch_boundary():
    original=checkpoint();before=copy.deepcopy(original)
    current=run_serial.policy(64,4)
    with pytest.raises(ValueError,match='allow-device-change'):prepare_batch_change(original,current,True)
    with pytest.raises(ValueError,match='completed data epoch'):prepare_batch_change(checkpoint(16),current,True,True)
    migrated,receipt=prepare_batch_change(original,current,True,True)
    assert original==before
    for key in ['ema','optimizer_states','lr_schedulers']:assert migrated[key] is original[key]
    assert migrated['hyper_parameters']['config']['trainer']==dict(devices=4,accumulate_grad_batches=2)
    assert receipt['device_count_changed'] and receipt['global_batch_preserved']


def test_added_ranks_have_independent_reproducible_rng():
    before=torch.get_rng_state().clone();a=added_rank_rng(1,28200,2);b=added_rank_rng(1,28200,3)
    assert torch.equal(before,torch.get_rng_state())
    assert torch.equal(a['torch'],added_rank_rng(1,28200,2)['torch'])
    assert not torch.equal(a['torch'],b['torch']) and not torch.equal(a['loader'],b['loader'])


@pytest.mark.parametrize('task',['sudoku','zebra'])
def test_serial_queue_uses_all_four_devices_and_alternates_without_barrier(tmp_path,monkeypatch,task):
    root,_=saved_runs(tmp_path,task);events=[]
    def fake_run(plans,*args):
        assert len(plans)==1
        plan=plans[0];cmd=plan['command'];stage=cmd[cmd.index('--stage')+1];variant=plan['variant']
        assert plan['gpus']=='0,1,2,3'
        if stage=='train':
            assert cmd[cmd.index('--devices')+1]=='4' and '--allow-device-change' in cmd
            epoch=int(cmd[cmd.index('--target-steps')+1])//TASKS[task]['steps_per_epoch']
            path=save_checkpoint(Path(plan['run'])/'checkpoints'/f'epoch{epoch-1}.ckpt',task,variant,epoch,64)
            payload=torch.load(path,weights_only=False)
            payload['puzzle_data_cursor']['batch_policy']=run_serial.policy(64,4);torch.save(payload,path)
        else:
            epoch=int(Path(plan['run']).name.split('-')[-1])
            Path(plan['run']).mkdir(parents=True,exist_ok=True)
            (Path(plan['run'])/'samples_test.json').write_text(json.dumps({'eval_metrics':dict(
                puzzle_accuracy=.5,n_correct_puzzles=640,n_total_puzzles=1280)}))
        events.append((stage,variant,epoch))
    monkeypatch.setattr(run_serial,'run_parallel',fake_run)
    monkeypatch.setattr('puzzle_recurrence.run_pair.run_parallel',fake_run)
    monkeypatch.setattr('puzzle_recurrence.monitor.refresh',lambda *args:None)
    monkeypatch.setattr(sys,'argv',['run_serial','--tasks',task,'--resume','--output-root',str(tmp_path),
        '--epochs','15','--allow-device-change'])
    run_serial.main()
    attention,recurrent=run_serial.VARIANTS
    assert [e for e in events if e[0]=='train']==[('train',recurrent,9),('train',attention,12),('train',recurrent,12),
        ('train',attention,15),('train',recurrent,15)]
    assert events.index(('evaluate',attention,12))<events.index(('train',recurrent,12))
    assert latest_checkpoint(root/recurrent)['batch_policy']==run_serial.policy(64,4)
    events.clear();run_serial.main();assert not events


def test_boundary_checkpoint_wins_over_first_microbatch_at_same_update(tmp_path):
    run=tmp_path/'run';variant='trajectory_recurrent'
    boundary=save_checkpoint(run/'checkpoints/7-28200.ckpt','sudoku',variant,8)
    partial=save_checkpoint(run/'checkpoints/8-28200.ckpt','sudoku',variant,8)
    payload=torch.load(partial,weights_only=False)
    payload['puzzle_data_cursor']['cursor'].update(rows=32,batches=1);torch.save(payload,partial)
    assert latest_checkpoint(run)['path']==str(boundary.resolve())
