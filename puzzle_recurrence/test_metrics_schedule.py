import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from puzzle_recurrence.schedule import milestones,paired_microbatch
from puzzle_recurrence.results import record_generation,history
from puzzle_recurrence.batch_change import prepare_batch_change
from puzzle_recurrence.monitor import read_rows,refresh


def info(step,batch=32,rows=0):
    return dict(step=step,cursor=dict(epoch=step//3525,rows=rows,batches=0,total_batches=32),
        batch_policy=dict(batch=batch,global_batch=512,devices=2,accumulation=512//(2*batch)))

def test_old_generation_schedule_is_preserved():
    assert milestones(20)==[3,6,9,12,15,18,20]
    assert milestones(40)==list(range(3,40,3))+[40]

def test_matched_batch_growth_waits_for_both_arms():
    assert paired_microbatch(64,[info(3525),info(3525)])==(64,None)
    assert paired_microbatch(64,[info(7050),info(3525)])[0]==32
    assert paired_microbatch(64,[info(3525,rows=8),info(3525,rows=8)])[0]==32
    assert paired_microbatch(64,[None,None])==(64,None)
    with pytest.raises(ValueError):paired_microbatch(64,[info(3525,64),info(3525,32)])

def checkpoint(rows=0):
    p=info(3525,rows=rows)
    return dict(global_step=3525,puzzle_data_cursor=p,
        hyper_parameters=dict(config=dict(loader=dict(batch_size=32),trainer=dict(accumulate_grad_batches=8))),
        optimizer_states=[{'state':'preserve'}],ema={'num_updates':3525},lr_schedulers=[{'last_epoch':3525}])

def test_boundary_growth_preserves_optimizer_ema_and_scheduler():
    old=checkpoint();saved=copy.deepcopy(old)
    new,receipt=prepare_batch_change(old,dict(batch=64,global_batch=512,devices=2,accumulation=4),True)
    assert old==saved
    assert new['optimizer_states'] is old['optimizer_states'] and new['ema'] is old['ema']
    assert new['lr_schedulers'] is old['lr_schedulers']
    assert new['hyper_parameters']['config']['loader']['batch_size']==64
    assert receipt['global_batch_preserved']
    with pytest.raises(ValueError):prepare_batch_change(checkpoint(16),dict(batch=64,global_batch=512,devices=2,accumulation=4),True)
    with pytest.raises(ValueError):prepare_batch_change(old,dict(batch=64,global_batch=512,devices=2,accumulation=4),False)

def test_generation_history_uses_same_old_metrics(tmp_path):
    run=tmp_path/'trajectory_attention/generation/epoch-003';run.mkdir(parents=True)
    metrics=dict(puzzle_accuracy=.75,mean_row_accuracy=.9,mean_cell_accuracy=.95,n_correct_puzzles=960,n_total_puzzles=1280)
    (run/'samples_20261010.json').write_text(json.dumps({'eval_metrics':metrics}))
    record_generation(run,3,10575,tmp_path/'model.ckpt')
    rows=history(tmp_path);assert rows[0]['accuracy']==.75 and rows[0]['correct']==960 and rows[0]['n']==1280
    assert rows[0]['row_accuracy']==.9 and rows[0]['cell_accuracy']==.95
    with pytest.raises(ValueError):record_generation(run,3,10575,tmp_path/'model.ckpt',4)

def test_monitor_reads_incomplete_csv_and_writes_live_figures(tmp_path):
    root=tmp_path/'sudoku/three-state-ablation'
    for variant in ('trajectory_attention','trajectory_recurrent'):
        metrics=root/variant/'local_metrics';metrics.mkdir(parents=True)
        (metrics/'train.csv').write_text('optimizer_step,main_elbo\n3525,2.1\n7050,1.9\n7051,')
        (metrics/'validation.csv').write_text('optimizer_step,val_nll\n3525,2.0\n7050,1.8\n')
        (metrics/'trajectory_validation.csv').write_text('optimizer_step,val/trajectory_center_accuracy\n3525,0.2\n7050,0.3\n')
    snapshot=refresh(root,'sudoku')
    assert snapshot['variants']['trajectory_attention']['latest_training_step']==7050
    assert (root/'performance.png').is_file() and (root/'snapshot.json').is_file()
    assert len(read_rows(root/'trajectory_attention/local_metrics/train.csv'))==2


def test_pair_runner_evaluates_both_arms_at_each_three_epoch_milestone(tmp_path,monkeypatch):
    import sys
    import torch
    from puzzle_recurrence import run_pair
    events=[]
    def fake_parallel(plans,logs,root,status):
        for plan in plans:
            command=plan['command'];stage=command[command.index('--stage')+1]
            directory=Path(plan['run']);directory.mkdir(parents=True,exist_ok=True)
            if stage=='train':
                step=int(command[command.index('--target-steps')+1]);epoch=step//3525
                batch=int(command[command.index('--microbatch')+1])
                (directory/'checkpoints').mkdir(exist_ok=True)
                torch.save(dict(global_step=step,puzzle_data_cursor=dict(variant=plan['variant'],
                    cursor=dict(epoch=epoch,rows=0,batches=0,total_batches=epoch*100),
                    batch_policy=dict(batch=batch,global_batch=512,devices=2,accumulation=512//(2*batch)))),
                    directory/'checkpoints'/f'step-{step}.ckpt')
            else:
                epoch=int(directory.name.split('-')[-1])
                (directory/'samples_20261010.json').write_text(json.dumps({'eval_metrics':dict(
                    puzzle_accuracy=epoch/10,mean_row_accuracy=.9,mean_cell_accuracy=.95,
                    n_correct_puzzles=int(1280*epoch/10),n_total_puzzles=1280)}))
            events.append((stage,plan['variant'],epoch))
    monkeypatch.setattr(run_pair,'run_parallel',fake_parallel)
    monkeypatch.setattr('puzzle_recurrence.monitor.refresh',lambda *a,**kw:None)
    monkeypatch.setattr(sys,'argv',['run_pair','--tasks','sudoku','--epochs','6','--output-root',str(tmp_path),'--microbatch','64'])
    run_pair.main()
    assert [e[2] for e in events if e[0]=='train']==[3,3,6,6]
    assert [e[2] for e in events if e[0]=='evaluate']==[3,3,6,6]
    rows=history(tmp_path/'sudoku/three-state-ablation')
    assert len(rows)==4 and {r['epoch'] for r in rows}=={3,6}
