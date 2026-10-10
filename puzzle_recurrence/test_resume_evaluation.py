"""Check historical evaluation ordering and absolute milestones after restart."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from puzzle_recurrence import run_pair
from puzzle_recurrence.resume_with_evaluation import resume_command
from puzzle_recurrence.schedule import TASKS
from puzzle_recurrence.results import history


def save_checkpoint(path,task,variant,epoch,batch=32):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(global_step=epoch*TASKS[task]['steps_per_epoch'],
        puzzle_data_cursor=dict(task='sudoku-puzzle' if task=='sudoku' else 'zebra',variant=variant,
            cursor=dict(epoch=epoch,rows=0,batches=0,total_batches=epoch*100),
            batch_policy=dict(batch=batch,global_batch=512,devices=2,accumulation=512//(2*batch)))),path)
    return path


def saved_runs(tmp_path,task):
    root=tmp_path/task/'three-state-ablation';attention,recurrent=run_pair.VARIANTS
    # The attention epoch6 backup lives outside the rolling checkpoint folder.
    backup=save_checkpoint(tmp_path/'backup/epoch6.ckpt',task,attention,7)
    for epoch in (8,9,10):save_checkpoint(root/attention/'checkpoints'/f'epoch{epoch-1}.ckpt',task,attention,epoch)
    for epoch in (6,7,8):save_checkpoint(root/recurrent/'checkpoints'/f'epoch{epoch-1}.ckpt',task,recurrent,epoch)
    return root,backup


def arguments(tmp_path,task,backup):
    return SimpleNamespace(task=task,output_root=tmp_path,attention_epoch6=backup,recurrent_epoch6=None,
        attention_epoch8=None,microbatch=64,workers=0,attention_gpus='0,1',recurrent_gpus='2,3',epochs=15,dry_run=False)


@pytest.mark.parametrize('task',['sudoku','zebra'])
def test_initial_evaluations_then_latest_resume_and_absolute_milestones(tmp_path,monkeypatch,task):
    root,backup=saved_runs(tmp_path,task);events=[]
    command=resume_command(arguments(tmp_path,task,backup))
    def fake_parallel(plans,logs,repo,status):
        for plan in plans:
            cmd=plan['command'];stage=cmd[cmd.index('--stage')+1];variant=plan['variant']
            source=Path(cmd[cmd.index('--resume')+1])
            source_info=run_pair.checkpoint_info(source)
            if stage=='train':
                target=int(cmd[cmd.index('--target-steps')+1]);epoch=target//TASKS[task]['steps_per_epoch']
                batch=int(cmd[cmd.index('--microbatch')+1])
                assert target>source_info['step']
                save_checkpoint(Path(plan['run'])/'checkpoints'/f'epoch{epoch-1}.ckpt',task,variant,epoch,batch)
            else:
                epoch=source_info['cursor']['epoch'];batch=None
                directory=Path(plan['run']);directory.mkdir(parents=True,exist_ok=True)
                (directory/'samples_test.json').write_text(json.dumps({'eval_metrics':dict(puzzle_accuracy=.5,
                    mean_row_accuracy=.7,mean_cell_accuracy=.8,n_correct_puzzles=640,n_total_puzzles=1280)}))
            events.append((stage,variant,epoch,source_info['cursor']['epoch'],batch))
    monkeypatch.setattr(run_pair,'run_parallel',fake_parallel)
    monkeypatch.setattr('puzzle_recurrence.monitor.refresh',lambda *a,**kw:None)
    monkeypatch.setattr(sys,'argv',command[3:])
    run_pair.main()
    attention,recurrent=run_pair.VARIANTS
    assert events[:3]==[('evaluate',attention,7,7,None),('evaluate',recurrent,7,7,None),('evaluate',attention,9,9,None)]
    trains=[e for e in events if e[0]=='train']
    assert trains==[('train',recurrent,9,8,32),('train',attention,12,10,32),('train',recurrent,12,9,32),
                    ('train',attention,15,12,64),('train',recurrent,15,12,64)]
    evaluations=[(e[1],e[2]) for e in events if e[0]=='evaluate']
    assert evaluations.count((attention,9))==1
    rows=history(root)
    assert len(rows)==8 and {r['epoch'] for r in rows}=={7,9,12,15}
    assert backup.exists()
    # A retry uses receipts, resumes latest state, and performs no GPU work.
    events.clear();run_pair.main();assert events==[]


def test_failed_initial_evaluation_prevents_training(tmp_path,monkeypatch):
    _,backup=saved_runs(tmp_path,'sudoku');events=[]
    cmd=resume_command(arguments(tmp_path,'sudoku',backup))
    def fail(plans,*a):
        events.extend(p['command'][p['command'].index('--stage')+1] for p in plans)
        raise RuntimeError('Evaluation failed')
    monkeypatch.setattr(run_pair,'run_parallel',fail);monkeypatch.setattr(sys,'argv',cmd[3:])
    with pytest.raises(RuntimeError,match='Evaluation failed'):run_pair.main()
    assert events==['evaluate','evaluate']


def test_wrong_backup_or_missing_checkpoint_fails_before_gpu_work(tmp_path):
    root,backup=saved_runs(tmp_path,'sudoku')
    args=arguments(tmp_path,'sudoku',backup)
    args.attention_epoch6=root/'trajectory_attention/checkpoints/epoch7.ckpt'
    with pytest.raises(ValueError,match='completed epochs7,7,9'):resume_command(args)
    args.attention_epoch6=backup
    (root/'trajectory_recurrent/checkpoints/epoch6.ckpt').unlink()
    with pytest.raises(FileNotFoundError,match='--recurrent-epoch6'):resume_command(args)
    with pytest.raises(ValueError,match='another task or variant'):
        run_pair.initial_checkpoints([['sudoku','trajectory_recurrent',str(backup)]],['sudoku'])


def test_dry_run_reports_two_remaining_epochs_for_attention(tmp_path,monkeypatch,capsys):
    _,backup=saved_runs(tmp_path,'sudoku')
    args=arguments(tmp_path,'sudoku',backup);args.dry_run=True
    cmd=resume_command(args);capsys.readouterr()
    monkeypatch.setattr(run_pair,'run_parallel',lambda *a:pytest.fail('Dry run launched GPU work'))
    monkeypatch.setattr(sys,'argv',cmd[3:]);run_pair.main()
    preview=json.loads(capsys.readouterr().out)[0]
    assert len(preview['initial_evaluations'])==3
    assert [r['completed_epoch_target'] for r in preview['next_training']]==[12,9]
    assert [r['cursor']['epoch'] for r in preview['resume_checkpoints']]==[10,8]


def test_checkpoint_ties_prefer_periodic_resume_file(tmp_path):
    from puzzle_recurrence.schedule import latest_checkpoint,checkpoint_at
    run=tmp_path/'trajectory_recurrent'
    periodic=save_checkpoint(run/'checkpoints/7-28200.ckpt','sudoku','trajectory_recurrent',8)
    save_checkpoint(run/'checkpoints/best.ckpt','sudoku','trajectory_recurrent',8)
    assert latest_checkpoint(run)['path']==str(periodic.resolve())
    assert checkpoint_at(run,28200)['path']==str(periodic.resolve())
    # The maximum optimizer step still takes priority over filename preference.
    best=save_checkpoint(run/'checkpoints/best.ckpt','sudoku','trajectory_recurrent',9)
    assert latest_checkpoint(run)['path']==str(best.resolve())


def test_failed_child_reports_actual_traceback(tmp_path):
    plan=dict(variant='trajectory_attention',gpus='0,1',
        command=[sys.executable,'-c',"raise TypeError('unexpected keyword argument vocab_size')"])
    log=tmp_path/'generation.log'
    with pytest.raises(RuntimeError,match='unexpected keyword argument vocab_size') as error:
        run_pair.run_parallel([plan],[log],tmp_path,tmp_path/'pair.json')
    assert 'trajectory_attention' in str(error.value) and str(log) in str(error.value)
