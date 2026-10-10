"""Train matched puzzle arms on separate GPU pairs, evaluating every three epochs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from puzzle_recurrence.devices import selected_gpu_ids
from puzzle_recurrence.schedule import TASKS,milestones,latest_checkpoint,checkpoint_at,checkpoint_info,paired_microbatch
from puzzle_recurrence.results import atomic_json,record_generation,history

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('trajectory_attention','trajectory_recurrent')


def initial_checkpoints(specifications,tasks):
    """Validate all requested initial evaluations before launching any GPU work."""
    result={task:[] for task in tasks};seen=set()
    for task,variant,path in specifications:
        if task not in result or variant not in VARIANTS:
            raise ValueError('Initial evaluation requires a selected task and a paired variant')
        info=checkpoint_info(Path(path).expanduser())
        expected_task='sudoku-puzzle' if task=='sudoku' else 'zebra'
        if info['variant']!=variant or info['task']!=expected_task:
            raise ValueError('Initial checkpoint belongs to another task or variant: '+path)
        epoch,remainder=divmod(info['step'],TASKS[task]['steps_per_epoch'])
        if epoch<1 or remainder or info['cursor']['epoch']!=epoch or info['cursor']['rows'] or info['cursor']['batches']:
            raise ValueError('Initial evaluation requires a completed epoch checkpoint: '+path)
        key=(task,variant,epoch)
        if key in seen:raise ValueError('Duplicate initial checkpoint evaluation')
        seen.add(key);result[task].append(dict(variant=variant,epoch=epoch,checkpoint=info))
    return result


def evaluation_plan(task,root,variant,ids,epoch,info,workers):
    generation=root/variant/'generation'/f'epoch-{epoch:03d}'
    complete=generation/'complete.json'
    if complete.exists():
        record=json.loads(complete.read_text())
        if record['step']!=info['step'] or Path(record['checkpoint']).resolve()!=Path(info['path']).resolve():
            raise ValueError('Existing evaluation uses a different checkpoint: '+str(complete))
        return None
    cmd=[sys.executable,'-u','-m','puzzle_recurrence.entrypoint','--task',task,'--variant',variant,'--stage','evaluate',
        '--resume',info['path'],'--run',str(generation),'--eval-batches','10','--eval-batch-size','128','--workers',str(workers)]
    return dict(variant=variant,gpus=','.join(ids),command=cmd,run=str(generation))


def evaluate_checkpoints(task,root,pairs,records,workers,stage='generation'):
    """Group equal epochs on disjoint pairs. Failures stop before training."""
    for epoch in sorted({r['epoch'] for r in records}):
        plans=[];receipts=[]
        for record in records:
            if record['epoch']!=epoch:continue
            variant=record['variant'];info=record['checkpoint']
            plan=evaluation_plan(task,root,variant,pairs[VARIANTS.index(variant)],epoch,info,workers)
            if plan:
                plans.append(plan);receipts.append((Path(plan['run']),info))
        if plans:
            atomic_json(root/'current.json',dict(stage=stage,epoch=epoch,step=epoch*TASKS[task]['steps_per_epoch'],controller_pid=os.getpid()))
            run_parallel(plans,[Path(p['run'])/'generation.log' for p in plans],ROOT,root/'pair.json')
            for generation,info in receipts:record_generation(generation,epoch,info['step'],info['path'])
    history(root)
    from puzzle_recurrence.monitor import refresh
    refresh(root,task)


def commands(task,pairs,microbatch,workers,root,resume=False,target_steps=None,allow_microbatch_change=False,sources=None):
    result=[]
    for index,(variant,ids) in enumerate(zip(VARIANTS,pairs)):
        run=root/task/'three-state-ablation'/variant
        cmd=[sys.executable,'-u','-m','puzzle_recurrence.entrypoint','--task',task,'--variant',variant,
             '--stage','train','--devices','2','--microbatch',str(microbatch),'--workers',str(workers),'--run',str(run)]
        if target_steps is not None:cmd += ['--target-steps',str(target_steps)]
        if allow_microbatch_change:cmd += ['--allow-microbatch-change']
        if sources is not None:
            if sources[index]:cmd += ['--resume',sources[index]['path']]
        elif resume and run.exists():
            checkpoint=run/'checkpoints/last.ckpt'
            if not checkpoint.exists():raise FileNotFoundError('No resume checkpoint: '+str(checkpoint))
            cmd += ['--resume',str(checkpoint.resolve())]
        result.append(dict(variant=variant,gpus=','.join(ids),command=cmd,run=str(run)))
    return result


def run_parallel(plans,logs,root,status):
    children=[];streams=[]
    try:
        for plan,path in zip(plans,logs):
            path.parent.mkdir(parents=True,exist_ok=True);stream=path.open('a');streams.append(stream)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=plan['gpus'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
            child=subprocess.Popen(plan['command'],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child);print(plan['variant']+' on GPUs '+plan['gpus']+'; log '+str(path),flush=True)
        atomic_json(status,dict(arms=plans,pids=[c.pid for c in children],controller_pid=os.getpid()))
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('One arm failed; inspect its log')
            time.sleep(2)
        if any(c.returncode for c in children):raise RuntimeError('A paired arm failed')
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for stream in streams:stream.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',nargs='+',choices=list(TASKS),default=['sudoku','zebra'])
    p.add_argument('--attention-gpus',default='0,1');p.add_argument('--recurrent-gpus',default='2,3')
    p.add_argument('--microbatch',type=int,default=32);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--allow-microbatch-change',action='store_true')
    p.add_argument('--evaluate',action='store_true',default=True,help='Generation evaluation is enabled by default')
    p.add_argument('--no-evaluate',action='store_false',dest='evaluate')
    p.add_argument('--evaluation-every',type=int,default=3);p.add_argument('--epochs',type=int)
    p.add_argument('--initial-evaluation',nargs=3,action='append',default=[],metavar=('TASK','VARIANT','CHECKPOINT'),
        help='Evaluate a saved epoch checkpoint before training; repeat for each checkpoint, including backups')
    args=p.parse_args()
    if args.initial_evaluation and (not args.resume or not args.evaluate):p.error('Initial evaluation requires --resume and enabled evaluation')
    def stop(signum,frame):raise KeyboardInterrupt('Pair runner stopped')
    signal.signal(signal.SIGTERM,stop)
    pairs=[selected_gpu_ids(args.attention_gpus),selected_gpu_ids(args.recurrent_gpus)]
    if any(len(ids)!=2 for ids in pairs) or set(pairs[0])&set(pairs[1]):p.error('Choose two disjoint GPU pairs')
    if args.microbatch<1 or 512%(2*args.microbatch):p.error('Global batch512 must divide by 2*microbatch')
    if len(set(args.tasks))!=len(args.tasks):p.error('Each task can appear once')
    output=args.output_root.resolve()
    initial=initial_checkpoints(args.initial_evaluation,args.tasks)
    if args.dry_run:
        preview=[]
        for task in args.tasks:
            root=output/task/'three-state-ablation'
            planned=milestones(args.epochs or TASKS[task]['epochs'],args.evaluation_every)
            latest=[latest_checkpoint(root/v) for v in VARIANTS] if args.resume else [None,None]
            batch,reason=paired_microbatch(args.microbatch,latest);next_training=[]
            for index,info in enumerate(latest):
                epoch=next((e for e in planned if info is None or info['step']<e*TASKS[task]['steps_per_epoch']),None)
                if epoch is not None:
                    plan=commands(task,pairs,batch,args.workers,output,target_steps=epoch*TASKS[task]['steps_per_epoch'],
                        allow_microbatch_change=args.allow_microbatch_change,sources=latest)[index]
                    next_training.append(dict(completed_epoch_target=epoch,**plan))
            preview.append(dict(task=task,initial_evaluations=[evaluation_plan(task,root,
                r['variant'],pairs[VARIANTS.index(r['variant'])],r['epoch'],r['checkpoint'],args.workers) for r in initial[task]],
                epochs=planned,resume_checkpoints=latest,next_training=next_training,microbatch_reason=reason))
        print(json.dumps(preview,indent=2));return
    for task in args.tasks:
        root=output/task/'three-state-ablation';runs=[root/v for v in VARIANTS]
        if not args.resume and any(run.exists() for run in runs):raise FileExistsError('Choose --resume or a fresh --output-root')
        root.mkdir(parents=True,exist_ok=True)
        latest=[latest_checkpoint(run) for run in runs]
        if args.resume and any(run.exists() and info is None for run,info in zip(runs,latest)):
            raise FileNotFoundError('An existing arm has no checkpoint; preserve it and use a fresh output root')
        if initial[task]:
            if any(info is None for info in latest):raise FileNotFoundError('Initial evaluations require both latest training checkpoints')
            if any(r['checkpoint']['step']>latest[VARIANTS.index(r['variant'])]['step'] for r in initial[task]):
                raise ValueError('Initial evaluation checkpoint is newer than the training resume checkpoint')
            budget=(args.epochs or TASKS[task]['epochs'])*TASKS[task]['steps_per_epoch']
            if any(info['step']>budget for info in latest):raise ValueError('Total epoch budget precedes a latest resume checkpoint')
            evaluate_checkpoints(task,root,pairs,initial[task],args.workers,stage='initial_generation')
            # Evaluation checkpoints are never used as training resume sources.
        planned=milestones(args.epochs or TASKS[task]['epochs'],args.evaluation_every)
        if initial[task]:
            # Historical checkpoints were explicitly selected above. Continue
            # the absolute schedule from the earliest current training state.
            planned=[e for e in planned if e*TASKS[task]['steps_per_epoch']>=min(info['step'] for info in latest)]
        for epoch in planned:
            target=epoch*TASKS[task]['steps_per_epoch']
            actual_batch,reason=paired_microbatch(args.microbatch,latest)
            if reason:print(reason+f'; using microbatch{actual_batch} for this chunk.',flush=True)
            if any(info and info['batch_policy']['batch']!=actual_batch for info in latest) and not args.allow_microbatch_change:
                raise ValueError('Use --allow-microbatch-change to change both arms at an epoch boundary')
            plans=commands(task,pairs,actual_batch,args.workers,output,target_steps=target,
                allow_microbatch_change=args.allow_microbatch_change,sources=latest)
            pending=[(plan,root/(plan['variant']+'.log')) for plan,info in zip(plans,latest) if info is None or info['step']<target]
            if pending:
                atomic_json(root/'current.json',dict(stage='training',epoch_target=epoch,target_step=target,microbatch=actual_batch,
                    requested_microbatch=args.microbatch,controller_pid=os.getpid()))
                run_parallel([x[0] for x in pending],[x[1] for x in pending],ROOT,root/'pair.json')
                latest=[latest_checkpoint(run) for run in runs]
                if any(info is None or info['step']<target for info in latest):raise RuntimeError('Chunk did not reach its milestone')
            if args.evaluate:
                records=[]
                for variant,run,ids in zip(VARIANTS,runs,pairs):
                    generation=run/'generation'/f'epoch-{epoch:03d}'
                    if (generation/'complete.json').exists():continue
                    info=checkpoint_at(run,target)
                    if info is None:
                        # Never label a newer checkpoint as an earlier epoch.
                        print(f'{variant}: epoch{epoch} checkpoint unavailable; no retrospective accuracy is invented.',flush=True)
                        continue
                    records.append(dict(variant=variant,epoch=epoch,checkpoint=info))
                evaluate_checkpoints(task,root,pairs,records,args.workers)
                print(f'{task}: epoch{epoch} generation metrics saved.',flush=True)
        atomic_json(root/'current.json',dict(stage='complete',epochs=planned[-1],controller_pid=os.getpid()))
        atomic_json(root/'pair-complete.json',dict(task=task,epochs=planned,variants=list(VARIANTS)))

if __name__=='__main__':main()
