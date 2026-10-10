"""Alternate attention and recurrence on all four GPUs at absolute epoch milestones."""
import argparse
import json
from pathlib import Path
import os
import signal

from puzzle_recurrence.run_pair import ROOT,VARIANTS,commands,run_parallel,evaluate_checkpoints,initial_checkpoints
from puzzle_recurrence.schedule import TASKS,milestones,latest_checkpoint,checkpoint_at
from puzzle_recurrence.devices import selected_gpu_ids
from puzzle_recurrence.results import atomic_json


def policy(batch,devices):
    if batch<1 or 512%(batch*devices):raise ValueError('Global batch512 must divide by devices*microbatch')
    return dict(batch=batch,devices=devices,global_batch=512,accumulation=512//(batch*devices))


def validate_resume(info,current,allow_batch,allow_devices,task,variant):
    if info is None:return
    expected='sudoku-puzzle' if task=='sudoku' else 'zebra'
    if info['variant']!=variant or info['task']!=expected:raise ValueError('Resume checkpoint belongs to another task or variant')
    old=info['batch_policy']
    if old==current:return
    if info['cursor']['rows'] or info['cursor']['batches']:raise ValueError('GPU/batch migration requires a completed epoch checkpoint')
    if old['global_batch']!=current['global_batch']:raise ValueError('Global batch must remain unchanged')
    if old['devices']!=current['devices'] and not allow_devices:raise ValueError('Use --allow-device-change to resume with four GPUs')
    if old['batch']!=current['batch'] and not (allow_batch or allow_devices):raise ValueError('Use --allow-microbatch-change')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',nargs='+',choices=list(TASKS),default=['sudoku','zebra'])
    p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--microbatch',type=int,default=64)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--output-root',type=Path,default=ROOT/'outputs')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--allow-microbatch-change',action='store_true');p.add_argument('--allow-device-change',action='store_true')
    p.add_argument('--evaluation-every',type=int,default=3);p.add_argument('--epochs',type=int)
    p.add_argument('--initial-evaluation',nargs=3,action='append',default=[],metavar=('TASK','VARIANT','CHECKPOINT'))
    args=p.parse_args();ids=selected_gpu_ids(args.gpus)
    if len(ids)!=4:p.error('Select four physical GPU IDs')
    if len(set(args.tasks))!=len(args.tasks):p.error('Each task can appear once')
    if args.initial_evaluation and not args.resume:p.error('Initial evaluation requires --resume')
    current=policy(args.microbatch,len(ids));output=args.output_root.expanduser().resolve()
    initial=initial_checkpoints(args.initial_evaluation,args.tasks)
    def stop(signum,frame):raise KeyboardInterrupt('Serial queue stopped')
    signal.signal(signal.SIGTERM,stop)
    for task in args.tasks:
        root=output/task/'three-state-ablation';runs=[root/v for v in VARIANTS]
        if not args.resume and any(run.exists() for run in runs):raise FileExistsError('Choose --resume or a fresh --output-root')
        latest=[latest_checkpoint(run) for run in runs]
        if args.resume and any(run.exists() and info is None for run,info in zip(runs,latest)):
            raise FileNotFoundError('An existing arm has no checkpoint')
        for variant,info in zip(VARIANTS,latest):validate_resume(info,current,args.allow_microbatch_change,args.allow_device_change,task,variant)
        planned=milestones(args.epochs or TASKS[task]['epochs'],args.evaluation_every)
        budget=planned[-1]*TASKS[task]['steps_per_epoch']
        if any(info and info['step']>budget for info in latest):raise ValueError('Total epoch budget precedes a latest checkpoint')
        # Continue the absolute schedule from the earlier arm. No catch-up
        # barrier, retrospective missing checkpoints, or rewind of the later arm.
        start=min(info['step'] if info else 0 for info in latest)
        planned=[e for e in planned if e*TASKS[task]['steps_per_epoch']>=start]
        if args.dry_run:
            steps=[];virtual=list(latest)
            for epoch in planned:
                target=epoch*TASKS[task]['steps_per_epoch']
                for index,variant in enumerate(VARIANTS):
                    info=virtual[index]
                    if info is None or info['step']<target:
                        steps.append(dict(variant=variant,from_step=info['step'] if info else 0,completed_epoch_target=epoch,
                            devices=4,gpus=ids,policy=current))
                        virtual[index]=dict(step=target)
            print(json.dumps(dict(task=task,resume_checkpoints=latest,initial_evaluations=initial[task],sequence=steps),indent=2));continue
        for record in initial[task]:
            index=VARIANTS.index(record['variant'])
            if latest[index] is None or record['checkpoint']['step']>latest[index]['step']:
                raise ValueError('Initial evaluation checkpoint is newer than latest training checkpoint')
            evaluate_checkpoints(task,root,[ids,ids],[record],args.workers,stage='initial_generation')
        for epoch in planned:
            target=epoch*TASKS[task]['steps_per_epoch']
            for index,(variant,run) in enumerate(zip(VARIANTS,runs)):
                info=latest[index]
                if info is None or info['step']<target:
                    plan=commands(task,[ids,ids],args.microbatch,args.workers,output,target_steps=target,sources=latest,
                        devices=4,allow_microbatch_change=args.allow_microbatch_change,allow_device_change=args.allow_device_change)[index]
                    atomic_json(root/'current.json',dict(stage='training',variant=variant,mode='serial',epoch_target=epoch,
                        target_step=target,policy=current,controller_pid=os.getpid()))
                    run_parallel([plan],[root/(variant+'.log')],ROOT,root/'pair.json')
                    latest[index]=latest_checkpoint(run)
                    if latest[index] is None or latest[index]['step']!=target:raise RuntimeError('Serial arm did not reach its exact milestone')
                generation=run/'generation'/f'epoch-{epoch:03d}'
                if (generation/'complete.json').exists():continue
                checkpoint=checkpoint_at(run,target)
                if checkpoint is None:
                    print(f'{variant}: completed epoch{epoch} checkpoint unavailable; skipping its old evaluation.',flush=True);continue
                evaluate_checkpoints(task,root,[ids,ids],[dict(variant=variant,epoch=epoch,checkpoint=checkpoint)],args.workers)
        atomic_json(root/'current.json',dict(stage='complete',mode='serial',epochs=planned[-1],controller_pid=os.getpid()))
        atomic_json(root/'pair-complete.json',dict(task=task,epochs=planned,variants=list(VARIANTS),mode='serial'))


if __name__=='__main__':main()
