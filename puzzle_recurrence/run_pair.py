"""Run the matched attention and RM arms on two disjoint GPU pairs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from puzzle_recurrence.devices import selected_gpu_ids

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('trajectory_attention','trajectory_recurrent')


def commands(task,pairs,microbatch,workers,root,resume=False):
    result=[]
    for variant,ids in zip(VARIANTS,pairs):
        run=root/task/'three-state-ablation'/variant
        cmd=[sys.executable,'-u','-m','puzzle_recurrence.entrypoint','--task',task,'--variant',variant,
             '--stage','train','--devices','2','--microbatch',str(microbatch),'--workers',str(workers),'--run',str(run)]
        if resume and run.exists():
            checkpoint=run/'checkpoints/last.ckpt'
            if not checkpoint.exists():raise FileNotFoundError('No resume checkpoint: '+str(checkpoint))
            cmd += ['--resume',str(checkpoint.resolve())]
        result.append(dict(variant=variant,gpus=','.join(ids),command=cmd,run=str(run)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',nargs='+',choices=['sudoku','zebra'],default=['sudoku','zebra'])
    p.add_argument('--attention-gpus',default='0,1');p.add_argument('--recurrent-gpus',default='2,3')
    p.add_argument('--microbatch',type=int,default=32);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--evaluate',action='store_true',help='Evaluate completed-puzzle generation after both training arms finish')
    args=p.parse_args()
    def stop(signum,frame):raise KeyboardInterrupt('Pair runner stopped')
    signal.signal(signal.SIGTERM,stop)
    pairs=[selected_gpu_ids(args.attention_gpus),selected_gpu_ids(args.recurrent_gpus)]
    if any(len(ids)!=2 for ids in pairs) or set(pairs[0])&set(pairs[1]):p.error('Choose two disjoint pairs of physical GPU IDs')
    if args.microbatch<1 or 512%(2*args.microbatch):p.error('Global batch512 must divide by 2*microbatch')
    if len(set(args.tasks))!=len(args.tasks):p.error('Each task can appear once')
    output=args.output_root.resolve()
    if args.dry_run:
        print(json.dumps([dict(task=task,arms=commands(task,pairs,args.microbatch,args.workers,output,args.resume)) for task in args.tasks],indent=2));return
    for task in args.tasks:
        plans=commands(task,pairs,args.microbatch,args.workers,output,args.resume)
        if not args.resume:
            for plan in plans:
                if Path(plan['run']).exists():raise FileExistsError('Preserve previous trial; choose --output-root or --resume: '+plan['run'])
        status_dir=output/task/'three-state-ablation';status_dir.mkdir(parents=True,exist_ok=True)
        children=[];logs=[]
        try:
            for plan in plans:
                logpath=status_dir/(plan['variant']+'.log');log=logpath.open('a');logs.append(log)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=plan['gpus'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
                child=subprocess.Popen(plan['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(child)
                print(f'{task}: {plan["variant"]} on GPUs {plan["gpus"]}; log {logpath}',flush=True)
            (status_dir/'pair.json').write_text(json.dumps(dict(task=task,arms=plans,pids=[c.pid for c in children]),indent=2)+'\n')
            # Abort the sibling promptly if either arm fails.
            while any(c.poll() is None for c in children):
                if any(c.poll() not in (None,0) for c in children):raise RuntimeError('One arm failed; inspect its log')
                import time
                time.sleep(2)
            if any(c.returncode for c in children):raise RuntimeError('A paired arm failed')
            (status_dir/'pair-complete.json').write_text(json.dumps(dict(task=task,arms=plans),indent=2)+'\n')
            print(task+': both arms finished.',flush=True)
            if args.evaluate:
                children=[]
                for plan in plans:
                    checkpoint=Path(plan['run'])/'checkpoints/last.ckpt'
                    if not checkpoint.exists():raise FileNotFoundError('Final checkpoint missing: '+str(checkpoint))
                    evaluation=Path(plan['run'])/'evaluation/final'
                    command=[sys.executable,'-u','-m','puzzle_recurrence.entrypoint','--task',task,'--variant',plan['variant'],
                        '--stage','evaluate','--resume',str(checkpoint.resolve()),'--run',str(evaluation)]
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=plan['gpus'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                        OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                    log=(status_dir/(plan['variant']+'-evaluation.log')).open('a');logs.append(log)
                    children.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
                while any(c.poll() is None for c in children):
                    if any(c.poll() not in (None,0) for c in children):raise RuntimeError('A generation evaluation failed')
                    import time
                    time.sleep(2)
                if any(c.returncode for c in children):raise RuntimeError('A generation evaluation failed')
                print(task+': both generation evaluations finished.',flush=True)
        finally:
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            for log in logs:log.close()

if __name__=='__main__':main()
