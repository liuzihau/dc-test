"""Stop only the verified MDM job after a requested validation and checkpoint."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
from owt.continuation import ROOT,digest
from owt.research import atomic_write,read_json,read_csv,timestamp

RUN=ROOT/'outputs/owt/continuation-15000/mdm'


def write(path,value):atomic_write(path,json.dumps(value,indent=2)+'\n')


def process(pid):
    folder=Path('/proc')/str(pid)
    try:
        text=(folder/'stat').read_text();fields=text[text.rfind(')')+2:].split()
        command=(folder/'cmdline').read_bytes().split(b'\0')
        return dict(pid=pid,state=fields[0],pgid=int(fields[2]),start_ticks=int(fields[19]),
            uid=folder.stat().st_uid,argv=[s.decode() for s in command if s])
    except FileNotFoundError:return None


def bind(pid,worker):
    p=process(pid)
    if not p or p['state']=='Z' or p['uid']!=os.getuid() or p['pgid']!=pid:
        raise ValueError('Expected owned live process-group leader')
    if 'owt.mdm_15000' not in p['argv'] or ('--worker' in p['argv'])!=worker:
        raise ValueError('PID does not belong to the selected MDM job')
    return p


def same_process(record):
    current=process(record['pid'])
    return bool(current and current['state']!='Z' and current['start_ticks']==record['start_ticks']
                and current['uid']==record['uid'] and current['pgid']==record['pgid'])


def descendants(pid):
    result=[];pending=[pid]
    while pending:
        current=pending.pop();record=process(current)
        if record:result.append(record)
        try:
            children=(Path('/proc')/str(current)/'task'/str(current)/'children').read_text()
            pending.extend(int(value) for value in children.split())
        except FileNotFoundError:pass
    return result


def checkpoint_audit(path,step):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    payload=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    if payload['global_step']!=step or payload['ema']['num_updates']!=step:
        raise ValueError('Checkpoint step/EMA mismatch')
    if payload['lr_schedulers'][0]['last_epoch']!=step:raise ValueError('Scheduler step mismatch')
    if {int(s['step']) for s in payload['optimizer_states'][0]['state'].values()}!={step}:
        raise ValueError('Optimizer step mismatch')
    if any(not {'exp_avg','exp_avg_sq'}.issubset(s) for s in payload['optimizer_states'][0]['state'].values()):
        raise ValueError('Missing optimizer moments')
    if payload['hyper_parameters']['config']['mechanisms']['np']['enabled']:
        raise ValueError('Expected MDM baseline')
    return dict(global_step=step,ema_updates=step,scheduler_step=step,adam_moments_verified=True,
        sha256=digest(path),bytes=path.stat().st_size,checkpoint=str(path.relative_to(ROOT)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args();request=read_json(args.request)
    if not request or request['run']!=str(RUN) or request['user_instruction']!='stop after the next validation':
        raise ValueError('Explicit stop request for this run is required')
    step=request['validation_step'];root=RUN.parent
    def status(stage,**details):write(root/'stop_status.json',dict(stage=stage,validation_step=step,
        watcher_pid=os.getpid(),updated_at=timestamp(),**details))
    with (root/'.stop-watcher.lock').open('a') as watcher:
        fcntl.flock(watcher,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controller=bind(request['controller_pid'],False);worker=bind(request['worker_pid'],True)
        write(root/'stop_bound_processes.json',dict(controller=controller,worker=worker))
        status('waiting_for_validation_and_checkpoint',controller_pid=controller['pid'],worker_pid=worker['pid'])
        checkpoint=RUN/'checkpoints'/f'step-{step:07d}.ckpt'
        while True:
            if not same_process(controller) or not same_process(worker):
                status('job_ended_before_scheduled_stop');return
            validation=next((row for row in read_csv(RUN/'local_metrics/validation.csv') if row['optimizer_step']==step),None)
            if validation and checkpoint.is_file():break
            time.sleep(2)
        # Freeze the owned group before checking the saved state, so no further
        # updates accumulate while verification reads the checkpoint.
        os.killpg(worker['pgid'],signal.SIGSTOP)
        os.kill(controller['pid'],signal.SIGSTOP)
        members=[r for r in descendants(worker['pid']) if r['pgid']==worker['pgid']]
        status('validation_complete_verifying_checkpoint',validation=validation)
        try:audit=checkpoint_audit(checkpoint,step)
        except BaseException as error:
            status('paused_for_checkpoint_review',error=str(error),trainer_paused=True)
            write(root/'queue.json',dict(stage='paused_for_checkpoint_review',user_requested_stop=True,
                validation_step=step,error=str(error),updated_at=timestamp()))
            raise
        # The frozen controller keeps the GPU lease until all owned training
        # processes are gone. It cannot misclassify the requested stop as failure.
        os.killpg(worker['pgid'],signal.SIGTERM);os.killpg(worker['pgid'],signal.SIGCONT)
        deadline=time.monotonic()+30
        while any(same_process(r) for r in members) and time.monotonic()<deadline:time.sleep(.5)
        forced=any(same_process(r) for r in members)
        if forced:
            try:os.killpg(worker['pgid'],signal.SIGKILL)
            except ProcessLookupError:pass
        deadline=time.monotonic()+10
        while any(same_process(r) for r in members) and time.monotonic()<deadline:time.sleep(.2)
        if any(same_process(r) for r in members):raise RuntimeError('An owned trainer process remains alive')
        os.kill(controller['pid'],signal.SIGTERM);os.kill(controller['pid'],signal.SIGCONT)
        deadline=time.monotonic()+10
        while same_process(controller) and time.monotonic()<deadline:time.sleep(.2)
        if same_process(controller):os.kill(controller['pid'],signal.SIGKILL)
        receipt=dict(stopped_at=timestamp(),user_requested_stop=True,validation_step=step,
            validation=validation,checkpoint_audit=audit,controller_pid=controller['pid'],worker_pid=worker['pid'],
            all_verified_trainer_processes_stopped=True,forced_termination_needed=forced,
            original_target_step=15000,checkpoint_preserved=True,automatic_restart=False)
        write(RUN/'stopped.json',receipt)
        write(root/'queue.json',dict(stage='stopped_after_validation',variant='mdm',
            validation_step=step,checkpoint=audit['checkpoint'],user_requested_stop=True,updated_at=timestamp()))
        state_path=ROOT/'outputs/research-notes/research_loop_state_20261001.json'
        state=read_json(state_path)
        if state:
            state.setdefault('mdm_15000_continuation',{}).update(status='stopped_after_validation',
                validation_step=step,stop_receipt=str((RUN/'stopped.json').relative_to(ROOT)))
            state['next_work']='User-requested MDM stop completed after validation; no training restart selected.'
            state['updated_at']=timestamp();write(state_path,state)
        status('stopped_after_validation',checkpoint=audit['checkpoint'],validation=validation)


if __name__=='__main__':main()
