#!/usr/bin/env python3
"""Pause ONLY an old queue controller; adopt it after its current worker ends.

The existing training worker is never signalled. After it exits successfully,
retire the old controller and exec the launcher with the requested schedule.
PID birth times protect against PID reuse. A normal interruption while waiting
resumes the old controller instead of stranding it. Runs inside its own tmux.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time


def process(pid):
    base = Path('/proc')/str(pid)
    try:
        fields = (base/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid, state=fields[0], parent=int(fields[1]), birth=fields[19],
                    argv=(base/'cmdline').read_bytes().decode().strip('\0').split('\0'))
    except FileNotFoundError:
        return None


def same_process(snapshot):
    current = process(snapshot['pid'])
    return current if current and current['birth']==snapshot['birth'] else None


def write_json(path, value):
    temp = path.with_suffix('.pending.json')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue-pid',type=int,required=True)
    p.add_argument('--worker-pid',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    schedule = output/'requested_schedule.json'
    if not schedule.is_file():
        raise ValueError('Write and validate requested_schedule.json before handoff')
    status = json.loads((output/'status.json').read_text())
    queue, worker = process(args.queue_pid), process(args.worker_pid)
    if (not queue or not worker or status['pid']!=args.queue_pid
            or worker['parent']!=args.queue_pid or 'reasoning.second_epoch_queue' not in queue['argv']
            or 'torch.distributed.run' not in worker['argv']
            or str(output) not in queue['argv'] or worker['argv'] != status['command']):
        raise ValueError('Queue/worker identity or active-stage contract mismatch')
    run = Path(worker['argv'][worker['argv'].index('--run-dir')+1])
    plan = json.loads((output/'plan.json').read_text())
    current_job = next(j for j in plan if Path(j['run']).resolve()==run.resolve())
    spec = json.loads(schedule.read_text())
    if spec['jobs'][0] != {k:current_job[k] for k in ('task','variant')}:
        raise ValueError('First scheduled job must finish/evaluate the active worker')
    receipt = output/'handoff_status.json'
    paused, retired = False, False
    def interrupted(sig, frame):
        raise KeyboardInterrupt('Handoff interrupted '+str(sig))
    signal.signal(signal.SIGTERM, interrupted)
    with (output/'handoff.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            os.kill(queue['pid'],signal.SIGSTOP)
            paused = True
            write_json(receipt,dict(status='waiting_for_current_training',pid=os.getpid(),
                queue=queue,worker=worker,run=str(run),schedule=str(schedule),
                note='Only old controller paused; existing DDP worker continues unchanged.'))
            print('Old controller paused; training worker continues:',worker['pid'],flush=True)
            while True:
                current = same_process(worker)
                if current is None or current['state']=='Z':
                    break
                if not same_process(queue):
                    raise RuntimeError('Old controller disappeared during handoff')
                time.sleep(15)
            completed = json.loads((run/'status.json').read_text())
            if (completed['status']!='finished' or completed['max_steps']!=current_job['target_step']
                    or completed['examples_seen']!=3*current_job['examples']):
                raise RuntimeError('Worker did not finish its exact epoch budget')
            # The child has exited. TERM now executes the old controller's
            # cleanup without ever terminating a live GPU training process.
            if same_process(queue):
                os.kill(queue['pid'],signal.SIGTERM)
                os.kill(queue['pid'],signal.SIGCONT)
            paused = False
            for _ in range(60):
                current = same_process(queue)
                if current is None or current['state']=='Z':
                    break
                time.sleep(1)
            else:
                raise RuntimeError('Old controller did not release ownership; no replacement started')
            retired = True
            write_json(receipt,dict(status='controller_replaced',pid=os.getpid(),
                completed_training=str(run),schedule=str(schedule)))
            env = dict(os.environ,DCACHE_PYTHON=sys.executable,DCACHE_QUEUE_SCHEDULE=str(schedule),
                       DCACHE_THIRD_EPOCH_DIR=str(output))
            os.chdir(root)
            os.execvpe('bash',['bash',str(root/'scripts/reasoning/run_split_third_epoch_2x3090.sh'),'run'],env)
        except BaseException as error:
            if paused and not retired and same_process(queue):
                os.kill(queue['pid'],signal.SIGCONT)
            write_json(receipt,dict(status='failed',error=str(error),old_controller_resumed=paused and not retired))
            raise


if __name__=='__main__':
    main()
