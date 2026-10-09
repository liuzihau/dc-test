"""Serialize the user-selected MDM and A continuations after distance-two training."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from owt.continuation import ROOT, RUN_ROOT, A, DISTANCE, ORIGINALS, verify_selection
from owt.research import atomic_write, read_csv, read_json, timestamp

LOCK = ROOT/'outputs/owt/mdm-np-5k/.queue.lock'
QUEUE = ROOT/RUN_ROOT/'queue.json'
DISTANCE_ROOT = ROOT/'outputs/owt/transformer-np-5k'


def write_status(stage, **details):
    atomic_write(QUEUE, json.dumps(dict(stage=stage, order=['mdm', A], resume_step=5000,
        target_step=7500, updated_at=timestamp(), controller_pid=os.getpid(),
        physical_gpus=[2,3], **details), indent=2)+'\n')


def distance_ready():
    queue = read_json(DISTANCE_ROOT/'distance2_queue.json') or {}
    if queue.get('stage')=='complete_waiting_for_scientific_review':
        if (read_json(DISTANCE_ROOT/DISTANCE/'complete.json') or {}).get('optimizer_step')!=5000:
            raise ValueError('Distance-two completion receipt is missing')
        return True
    if any(word in queue.get('stage','') for word in ('failed','blocked')):
        raise RuntimeError('Distance-two job failed; inspect before continuing')
    if queue.get('controller_pid'):
        try: os.kill(int(queue['controller_pid']),0)
        except ProcessLookupError:
            raise RuntimeError('Distance-two controller stopped before completion')
    return False


def prepare_run(variant):
    run = ROOT/RUN_ROOT/variant
    if run.exists() and any(run.iterdir()):
        raise RuntimeError('Continuation output already exists; inspect before retrying: '+str(run))
    run.mkdir(parents=True,exist_ok=True)
    destination=run/'local_metrics';destination.mkdir()
    names=['train.csv','validation.csv']
    if variant==A: names+=['gradient_norms.csv','source_pairs.csv']
    for name in names:
        source=ROOT/ORIGINALS[variant]/'local_metrics'/name
        rows=read_csv(source)
        if not rows or rows[-1]['optimizer_step']!=5000:
            raise ValueError('Original metric prefix does not finish at5000: '+str(source))
        shutil.copy2(source,destination/name)
    atomic_write(run/'metric_prefix.json',json.dumps(dict(original=str(ORIGINALS[variant]),
        optimizer_step=5000,copied_files=names,elapsed_seconds_restart_at_continuation=True),indent=2)+'\n')
    return run


def verify_startup(run, variant):
    rows=[r for r in read_csv(run/'local_metrics/train.csv') if r['optimizer_step']>5000][:3]
    if len(rows)<3: return None
    if [r['optimizer_step'] for r in rows]!=[5001,5002,5003]:
        raise ValueError('Continuation did not begin at optimizer step5001')
    for row in rows:
        expected=row['main_elbo'] if variant=='mdm' else row['main_elbo']+.25*(row['np_prev']+row['np_next'])
        if not math.isclose(row['objective'],expected,rel_tol=2e-6,abs_tol=2e-5):
            raise ValueError('Continuation loss or A auxiliary coefficient changed')
        if not math.isclose(row['learning_rate'],.0003,rel_tol=1e-8):
            raise ValueError('Continuation learning rate changed')
    for rank in (0,1):
        restored=read_json(run/f'resume-rank{rank}.json')
        trace=read_json(run/f'first-batches-rank{rank}.json')
        if not restored or not trace or len(trace)<3: return None
        if restored['restored']['step']!=5000 or restored['continuation_seed']!=750001+rank:
            raise ValueError('Resume audit or per-rank seed differs')
        if [r['optimizer_step'] for r in trace]!=[5001,5002,5003] or any('noisy_sha256' not in r for r in trace):
            raise ValueError('Incomplete matched-input audit')
        if variant==A:
            baseline=read_json(ROOT/RUN_ROOT/'mdm'/f'first-batches-rank{rank}.json')
            if trace!=baseline:
                raise ValueError('MDM/A continuation batches or corruption differ')
    if variant==A:
        pairs=[r for r in read_csv(run/'local_metrics/source_pairs.csv') if r['optimizer_step']>5000][:3]
        if len(pairs)<3: return None
        for row in pairs:
            for direction in ('prev','next'):
                for b in range(5):
                    prefix=f'{direction}_maskbin{b}_'
                    if not row[prefix+'selected']==row[prefix+'selected_masked']==row[prefix+'masked_source']:
                        raise ValueError('A used a revealed source or changed its pair rule')
    receipt=dict(verified_at=timestamp(),variant=variant,steps_checked=[5001,5002,5003],
        model_optimizer_scheduler_ema_and_cursor_restored=True,
        matched_new_random_stream=True,matched_batches_and_masks_checked_against_mdm=variant==A,
        objective_verified=True,learning_rate=.0003)
    atomic_write(run/'startup_review.json',json.dumps(receipt,indent=2)+'\n')
    return receipt


def verify_finished(run,variant):
    complete=read_json(run/'complete.json') or {}
    if complete.get('optimizer_step')!=7500 or complete.get('resumed_from')!=5000:
        raise RuntimeError('Continuation did not finish at7500')
    if not Path(complete['checkpoint']).is_file():
        raise RuntimeError('Final continuation checkpoint is missing')
    for name in ('train.csv','validation.csv'):
        rows=read_csv(run/'local_metrics'/name)
        if not rows or rows[-1]['optimizer_step']!=7500:
            raise RuntimeError('Continuation final metrics are missing: '+name)
    if not verify_startup(run,variant):
        raise RuntimeError('Continuation startup verification is incomplete')


def stop_child(child):
    if child.poll() is not None: return
    os.killpg(child.pid,signal.SIGTERM)
    try: child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=30)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    args=parser.parse_args()
    verify_selection(args.selection)
    (ROOT/RUN_ROOT).mkdir(parents=True,exist_ok=True)
    with (ROOT/RUN_ROOT/'.controller.lock').open('a') as controller:
        fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            write_status('queued_waiting_for_distance2',selection=str(args.selection.resolve()))
            while not distance_ready(): time.sleep(30)
            with LOCK.open('a') as lock:
                while True:
                    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:
                        write_status('queued_waiting_for_GPU_lock');time.sleep(30)
                verify_selection(args.selection,verify_checkpoints=True)
                if not distance_ready(): raise RuntimeError('Distance-two completion changed')
                for variant in ('mdm',A):
                    run=ROOT/RUN_ROOT/variant
                    if (read_json(run/'complete.json') or {}).get('optimizer_step')==7500:
                        verify_finished(run,variant);continue
                    verify_selection(args.selection)
                    run=prepare_run(variant)
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',OMP_NUM_THREADS='2',
                        MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                        OWT_CONTINUATION_AUTHORIZED=str(args.selection.resolve()))
                    command=[sys.executable,'-u','-m','owt.continuation_entrypoint',
                        '--selection',str(args.selection.resolve()),'--variant',variant]
                    with (run/'train.log').open('a') as log:
                        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,
                            stderr=subprocess.STDOUT,start_new_session=True)
                        reviewed=False
                        write_status('training',variant=variant,worker_pid=child.pid,startup_verified=False)
                        try:
                            while child.poll() is None:
                                if not reviewed and verify_startup(run,variant):
                                    reviewed=True
                                    write_status('training',variant=variant,worker_pid=child.pid,startup_verified=True)
                                time.sleep(30)
                            result=child.wait()
                            if result: raise RuntimeError(f'{variant} exited with status{result}')
                            verify_finished(run,variant)
                        except BaseException:
                            stop_child(child);raise
                write_status('complete',completed=['mdm',A],next_training_launched=False)
        except BaseException as error:
            write_status('failed_requires_review',error=str(error));raise


if __name__=='__main__': main()
