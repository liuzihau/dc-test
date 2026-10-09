"""Queue only A+distance-two after B's already planned diagnostics finish."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from owt.research import ROOT, atomic_write, read_json, read_csv, timestamp
from owt.transformer_np_control_schedule import A, B, verify_selection as verify_b_selection, digest

VARIANT = 'mdm_np_zero_init_transformer_distance2_masked_source'
VARIANTS = (VARIANT,)
RUN_ROOT = Path('outputs/owt/transformer-np-5k')
LOCK = Path('outputs/owt/mdm-np-5k/.queue.lock')
FILES = ('owt/transformer_np_distance.py', 'owt/transformer_np_distance_metrics.py',
    'owt/transformer_np_distance_entrypoint.py', 'owt/transformer_np_distance_schedule.py',
    'owt/transformer_np_distance_preflight.py', 'owt/test_transformer_np_distance.py',
    'owt/configs/'+VARIANT+'.yaml')
QUEUE = RUN_ROOT / 'distance2_queue.json'


def verify_selection(path, variant=VARIANT):
    selection = read_json(path) or {}
    if (variant != VARIANT or selection.get('selected_variant') != VARIANT
            or selection.get('execution_ready') is not True or selection.get('optimizer_steps') != 5000
            or selection.get('near_weight_per_direction') != .25
            or selection.get('far_weight_per_direction') != .1
            or selection.get('offsets') != [-1,1,-2,2] or selection.get('source_policy') != 'masked_source'
            or selection.get('run_root') != str(RUN_ROOT)
            or selection.get('wait_for_B_final_diagnostics') is not True
            or not selection.get('user_instruction')):
        raise ValueError('Require the user-selected A+distance-two design')
    b_selection = verify_b_selection(ROOT / 'outputs/research-notes/transformer_np_control_selection_20261004.json')
    pins = selection.get('source_sha256', {})
    if not (set(b_selection['source_sha256']) | set(FILES)).issubset(pins):
        raise ValueError('Pin the isolated extension and preserve all A/B dependencies')
    if any(pins[name] != expected for name,expected in b_selection['source_sha256'].items()):
        raise ValueError('The extension cannot replace A/B source pins')
    for name, expected in pins.items():
        if digest(ROOT / name) != expected: raise ValueError('Selected extension changed: '+name)
    for name, expected in selection.get('evidence_sha256', {}).items():
        if digest(ROOT / name) != expected: raise ValueError('Selected evidence changed: '+name)
    return selection


def b_ready():
    """B outcome does not select this recipe; completion only serializes jobs."""
    if (read_json(ROOT / RUN_ROOT / B / 'complete.json') or {}).get('optimizer_step') != 5000:
        return False
    status = read_json(ROOT / RUN_ROOT / 'control_followup.json') or {}
    if status.get('stage') != 'B_final_report_ready_scientific_review_pending': return False
    protocol_path = ROOT / 'outputs/research-notes/transformer_np_diagnostic_protocol_20261004.json'
    protocol = read_json(protocol_path)
    collection = ROOT / protocol['outputs'][B]['collection'] / 'summary.json'
    report = read_json(ROOT / protocol['outputs'][B]['report'] / 'summary.json') or {}
    if (report.get('preflight') is not False or report.get('variant') != B
            or report.get('protocol_sha256') != digest(protocol_path)
            or report.get('collection_sha256') != digest(collection)
            or report.get('row_ids') != list(range(1024))
            or [c.get('cell') for c in report.get('cells', [])] != protocol['conditions']
            or report.get('training_evidence', {}).get('finite_updates') != 5000
            or report.get('paired_source_arm_available') is not True):
        raise ValueError('B final native report failed completion verification')
    return True


def entrypoint_authorization():
    path = os.environ.get('NP_TRANSFORMER_DISTANCE_SELECTION')
    if not path or os.environ.get('NP_TRANSFORMER_DISTANCE_VARIANT') != VARIANT:
        raise RuntimeError('Use transformer_np_distance_schedule for this queued extension')
    selection = verify_selection(path)
    if not b_ready(): raise RuntimeError('B final diagnostics must finish before this trial starts')
    return selection


def verify_startup(run, selection_path):
    trajectories = {name:read_csv(run/'local_metrics'/name)[:3]
        for name in ('train.csv','gradient_norms.csv','source_pairs.csv')}
    if any(len(rows)<3 for rows in trajectories.values()): return None
    if any([r['optimizer_step'] for r in rows] != [1,2,3] for rows in trajectories.values()):
        raise ValueError('Fresh aligned startup required')
    reference = ROOT / RUN_ROOT / A
    if trajectories['train.csv'][0]['main_elbo'] != read_csv(reference/'local_metrics/train.csv')[0]['main_elbo']:
        raise ValueError('Initial main prediction differs from A')
    for row in trajectories['train.csv']:
        expected = row['main_elbo'] + .25*(row['np_prev']+row['np_next']) + .1*(row['np_prev2']+row['np_next2'])
        if abs(row['objective']-expected)>2e-5: raise ValueError('Four-head objective does not reconstruct')
    for row in trajectories['gradient_norms.csv']:
        norm = sum(row[k]**2 for k in ('shared_trunk_l2','main_readout_l2','neighbor_readouts_l2','neighbor_processing_l2'))**.5
        if not math.isclose(norm,row['joint_l2'],rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError('Joint gradient partition differs')
    a_pairs = read_csv(reference/'local_metrics/source_pairs.csv')[:3]
    for old,new in zip(a_pairs,trajectories['source_pairs.csv']):
        for direction in ('prev','next'):
            for b in range(5):
                key=f'{direction}_maskbin{b}_'
                for field in ('eligible','masked_source','selected','selected_masked'):
                    if old[key+field]!=new[key+field]: raise ValueError('Near pair counts differ from A')
                for field in ('eligible_weight_mass','selected_weight_mass','masked_weight_mass'):
                    if abs(old[key+field]-new[key+field]) > .001 + 2e-6*abs(old[key+field]):
                        raise ValueError('Near weight mass differs from A')
        for direction in ('prev','next','prev2','next2'):
            for b in range(5):
                key=f'{direction}_maskbin{b}_'
                if not new[key+'selected']==new[key+'selected_masked']==new[key+'masked_source']:
                    raise ValueError('A distance head used a revealed endpoint')
    receipt=dict(verified_at=timestamp(),variant=VARIANT,steps_checked=3,initial_main_loss_matches_A=True,
        near_pair_counts_and_weight_mass_match_A=True,all_four_heads_both_masked=True,
        objective_and_gradient_partition_verified=True,peak_allocated_gib=max(r['peak_allocated_gib'] for r in trajectories['train.csv']),
        selection_sha256=digest(selection_path),trajectories=trajectories,startup_is_mechanics_not_efficacy=True)
    atomic_write(run/'startup_review.json',json.dumps(receipt,indent=2)+'\n')
    return receipt


def write_status(stage, **details):
    atomic_write(ROOT / QUEUE, json.dumps(dict(stage=stage,variant=VARIANT,
        updated_at=timestamp(),controller_pid=os.getpid(),**details),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    args=parser.parse_args(); selection=verify_selection(args.selection,args.variant)
    (ROOT / RUN_ROOT).mkdir(parents=True,exist_ok=True)
    # Prevent duplicate controllers without taking B's GPU lock while waiting.
    with (ROOT / RUN_ROOT / '.distance2-controller.lock').open('a') as controller:
        fcntl.flock(controller,fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_status('queued_waiting_for_B_final_diagnostics',selection=str(args.selection.resolve()),
            near_weight=.25,far_weight=.1,steps=5000,physical_gpus=[2,3])
        while not b_ready():
            q=read_json(ROOT / RUN_ROOT / 'control_queue.json') or {}
            f=read_json(ROOT / RUN_ROOT / 'control_followup.json') or {}
            if q.get('stage')=='failed_requires_review' or f.get('stage')=='training_failed_requires_review':
                write_status('blocked_B_failed_requires_review'); raise RuntimeError('B failed; preserve queued extension')
            if f.get('pid') and f.get('stage') not in ('B_final_report_ready_scientific_review_pending',):
                try: os.kill(int(f['pid']),0)
                except ProcessLookupError:
                    write_status('blocked_B_follower_stopped_requires_review'); raise RuntimeError('B diagnostics follower stopped')
            time.sleep(30)
        verify_selection(args.selection,args.variant)
        with (ROOT / LOCK).open('a') as lock:
            while True:
                try: fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError: write_status('queued_waiting_for_GPU_lock'); time.sleep(30)
            verify_selection(args.selection,args.variant)
            if not b_ready(): raise RuntimeError('B completion evidence changed')
            run=ROOT / RUN_ROOT / VARIANT
            if (read_json(run/'complete.json') or {}).get('optimizer_step')==5000:
                write_status('complete_waiting_for_scientific_review'); return
            resume=run/'checkpoints/last.ckpt'
            if run.exists() and any(run.iterdir()) and not resume.is_file():
                raise RuntimeError('Partial extension without checkpoint; inspect before restarting')
            run.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-u','-m','owt.transformer_np_distance_entrypoint','--variant',VARIANT,
                '--run',str(run),'--np-weight','.25','--far-weight','.1']
            if resume.is_file(): command+=['--resume',str(resume)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                NP_TRANSFORMER_DISTANCE_SELECTION=str(args.selection.resolve()),NP_TRANSFORMER_DISTANCE_VARIANT=VARIANT)
            with (run/'train.log').open('a') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                reviewed=(run/'startup_review.json').exists()
                write_status('training',worker_pid=child.pid,startup_verified=reviewed)
                try:
                    while child.poll() is None:
                        if not reviewed and verify_startup(run,args.selection):
                            reviewed=True; write_status('training',worker_pid=child.pid,startup_verified=True)
                        time.sleep(30)
                except Exception:
                    os.killpg(child.pid,signal.SIGTERM)
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL); child.wait(timeout=30)
                    write_status('failed_startup_requires_review'); raise
                result=child.wait()
            if result or (read_json(run/'complete.json') or {}).get('optimizer_step')!=5000:
                write_status('failed_requires_review',exit_code=result); raise RuntimeError('Extension failed; no fallback trial started')
            write_status('complete_waiting_for_scientific_review',next_training_launched=False)


if __name__=='__main__': main()
