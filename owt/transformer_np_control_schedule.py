"""Execute only the user-selected B trial after A's final scientific review."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from owt.research import ROOT, atomic_write
from owt.transformer_np_schedule import digest

A = 'mdm_np_zero_init_transformer_masked_source'
B = 'mdm_np_zero_init_transformer_pair_count_control'
VARIANTS = (B,)
RUN_ROOT = Path('outputs/owt/transformer-np-5k')
LOCK = Path('outputs/owt/mdm-np-5k/.queue.lock')
REVIEW = 'outputs/research-notes/transformer_np_A_final_review_20261004.json'
IMPLEMENTATION_FILES = ('owt/transformer_np_control.py', 'owt/transformer_np_control_entrypoint.py',
    'owt/transformer_np_control_schedule.py', 'owt/test_transformer_np_control.py',
    'owt/transformer_np_control_preflight.py', 'owt/transformer_np_control_followup.py', 'owt/configs/'+B+'.yaml')


def verify_selection(path, variant=B):
    selection = json.loads(Path(path).read_text())
    if (variant != B or selection.get('selected_variant') != B
            or selection.get('execution_ready') is not True
            or selection.get('scientific_review_complete') is not True
            or selection.get('np_weight_per_direction') != .25
            or selection.get('source_policy') != 'matched_pair_count'
            or selection.get('optimizer_steps') != 5000
            or selection.get('run_root') != str(RUN_ROOT)
            or selection.get('user_instruction') != 'do B first and during that we can derive some idea'):
        raise ValueError('Require the reviewed user-selected B5000 configuration')
    original = json.loads((ROOT / 'outputs/research-notes/np_transformer_launch_selection_20261003.json').read_text())
    pins = selection.get('source_sha256', {})
    if not (set(original['source_sha256']) | set(IMPLEMENTATION_FILES)).issubset(pins):
        raise ValueError('B selection must preserve the A implementation pins')
    if any(pins[name] != expected for name, expected in original['source_sha256'].items()):
        raise ValueError('B cannot replace the frozen A dependency pins')
    for name, expected in pins.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Selected B dependency changed: ' + name)
    for name, expected in selection.get('evidence_sha256', {}).items():
        if digest(ROOT / name) != expected:
            raise ValueError('Reviewed B evidence changed: ' + name)
    review = json.loads((ROOT / REVIEW).read_text())
    if (REVIEW not in selection.get('evidence_sha256', {})
            or review.get('scientific_review_complete') is not True or review.get('variant') != A
            or review.get('optimizer_step') != 5000 or len(review.get('cells', [])) != 5
            or review.get('report_sha256') != digest(ROOT / review['report'])
            or review.get('collection_sha256') != digest(ROOT / review['collection'])):
        raise ValueError('Require A final native scientific review before B')
    complete = json.loads((ROOT / RUN_ROOT / A / 'complete.json').read_text())
    if complete.get('optimizer_step') != 5000:
        raise ValueError('A has not completed5000')
    return selection


def entrypoint_authorization():
    path = os.environ.get('NP_TRANSFORMER_CONTROL_SELECTION')
    variant = os.environ.get('NP_TRANSFORMER_CONTROL_VARIANT')
    if not path or variant != B:
        raise RuntimeError('Use transformer_np_control_schedule with the reviewed B selection')
    return verify_selection(path, variant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    args = parser.parse_args(); selection = verify_selection(args.selection, args.variant)
    with (ROOT / LOCK).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_selection(args.selection, args.variant)
        run = ROOT / RUN_ROOT / B
        if (run / 'complete.json').exists():
            if json.loads((run / 'complete.json').read_text()).get('optimizer_step') != 5000:
                raise ValueError('Unexpected B completion')
            return
        resume = run / 'checkpoints/last.ckpt'
        if run.exists() and any(run.iterdir()) and not resume.is_file():
            raise ValueError('Partial B without checkpoint; inspect before restarting')
        run.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', '-m', 'owt.transformer_np_control_entrypoint',
            '--variant', B, '--run', str(run), '--np-weight', '.25']
        if resume.is_file(): command += ['--resume', str(resume)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='2,3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2', NP_TRANSFORMER_CONTROL_SELECTION=str(args.selection.resolve()),
            NP_TRANSFORMER_CONTROL_VARIANT=B)
        queue = ROOT / RUN_ROOT / 'control_queue.json'
        with (run / 'train.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            atomic_write(queue, json.dumps(dict(stage='training', variant=B,
                controller_pid=os.getpid(), worker_pid=child.pid))+'\n')
            result = child.wait()
        if result or not (run / 'complete.json').is_file() or json.loads((run / 'complete.json').read_text()).get('optimizer_step') != 5000:
            atomic_write(queue, json.dumps(dict(stage='failed_requires_review', variant=B, exit_code=result))+'\n')
            raise RuntimeError('B did not finish exactly5000')
        atomic_write(queue, json.dumps(dict(stage='complete_waiting_for_scientific_review', variant=B))+'\n')


if __name__ == '__main__':
    main()
