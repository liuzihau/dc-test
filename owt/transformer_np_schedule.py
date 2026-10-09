"""Launch a selected transformer-NP trial serially under the existing GPU lock."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from owt.research import ROOT, atomic_write

VARIANTS = ('mdm_np_zero_init_transformer_target_only',
            'mdm_np_zero_init_transformer_masked_source')
RUN_ROOT = Path('outputs/owt/transformer-np-5k')
LOCK = Path('outputs/owt/mdm-np-5k/.queue.lock')
REQUIRED_EVIDENCE = (
    'outputs/owt/mdm-np-5k/mdm_np_zero_init_masked_source/complete.json',
    'outputs/analysis/mdm_np_zero_init_masked_source-comparison-5000/summary.json',
    'outputs/owt/mdm-np-5k/mdm_np_zero_init_pair_count_control/complete.json',
    'outputs/analysis/mdm_np_zero_init_pair_count_control-comparison-5000/summary.json',
)
MASKED_REVIEW = 'outputs/research-notes/masked_source_final_review_20261003.json'
IMPLEMENTATION_FILES = (
    'owt/transformer_np.py', 'owt/transformer_np_model.py',
    'owt/transformer_np_metrics.py', 'owt/transformer_np_entrypoint.py',
    'owt/transformer_np_schedule.py', 'owt/test_transformer_np.py',
    'owt/test_transformer_np_schedule.py',
    'owt/transformer_np_preflight.py',
    *('owt/configs/' + variant + '.yaml' for variant in VARIANTS),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def required_evidence(selection, variant, weight):
    order = selection.get('experiment_order', 'count_control_first')
    if order == 'count_control_first':
        return REQUIRED_EVIDENCE
    if order != 'transformer_first':
        raise ValueError('Unknown experiment order')
    if (selection.get('count_control_deferred_by_user') is not True
            or not isinstance(selection.get('user_instruction'), str)
            or not selection['user_instruction'].strip()):
        raise ValueError('Transformer-first order requires the recorded user instruction')
    if (variant != VARIANTS[1] or weight != .25
            or selection.get('linear_reference') != 'mdm_np_zero_init_masked_source'):
        raise ValueError('Transformer-first pilot must match the reviewed both-masked0.25 linear reference')
    return (*REQUIRED_EVIDENCE[:2], MASKED_REVIEW)


def verify_selection(path, variant):
    selection = json.loads(Path(path).read_text())
    if (selection.get('execution_ready') is not True
            or selection.get('scientific_review_complete') is not True
            or selection.get('selected_variant') != variant or variant not in VARIANTS):
        raise ValueError('Transformer design is prepared; final weight/condition selection is pending')
    weight = selection.get('np_weight_per_direction')
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError('Select a positive finite NP weight per direction')
    if selection.get('run_root') != str(RUN_ROOT):
        raise ValueError('Use the isolated transformer NP output directory')
    pins = selection.get('source_sha256', {})
    base_pins = json.loads((ROOT / 'outputs/research-notes/source_pairing_diagnostic_protocol_20261002.json').read_text())['source_sha256']
    required_pins = set(IMPLEMENTATION_FILES) | set(base_pins)
    if not required_pins.issubset(pins):
        raise ValueError('Selection must pin the implementation and existing training/diagnostic dependencies')
    for filename in required_pins:
        if digest(ROOT / filename) != pins[filename]:
            raise ValueError('Selected implementation changed: ' + filename)
    for filename in required_evidence(selection, variant, weight):
        expected = selection.get('evidence_sha256', {}).get(filename)
        artifact = ROOT / filename
        if not expected or not artifact.is_file() or digest(artifact) != expected:
            raise ValueError('Pending or changed reviewed condition evidence: ' + filename)
        evidence = json.loads(artifact.read_text())
        if filename == MASKED_REVIEW:
            report = ROOT / REQUIRED_EVIDENCE[1]
            if (evidence.get('scientific_review_complete') is not True
                    or evidence.get('variant') != 'mdm_np_zero_init_masked_source'
                    or evidence.get('report_sha256') != digest(report)
                    or evidence.get('primary_validation', {}).get(
                        'mdm_np_zero_init_masked_source', {}).get('optimizer_step') != 5000):
                raise ValueError('Require the scientific review of the matched linear reference')
        elif filename.endswith('complete.json'):
            if evidence.get('optimizer_step') != 5000:
                raise ValueError('Condition trial has not completed 5000 steps')
        elif evidence.get('preflight') is not False or len(evidence.get('cells', [])) != 5:
            raise ValueError('Require the real five-condition final diagnostic report')
    return selection


def entrypoint_authorization():
    path, variant = os.environ.get('NP_TRANSFORMER_SELECTION'), os.environ.get('NP_TRANSFORMER_VARIANT')
    if not path or not variant:
        raise RuntimeError('Use transformer_np_schedule after final loss-condition selection')
    return verify_selection(path, variant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    args = parser.parse_args()
    selection = verify_selection(args.selection, args.variant)
    with (ROOT / LOCK).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_selection(args.selection, args.variant)
        run = ROOT / RUN_ROOT / args.variant
        if (run / 'complete.json').exists():
            if json.loads((run / 'complete.json').read_text()).get('optimizer_step') != 5000:
                raise ValueError('Unexpected prior completion')
            return
        resume = run / 'checkpoints/last.ckpt'
        if run.exists() and any(run.iterdir()) and not resume.is_file():
            raise ValueError('Partial run without a checkpoint; inspect before restarting')
        run.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', '-m', 'owt.transformer_np_entrypoint',
                   '--variant', args.variant, '--run', str(run),
                   '--np-weight', str(selection['np_weight_per_direction'])]
        if resume.is_file():
            command += ['--resume', str(resume)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='2,3', OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   NP_TRANSFORMER_SELECTION=str(args.selection.resolve()), NP_TRANSFORMER_VARIANT=args.variant)
        with (run / 'train.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            atomic_write(ROOT / RUN_ROOT / 'queue.json', json.dumps(dict(
                stage='training', variant=args.variant, controller_pid=os.getpid(), worker_pid=child.pid)) + '\n')
            result = child.wait()
        if result or not (run / 'complete.json').is_file() or json.loads((run / 'complete.json').read_text()).get('optimizer_step') != 5000:
            raise RuntimeError('Transformer NP failed or did not finish exactly 5000 steps')
        atomic_write(ROOT / RUN_ROOT / 'queue.json', json.dumps(dict(
            stage='complete_waiting_for_scientific_review', variant=args.variant)) + '\n')


if __name__ == '__main__':
    main()
