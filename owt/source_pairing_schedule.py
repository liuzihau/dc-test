"""CPU-only, explicitly reviewed source-policy trial under the shared GPU lock."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from owt.research import ROOT, atomic_write

VARIANTS = ('mdm_np_zero_init_masked_source', 'mdm_np_zero_init_pair_count_control')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def verify_authorization(protocol_path, review_path, variant):
    protocol = load(protocol_path)
    if variant not in VARIANTS or protocol.get('variants') != list(VARIANTS):
        raise ValueError('Registered zero-initialized source-policy variants required')
    for filename, expected in protocol['source_sha256'].items():
        if digest(ROOT/filename) != expected:
            raise ValueError('Registered source changed: '+filename)
    review = load(review_path)
    if (review.get('scientific_review_complete') is not True
            or review.get('source_protocol_sha256') != digest(protocol_path)
            or variant not in review.get('selected_variants', [])):
        raise ValueError('Final scientific review has not selected this source-policy arm')
    required = protocol['required_final_evidence']
    for filename in required + [review['scientific_memo']]:
        if digest(ROOT/filename) != review.get('evidence_sha256', {}).get(filename):
            raise ValueError('Missing or changed reviewed final evidence: '+filename)
    completion = load(ROOT/required[0])
    collection, report = (load(ROOT/filename) for filename in required[1:3])
    low_protocol = load(ROOT/protocol['low_weight_protocol'])
    low_digest = digest(ROOT/protocol['low_weight_protocol'])
    if completion.get('optimizer_step') != 5000:
        raise ValueError('Existing lower-weight trial has not finished 5000 steps')
    for artifact in (collection, report):
        if (artifact.get('preflight') or artifact.get('protocol_sha256') != low_digest
                or artifact.get('row_ids') != low_protocol['row_ids']
                or len(artifact.get('cells', [])) != 5):
            raise ValueError('Registered real final five-cell diagnostics are incomplete')
    if (report.get('fitting_performed') is not False or not report.get('training_evidence')
            or report.get('fixed_lambda') != low_protocol['fixed_fusion']['lambda_weight']
            or report.get('scoring_row_ids') != low_protocol['fixed_fusion']['scoring_row_ids']):
        raise ValueError('Final primary/fixed-readout comparison is incomplete')
    return protocol


def entrypoint_authorization():
    """Run before importing Torch, including Lightning's spawned second rank."""
    names = ('NP_SOURCE_PROTOCOL', 'NP_SOURCE_REVIEW', 'NP_SOURCE_VARIANT')
    if not all(os.environ.get(name) for name in names):
        raise RuntimeError('Use the reviewed source_pairing_schedule; direct launch is disabled')
    return verify_authorization(*(os.environ[name] for name in names))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    args = parser.parse_args()
    protocol = verify_authorization(args.protocol, args.review, args.variant)
    root = ROOT/protocol['run_root']
    run = root/args.variant
    with (root/'.queue.lock').open('a') as lock:
        # Busy means defer, never start a second trainer or an unattended waiter.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_authorization(args.protocol, args.review, args.variant)
        complete = run/'complete.json'
        if complete.exists():
            if load(complete).get('optimizer_step') != 5000:
                raise ValueError('Unexpected prior completion; inspect before continuing')
            return
        run.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', '-m', 'owt.source_pairing_entrypoint',
                   '--variant', args.variant, '--run', str(run)]
        resume = run/'checkpoints/last.ckpt'
        if resume.exists():
            command += ['--resume', str(resume)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='2,3', OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   NP_SOURCE_PROTOCOL=str(args.protocol.resolve()),
                   NP_SOURCE_REVIEW=str(args.review.resolve()), NP_SOURCE_VARIANT=args.variant)
        with (run/'train.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            atomic_write(root/'source_pairing_queue.json', json.dumps(dict(
                stage='training', variant=args.variant, controller_pid=os.getpid(), worker_pid=child.pid))+'\n')
            result = child.wait()
        if result or not complete.exists() or load(complete).get('optimizer_step') != 5000:
            raise RuntimeError('Source-policy training failed or did not finish exactly 5000 steps')
        atomic_write(root/'source_pairing_queue.json', json.dumps(dict(
            stage='complete_waiting_for_scientific_review', variant=args.variant))+'\n')


if __name__ == '__main__':
    main()
