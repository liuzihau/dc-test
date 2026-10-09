"""Bounded overnight public-encoding diagnostic; never interrupts another job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .runner import atomic_json


ROOT = Path(__file__).resolve().parents[1]


def predecessor_complete(path):
    try:
        status = json.loads((path / 'status.json').read_text())
        generation = json.loads((path / 'generation/test-step-000020013.json').read_text())
        audit = json.loads((path / 'clue_audit.json').read_text())
    except (OSError, ValueError):
        return False
    return (status.get('status') in ('paused', 'finished') and status['step'] == 20013
            and generation['step'] == audit['step'] == 20013)


def gpu_idle(gpu):
    # Inspect the physical UUID, independent of CUDA_VISIBLE_DEVICES.
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader,nounits'], text=True)
    return not any(line.split(',')[0].strip() == uuid for line in processes.splitlines())


def sanity_passed(run):
    try:
        path = max((run / 'sanity').glob('step-*.json'))
        result = json.loads(path.read_text())
        return (result['split'] == 'training_memorization_diagnostic'
                and result['puzzles'] == 32 and result['greedy_rollout_exact'] >= .9
                and result['greedy_rollout_pad_fraction'] == 0.)
    except (OSError, ValueError, KeyError):
        return False


def train_command(encoding, run, stop, *, smoke=False, overfit=False):
    result = [sys.executable, '-u', '-m', 'reasoning.tfw_runner',
        '--data-dir', str(ROOT / '.cache/reasoning/zebra-benchmark-full-v1'),
        '--run-dir', str(run), '--global-batch', '128', '--micro-batch', '32',
        '--lr', '0.0003', '--schedule-epochs', '300', '--logit-shift', '0',
        '--target-region', 'answer', '--padding-attention', 'masked',
        '--zebra-encoding', encoding, '--full-mask-probability', '0',
        '--stop-after-steps', str(stop), '--precision', 'bf16',
        '--val-every', '500', '--save-every', '500', '--save-seconds', '1200',
        '--log-every', '1' if smoke else '10',
        '--validation-examples', '4' if smoke else '128' if overfit else '1000',
        '--eval-batch-size', '32', '--generation-examples', '1000',
        '--generation-every', '0' if smoke or overfit else '5000']
    if overfit:
        result += ['--overfit-examples', '32', '--sanity-every', '500']
    elif not smoke:
        result += ['--final-generation']
    return result


def run(args):
    if args.gpu not in (2, 3):
        raise ValueError('This overnight queue is restricted to physical GPUs 2 and 3')
    run = ROOT / 'outputs/reasoning' / f'zebra-encoding-{args.encoding}-gpu{args.gpu}'
    runtime = ROOT / '.cache/runtime/reasoning-zebra-encoding' / f'gpu{args.gpu}'
    runtime.mkdir(parents=True, exist_ok=True)
    lock = (runtime / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = runtime / 'queue_status.json'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4')
    for key, leaf in dict(TMPDIR='tmp', TMP='tmp', TEMP='tmp', MPLCONFIGDIR='matplotlib',
                          CUDA_CACHE_PATH='cuda', TORCHINDUCTOR_CACHE_DIR='inductor', TRITON_CACHE_DIR='triton').items():
        path = runtime / leaf
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['XDG_CACHE_HOME'] = str(ROOT / '.cache')
    def record(status, **kwargs):
        payload = dict(status=status, gpu=args.gpu, encoding=args.encoding, pid=os.getpid(),
                       updated_ns=time.time_ns(), run_dir=str(run), **kwargs)
        atomic_json(state, payload)
        print(json.dumps(payload), flush=True)
    def command(argv, phase):
        record('running', phase=phase, command=argv)
        subprocess.run(argv, cwd=ROOT, env=env, check=True)
    try:
        deadline = time.monotonic() + 6 * 3600
        while True:
            complete = predecessor_complete(Path(args.predecessor))
            idle = gpu_idle(args.gpu)
            if complete and idle:
                break
            record('waiting', predecessor_complete=complete, gpu_idle=idle,
                   prerequisite=str(Path(args.predecessor).resolve()))
            if time.monotonic() >= deadline:
                raise TimeoutError('Predecessor/GPU not ready after six hours; no jobs interrupted')
            time.sleep(30)
        # Isolated real-model full-batch smoke, including save and resume.
        smoke = runtime / 'smoke'
        for step in (1, 2):
            command(train_command(args.encoding, smoke, step, smoke=True), f'smoke-{step}')
        # Memorization only: this is never reported as held-out reasoning.
        sanity = ROOT / 'outputs/reasoning' / f'zebra-encoding-{args.encoding}-overfit32-gpu{args.gpu}'
        command(train_command(args.encoding, sanity, 1500, overfit=True), 'overfit32-1500')
        if not sanity_passed(sanity):
            command(train_command(args.encoding, sanity, 3000, overfit=True), 'overfit32-3000')
        if not sanity_passed(sanity):
            raise RuntimeError('Tiny TRAIN memorization gate failed; refusing an expensive full-data run')
        command(train_command(args.encoding, run, 40026), 'fresh-full-data-six-epochs')
        command([sys.executable, '-m', 'reasoning.tfw_clue_audit', '--checkpoint', str(run/'checkpoints/last.pt'),
                 '--data-dir', str(ROOT/'.cache/reasoning/zebra-benchmark-full-v1'),
                 '--output', str(run/'clue_audit.json')], 'final-clue-audit')
        command([sys.executable, 'scripts/reasoning/report_zebra_encoding.py', '--gpu', str(args.gpu)], 'final-report')
        record('finished', step=40026)
    except BaseException as error:
        record('failed', error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--encoding', choices=('answer_relative', 'typed_coordinates'), required=True)
    parser.add_argument('--predecessor', required=True)
    run(parser.parse_args())
