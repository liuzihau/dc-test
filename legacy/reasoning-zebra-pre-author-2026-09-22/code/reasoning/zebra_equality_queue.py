"""Bounded corrected equality-only experiment; independent of flawed v1 runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .runner import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'.cache/reasoning/zebra-equality-control-v2'


def train_command(encoding, run, stop, smoke=False):
    return [sys.executable, '-u', '-m', 'reasoning.tfw_runner',
        '--data-dir', str(DATA), '--run-dir', str(run), '--zebra-encoding', encoding,
        '--logit-shift', '0', '--target-region', 'answer', '--padding-attention', 'masked',
        '--global-batch', '128', '--micro-batch', '32', '--lr', '.0003',
        '--schedule-epochs', '300', '--stop-after-steps', str(stop), '--val-every', '250',
        '--validation-examples', '6' if smoke else '128', '--save-every', '250',
        '--save-seconds', '1200', '--log-every', '10', '--generation-every', '0',
        '--no-epoch-end-generation', '--seed', '1']


def work(args):
    encoding = 'answer_relative' if args.gpu == 2 else 'typed_coordinates'
    run = ROOT/f'outputs/reasoning/zebra-equality-v2-{encoding}-gpu{args.gpu}'
    runtime = ROOT/f'.cache/runtime/reasoning-zebra-equality-v2/gpu{args.gpu}'
    runtime.mkdir(parents=True, exist_ok=True)
    if json.loads((DATA/'manifest.json').read_text())['source']['kind'] != 'synthetic_equality_control_v2':
        raise ValueError('Wrong equality-control dataset')
    status, started = runtime/'queue_status.json', time.time()
    def execute(stage, command):
        log = run/'console'/f'{time.time_ns()}-{stage}.log'; log.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(status, dict(status='running', stage=stage, gpu=args.gpu, encoding=encoding,
            run_dir=str(run), log=str(log), pid=os.getpid(), stop_step=1500, command=command))
        print(f'\nSTAGE {stage}: {log}', flush=True)
        with log.open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True); stream.write(line); stream.flush()
            code = process.wait()
        if code: raise RuntimeError(f'{stage} failed with exit {code}; see {log}')
    try:
        smoke = runtime/'smoke'
        execute('smoke-one', train_command(encoding, smoke, 1, True))
        execute('smoke-resume-two', train_command(encoding, smoke, 2, True))
        execute('smoke-evaluate', [sys.executable, '-u', '-m', 'reasoning.zebra_binding',
            'evaluate', '--data-dir', str(DATA), '--checkpoint', str(smoke/'checkpoints/last.pt'),
            '--split', 'validation', '--output', str(smoke/'binding/validation-000000002.json')])
        for stop in (250, 750, 1500):
            splits = ('validation', 'test') if stop == 1500 else ('validation',)
            reports = [run/'binding'/f'{split}-{stop:09d}.json' for split in splits]
            if all(path.exists() for path in reports):
                for split, path in zip(splits, reports):
                    saved = json.loads(path.read_text())
                    if (saved['step'] != stop or saved['split'] != split
                            or saved['data_sha256'] != digest(DATA/'manifest.json')
                            or saved['model_config']['zebra_encoding'] != encoding):
                        raise ValueError('Existing equality stage provenance differs')
                continue
            last = run/'checkpoints/last.pt'
            if last.exists() and json.loads(last.resolve().with_suffix('.pt.json').read_text())['step'] > stop:
                raise ValueError('Cannot replace a missing historical report with later weights')
            execute(f'train-to-{stop}', train_command(encoding, run, stop))
            for split in splits:
                execute(f'{split}-{stop}', [sys.executable, '-u', '-m', 'reasoning.zebra_binding',
                    'evaluate', '--data-dir', str(DATA), '--checkpoint', str(run/'checkpoints/last.pt'),
                    '--split', split, '--output', str(run/'binding'/f'{split}-{stop:09d}.json')])
        atomic_json(status, dict(status='finished', gpu=args.gpu, encoding=encoding,
            run_dir=str(run), stop_step=1500, elapsed_seconds=time.time()-started))
        print('FINISHED: corrected equality control; no further work queued.', flush=True)
    except Exception as error:
        atomic_json(status, dict(status='failed', gpu=args.gpu, encoding=encoding,
            run_dir=str(run), error=str(error), elapsed_seconds=time.time()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(2, 3), required=True)
    work(parser.parse_args())
