"""Bounded single-GPU clue-binding diagnostic, using the existing MDM trainer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .runner import atomic_json, digest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '.cache/reasoning/zebra-clue-binding-v1'


def train_command(encoding, run, stop, smoke=False):
    return [sys.executable, '-u', '-m', 'reasoning.tfw_runner',
        '--data-dir', str(DATA), '--run-dir', str(run),
        '--zebra-encoding', encoding, '--logit-shift', '0',
        '--target-region', 'answer', '--padding-attention', 'masked',
        '--global-batch', '128', '--micro-batch', '32', '--lr', '.0003',
        '--schedule-epochs', '300', '--stop-after-steps', str(stop),
        '--val-every', '500', '--validation-examples', '8' if smoke else '128',
        '--save-every', '500', '--save-seconds', '1200', '--log-every', '10',
        '--generation-every', '0', '--no-epoch-end-generation', '--seed', '1']


def work(args):
    encoding = 'answer_relative' if args.gpu == 2 else 'typed_coordinates'
    run = ROOT / f'outputs/reasoning/zebra-binding-{encoding}-gpu{args.gpu}'
    runtime = ROOT / '.cache/runtime/reasoning-zebra-binding' / f'gpu{args.gpu}'
    runtime.mkdir(parents=True, exist_ok=True)
    if json.loads((DATA/'manifest.json').read_text())['source']['kind'] != 'synthetic_clue_binding_v1':
        raise ValueError('Wrong dataset selected')
    started = time.time()
    status_path = runtime/'queue_status.json'
    def execute(stage, command):
        path = run/'console'/f'{time.time_ns()}-{stage}.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(status_path, dict(status='running', stage=stage, gpu=args.gpu,
            encoding=encoding, pid=os.getpid(), run_dir=str(run), log=str(path), started=started,
            stop_step=3000, command=command))
        print(f'\nSTAGE {stage}: {path}', flush=True)
        with path.open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True); stream.write(line); stream.flush()
            code = process.wait()
        if code:
            raise RuntimeError(f'{stage} failed with exit {code}; see {path}')
    try:
        smoke = runtime/'smoke'
        execute('smoke-one', train_command(encoding, smoke, 1, True))
        execute('smoke-resume-two', train_command(encoding, smoke, 2, True))
        execute('smoke-evaluation', [sys.executable, '-u', '-m', 'reasoning.zebra_binding',
            'evaluate', '--data-dir', str(DATA), '--checkpoint', str(smoke/'checkpoints/last.pt'),
            '--split', 'validation', '--output', str(smoke/'binding/validation-000000002.json')])
        # These run directories differ: tiny smoke weights NEVER initialize training.
        for stop in (500, 1500, 3000):
            splits = ('validation', 'test') if stop == 3000 else ('validation',)
            reports = [run/'binding'/f'{split}-{stop:09d}.json' for split in splits]
            complete = all(path.exists() for path in reports)
            if complete:
                for split, path in zip(splits, reports):
                    saved = json.loads(path.read_text())
                    if (saved['step'] != stop or saved['split'] != split
                            or saved['data_sha256'] != digest(DATA/'manifest.json')
                            or saved['model_config']['zebra_encoding'] != encoding):
                        raise ValueError('Completed stage provenance differs')
                print(f'Stage {stop} already evaluated; preserving historical reports.', flush=True)
                continue
            last = run/'checkpoints/last.pt'
            if last.exists():
                receipt = json.loads(last.resolve().with_suffix('.pt.json').read_text())
                if receipt['step'] > stop:
                    raise ValueError('Missing historical stage report; cannot substitute a later checkpoint')
            execute(f'train-to-{stop}', train_command(encoding, run, stop))
            for split in splits:
                execute(f'{split}-{stop}', [sys.executable, '-u', '-m', 'reasoning.zebra_binding',
                    'evaluate', '--data-dir', str(DATA), '--checkpoint', str(run/'checkpoints/last.pt'),
                    '--split', split, '--output', str(run/'binding'/f'{split}-{stop:09d}.json')])
            execute(f'plot-{stop}', [sys.executable, '-u', 'scripts/reasoning/report_zebra_binding.py',
                '--encoding', encoding])
        atomic_json(status_path, dict(status='finished', gpu=args.gpu, encoding=encoding,
            run_dir=str(run), stop_step=3000, elapsed_seconds=time.time()-started))
        print('FINISHED: bounded synthetic diagnostic, no additional training queued.', flush=True)
    except Exception as error:
        atomic_json(status_path, dict(status='failed', gpu=args.gpu, encoding=encoding,
            error=str(error), run_dir=str(run), elapsed_seconds=time.time()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(2, 3), required=True)
    work(parser.parse_args())
