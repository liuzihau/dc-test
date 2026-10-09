"""Bounded, restartable 5k->10k Zebra continuation; original runs are immutable.

This diagnoses undertraining on OUR synthetic pilot, not paper reproduction.
Five variants retain their own full-state optimizer/data/RNG contracts. No
test-set score controls training duration or hyperparameter selection.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from .baseline_queue import BaselineQueue, ROOT, CLI, atomic_json, digest, parser as base_parser

VARIANTS = ('vanilla', 'mdm', 'mdm_aux', 'both', 'both_aux')


def verified_checkpoint(directory, step=None):
    path = (Path(directory) / 'checkpoints/last.pt').resolve(strict=True)
    receipt = json.loads(path.with_suffix('.pt.json').read_text())
    if (receipt['file'] != path.name or receipt['size'] != path.stat().st_size
            or receipt['sha256'] != digest(path)
            or (step is not None and receipt['step'] != step)):
        raise ValueError('Wrong-step or corrupt checkpoint: ' + str(path))
    return path, receipt


def fork_run(source, destination, step=5000):
    """Copy a verified immutable checkpoint; never hardlink mutable run files."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('Continuation must not modify the original run')
    path, receipt = verified_checkpoint(source, step)
    contract = json.loads((source / 'contract.json').read_text())
    provenance = dict(version=1, source_run=str(source), source_checkpoint=str(path),
                      source_sha256=receipt['sha256'], source_step=step,
                      source_contract_sha256=digest(source / 'contract.json'),
                      full_state=True, scientific_change='additional optimizer updates only')
    if destination.exists():
        if (json.loads((destination / 'continuation_source.json').read_text()) != provenance
                or json.loads((destination / 'contract.json').read_text()) != contract):
            raise ValueError('Existing continuation provenance/contract differs')
        _, resumed = verified_checkpoint(destination)
        if resumed['step'] < step:
            raise ValueError('Continuation checkpoint predates its source')
        return provenance
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix='.' + destination.name + '.', dir=destination.parent))
    # Failed copies remain recoverable, never publish a half-built run directory.
    checkpoints = scratch / 'checkpoints'
    checkpoints.mkdir()
    shutil.copy2(path, checkpoints / path.name)
    shutil.copy2(path.with_suffix('.pt.json'), checkpoints / (path.name + '.json'))
    (checkpoints / 'last.pt').symlink_to(path.name)
    verified_checkpoint(scratch, step)
    atomic_json(scratch / 'contract.json', contract)
    atomic_json(scratch / 'continuation_source.json', provenance)
    scratch.rename(destination)
    return provenance


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('action', choices=('plan', 'run', 'status'))
    result.add_argument('--gpu', default='0')
    result.add_argument('--data-root', type=Path, default=ROOT / '.cache/reasoning')
    result.add_argument('--output-root', type=Path, default=ROOT / 'outputs/reasoning')
    result.add_argument('--queue-dir', type=Path,
                        default=ROOT / 'outputs/reasoning/queues/zebra-continuation-5k-10k-v1')
    result.add_argument('--hours', type=float, default=12.0,
                        help='Persistent wall-clock cap; includes downtime across restarts')
    result.add_argument('--audit-examples', type=int, default=128)
    result.add_argument('--official-data-dir', type=Path,
                        help='Optional separately prepared source-data-aligned Zebra set; run a fresh vanilla5k reference')
    result.add_argument('--official-train-file', type=Path)
    result.add_argument('--official-test-file', type=Path)
    return result


class ZebraContinuation:
    def __init__(self, args):
        if (not math.isfinite(args.hours) or args.hours <= 0
                or args.audit_examples < 1 or args.audit_examples > 1000):
            raise ValueError('Positive time cap and 1..1000 audit examples required')
        if bool(args.official_train_file) != bool(args.official_test_file):
            raise ValueError('Supply both official source files or neither')
        if args.official_train_file is not None and args.official_data_dir is None:
            raise ValueError('Source import requires a separate official-data-dir')
        self.args = args
        self.directory = args.queue_dir.resolve()
        self.queues = {}
        for suite in ('baselines', 'memory'):
            parsed = base_parser().parse_args([
                'run', '--suite', suite, '--max-steps', '10000', '--gpu', args.gpu,
                '--micro-batch', '128', '--global-batch', '128', '--seed', '1',
                '--data-root', str(args.data_root), '--output-root', str(args.output_root),
                '--queue-dir', str(self.directory)])
            self.queues[suite] = BaselineQueue(parsed)
        self.base = self.queues['baselines']
        self.env = self.base.env
        self.spec = dict(version=1, task='zebra', variants=list(VARIANTS),
                         source_step=5000, target_step=10000, round_steps=1000,
                         hours=args.hours, audit_examples=args.audit_examples,
                         data_root=str(args.data_root.resolve()),
                         output_root=str(args.output_root.resolve()), gpu=args.gpu,
                         training_change='none; full-state continuation',
                         selection='fixed budget; no test-driven stopping',
                         paper_equivalence=False)
        if args.official_data_dir is not None:
            self.spec['official_reference'] = dict(data_dir=str(args.official_data_dir.resolve()),
                                                  variant='vanilla', steps=5000,
                                                  task='zebra-official', exact_reproduction=False)
            if args.official_train_file is not None:
                self.spec['official_reference'].update(train_file=str(args.official_train_file.resolve()),
                                                       test_file=str(args.official_test_file.resolve()))

    def queue(self, variant):
        return self.queues['memory' if variant in ('both', 'both_aux') else 'baselines']

    def source(self, variant):
        return self.queue(variant).run_dir('zebra', variant)

    def destination(self, variant):
        source = self.source(variant)
        return source.with_name(source.name + '-continuation10k-v1')

    def status(self, state, **fields):
        atomic_json(self.directory / 'status.json', dict(status=state, pid=os.getpid(),
                    updated_ns=time.time_ns(), **fields))

    def execute(self, command, stage, gpu=False):
        remaining = self.deadline - time.time()
        if remaining <= 0:
            raise TimeoutError('Persistent wall-clock budget exhausted')
        if gpu:
            self.base.gpu_check()
        console = self.directory / 'console'
        console.mkdir(exist_ok=True)
        path = console / f'{time.time_ns()}-{stage}.log'
        self.status('running', stage=stage, command=command, console=str(path),
                    deadline_utc_seconds=self.deadline)
        print(f'[{stage}] log: {path}', flush=True)
        with path.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=self.env,
                                     stdout=stream, stderr=subprocess.STDOUT)
            try:
                code = child.wait(timeout=max(0.01, remaining))
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                raise
        if code:
            raise RuntimeError(f'{stage} exited {code}; see {path}')

    def audit(self, variant, directory, step):
        checkpoint, receipt = verified_checkpoint(directory, step)
        output = self.directory / 'audits' / f'{variant}-{step}-{receipt["sha256"][:12]}.json'
        if output.exists():
            # Results are atomic, provenance-bearing products, not GPU checkpoints.
            saved = json.loads(output.read_text())
            if (saved['checkpoint_sha256'] != receipt['sha256']
                    or saved['arguments']['examples'] != self.args.audit_examples
                    or saved['arguments']['seed'] != 2026
                    or saved['arguments']['batch_size'] != 8
                    or saved['arguments']['skip_generation']
                    or saved['data_sha256'] != digest(self.base.data_dir('zebra') / 'manifest.json')
                    or saved['step'] != step
                    or saved['protocol']['splits'] != ['train', 'validation']
                    or saved['protocol']['ratios'] != [1.0, 0.7, 0.3]
                    or any('permutation_shortcut' not in saved['splits'][s]['cold']
                           for s in ('train', 'validation'))):
                raise ValueError('Existing diagnostic checkpoint hash differs')
            return
        self.execute([sys.executable, str(ROOT / 'scripts/reasoning/audit_zebra.py'),
                      '--checkpoint', str(checkpoint), '--trust-checkpoint',
                      '--data-dir', str(self.base.data_dir('zebra')), '--output', str(output),
                      '--device', 'cuda', '--examples', str(self.args.audit_examples),
                      '--batch-size', '8', '--seed', '2026'], f'audit-{variant}-{step}', gpu=True)

    def report(self, final=False):
        runs = [str(self.destination(v)) for v in VARIANTS]
        if all((Path(run) / 'logs').exists() for run in runs):
            self.execute([sys.executable, str(CLI), 'plot', '--runs', *runs,
                          '--labels', *VARIANTS, '--train-metric', 'train/base_loss',
                          '--val-label', 'Cold validation NLL (not solving accuracy)',
                          '--output', str(self.directory / 'figures/zebra.png')], 'plot')
        if final:
            self.execute([sys.executable, str(ROOT / 'scripts/reasoning/summarize_reasoning.py'),
                          '--runs', *runs, '--output-dir', str(self.directory / 'report'),
                          '--step', '10000'], 'report-final')

    def audit_report(self):
        self.execute([sys.executable, str(ROOT / 'scripts/reasoning/summarize_zebra_audits.py'),
                      '--audit-dir', str(self.directory / 'audits'), '--output-dir',
                      str(self.directory / 'audit_report')], 'report-audits')

    def official_reference(self):
        """Fresh reference, separate data/model vocabulary; never mix with pilot."""
        data = self.args.official_data_dir
        if data is None:
            return
        if not (data / 'manifest.json').exists() and self.args.official_train_file is not None:
            self.execute([sys.executable, str(ROOT / 'scripts/reasoning/import_official_zebra.py'),
                          '--train-file', str(self.args.official_train_file),
                          '--test-file', str(self.args.official_test_file), '--output-dir', str(data),
                          '--train-size', '20000', '--valid-size', '1000', '--test-size', '1000',
                          '--seed', '17'], 'import-official-source')
        manifest = json.loads((data / 'manifest.json').read_text())
        if manifest.get('schema_version') != 1 or manifest['task'] != 'zebra-official':
            raise ValueError('Official reference requires separately normalized source data')
        if manifest.get('source', {}).get('kind') != 'shah_official_zebra_subset':
            raise ValueError('Official reference requires verified source-data provenance')
        for split in ('train', 'validation', 'test'):
            entry = manifest['splits'][split]
            if entry['filename'] != split + '.jsonl':
                raise ValueError('Unexpected official split filename')
            if digest(data / entry['filename']) != entry['sha256']:
                raise ValueError('Source-data split checksum mismatch')
        directory = self.args.output_root / 'zebra-official' / 'vanilla-source-v1-n20000-v1000-t1000-mb128-gb128-seed1'
        expected_counts = {'train': 20000, 'validation': 1000, 'test': 1000}
        if any(manifest['splits'][key]['records'] != count for key, count in expected_counts.items()):
            raise ValueError('Unexpected official-reference subset size; use a separately labelled experiment')
        command = [sys.executable, str(CLI), 'train', '--task', 'zebra-official',
                   '--variant', 'vanilla', '--data-dir', str(data), '--run-dir', str(directory),
                   '--size', 'mini', '--device', 'cuda', '--precision', 'bf16',
                   '--micro-batch', '128', '--global-batch', '128', '--max-steps', '5000',
                   '--seed', '1', '--gradient-mode', 'detached', '--no-robustness']
        self.execute(command, 'train-official-reference-5000', gpu=True)
        checkpoint, receipt = verified_checkpoint(directory, 5000)
        output = directory / f'evaluation-last-step5000-{receipt["sha256"][:12]}-n1000.json'
        if not output.exists():
            self.execute([sys.executable, str(CLI), 'evaluate', '--checkpoint', str(checkpoint),
                          '--data-dir', str(data), '--output', str(output), '--split', 'test',
                          '--protocol', 'generate', '--policy', 'top_prob', '--examples', '1000',
                          '--batch-size', '8', '--seed', '2026', '--device', 'cuda'],
                         'evaluate-official-reference-5000', gpu=True)
        result = json.loads(output.read_text())
        settings = result['arguments']
        if (result['step'] != 5000 or result['contract']['data_sha256'] != digest(data / 'manifest.json')
                or result['metrics']['num_examples'] != 1000 or settings['split'] != 'test'
                or settings['protocol'] != 'generate' or settings['policy'] != 'top_prob'
                or settings['seed'] != 2026 or settings['batch_size'] != 8
                or settings['examples'] != 1000 or settings.get('memory_condition', 'correct') != 'correct'
                or result['contract']['task'] != 'zebra-official'
                or result['contract']['variant'] != 'vanilla'
                or Path(result['checkpoint']).resolve() != checkpoint):
            raise ValueError('Official reference evaluation provenance mismatch')
        self.execute([sys.executable, str(CLI), 'plot', '--runs', str(directory),
                      '--train-metric', 'train/base_loss', '--output',
                      str(self.directory / 'official-reference/training.png')], 'plot-official-reference')
        atomic_json(self.directory / 'official-reference/result.json', dict(
            task='zebra-official', variant='vanilla', step=5000, metrics=result['metrics'],
            evaluation=str(output), data_sha256=digest(data / 'manifest.json'),
            checkpoint_sha256=receipt['sha256'],
            limit='Source-data-aligned small subset, NOT paper96.9 reproduction; do not compare accuracy to synthetic pilot'))
        self.execute([sys.executable, str(ROOT / 'scripts/reasoning/summarize_official_zebra.py'),
                      '--evaluation', str(output), '--data-dir', str(data), '--output-dir',
                      str(self.directory / 'official-reference')], 'report-official-reference')

    def run(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        runtime = ROOT / '.cache/runtime/reasoning'
        runtime.mkdir(parents=True, exist_ok=True)
        with (runtime / f'queue-gpu{self.args.gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            config = self.directory / 'queue_config.json'
            if config.exists() and json.loads(config.read_text()) != self.spec:
                raise ValueError('Changed queue specification; choose a separate queue directory')
            atomic_json(config, self.spec)
            previous_status = self.directory / 'status.json'
            if previous_status.exists() and json.loads(previous_status.read_text()).get('status') == 'finished':
                print('This fixed-budget queue already finished; no new GPU work.', flush=True)
                return
            budget_path = self.directory / 'budget.json'
            if not budget_path.exists():
                atomic_json(budget_path, dict(started_utc_seconds=time.time(),
                            deadline_utc_seconds=time.time() + self.args.hours * 3600))
            self.deadline = json.loads(budget_path.read_text())['deadline_utc_seconds']
            for key in ('TMPDIR', 'MPLCONFIGDIR', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH'):
                Path(self.env[key]).mkdir(parents=True, exist_ok=True)
            try:
                self.base.verify_data('zebra')
                for variant in VARIANTS:
                    fork_run(self.source(variant), self.destination(variant))
                    self.audit(variant, self.source(variant), 5000)
                self.audit_report()
                self.official_reference()
                # Round-robin: no model gets all the extra budget while others stay at5k.
                for target in range(6000, 10001, 1000):
                    for variant in VARIANTS:
                        directory = self.destination(variant)
                        _, receipt = verified_checkpoint(directory)
                        if receipt['step'] >= target:
                            continue
                        command = self.queue(variant).train_command('zebra', variant, directory)
                        command[command.index('--max-steps') + 1] = str(target)
                        self.execute(command, f'train-{variant}-{target}', gpu=True)
                        verified_checkpoint(directory, target)
                    self.report()
                for variant in VARIANTS:
                    directory = self.destination(variant)
                    self.audit(variant, directory, 10000)
                    queue = self.queue(variant)
                    # Existing evaluator verifies checkpoint/data/protocol on restart.
                    queue.subprocess = lambda command, stage: self.execute(command, stage, gpu=True)
                    queue.evaluate('zebra', variant, directory)
                self.audit_report()
                self.report(final=True)
                self.status('finished', target_step=10000, variants=list(VARIANTS),
                            original_5000_runs_preserved=True)
            except BaseException as error:
                self.status('stopped', error=str(error),
                            resume='Same command resumes committed optimizer boundaries within original time cap')
                raise


def main(argv=None):
    job = ZebraContinuation(parser().parse_args(argv))
    if job.args.action == 'run':
        job.run()
    elif job.args.action == 'status':
        print((job.directory / 'status.json').read_text())
    else:
        print(json.dumps(job.spec, indent=2))
        for variant in VARIANTS:
            print(variant, job.source(variant), '->', job.destination(variant))
