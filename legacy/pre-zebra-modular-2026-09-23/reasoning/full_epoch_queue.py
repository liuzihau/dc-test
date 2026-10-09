"""Ten finite full-data one-epoch runs, optionally after subset Zebra both_aux.

No model/data selection by test performance, no checkpoint reuse across methods,
no instance shutdown. Restarting resumes each run's own optimizer/data cursor.
Preparation is CPU-only and can overlap the previous GPU job. Handoff preserves
its final Zebra evaluation and retires ONLY the explicitly named old service.
"""
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

from .benchmark import PROTOCOL
from .benchmark_queue import ROOT, TASKS, VARIANTS, DISPLAY_NAMES, training_command, report
from .data import ReasoningDataset
from .runner import atomic_json, digest
from .zebra_continuation import verified_checkpoint
from .variants import SPLIT_VARIANTS, SPLIT_LABELS

CLI = ROOT / 'scripts/reasoning/run_reasoning.py'


def dataset_path(data_root, task):
    return Path(data_root) / (task + '-full-v1')


def full_training_command(task, variant, data, run, count):
    command = training_command(task, variant, data, run, math.ceil(count / 128), 128)
    return command + ['--one-epoch', '--validation-protocol', 'both',
                      '--validation-examples', '1000', '--eval-batch-size', '64']


def split_training_command(task, variant, data, run, count, devices=2,
                           micro_batch=8, eval_batch=8, robustness=True):
    command = [sys.executable, '-u']
    if devices > 1:
        command += ['-m', 'torch.distributed.run', '--standalone',
                    '--nproc_per_node', str(devices)]
    command += [str(CLI), 'train', '--suite', 'split', '--task', task,
                '--variant', variant, '--data-dir', str(data), '--run-dir', str(run),
                '--size', 'mini', '--one-epoch', '--global-batch', '128',
                '--micro-batch', str(micro_batch), '--gradient-mode', 'adjacent',
                '--seed', '1', '--eval-seed', '2026', '--val-every', '500',
                '--save-every', '500', '--save-seconds', '1200',
                '--validation-protocol', 'both', '--validation-examples', '1000',
                '--eval-batch-size', str(eval_batch)]
    if not robustness:
        command += ['--no-robustness']
    return command


def verify_prepared(data, frozen):
    datasets = {s: ReasoningDataset(data, s) for s in ('train', 'validation', 'test')}
    manifest = datasets['train'].manifest
    source = manifest['source']
    if (manifest['schema_version'] != 2 or source['kind'] != 'full_released_training_unique_puzzles'
            or source['frozen_manifest_sha256'] != digest(Path(frozen) / 'manifest.json')
            or source['counters']['retained_train'] != len(datasets['train'])):
        raise ValueError('Prepared full-data provenance differs')
    old = json.loads((Path(frozen) / 'manifest.json').read_text())
    for s in ('validation', 'test'):
        if manifest['splits'][s] != old['splits'][s] or len(datasets[s]) != 1000:
            raise ValueError('Full runs require the same frozen 1k validation/test sets')
    return manifest


def verify_subset_completion(directory, data_root):
    """Returns a handoff receipt only after final checkpoint AND paired test exist."""
    directory = Path(directory)
    run = directory / 'zebra-benchmark/both_aux'
    generation = run / 'generation.json'
    if not generation.exists():
        return None
    checkpoint, receipt = verified_checkpoint(run, 5000)
    result = json.loads(generation.read_text())
    frozen = Path(data_root) / 'zebra-benchmark-v2-n20000-v1000-t1000'
    test = ReasoningDataset(frozen, 'test')
    if (Path(result['checkpoint']).resolve() != checkpoint
            or result.get('benchmark_protocol') != PROTOCOL or result['step'] != 5000
            or result['contract']['task'] != 'zebra-benchmark'
            or result['contract']['variant'] != 'both_aux'
            or result['contract']['data_sha256'] != digest(frozen / 'manifest.json')
            or [r['id'] for r in result['examples']] != [r['id'] for r in test.records]):
        raise ValueError('Subset completion must include paired 1k-test final Zebra both_aux')
    metrics = result['metrics']
    if any(metrics[k] != v for k, v in dict(policy='top_prob', candidate_k=8,
            token_selection='paper', tokens_per_step=1, memory_condition='correct', seed=2026).items()):
        raise ValueError('Subset generation protocol differs')
    return dict(subset=str(directory.resolve()), checkpoint=receipt,
                generation_sha256=digest(generation), finished_ns=time.time_ns())


class FullEpochQueue:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.env = os.environ.copy()
        self.env['CUDA_VISIBLE_DEVICES'] = args.gpu
        self.handoff_path = self.output / 'handoff.json'
        self.report_only = args.action == 'report'
        self.handoff_done = self.report_only or self.handoff_path.exists() or args.after_subset is None
        self.last_handoff_poll = 0
        self.split = getattr(args, 'suite', 'legacy') == 'split'
        self.tasks = (tuple(reversed(TASKS)) if getattr(args, 'task_order', 'source') == 'sudoku-first'
                      else tuple(TASKS))
        self.variants = SPLIT_VARIANTS if self.split else VARIANTS
        self.labels = SPLIT_LABELS if self.split else DISPLAY_NAMES
        self.jobs = len(self.tasks) * len(self.variants)

    def handoff(self):
        if self.handoff_done:
            return True
        receipt = verify_subset_completion(self.args.after_subset, self.args.data_root)
        if receipt is None:
            # A stopped/expired predecessor is an actionable error, not an infinite wait.
            service = subprocess.run(['supervisorctl', 'status', 'dcache_benchmark_v2'],
                                     capture_output=True, text=True, timeout=30)
            if 'RUNNING' not in service.stdout:
                raise RuntimeError('Previous Zebra evaluation is incomplete and old service is not RUNNING: ' + service.stdout)
            return False
        status = subprocess.run(['supervisorctl', 'status', 'dcache_benchmark_v2'],
                                capture_output=True, text=True, timeout=30)
        if 'RUNNING' in status.stdout:
            subprocess.run(['supervisorctl', 'stop', 'dcache_benchmark_v2'], check=True, timeout=90)
        # Deployment sets old autostart=false; apply it only AFTER stopping safely.
        config = Path('/etc/supervisor/conf.d/dcache_benchmark_v2.conf').read_text()
        if 'autostart=false' not in config:
            raise RuntimeError('Disable old queue autostart before arming the handoff')
        subprocess.run(['supervisorctl', 'reread'], check=True, timeout=30)
        subprocess.run(['supervisorctl', 'update', 'dcache_benchmark_v2'], check=True, timeout=90)
        status = subprocess.run(['supervisorctl', 'status', 'dcache_benchmark_v2'],
                                capture_output=True, text=True, timeout=30)
        if 'RUNNING' in status.stdout:
            raise RuntimeError('Old service still running; will not share the training GPU')
        report(self.args.after_subset)
        atomic_json(self.handoff_path, dict(**receipt, skipped='pending 5k-subset Sudoku jobs',
                                           retired_service='dcache_benchmark_v2'))
        self.handoff_done = True
        print('Handoff complete: Zebra final evaluation retained; subset Sudoku queue retired.', flush=True)
        return True

    def execute(self, command, label, cpu=False):
        console = self.output / 'console'
        console.mkdir(exist_ok=True)
        logfile = console / (str(time.time_ns()) + '-' + label + '.log')
        if not self.report_only:
            atomic_json(self.output / 'status.json', dict(status='running', stage=label,
                        command=command, console=str(logfile), jobs=self.jobs))
        print(label + ': ' + str(logfile), flush=True)
        env = dict(self.env)
        if cpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        with logfile.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    if cpu and not self.handoff_done and time.monotonic() - self.last_handoff_poll >= 15:
                        self.handoff()
                        self.last_handoff_poll = time.monotonic()
                    time.sleep(2)
                if child.returncode:
                    raise RuntimeError('Stage failed: ' + label + '; see ' + str(logfile))
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
                raise

    def prepare(self):
        """Materialize and verify both full released datasets before GPU work."""
        a = self.args
        counts = {}
        for task in self.tasks:
            data = dataset_path(a.data_root, task)
            frozen = a.data_root / (task + '-v2-n20000-v1000-t1000')
            stem = task.split('-')[0]
            filenames = {'zebra': ('zebra-train.pickle.partial', 'zebra-test.pickle.partial'),
                         'sudoku': ('sudoku-train.npy.partial', 'sudoku-test.npy')}[stem]
            if not data.exists():
                self.execute([sys.executable, '-u', '-m', 'reasoning.full_data',
                    '--task', task, '--train-file', str(a.raw_root / filenames[0]),
                    '--test-file', str(a.raw_root / filenames[1]), '--frozen-dir', str(frozen),
                    '--output', str(data)], task + '-prepare', cpu=True)
            manifest = verify_prepared(data, frozen)
            counts[task] = manifest['splits']['train']['records']
        return counts

    def run(self):
        a = self.args
        with (self.output / 'queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            config = dict(version=1, tasks=list(self.tasks), variants=list(self.variants), epochs=1,
                          global_batch=128, micro_batch=128, seed=1, eval_seed=2026,
                          data_root=str(a.data_root.resolve()), raw_root=str(a.raw_root.resolve()),
                          after_subset=str(a.after_subset.resolve()) if a.after_subset else None,
                          validation_examples=1000, protocol=PROTOCOL)
            if self.split:
                config.update(version=2, suite='split', micro_batch=a.micro_batch,
                              world_size=a.devices, eval_batch=a.eval_batch,
                              robustness=not a.no_robustness)
            cfg = self.output / 'queue_config.json'
            if cfg.exists() and json.loads(cfg.read_text()) != config:
                raise ValueError('Queue contract differs; use another output directory')
            atomic_json(cfg, config)
            try:
                counts = self.prepare()
                plan = [dict(task=t, variant=v, label=self.labels[v], examples=counts[t],
                             optimizer_steps=math.ceil(counts[t] / 128),
                             final_batch=(counts[t] - 1) % 128 + 1,
                             data_sha256=digest(dataset_path(a.data_root, t) / 'manifest.json'))
                        for t in self.tasks for v in self.variants]
                atomic_json(self.output / 'plan.json', plan)
                while not self.handoff():
                    atomic_json(self.output / 'status.json', dict(status='waiting',
                                stage='previous-zebra-both_aux-final-evaluation', jobs=self.jobs))
                    time.sleep(15)
                for job in plan:
                    task, variant, count = job['task'], job['variant'], job['examples']
                    data = dataset_path(a.data_root, task)
                    run = self.output / task / variant
                    # Runner validates full optimizer/data contract even for an already complete run.
                    command = (split_training_command(task, variant, data, run, count,
                               a.devices, a.micro_batch, a.eval_batch, not a.no_robustness)
                               if self.split else full_training_command(task, variant, data, run, count))
                    self.execute(command, task + '-' + variant + '-train')
                    ckpt, _ = verified_checkpoint(run, job['optimizer_steps'])
                    status = json.loads((run / 'status.json').read_text())
                    if status['status'] != 'finished' or status['examples_seen'] != count:
                        raise ValueError('Run did not finish exactly one epoch')
                    if not (run / 'generation.json').exists():
                        self.execute([sys.executable, '-u', str(CLI), 'evaluate',
                            '--checkpoint', str(ckpt), '--data-dir', str(data),
                            '--output', str(run / 'generation.json'), '--examples', '1000',
                            '--batch-size', str(a.eval_batch if self.split else 32),
                            '--seed', '2026', '--policy', 'top_prob'],
                            task + '-' + variant + '-evaluate')
                    result = json.loads((run / 'generation.json').read_text())
                    expected_ids = [r['id'] for r in ReasoningDataset(data, 'test').records]
                    if (Path(result['checkpoint']).resolve() != ckpt or result['step'] != job['optimizer_steps']
                            or result['contract']['task'] != task or result['contract']['variant'] != variant
                            or result['contract']['data_sha256'] != job['data_sha256']
                            or [r['id'] for r in result['examples']] != expected_ids):
                        raise ValueError('Existing evaluation differs from the final one-epoch checkpoint/test')
                    self.make_report()
                atomic_json(self.output / 'status.json', dict(status='finished', jobs=self.jobs, plan=plan))
            except BaseException as error:
                atomic_json(self.output / 'failure.json', dict(error=str(error), time_ns=time.time_ns()))
                status_path = self.output / 'status.json'
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                atomic_json(status_path, dict(status, status='stopped', error=str(error), updated_ns=time.time_ns()))
                raise

    def make_report(self):
        report(self.output, title='Full released data, one epoch — frozen tests; final checkpoint',
               variants=self.variants, display_names=self.labels,
               suite='split' if self.split else None)
        for task in self.tasks:
            variants = [v for v in self.variants if any((self.output / task / v / 'logs').glob('attempt-*/metrics.csv'))]
            if not variants:
                continue
            for name, metric, label in (
                    ('training-cold', 'val/conditional_nll', 'Cold masked-token NLL (memory reset)'),
                    ('training-nested', 'val/nested_conditional_nll', 'Nested teacher-forced NLL (0.7 → 0.5 → 0.3 → 0.1)')):
                self.execute([sys.executable, str(CLI), 'plot', '--runs',
                    *[str(self.output / task / v) for v in variants], '--labels',
                    *[self.labels[v] for v in variants], '--output', str(self.output / task / (name + '.png')),
                    '--smooth', '60', '--val-metric', metric, '--val-label', label], task + '-' + name, cpu=True)

    def smoke(self):
        if not self.split:
            raise ValueError('Use --suite split for the split-attention smoke test')
        # Smoke is the public preflight entry point, so it must prepare missing
        # CPU-side datasets before attempting to verify or launch GPU work.
        self.prepare()
        smoke_root = self.output / ('smoke-' + str(time.time_ns()))
        for task in self.tasks:
            data = dataset_path(self.args.data_root, task)
            verify_prepared(data, self.args.data_root / (task + '-v2-n20000-v1000-t1000'))
            for variant in self.variants:
                command = split_training_command(task, variant, data, smoke_root / task / variant,
                    0, self.args.devices, self.args.micro_batch, self.args.eval_batch,
                    not self.args.no_robustness)
                command += ['--stop-after-steps', '1', '--validation-examples', '4', '--log-every', '1']
                if '_rm' in variant and not self.args.no_robustness:
                    # Force all optional forwards at the configured batch size.
                    command += ['--stress-memory-routes']
                self.execute(command, task + '-' + variant + '-smoke')
        atomic_json(self.output / 'status.json', dict(status='smoke_passed', output=str(smoke_root), jobs=self.jobs))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('run', 'report', 'plan', 'prepare', 'smoke'))
    p.add_argument('--data-root', type=Path, default=ROOT / '.cache/reasoning')
    p.add_argument('--raw-root', type=Path, default=ROOT / 'imports/official-reasoning/raw')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/reasoning/full-epoch-v1')
    p.add_argument('--after-subset', type=Path, help='Wait for this subset queue; retire ONLY dcache_benchmark_v2 after final Zebra both_aux evaluation')
    p.add_argument('--gpu', default='0')
    p.add_argument('--suite', choices=('legacy', 'split'), default='legacy')
    p.add_argument('--devices', type=int, default=2)
    p.add_argument('--micro-batch', type=int, default=8)
    p.add_argument('--eval-batch', type=int, default=8)
    p.add_argument('--no-robustness', action='store_true')
    p.add_argument('--task-order', choices=('source', 'sudoku-first'), default='source')
    a = p.parse_args(argv)
    if a.devices < 1 or a.micro_batch < 1 or 128 % (a.devices * a.micro_batch) or a.eval_batch < 1:
        raise ValueError('devices * micro_batch must divide global batch 128; positive evaluation batch required')
    if a.suite == 'split' and len(a.gpu.split(',')) != a.devices:
        raise ValueError('--gpu must list exactly --devices visible GPU IDs')
    variants = SPLIT_VARIANTS if a.suite == 'split' else VARIANTS
    labels = SPLIT_LABELS if a.suite == 'split' else DISPLAY_NAMES
    if a.action == 'plan':
        print(json.dumps([dict(task=t, variant=v, label=labels[v], epochs=1, global_batch=128,
                              command=split_training_command(t, v, dataset_path(a.data_root, t),
                                  a.output / t / v, 0, a.devices, a.micro_batch, a.eval_batch,
                                  not a.no_robustness) if a.suite == 'split' else None)
                          for t in ((tuple(reversed(TASKS)) if a.task_order == 'sudoku-first' else TASKS))
                          for v in variants], indent=2)); return
    queue = FullEpochQueue(a)
    def interrupted(signum, _frame):
        raise KeyboardInterrupt('Queue interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    if a.action == 'report':
        queue.make_report()
    elif a.action == 'prepare':
        counts = queue.prepare()
        print(json.dumps({'status': 'prepared', 'train_examples': counts}, indent=2))
    elif a.action == 'smoke':
        queue.smoke()
    else:
        queue.run()


if __name__ == '__main__':
    main()
