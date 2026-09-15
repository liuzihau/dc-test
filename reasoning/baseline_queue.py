"""Sequential, restartable H100 reasoning suites; no GPU work on import.

The original baseline suite/paths/contracts are preserved. The opt-in memory
suite uses corrected merged attention and separate scientific run directories.
"""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'scripts/reasoning/run_reasoning.py'
ENTRY = ROOT / 'scripts/reasoning/run_baseline_queue.py'
TASKS = ('sudoku', 'zebra')
VARIANTS = ('vanilla', 'mdm', 'mdm_aux')
MEMORY_VARIANTS = ('both', 'both_aux')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.' + path.name,
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('action', choices=('plan', 'smoke', 'run', 'tmux', 'status'))
    result.add_argument('--suite', choices=('baselines', 'memory'), default='baselines')
    result.add_argument('--micro-batch', type=int, default=int(os.getenv('REASONING_MICRO_BATCH', '128')))
    result.add_argument('--global-batch', type=int, default=int(os.getenv('REASONING_GLOBAL_BATCH', '128')))
    result.add_argument('--max-steps', type=int, default=int(os.getenv('REASONING_MAX_STEPS', '5000')))
    result.add_argument('--seed', type=int, default=int(os.getenv('REASONING_SEED', '1')))
    result.add_argument('--gpu', default=os.getenv('REASONING_GPU_IDS', '0'))
    result.add_argument('--train-examples', type=int, default=20000)
    result.add_argument('--valid-examples', type=int, default=1000)
    result.add_argument('--test-examples', type=int, default=1000)
    result.add_argument('--eval-examples', type=int, default=1000)
    result.add_argument('--data-root', type=Path, default=ROOT / '.cache/reasoning')
    result.add_argument('--output-root', type=Path, default=ROOT / 'outputs/reasoning')
    result.add_argument('--queue-dir', type=Path)
    return result


class BaselineQueue:
    def __init__(self, args):
        self.args = args
        self.memory_suite = args.suite == 'memory'
        self.variants = MEMORY_VARIANTS if self.memory_suite else VARIANTS
        args.data_root = args.data_root.resolve()
        args.output_root = args.output_root.resolve()
        for name in ('micro_batch', 'global_batch', 'max_steps', 'train_examples',
                     'valid_examples', 'test_examples', 'eval_examples'):
            if getattr(args, name) < 1:
                raise ValueError(name + ' must be positive')
        if args.global_batch % args.micro_batch:
            raise ValueError('global_batch must be divisible by micro_batch; do not change global batch to fit silently')
        if args.seed < 0 or not args.gpu.isdigit():
            raise ValueError('Use a nonnegative seed and one numeric physical GPU ID')
        visible = os.getenv('CUDA_VISIBLE_DEVICES', '')
        if visible and args.gpu not in [item.strip() for item in visible.split(',')]:
            raise ValueError('Requested GPU conflicts with inherited CUDA_VISIBLE_DEVICES; select an allocated GPU explicitly')
        if args.eval_examples > args.test_examples:
            raise ValueError('eval_examples cannot exceed the prepared test split')
        if int(os.getenv('WORLD_SIZE', '1')) != 1 or int(os.getenv('RANK', '0')) != 0 or int(os.getenv('LOCAL_RANK', '0')) != 0:
            raise ValueError('Launch this single-GPU queue outside torchrun/DDP')
        self.label = f'pilot-v1-n{args.train_examples}-v{args.valid_examples}-t{args.test_examples}'
        self.batch_label = f'mb{args.micro_batch}-gb{args.global_batch}-seed{args.seed}'
        suite_label = 'memory-current-preserving' if self.memory_suite else 'baselines'
        self.directory = (args.queue_dir or args.output_root / 'queues' /
                          f'sudoku-zebra-{suite_label}-h100-{self.label}-{self.batch_label}').resolve()
        self.spec = dict(version=1, tasks=list(TASKS), variants=list(self.variants),
                         micro_batch=args.micro_batch, global_batch=args.global_batch,
                         max_steps=args.max_steps, seed=args.seed, gpu=args.gpu,
                         train_examples=args.train_examples, valid_examples=args.valid_examples,
                         test_examples=args.test_examples, eval_examples=args.eval_examples,
                         data_root=str(args.data_root.resolve()), output_root=str(args.output_root.resolve()),
                         size='mini', precision='bf16', gradient_mode='detached',
                         no_robustness=True, generation_checkpoint='last', evaluation_seed=2026)
        if self.memory_suite:
            self.spec.update(suite='memory', gradient_mode='adjacent', no_robustness=False,
                             merged_policy='current_preserving', gate_enabled=False,
                             cache_only_probability=0.0, current_only_probability=0.05,
                             final_dropout=0.10, identity_probability=0.25,
                             identity_weight=0.10, identity_margin=0.05,
                             identity_final_probability=0.50, neighbor_weight=0.5,
                             smoke_stress_memory_routes=True)
        self.env = dict(os.environ)
        for name in list(self.env):
            if name.startswith(('TORCHELASTIC_', 'MASTER_')):
                self.env.pop(name)
        runtime = ROOT / '.cache/runtime/reasoning'
        self.env.update(CUDA_VISIBLE_DEVICES=args.gpu, WORLD_SIZE='1', RANK='0', LOCAL_RANK='0',
                        TMPDIR=str(runtime / 'tmp'), TMP=str(runtime / 'tmp'), TEMP=str(runtime / 'tmp'),
                        MPLCONFIGDIR=str(runtime / 'matplotlib'),
                        TORCHINDUCTOR_CACHE_DIR=str(runtime / 'inductor'),
                        TRITON_CACHE_DIR=str(runtime / 'triton'), CUDA_CACHE_PATH=str(runtime / 'cuda'),
                        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')

    def data_dir(self, task):
        return self.args.data_root.resolve() / f'{task}-{self.label}'

    def run_dir(self, task, variant):
        policy = '-current-preserving' if self.memory_suite else ''
        return self.args.output_root.resolve() / task / f'{variant}-h100{policy}-{self.label}-{self.batch_label}'

    def compatible_baseline_runs(self, task):
        """Read-only plot overlays, never train or resume an existing control.

        Match the complete training contract except the documented intervention
        (memory/auxiliary/trajectory), and match the fixed validation protocol.
        Missing controls are fine; incompatible controls are visibly skipped.
        """
        if not self.memory_suite:
            return []
        reference_dir = self.run_dir(task, 'both')
        reference_contract = reference_dir / 'contract.json'
        reference_launch = reference_dir / 'launch.json'
        if not reference_contract.exists() or not reference_launch.exists():
            return []
        contract = json.loads(reference_contract.read_text())
        launch = json.loads(reference_launch.read_text())
        if contract['data_sha256'] != digest(self.data_dir(task) / 'manifest.json'):
            raise ValueError('Memory plot reference differs from the verified shared dataset')
        result = []
        for variant in VARIANTS:
            directory = (self.args.output_root / task /
                         f'{variant}-h100-{self.label}-{self.batch_label}')
            if not directory.exists():
                continue
            try:
                actual = json.loads((directory / 'contract.json').read_text())
                old_launch = json.loads((directory / 'launch.json').read_text())
                expected = copy.deepcopy(contract)
                expected['variant'] = variant
                model = expected['model_config']
                model.pop('merged_policy', None)
                model.update(memory_mode='none',
                             attention_mode='vanilla' if variant == 'vanilla' else 'merged',
                             neighbors=variant == 'mdm_aux', gradient_mode='detached',
                             trajectory='single' if variant == 'vanilla' else 'five',
                             gate_enabled=True, cache_only_probability=0.0,
                             current_only_probability=0.0, final_dropout=0.0,
                             identity_probability=0.0)
                validation_keys = ('eval_seed', 'eval_batch_size', 'validation_examples')
                if actual != expected or any(old_launch[k] != launch[k] for k in validation_keys):
                    raise ValueError('training or fixed-validation contract differs')
                if not any((directory / 'logs').glob('attempt-*/metrics.csv')):
                    raise ValueError('no recorded metrics')
                result.append(str(directory))
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(f'Skipping incompatible baseline plot overlay {directory}: {error}', flush=True)
        return result

    def status(self, state, **fields):
        atomic_json(self.directory / 'status.json', dict(status=state, pid=os.getpid(),
                    updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields))

    def subprocess(self, command, stage):
        if len(command) > 2 and command[2] in ('train', 'evaluate'):
            # Preparation/evaluation can take time: a check at queue startup
            # alone cannot detect jobs launched by others between stages.
            self.gpu_check()
        self.status('running', stage=stage, command=command)
        print(f'\n[{stage}] {shlex.join(command)}', flush=True)
        path = self.directory / 'console' / f'{time.time_ns()}-{stage}.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    stream.write(line)
                    stream.flush()
                code = process.wait()
            except BaseException:
                # Only stop the child launched by this queue, never unrelated GPU jobs.
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
        if code:
            raise RuntimeError(f'{stage} failed (exit {code}); queue stopped. See {path}')

    def gpu_check(self):
        # Before this process creates any CUDA context. Never evict an existing job.
        base = ['nvidia-smi', '-i', self.args.gpu]
        info = subprocess.check_output(base + ['--query-gpu=name,memory.total',
                                               '--format=csv,noheader'], text=True).strip()
        busy = subprocess.check_output(base + ['--query-compute-apps=pid',
                                               '--format=csv,noheader,nounits'], text=True).strip()
        if busy:
            raise RuntimeError(f'GPU {self.args.gpu} already has compute processes: {busy}. Finish/stop your old run before launching the queue.')
        print(f'GPU {self.args.gpu}: {info}', flush=True)
        atomic_json(self.directory / 'gpu.json', dict(gpu=self.args.gpu, description=info))

    def verify_data(self, task):
        directory = self.data_dir(task)
        manifest = json.loads((directory / 'manifest.json').read_text())
        expected = dict(train=self.args.train_examples, validation=self.args.valid_examples,
                        test=self.args.test_examples)
        if (manifest.get('schema_version') != 1 or manifest.get('task') != task
                or manifest.get('seed') != 17 or manifest.get('source', {}).get('kind') != 'synthetic_pilot'):
            raise ValueError(f'{directory}: not the requested seed-17 pilot dataset; nothing overwritten')
        for split, count in expected.items():
            entry = manifest['splits'][split]
            if entry['filename'] != split + '.jsonl' or entry['records'] != count:
                raise ValueError(f'{directory}: unexpected {split} count/file; do not reuse a small demo')
            if digest(directory / entry['filename']) != entry['sha256']:
                raise ValueError(f'{directory}: checksum mismatch in {split}')
        return manifest

    def prepare(self):
        for task in TASKS:
            directory = self.data_dir(task)
            if not (directory / 'manifest.json').exists():
                self.subprocess([sys.executable, str(CLI), 'prepare', '--task', task,
                    '--output', str(directory), '--train-size', str(self.args.train_examples),
                    '--valid-size', str(self.args.valid_examples), '--test-size', str(self.args.test_examples),
                    '--seed', '17'], f'prepare-{task}')
            self.verify_data(task)
            print(f'Reusing verified shared data: {directory}', flush=True)

    def train_command(self, task, variant, directory, smoke=False):
        command = [sys.executable, str(CLI), 'train', '--task', task, '--variant', variant,
                   '--data-dir', str(self.data_dir(task)), '--run-dir', str(directory),
                   '--size', 'mini', '--device', 'cuda', '--precision', 'bf16',
                   '--micro-batch', str(self.args.micro_batch), '--global-batch', str(self.args.global_batch),
                   '--max-steps', '2' if smoke else str(self.args.max_steps), '--seed', str(self.args.seed),
                   '--gradient-mode', 'adjacent' if self.memory_suite else 'detached']
        if self.memory_suite:
            if variant not in MEMORY_VARIANTS:
                raise ValueError('The memory suite only trains both and both_aux')
            command += ['--merged-policy', 'current_preserving', '--neighbor-weight', '0.5']
        else:
            command += ['--no-robustness']
        if smoke:
            command += ['--fresh', '--val-every', '1', '--save-every', '1', '--save-seconds', '0',
                        '--log-every', '1', '--validation-examples', '8', '--eval-batch-size', '8']
            if self.memory_suite:
                # Force the expensive identity reference on every smoke update;
                # stress checkpoints/configs are never production checkpoints.
                command += ['--stress-memory-routes']
        return command

    def checkpoint(self, directory, expected_step):
        path = (directory / 'checkpoints/last.pt').resolve(strict=True)
        receipt = json.loads(path.with_suffix('.pt.json').read_text())
        if (receipt['step'] != expected_step or receipt['size'] != path.stat().st_size
                or receipt['sha256'] != digest(path)):
            raise ValueError(f'Incomplete/wrong-step checkpoint: {path}')
        return path, receipt

    def smoke(self):
        summary = {}
        for task in TASKS:
            # Reuse queue-owned scratch runs with the trainer's latest-three
            # retention. Never accumulate new full-model checkpoint directories
            # on every restart, and never reuse a production checkpoint here.
            directory = self.directory / 'smoke' / task
            last = directory / 'checkpoints/last.pt'
            start = 0
            has_checkpoint = last.exists() or last.is_symlink()
            if has_checkpoint:
                target = last.resolve(strict=True)
                start = int(json.loads(target.with_suffix('.pt.json').read_text())['step'])
                self.checkpoint(directory, start)
            expected_step = start + 2
            # The heaviest variant covers both full sequence lengths.
            smoke_variant = 'both_aux' if self.memory_suite else 'mdm_aux'
            command = self.train_command(task, smoke_variant, directory, smoke=True)
            command[command.index('--max-steps') + 1] = str(expected_step)
            # A first-update OOM can leave contract.json but no checkpoint.
            # The normal strict contract path can safely retry that directory;
            # --fresh would reject it before another forward could run.
            command.remove('--fresh')
            self.subprocess(command, f'smoke-{task}')
            self.checkpoint(directory, expected_step)
            rows = []
            for path in sorted((directory / 'logs').glob('attempt-*/metrics.csv'))[-1:]:
                with path.open() as stream:
                    rows.extend(csv.DictReader(stream))
            summary[task] = dict(run=str(directory), optimizer_steps=2, size='mini',
                                 micro_batch=self.args.micro_batch, global_batch=self.args.global_batch,
                                 start_step=start, end_step=expected_step,
                                 logged_updates=[row for row in rows if row.get('seconds_per_update')
                                                 and start < int(float(row['step'])) <= expected_step])
            if self.memory_suite:
                summary[task].update(variant=smoke_variant, merged_policy='current_preserving',
                                     gradient_mode='adjacent', stress_memory_routes=True)
            for row in summary[task]['logged_updates']:
                print(f'{task} full-size smoke: step {row["step"]}, {row["seconds_per_update"]} s/update, '
                      f'peak allocated {row.get("peak_cuda_allocated_gib", "not logged")} GiB', flush=True)
        atomic_json(self.directory / 'full_model_smoke.json', summary)

    def evaluate(self, task, variant, directory):
        checkpoint, receipt = self.checkpoint(directory, self.args.max_steps)
        output = directory / f'evaluation-last-step{self.args.max_steps}-{receipt["sha256"][:12]}-n{self.args.eval_examples}.json'
        if not output.exists():
            self.subprocess([sys.executable, str(CLI), 'evaluate', '--checkpoint', str(checkpoint),
                '--data-dir', str(self.data_dir(task)), '--device', 'cuda', '--output', str(output),
                '--split', 'test', '--protocol', 'generate', '--policy', 'top_prob', '--examples', str(self.args.eval_examples),
                '--batch-size', '8', '--seed', '2026'], f'evaluate-{task}-{variant}')
        result = json.loads(output.read_text())
        settings = result['arguments']
        if (result['step'] != self.args.max_steps or Path(result['checkpoint']).resolve() != checkpoint
                or result['contract']['data_sha256'] != digest(self.data_dir(task) / 'manifest.json')
                or result['contract']['variant'] != variant
                or settings['split'] != 'test'
                or settings['examples'] != self.args.eval_examples or settings['protocol'] != 'generate'
                or settings['policy'] != 'top_prob' or settings['seed'] != 2026
                or settings.get('memory_condition', 'correct') != 'correct'
                or settings['batch_size'] != 8 or result['metrics']['num_examples'] != self.args.eval_examples):
            raise ValueError(f'Evaluation metadata does not match this queue: {output}')
        print(f'Validated solving result: {output}', flush=True)

    def run(self, smoke_only=False):
        self.directory.mkdir(parents=True, exist_ok=True)
        lock_path = ROOT / '.cache/runtime/reasoning' / f'queue-gpu{self.args.gpu}.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('Another reasoning queue owns this GPU lock') from error
            config_path = self.directory / 'queue_config.json'
            if config_path.exists() and json.loads(config_path.read_text()) != self.spec:
                raise ValueError('Queue settings changed; use a new --queue-dir instead of mixing trials')
            atomic_json(config_path, self.spec)
            for key in ('TMPDIR', 'MPLCONFIGDIR', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH'):
                Path(self.env[key]).mkdir(parents=True, exist_ok=True)
            try:
                self.gpu_check()
                self.prepare()
                self.smoke()
                if smoke_only:
                    self.status('smoke_passed', experiments_started=0)
                    return
                for task in TASKS:
                    finished_runs = []
                    for variant in self.variants:
                        directory = self.run_dir(task, variant)
                        # Always let the trainer verify the full resume contract.
                        # It returns immediately for an already completed run.
                        self.subprocess(self.train_command(task, variant, directory), f'train-{task}-{variant}')
                        self.checkpoint(directory, self.args.max_steps)
                        finished_runs.append(str(directory))
                        overlays = self.compatible_baseline_runs(task)
                        plot_command = [sys.executable, str(CLI), 'plot', '--runs', *overlays, *finished_runs,
                            '--output', str(self.directory / 'figures' / f'{task}.png'),
                            '--train-metric', 'train/base_loss']
                        if self.memory_suite:
                            plot_command += ['--val-label', 'Cold validation NLL (10/30/50/70% masks)',
                                             '--labels', *[Path(run).name.split('-h100', 1)[0]
                                                           for run in [*overlays, *finished_runs]]]
                        self.subprocess(plot_command, f'plot-{task}-{variant}')
                        self.evaluate(task, variant, directory)
                self.status('finished', experiments_completed=len(TASKS)*len(self.variants), max_steps=self.args.max_steps)
            except BaseException as error:
                self.status('stopped', error=str(error), resume='Rerun the same queue command; no automatic batch reduction')
                raise

    def plan(self):
        print(f'Microbatch {self.args.micro_batch}; global batch {self.args.global_batch}; '
              f'accumulation {self.args.global_batch // self.args.micro_batch}. GPU {self.args.gpu}.')
        smoke_variant = 'both_aux (forced identity reference)' if self.memory_suite else 'mdm_aux'
        print(f'Full mini-model preflight: Sudoku and Zebra, {smoke_variant}, two AdamW updates each.')
        if self.memory_suite:
            print('Corrected merged attention + adjacent DCache + detached final state. '
                  'Previous-V gate OFF, cache-only OFF, current-only 5%, final dropout 10%, '
                  'identity probability 25%. Auxiliary weight 0.5 for both_aux only.')
        for index, (task, variant) in enumerate(((t, v) for t in TASKS for v in self.variants), 1):
            print(f'{index}. {task}/{variant}: {self.args.max_steps} updates -> plot -> '
                  f'{self.args.eval_examples}-example generation\n   {self.run_dir(task, variant)}')
        print(f'Queue status/console/figures: {self.directory}')
        print('All are fresh batch-labelled runs unless their OWN compatible checkpoint exists. '
              + ('No baseline or OWT checkpoints are imported.' if self.memory_suite else 'No memory models.'))

    def tmux(self):
        socket = ROOT / '.cache/tmux/reasoning-queue.sock'
        socket.parent.mkdir(parents=True, exist_ok=True)
        if len(str(socket)) > 100:
            raise ValueError('Use a shorter checkout path for the tmux socket')
        session = f'reasoning-{self.args.suite}-gpu{self.args.gpu}-{hashlib.sha256(str(self.directory).encode()).hexdigest()[:8]}'
        command = [sys.executable, str(ENTRY), 'run']
        if self.memory_suite:
            command += ['--suite', 'memory']
        for name in ('micro_batch', 'global_batch', 'max_steps', 'seed', 'gpu', 'train_examples',
                     'valid_examples', 'test_examples', 'eval_examples', 'data_root', 'output_root'):
            command += ['--' + name.replace('_', '-'), str(getattr(self.args, name))]
        command += ['--queue-dir', str(self.directory)]
        prefix = ['tmux', '-S', str(socket)]
        if subprocess.run(prefix + ['has-session', '-t', session], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode:
            launch = ['env', 'WORLD_SIZE=1', 'RANK=0', 'LOCAL_RANK=0',
                      'PATH=' + os.environ.get('PATH', '')]
            if 'LD_LIBRARY_PATH' in os.environ:
                launch.append('LD_LIBRARY_PATH=' + os.environ['LD_LIBRARY_PATH'])
            launch += command
            subprocess.run(prefix + ['new-session', '-d', '-s', session, '-c', str(ROOT),
                                    shlex.join(launch)], check=True)
        print('Attach: ' + shlex.join(prefix + ['attach', '-t', session]))
        print(f'Status: {self.directory / "status.json"}')


def main(argv=None):
    queue = BaselineQueue(parser().parse_args(argv))
    if queue.args.action == 'plan':
        queue.plan()
    elif queue.args.action == 'status':
        path = queue.directory / 'status.json'
        print(path.read_text() if path.exists() else f'No queue status yet: {path}')
    elif queue.args.action == 'tmux':
        queue.plan()
        queue.tmux()
    else:
        queue.run(smoke_only=queue.args.action == 'smoke')
