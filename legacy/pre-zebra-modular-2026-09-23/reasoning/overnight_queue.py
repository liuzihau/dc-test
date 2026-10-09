"""Wait for an existing reasoning queue, refresh CPU reports, then resume safely.

This coordinator never terminates another process, changes the experiment
recipe, or acquires the GPU queue lock while waiting for its predecessor.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from reasoning.baseline_queue import BaselineQueue, ROOT, TASKS, VARIANTS, atomic_json, parser as base_parser


def parser():
    result = base_parser()
    result.description = __doc__
    result.set_defaults(suite='memory')
    result.add_argument('--wait-pid', type=int)
    result.add_argument('--wait-pid-starttime', type=int,
                        help='Field 22 from /proc/PID/stat, guards against PID reuse')
    result.add_argument('--max-wait-seconds', type=float, default=86400)
    result.add_argument('--idle-seconds', type=float, default=30)
    result.add_argument('--poll-seconds', type=float, default=10)
    result.add_argument('--report-every-seconds', type=float, default=300)
    result.add_argument('--report-output', type=Path)
    return result


def process_identity(pid):
    """Return (state, start ticks); missing or zombie processes are not running."""
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    # Linux comm may itself contain spaces and ')'. Fields after its final ')'
    # start at field 3 (state); starttime is field 22.
    tail = stat.rsplit(') ', 1)[1].split()
    return tail[0], int(tail[19])


class OvernightQueue(BaselineQueue):
    def __init__(self, args):
        super().__init__(args)
        if not self.memory_suite:
            raise ValueError('This coordinator resumes only the corrected memory suite')
        if args.action not in ('plan', 'run', 'status'):
            raise ValueError('Use plan, run or status; launch run under your external supervisor')
        for field in ('max_wait_seconds', 'poll_seconds', 'report_every_seconds'):
            if getattr(args, field) <= 0:
                raise ValueError(field + ' must be positive')
        if args.idle_seconds < 0 or args.poll_seconds > 60:
            raise ValueError('idle_seconds must be nonnegative; poll_seconds must be <=60')
        if (args.wait_pid is None) != (args.wait_pid_starttime is None):
            raise ValueError('Supply both --wait-pid and --wait-pid-starttime, or neither')
        if args.wait_pid is not None and (args.wait_pid < 1 or args.wait_pid_starttime < 1):
            raise ValueError('PID and process start ticks must be positive')
        self.report_output = (args.report_output or self.directory / 'report').resolve()
        self.report_last = None
        self.verification_only = False

    def coordinator_status(self, state, **fields):
        atomic_json(self.directory / 'overnight_status.json', dict(
            status=state, pid=os.getpid(), updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            wait_pid=self.args.wait_pid, wait_pid_starttime=self.args.wait_pid_starttime,
            report_output=str(self.report_output), **fields))

    def report_runs(self):
        # The reporter groups/checks contracts. Include completed baselines even
        # before the first memory run for that task has created its contract.
        runs = []
        for task in TASKS:
            for variant in VARIANTS:
                candidate = (self.args.output_root / task /
                    f'{variant}-h100-{self.label}-{self.batch_label}')
                if (candidate / 'contract.json').exists():
                    runs.append(str(candidate))
            for variant in self.variants:
                candidate = self.run_dir(task, variant)
                if (candidate / 'contract.json').exists():
                    runs.append(str(candidate))
        return runs

    def refresh_report(self, force=False):
        now = time.monotonic()
        if not force and self.report_last is not None and now - self.report_last < self.args.report_every_seconds:
            return
        self.report_last = now
        runs = self.report_runs()
        if not runs:
            return
        command = [sys.executable, str(ROOT / 'scripts/reasoning/summarize_reasoning.py'),
                   '--runs', *runs, '--output-dir', str(self.report_output),
                   '--step', str(self.args.max_steps)]
        cpu_env = dict(self.env, CUDA_VISIBLE_DEVICES='')
        path = self.directory / 'console' / f'{time.time_ns()}-overnight-report.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Refreshing CPU-only report: {self.report_output}', flush=True)
        # Reporting failure never kills/resets an already-running training job.
        try:
            with path.open('x') as stream:
                completed = subprocess.run(command, cwd=ROOT, env=cpu_env, stdout=stream,
                                           stderr=subprocess.STDOUT, timeout=180)
            if completed.returncode:
                raise RuntimeError(f'Reporter exited {completed.returncode}; see {path}')
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            atomic_json(self.directory / 'report_error.json', dict(error=str(error), log=str(path)))
            print(f'Report refresh failed without interrupting training: {error}', flush=True)

    def predecessor_running(self):
        if self.args.wait_pid is None:
            return False
        current = process_identity(self.args.wait_pid)
        return (current is not None and current[0] not in ('Z', 'X')
                and current[1] == self.args.wait_pid_starttime)

    def gpu_busy(self):
        return bool(subprocess.check_output(
            ['nvidia-smi', '-i', self.args.gpu, '--query-compute-apps=pid',
             '--format=csv,noheader,nounits'], text=True, timeout=20).strip())

    def subprocess(self, command, stage):
        if self.verification_only:
            raise FileNotFoundError('Required evaluation is not complete: ' + stage)
        return super().subprocess(command, stage)

    def all_complete(self):
        """Verify existing final checkpoints/evaluations without launching children."""
        config = self.directory / 'queue_config.json'
        if not config.exists():
            return False
        if json.loads(config.read_text()) != self.spec:
            raise ValueError('Existing memory queue settings differ; do not mix experiment contracts')
        # Cheap readiness checks precede hashing the large checkpoint files.
        # Skipping train() also skips its normal contract verification, so bind
        # each evaluation to the full recorded run contract and corrected policy.
        for task in TASKS:
            for variant in self.variants:
                directory = self.run_dir(task, variant)
                try:
                    contract = json.loads((directory / 'contract.json').read_text())
                    target = (directory / 'checkpoints/last.pt').resolve(strict=True)
                    receipt = json.loads(target.with_suffix('.pt.json').read_text())
                    if receipt['step'] != self.args.max_steps:
                        return False
                    output = directory / (f'evaluation-last-step{self.args.max_steps}-'
                        f'{receipt["sha256"][:12]}-n{self.args.eval_examples}.json')
                    evaluation = json.loads(output.read_text())
                except FileNotFoundError:
                    return False
                expected = dict(task=task, variant=variant, micro_batch=self.args.micro_batch,
                                global_batch=self.args.global_batch, world_size=1, seed=self.args.seed,
                                precision='bf16', device_type='cuda')
                model_expected = dict(memory_mode='both', attention_mode='merged',
                    merged_policy='current_preserving', gradient_mode='adjacent', trajectory='five',
                    neighbors=variant == 'both_aux', neighbor_weight=0.5,
                    gate_enabled=False, cache_only_probability=0.0,
                    current_only_probability=0.05, final_dropout=0.10,
                    identity_probability=0.25, identity_weight=0.10, identity_margin=0.05,
                    identity_final_probability=0.50, hidden_size=512, n_heads=8, n_layers=6)
                if (any(contract.get(key) != value for key, value in expected.items())
                        or any(contract.get('model_config', {}).get(key) != value
                               for key, value in model_expected.items())
                        or evaluation.get('contract') != contract):
                    raise ValueError(f'Completed-run contract is not the corrected memory trial: {directory}')
        self.verification_only = True
        try:
            for task in TASKS:
                for variant in self.variants:
                    super().evaluate(task, variant, self.run_dir(task, variant))
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
            return False
        finally:
            self.verification_only = False
        return True

    def wait_for_handoff(self):
        started = time.monotonic()
        idle_since = None
        while True:
            self.refresh_report()
            predecessor = self.predecessor_running()
            if predecessor:
                idle_since = None
                state = 'waiting_for_predecessor'
            else:
                # Completed queues require no GPU, even if a new unrelated job
                # has since claimed it. Do not run smoke again after completion.
                if self.all_complete():
                    return True
                busy = self.gpu_busy()
                now = time.monotonic()
                if busy:
                    idle_since = None
                elif idle_since is None:
                    idle_since = now
                state = 'waiting_for_gpu_idle' if busy else 'confirming_gpu_idle'
                if idle_since is not None and now - idle_since >= self.args.idle_seconds:
                    return False
            elapsed = time.monotonic() - started
            self.coordinator_status(state, elapsed_wait_seconds=elapsed)
            if elapsed >= self.args.max_wait_seconds:
                raise TimeoutError('Handoff wait expired; no existing processes were stopped')
            time.sleep(min(self.args.poll_seconds, self.args.max_wait_seconds - elapsed))

    def evaluate(self, task, variant, directory):
        super().evaluate(task, variant, directory)
        self.refresh_report(force=True)

    def run_after_wait(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        # Independent coordinator lock: the original queue keeps its own GPU
        # lock until it exits; we never block it by taking that lock prematurely.
        with (self.directory / '.overnight-coordinator.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                done = self.wait_for_handoff()
                if not done:
                    self.coordinator_status('resuming_corrected_memory_queue')
                    super().run()
                self.refresh_report(force=True)
                self.coordinator_status('finished', experiments_completed=len(TASKS)*len(self.variants),
                                        resumed_training=not done)
            except BaseException as error:
                # Summarize completed work on failure; the reporter is CPU-only.
                self.refresh_report(force=True)
                self.coordinator_status('stopped', error=str(error))
                raise


def main(argv=None):
    job = OvernightQueue(parser().parse_args(argv))
    if job.args.action == 'plan':
        job.plan()
        print(f'Wait for PID {job.args.wait_pid}, start ticks {job.args.wait_pid_starttime}; '
              f'maximum wait {job.args.max_wait_seconds}s. Reports: {job.report_output}')
    elif job.args.action == 'status':
        path = job.directory / 'overnight_status.json'
        print(path.read_text() if path.exists() else f'No overnight status yet: {path}')
    else:
        job.run_after_wait()
