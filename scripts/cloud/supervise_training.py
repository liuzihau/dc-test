#!/usr/bin/env python3
"""Persist launch settings and supervise one existing instance; no cloud API calls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

KEYS = ('DCACHE_PYTHON', 'DCACHE_RESUME_CKPT', 'DCACHE_TRANSFER_MANIFEST',
        'DCACHE_DATA_DIR', 'DCACHE_RUN_DIR', 'DCACHE_CUDA_VISIBLE_DEVICES',
        'DCACHE_MAX_STEPS', 'DCACHE_NUM_WORKERS', 'DCACHE_CPU_THREADS',
        'DCACHE_MICRO_BATCH', 'DCACHE_RECOVERY_SECONDS')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + uuid.uuid4().hex)
    with temp.open('x') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def retryable(code, tail):
    if code == 0:
        return False
    fatal = ('out of memory', 'no space left', 'insufficient recovery checkpoint space',
             'permission denied', 'permissionerror', 'recovery_fatal', 'traceback (most recent call last)')
    # Known GPU/transport failures may include a traceback. Explicit resource /
    # permission failures still stop rather than hammer an unchanged bad setup.
    lower = tail.lower()
    if any(x in lower for x in fatal[:-1]):
        return False
    transient = ('cuda unavailable', 'nccl', 'device has fallen off', 'device is lost',
                 'driver shutting down', 'connection reset', 'connection timed out')
    return any(x in lower for x in transient) or (code in (137, 143, -9, -15) and 'traceback' not in lower)


def completed_output(output, target, tail):
    if 'Training target already complete; nothing to restart.' in tail:
        return True  # Emitted only after checkpoint verification by the launcher.
    for path in (Path(output) / 'checkpoints').glob('recovery-*.ckpt.json'):
        try:
            meta = json.loads(path.read_text())
            if (meta['completed'] is True and meta['step'] >= target
                    and Path(meta['file']).name == meta['file']
                    and (path.parent / meta['file']).is_file()):
                return True
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return False


def run(config_path, delay=120, retries=3):
    cfg = json.loads(Path(config_path).read_text())
    if cfg['version'] != 1:
        raise ValueError('Unsupported supervisor configuration')
    repo = Path(cfg['repo'])
    env = dict(os.environ)
    env.update({k: str(v) for k, v in cfg['env'].items() if k in KEYS})
    env['DCACHE_RECOVERY_ENABLED'] = '1'
    output = Path(env['DCACHE_RUN_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    logs = output / 'supervisor_logs'
    logs.mkdir(exist_ok=True)
    with (output / '.supervisor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Supervisor already running for this output directory.', flush=True)
            return 0
        status = output / 'supervisor_status.json'
        for attempt in range(retries + 1):
            log = logs / f'{time.time_ns()}.log'
            write_json(status, {'state': 'RUNNING', 'pid': os.getpid(),
                               'attempt': attempt, 'log': str(log), 'time': time.time()})
            print(f'Launching recovery attempt {attempt}; log: {log}', flush=True)
            with log.open('xb') as f:
                proc = subprocess.run(['bash', str(repo / 'scripts/cloud/lightning_h100.sh'), 'train'],
                                      cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                      stdout=f, stderr=subprocess.STDOUT)
            with log.open('rb') as f:
                f.seek(max(0, log.stat().st_size - 128 * 1024))
                tail = f.read().decode(errors='replace')
            if proc.returncode == 0:
                done = completed_output(output, int(env['DCACHE_MAX_STEPS']), tail)
                write_json(status, {'state': 'COMPLETE' if done else 'STOPPED_EARLY',
                                    'log': str(log), 'time': time.time()})
                return 0 if done else 1
            retry = attempt < retries and retryable(proc.returncode, tail)
            write_json(status, {'state': 'RETRY_WAIT' if retry else 'STOPPED_ERROR',
                               'exit_code': proc.returncode, 'log': str(log), 'time': time.time()})
            if not retry:
                print(f'Stopped on error {proc.returncode}; inspect {log}', flush=True)
                return proc.returncode if proc.returncode > 0 else 1
            print(f'Transient failure; retrying in {delay}s.', flush=True)
            time.sleep(delay)
    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['configure', 'run'])
    p.add_argument('--config', required=True)
    p.add_argument('--repo')
    p.add_argument('--retry-delay', type=int, default=120)
    p.add_argument('--max-retries', type=int, default=3)
    a = p.parse_args()
    if a.retry_delay < 1 or a.max_retries < 0:
        p.error('Positive delay and nonnegative retry limit required')
    if a.action == 'configure':
        if not a.repo:
            p.error('--repo is required for configure')
        selected = {k: os.environ[k] for k in KEYS if k in os.environ}
        for key in ('DCACHE_PYTHON', 'DCACHE_RUN_DIR', 'DCACHE_DATA_DIR',
                    'DCACHE_RESUME_CKPT', 'DCACHE_TRANSFER_MANIFEST'):
            if key not in selected:
                p.error(f'Missing required launch environment {key}')
        write_json(a.config, {'version': 1, 'repo': str(Path(a.repo).resolve()), 'env': selected})
        print(f'Saved restart settings to {a.config}; no process started.')
        print(json.dumps(selected, indent=2))
        return 0
    return run(a.config, a.retry_delay, a.max_retries)


if __name__ == '__main__':
    sys.exit(main())
