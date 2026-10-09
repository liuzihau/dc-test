#!/usr/bin/env python3
"""Schedule independent, resumable evaluation jobs on two physical GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'outputs/eval-urgent-recurrence-5k'
CHECKPOINTS = {
  'five-forward': 'outputs/owt-dcache-final-state-pretrain-5k-2x3090/checkpoints/0-5000.ckpt',
  'two-forward': 'outputs/owt-dcache-two-forward-pretrain-5k-2x3090/checkpoints/0-5000.ckpt',
  'dcache-v2': 'outputs/owt-dcache-v2-pretrain-5k-2x3090/checkpoints/0-5000.ckpt',
  'objective': 'outputs/owt-mdlm-objective-matched-5k/checkpoints/last.ckpt',
  'vanilla': 'outputs/owt-mdlm-pretrain-5k-2x3090/checkpoints/0-5000.ckpt',
}


def now():
  return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2) + '\n')
  temporary.replace(path)


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--gpus', nargs='+', default=['2', '3'])
  parser.add_argument('--hours', type=float, default=24.0)
  parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument('--data-dir', type=Path,
                      default=ROOT / '.cache/huggingface')
  parser.add_argument('--examples', type=int, default=800)
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--num-workers', type=int, default=2)
  parser.add_argument('--seeds', type=int, nargs='+',
                      default=[20260812, 20260906, 20260907])
  parser.add_argument('--variants', nargs='+', choices=list(CHECKPOINTS),
                      default=list(CHECKPOINTS))
  parser.add_argument('--families', nargs='+',
                      choices=['transitions', 'repeat', 'generated'],
                      default=['transitions', 'repeat', 'generated'])
  parser.add_argument('--ratios', type=float, nargs='+',
                      default=[0.05, 0.10, 0.20, 0.30, 0.50, 0.70])
  parser.add_argument('--jumps', type=float, nargs='+',
                      default=[0.025, 0.05, 0.10, 0.20])
  parser.add_argument('--repeat-steps', type=int, nargs='+', default=[1, 2, 4, 8])
  parser.add_argument('--bootstrap-samples', type=int, default=2000)
  parser.add_argument('--preflight-only', action='store_true')
  parser.add_argument('--plot-only', action='store_true')
  return parser.parse_args()


def preflight(args):
  if not math.isfinite(args.hours) or args.hours <= 0 or len(set(args.gpus)) != len(args.gpus):
    raise ValueError('Use positive hours and distinct physical GPU IDs')
  if any(not gpu.isdigit() for gpu in args.gpus):
    raise ValueError('--gpus accepts physical numeric GPU IDs')
  if args.batch_size < 3 or args.examples <= 0 or args.examples % args.batch_size:
    raise ValueError('Use batch-size >=3 and a positive divisible example count')
  if args.num_workers < 0 or args.bootstrap_samples < 2:
    raise ValueError('Invalid worker/bootstrap count')
  if len(set(args.seeds)) != len(args.seeds):
    raise ValueError('Duplicate seeds would repeat the same experiment')
  if len(set(args.variants)) != len(args.variants):
    raise ValueError('Duplicate variants would create conflicting jobs')
  args.output_root = args.output_root.expanduser().resolve()
  args.data_dir = args.data_dir.expanduser().resolve()
  filesystem_path = args.output_root
  while not filesystem_path.exists():
    filesystem_path = filesystem_path.parent
  storage = os.statvfs(filesystem_path)
  free_gib = storage.f_bavail * storage.f_frsize / 2**30
  print(f'Output filesystem: {free_gib:.1f} GiB free, '
        f'{storage.f_favail:,} free inodes; temporary files: '
        f'{os.environ.get("TMPDIR", "system default")}', flush=True)
  if not args.plot_only and (free_gib < 5 or storage.f_favail < 250000):
    raise RuntimeError('Audit needs headroom: keep at least 5 GiB and 250,000 '
                       'free inodes on the output filesystem')
  dataset = args.data_dir / 'openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat'
  for filename in ('state.json', 'dataset_info.json'):
    if not (dataset / filename).is_file():
      raise FileNotFoundError(dataset / filename)
  for variant in args.variants:
    path = ROOT / CHECKPOINTS[variant]
    if not path.is_file():
      raise FileNotFoundError(path)
    print(f'{variant}: {path}', flush=True)
  if not args.plot_only:
    result = subprocess.run(
      ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total',
       '--format=csv,noheader,nounits'],
      capture_output=True, text=True, check=True)
    print(result.stdout.strip(), flush=True)
    available = {line.split(',')[0].strip() for line in result.stdout.splitlines()}
    if not set(args.gpus) <= available:
      raise ValueError(f'Requested GPUs are not present: {args.gpus}')
    # Check one CUDA device inside each worker's physical GPU mapping.
    for gpu in args.gpus:
      env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
      subprocess.run(
        [sys.executable, '-c',
         'import torch; assert torch.cuda.is_available(); '
         'assert torch.cuda.device_count() == 1; '
         'print("CUDA worker:", torch.cuda.get_device_name(0))'],
        env=env, check=True)
  print(f'Prepared {len(args.variants) * len(args.seeds)} jobs on GPUs '
        f'{",".join(args.gpus)}; limit {args.hours:g} hours; '
        f'{args.examples} documents x {len(args.seeds)} mask seeds.', flush=True)


def plot(args):
  command = [
    sys.executable, '-u', str(ROOT / 'scripts/eval/plot_recurrence_audit.py'),
    '--input-root', str(args.output_root),
    '--output-dir', str(args.output_root / 'comparison'),
    '--bootstrap-samples', str(args.bootstrap_samples),
    '--expected-variants', *args.variants,
    '--expected-seeds', *map(str, args.seeds),
  ]
  return subprocess.run(command, cwd=ROOT).returncode


def run(args):
  args.output_root.mkdir(parents=True, exist_ok=True)
  lock_path = args.output_root / '.runner.lock'
  lock_file = lock_path.open('a+')
  try:
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except BlockingIOError:
    lock_file.close()
    raise RuntimeError(f'Another runner already owns {args.output_root}')
  logs = args.output_root / 'logs'
  logs.mkdir(exist_ok=True)
  deadline = time.monotonic() + args.hours * 3600
  stop = threading.Event()
  status_lock = threading.Lock()
  processes = {}
  jobs = queue.Queue()
  status = {'started_at': now(), 'status': 'running', 'hours': args.hours,
            'gpus': args.gpus, 'jobs': {}}
  for seed in args.seeds:
    for variant in args.variants:
      key = f'{variant}/seed-{seed}'
      status['jobs'][key] = {'status': 'pending'}
      jobs.put((variant, seed, key))
  atomic_json(args.output_root / 'RUN_STATUS.json', status)

  def request_stop(signum, frame):
    stop.set()
    print('Stop requested; asking active evaluations to save their current '
          'batch before exiting.', flush=True)
    for process in list(processes.values()):
      if process.poll() is None:
        process.send_signal(signal.SIGTERM)

  old_handlers = {sig: signal.signal(sig, request_stop)
                  for sig in (signal.SIGINT, signal.SIGTERM)}

  def update(key, **values):
    with status_lock:
      status['jobs'][key].update(values)
      status['updated_at'] = now()
      atomic_json(args.output_root / 'RUN_STATUS.json', status)

  def worker(gpu):
    while not stop.is_set() and time.monotonic() < deadline:
      try:
        variant, seed, key = jobs.get_nowait()
      except queue.Empty:
        return
      remaining = deadline - time.monotonic()
      if remaining < 30:
        return
      directory = args.output_root / variant / f'seed-{seed}'
      log_path = logs / f'{variant}-seed-{seed}.log'
      command = [
        sys.executable, '-u', str(ROOT / 'scripts/eval/eval_recurrence_audit.py'),
        '--variant', variant, '--checkpoint', str(ROOT / CHECKPOINTS[variant]),
        '--data-dir', str(args.data_dir), '--output-dir', str(directory),
        '--examples', str(args.examples), '--batch-size', str(args.batch_size),
        '--num-workers', str(args.num_workers), '--seed', str(seed),
        '--device', 'cuda:0', '--max-hours', str(remaining / 3600),
        '--families', *args.families,
        '--ratios', *map(str, args.ratios),
        '--jumps', *map(str, args.jumps),
        '--repeat-steps', *map(str, args.repeat_steps),
      ]
      update(key, status='running', gpu=gpu, started_at=now(), log=str(log_path))
      print(f'GPU {gpu}: {key}; log {log_path}', flush=True)
      try:
        with log_path.open('a', buffering=1) as handle:
          handle.write(f'\n=== Invocation {now()} ===\n')
          process = subprocess.Popen(
            command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
            stdout=handle, stderr=subprocess.STDOUT)
          processes[gpu] = process
          code = process.wait()
          processes.pop(gpu, None)
        progress_path = directory / 'STATUS.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        complete = code == 0 and progress.get('status') == 'complete'
        paused = code == 75 or progress.get('status') in ('paused', 'partial')
        result = 'complete' if complete else 'paused' if paused else 'failed'
        update(key, status=result, exit_code=code, finished_at=now())
        print(f'GPU {gpu}: {key} {result}', flush=True)
        if result == 'paused':
          return
      except Exception as error:
        update(key, status='failed', error=str(error), finished_at=now())
        print(f'GPU {gpu}: {key} failed: {error}', flush=True)
      finally:
        jobs.task_done()

  try:
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
      futures = [executor.submit(worker, gpu) for gpu in args.gpus]
      for future in futures:
        future.result()
    all_jobs = list(status['jobs'].values())
    failed = any(job['status'] == 'failed' for job in all_jobs)
    complete = all(job['status'] == 'complete' for job in all_jobs)
    status['status'] = 'failed' if failed else 'complete' if complete else 'paused'
    status['finished_at'] = now()
    atomic_json(args.output_root / 'RUN_STATUS.json', status)
    has_parts = any(args.output_root.glob('*/seed-*/parts/*/*.csv'))
    if has_parts:
      plot_code = plot(args)
      status['plot_exit_code'] = plot_code
      atomic_json(args.output_root / 'RUN_STATUS.json', status)
    else:
      plot_code = 0
      print('No completed batches yet; plotting will be available after resume.')
    print(f'Audit {status["status"]}. Status: {args.output_root / "RUN_STATUS.json"}',
          flush=True)
    if not complete:
      print('Run the same command to resume missing work.', flush=True)
    return 1 if failed or plot_code else 0
  finally:
    for sig, previous in old_handlers.items():
      signal.signal(sig, previous)
    fcntl.flock(lock_file, fcntl.LOCK_UN)
    lock_file.close()


def main():
  args = parse_args()
  preflight(args)
  if args.preflight_only:
    return 0
  if args.plot_only:
    return plot(args)
  return run(args)


if __name__ == '__main__':
  sys.exit(main())
