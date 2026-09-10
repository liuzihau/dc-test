#!/usr/bin/env python3
"""Run paired reveal-schedule diagnostics on independent physical GPU workers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
sys.path.insert(0, str(ROOT))
from scripts.eval.run_recurrence_audit import (  # noqa: E402
  CHECKPOINTS, atomic_json, now, preflight,
)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--gpus', nargs='+', default=['2', '3'])
  parser.add_argument('--hours', type=float, default=24.0)
  parser.add_argument('--output-root', type=Path,
                      default=ROOT / 'outputs/eval-anchor-schedule-5k')
  parser.add_argument('--data-dir', type=Path, default=ROOT / '.cache/huggingface')
  parser.add_argument('--examples', type=int, default=800)
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--num-workers', type=int, default=2)
  parser.add_argument('--seeds', type=int, nargs='+',
                      default=[20260812, 20260906, 20260907])
  parser.add_argument('--variants', nargs='+', choices=list(CHECKPOINTS),
                      default=list(CHECKPOINTS))
  parser.add_argument('--sources', nargs='+', choices=['teacher', 'model'],
                      default=['teacher', 'model'])
  parser.add_argument('--initial-mask-ratios', type=float, nargs='+',
                      default=[.5, 1.0])
  parser.add_argument('--probe-count', type=int, default=128)
  parser.add_argument('--stride', type=int, default=32)
  parser.add_argument('--score-rounds', type=int, nargs='+',
                      default=[0, 1, 2, 4, 8, 16, 24, 32])
  parser.add_argument('--bootstrap-samples', type=int, default=2000)
  parser.add_argument('--preflight-only', action='store_true')
  parser.add_argument('--plot-only', action='store_true')
  return parser.parse_args(argv)


def validate_protocol(args):
  if not 1 <= args.stride <= 1024 or 1024 % args.stride:
    raise ValueError('Stride must be a divisor of sequence length 1024')
  if not 1 <= args.probe_count < 1023:
    raise ValueError('Probe count must be between 1 and 1022')
  if not all(math.isfinite(x) and 0 < x <= 1 for x in args.initial_mask_ratios):
    raise ValueError('Initial mask ratios must be in (0, 1]')
  if min(round(x * 1023) for x in args.initial_mask_ratios) <= args.probe_count:
    raise ValueError('Each initial mask must contain probes plus revealable tokens')
  if not all(0 <= x <= args.stride for x in args.score_rounds):
    raise ValueError('Score rounds must lie between zero and stride')
  if 0 not in args.score_rounds or args.stride not in args.score_rounds:
    raise ValueError('Score rounds must include initial and final round')
  for name in ('initial_mask_ratios', 'score_rounds', 'sources'):
    values = getattr(args, name)
    if len(values) != len(set(values)):
      raise ValueError(f'Duplicate entries in {name}')


def plot(args):
  return subprocess.run([
    sys.executable, '-u', str(ROOT / 'scripts/eval/plot_anchor_schedule.py'),
    '--input-root', str(args.output_root),
    '--output-dir', str(args.output_root / 'comparison'),
    '--bootstrap-samples', str(args.bootstrap_samples),
    '--expected-variants', *args.variants,
    '--expected-seeds', *map(str, args.seeds),
  ], cwd=ROOT).returncode


def evaluator_command(args, variant, seed, remaining_hours):
  return [
    sys.executable, '-u', str(ROOT / 'scripts/eval/eval_anchor_schedule.py'),
    '--variant', variant, '--checkpoint', str(ROOT / CHECKPOINTS[variant]),
    '--data-dir', str(args.data_dir),
    '--output-dir', str(args.output_root / variant / f'seed-{seed}'),
    '--examples', str(args.examples), '--batch-size', str(args.batch_size),
    '--num-workers', str(args.num_workers), '--seed', str(seed),
    '--device', 'cuda:0', '--max-hours', str(remaining_hours),
    '--probe-count', str(args.probe_count), '--stride', str(args.stride),
    '--sources', *args.sources,
    '--initial-mask-ratios', *map(str, args.initial_mask_ratios),
    '--score-rounds', *map(str, args.score_rounds),
  ]


def run(args):
  args.output_root.mkdir(parents=True, exist_ok=True)
  lock_file = (args.output_root / '.runner.lock').open('a+')
  try:
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except BlockingIOError as error:
    lock_file.close()
    raise RuntimeError(f'Another runner owns {args.output_root}') from error
  logs = args.output_root / 'logs'
  logs.mkdir(exist_ok=True)
  deadline = time.monotonic() + args.hours * 3600
  stopping = threading.Event()
  status_lock = threading.Lock()
  processes = {}
  jobs = queue.Queue()
  status = {
    'evaluation': 'anchor_schedule', 'status': 'running', 'started_at': now(),
    'hours': args.hours, 'gpus': args.gpus, 'jobs': {},
    'protocol': {key: getattr(args, key) for key in (
      'examples', 'batch_size', 'seeds', 'variants', 'sources',
      'initial_mask_ratios', 'probe_count', 'stride', 'score_rounds')},
  }
  for seed in args.seeds:
    for variant in args.variants:
      key = f'{variant}/seed-{seed}'
      status['jobs'][key] = {'status': 'pending'}
      jobs.put((variant, seed, key))
  atomic_json(args.output_root / 'RUN_STATUS.json', status)

  def stop(signum, frame):
    stopping.set()
    print('Stop requested; active workers will finish their atomic part.', flush=True)
    for process in list(processes.values()):
      try:
        if process.poll() is None:
          process.send_signal(signal.SIGTERM)
      except ProcessLookupError:
        pass

  previous_handlers = {number: signal.signal(number, stop)
                       for number in (signal.SIGINT, signal.SIGTERM)}

  def update(key, **values):
    with status_lock:
      status['jobs'][key].update(values)
      status['updated_at'] = now()
      atomic_json(args.output_root / 'RUN_STATUS.json', status)

  def worker(gpu):
    while not stopping.is_set() and deadline - time.monotonic() > 30:
      try:
        variant, seed, key = jobs.get_nowait()
      except queue.Empty:
        return
      directory = args.output_root / variant / f'seed-{seed}'
      log_path = logs / f'{variant}-seed-{seed}.log'
      update(key, status='running', gpu=gpu, log=str(log_path), started_at=now())
      print(f'GPU {gpu}: {key}; log {log_path}', flush=True)
      try:
        with log_path.open('a', buffering=1) as handle:
          handle.write(f'\n=== Invocation {now()} ===\n')
          process = subprocess.Popen(
            evaluator_command(args, variant, seed, (deadline - time.monotonic()) / 3600),
            cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
            stdout=handle, stderr=subprocess.STDOUT)
          processes[gpu] = process
          if stopping.is_set():
            process.send_signal(signal.SIGTERM)
          code = process.wait()
          processes.pop(gpu, None)
        progress_path = directory / 'STATUS.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        result = ('complete' if code == 0 and progress.get('status') == 'complete'
                  else 'paused' if code == 75 else 'failed')
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
    states = [job['status'] for job in status['jobs'].values()]
    failed = 'failed' in states
    complete = all(state == 'complete' for state in states)
    status['status'] = 'failed' if failed else 'complete' if complete else 'paused'
    status['evaluation_finished_at'] = now()
    atomic_json(args.output_root / 'RUN_STATUS.json', status)
    if any(args.output_root.glob('*/seed-*/parts/*/*.csv')):
      print('Generating paired reports and figures.', flush=True)
      plot_code = plot(args)
      status['plot_exit_code'] = plot_code
    else:
      plot_code = 0
      print('No completed parts yet. Rerun the same command to resume.', flush=True)
    status['finished_at'] = now()
    atomic_json(args.output_root / 'RUN_STATUS.json', status)
    print(f'Anchor audit {status["status"]}; status: '
          f'{args.output_root / "RUN_STATUS.json"}', flush=True)
    if not complete:
      print('Run the identical command to resume missing parts.', flush=True)
    return 1 if failed or plot_code else 0
  finally:
    for number, handler in previous_handlers.items():
      signal.signal(number, handler)
    fcntl.flock(lock_file, fcntl.LOCK_UN)
    lock_file.close()


def main(argv=None):
  args = parse_args(argv)
  validate_protocol(args)
  preflight(args)
  if args.preflight_only:
    return 0
  if args.plot_only:
    return plot(args)
  return run(args)


if __name__ == '__main__':
  raise SystemExit(main())
