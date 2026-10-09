#!/usr/bin/env python3
"""Move explicitly classified legacy artifacts out of the canonical outputs."""

import argparse
from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = REPO_ROOT / 'outputs'
ARCHIVE = REPO_ROOT / 'archive' / 'legacy_outputs_2026-08-25'

CANONICAL = {
  'owt-mdlm-pretrain-5k-2x3090',
  'owt-mdlm-objective-matched-5k',
  'owt-dcache-v2-pretrain-5k-2x3090',
  'owt-dcache-final-state-pretrain-5k-2x3090',
  'eval-three-way-5k-teacher-forced',
  'eval-v2-checkpoint-trend',
  'eval-v2-same-state-recurrence',
}

LEGACY = {
  'dcache-v2-ddp-health-2x3090-no-ckpt': 'DDP health check',
  'dcache-v2-ddp-health-4x3090': 'DDP health check',
  'dcache_full_profile': 'memory profile',
  'dcache_pretrain_profile': 'memory profile',
  'dcache_shifted_smoke': 'smoke checkpoint',
  'dcache_smoke': 'smoke checkpoint',
  'dcache_v2_profile_mb1': 'memory profile',
  'dcache_v2_profile_mb2': 'memory profile',
  'eval-5k-teacher-forced': 'DCache-v1 evaluation',
  'eval-objective-v2-5k-teacher-forced': 'superseded two-way evaluation',
  'eval-v2-5k-teacher-forced': 'superseded two-way evaluation',
  'openwebtext-train': 'superseded launch output',
  'owt-dcache-50k': 'obsolete early DCache trial',
  'owt-dcache-pretrain-5k-2x4090': 'DCache-v1 run',
  'owt-dcache-pretrain-ddp-smoke': 'DDP smoke run',
  'owt-dcache-v2-pretrain-5k-2x4090': 'duplicate DCache-v2 server copy',
  'owt-dcachehooping-pretrain-5k-2x3090': 'failed full DCachehooping run',
  'owt-dcachehooping-pretrain-5k-2x3090-exclusive':
    'incomplete full DCachehooping run',
  'owt-mdlm-objective-matched-smoke': 'objective smoke run',
  'owt-mdlm-pretrain-ddp-smoke': 'vanilla DDP smoke run',
  'current-3090-vs-4090-loss.png': 'obsolete hardware comparison',
  'dcache-final-state-live-early-step500-smooth60.png':
    'superseded live plot',
  'dcache-final-state-live-matched-step500-smooth60.png':
    'superseded live plot',
  'dcache-v2-final-health-t1.png': 'noncanonical T1 plot',
  'dcache-v2-final-health-t2.png': 'superseded standalone plot',
  'dcache-v2-final-validation-t0-t3-after-900.png':
    'superseded standalone plot',
  'dcache-v2-live-health-smooth30.png': 'obsolete smoothing',
  'dcache-v2-live-health-t1.png': 'noncanonical T1 plot',
  'dcache-v2-live-health.png': 'obsolete live plot',
  'dcache-v2-validation-t0-t3-after-900.png': 'obsolete live plot',
  'dcachehooping-live-matched-step300-smooth60.png':
    'failed full DCachehooping plot',
  'final-5k-3090-vs-4090-loss.png': 'obsolete hardware comparison',
  'vanilla-v2-objective-matched-live-t2-smooth60.png':
    'superseded three-way plot',
}

ROOT_LEGACY = {
  'metrics.csv': 'stray copied DCache-v1 metrics file',
}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--apply', action='store_true',
    help='Perform moves. Without this flag, only print the exact plan.')
  args = parser.parse_args()
  overlap = CANONICAL & set(LEGACY)
  if overlap:
    raise RuntimeError(f'Canonical entries cannot be archived: {overlap}')

  candidates = []
  for name, reason in LEGACY.items():
    source = OUTPUTS / name
    if source.exists():
      candidates.append((source, ARCHIVE / name, reason))
  for name, reason in ROOT_LEGACY.items():
    source = REPO_ROOT / name
    if source.exists():
      candidates.append((source, ARCHIVE / f'root-{name}', reason))
  for source, destination, reason in candidates:
    print(f'{source.relative_to(REPO_ROOT)} -> '
          f'{destination.relative_to(REPO_ROOT)} [{reason}]')
  if not args.apply:
    print(f'Dry run: {len(candidates)} artifacts; use --apply to move them.')
    return

  ARCHIVE.mkdir(parents=True, exist_ok=True)
  for source, destination, _ in candidates:
    if destination.exists():
      raise FileExistsError(f'Archive destination already exists: {destination}')
    shutil.move(str(source), str(destination))
  print(f'Archived {len(candidates)} artifacts under {ARCHIVE}')


if __name__ == '__main__':
  main()
