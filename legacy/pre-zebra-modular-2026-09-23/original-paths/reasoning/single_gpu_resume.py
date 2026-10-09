"""Explicit, audited multi-GPU -> single-GPU fork for plain MDM only.

Model, optimizer, global batch, microbatch and example cursor are preserved.
Only the world size and rank RNG list change. This is not bitwise replay.
The original run is never overwritten; stop it before invoking this module.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import torch

from .runner import atomic_json, digest, examples_at_step, load_checkpoint


def fork_single_gpu(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Destination must be separate from source')
    if destination.exists():
        raise FileExistsError('Destination already exists; resume it, do not overwrite it')
    with (source / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = (source / 'checkpoints/last.pt').resolve(strict=True)
        checkpoint = load_checkpoint(path)
        contract = json.loads((source / 'contract.json').read_text())
        model = contract['model_config']
        if (checkpoint['contract'] != contract or checkpoint['model_config'] != model
                or contract.get('suite') != 'split' or contract.get('variant') != 'mdm'
                or model.get('memory_mode') != 'none' or model.get('neighbors')
                or model.get('trajectory') != 'single'
                or model.get('attention_mode') != 'vanilla'):
            raise ValueError('Only a consistent plain single-forward MDM run is supported')
        world, micro, batch = (contract[k] for k in ('world_size', 'micro_batch', 'global_batch'))
        if world <= 1 or micro < 1 or batch % (world * micro):
            raise ValueError('Expected valid multi-GPU batch geometry')
        if (len(checkpoint['rng_by_rank']) != world
                or checkpoint['examples_seen'] != examples_at_step(checkpoint['step'], contract)):
            raise ValueError('Invalid RNG list or data cursor')
        updated = dict(contract, world_size=1)
        provenance = dict(
            source_run=str(source), source_checkpoint=str(path), source_sha256=digest(path),
            source_step=checkpoint['step'], examples_seen=checkpoint['examples_seen'],
            source_contract=contract, destination_contract=updated,
            change='world_size -> 1; retain micro/global batch; retain rank-zero RNG',
            preserved='model, optimizer, absolute step, data cursor, LR, sampler, epoch budget',
            bitwise_replay=False, created_ns=time.time_ns())
        checkpoint['contract'] = updated
        checkpoint['rng_by_rank'] = checkpoint['rng_by_rank'][:1]
        destination.parent.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix='.' + destination.name + '.', dir=destination.parent))
        checkpoints = scratch / 'checkpoints'
        checkpoints.mkdir()
        target = checkpoints / path.name
        torch.save(checkpoint, target)
        with target.open('rb') as stream:
            os.fsync(stream.fileno())
        atomic_json(target.with_suffix('.pt.json'), dict(
            file=target.name, step=checkpoint['step'], sha256=digest(target),
            size=target.stat().st_size, created_ns=time.time_ns()))
        (checkpoints / 'last.pt').symlink_to(target.name)
        atomic_json(scratch / 'contract.json', updated)
        atomic_json(scratch / 'geometry_migration.json', provenance)
        # Keep training history; normal resume-attempt semantics supersede any
        # source log rows after the saved checkpoint once training restarts.
        if (source / 'logs').exists():
            shutil.copytree(source / 'logs', scratch / 'logs')
        if (source / 'validation').exists():
            (scratch / 'validation').mkdir()
            for result in (source / 'validation').glob('step-*.json'):
                if int(result.stem.split('-')[1]) <= checkpoint['step']:
                    shutil.copy2(result, scratch / 'validation' / result.name)
        load_checkpoint(target)  # Verify the serialized full-state fork.
        scratch.rename(destination)
        return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--trust-checkpoint', required=True, action='store_true')
    args = parser.parse_args()
    print(json.dumps(fork_single_gpu(args.source, args.destination), indent=2))
