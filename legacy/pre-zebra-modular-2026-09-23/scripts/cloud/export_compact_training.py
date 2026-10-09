#!/usr/bin/env python3
"""Export original packed rows for a bounded, exact-order epoch-zero continuation.

Does NOT retokenize raw documents. A packed row can span documents, and the
original tokenizer/grouping batch boundaries are part of its identity.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import datasets
import numpy as np
import torch

from compact_training import DESCRIPTOR, digest, order_indices
from scripts.cloud.check_h100_resume import CACHE_NAMES, verify_transfer


def export(source, destination, manifest, checkpoint, end_step=5000):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError('Destination already exists; use a fresh directory (no overwrite)')
    verify_transfer(checkpoint, source, manifest, end_step, cpu_only=True)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    fit = ckpt['loops']['fit_loop']
    if fit['epoch_progress']['current']['completed'] != 0:
        raise ValueError('This exporter supports epoch zero only')
    start_step = ckpt['global_step']
    if start_step != manifest['checkpoint']['global_step']:
        raise ValueError('Export must start at the trusted source checkpoint')
    n = manifest['datasets']['train']['num_rows']
    batch = 512
    if not 0 < start_step < end_step or end_step * batch + 4096 >= n:
        raise ValueError('Invalid range or range reaches epoch boundary')
    # Include a small explicit prefetch allowance, NOT additional optimizer steps.
    stop = end_step * batch + 4096
    # Lightning seeds its ORIGINAL active DistributedSampler with config.seed.
    # The historical on_train_start replacement did not reset that live iterator.
    seed = int(manifest['checkpoint']['scientific_config']['seed'])
    order = order_indices(n, start_step * batch, stop, seed=seed)
    indices = np.sort(order)
    required = len(indices) * 1024 * 4 + manifest['datasets']['validation']['size_bytes']
    if shutil.disk_usage(destination.parent).free < required + 2 * 1024**3:
        raise ValueError('Insufficient export space (including 2 GiB reserve)')
    destination.mkdir()
    np.save(destination / 'training_order.npy', order, allow_pickle=False)
    np.save(destination / 'original_indices.npy', indices, allow_pickle=False)
    tokens = np.lib.format.open_memmap(destination / 'tokens.npy', mode='w+',
                                     dtype=np.int32, shape=(len(indices), 1024))
    data = datasets.load_from_disk(str(source / CACHE_NAMES['train'])).with_format('numpy')
    for begin in range(0, len(indices), 2048):
        end = min(begin + 2048, len(indices))
        rows = data[indices[begin:end].tolist()]
        values = np.asarray(rows['input_ids'])
        if (values.shape != (end-begin, 1024) or not np.issubdtype(values.dtype, np.integer)
                or values.min() < 0 or values.max() > np.iinfo(np.int32).max
                or not np.all(np.asarray(rows['attention_mask']) == 1)):
            raise ValueError('Source tokens/masks cannot be stored losslessly in compact format')
        tokens[begin:end] = values
        if begin % (2048 * 50) == 0:
            tokens.flush()
            print(f'Exported {end:,}/{len(indices):,} original packed rows', flush=True)
    tokens.flush()
    del tokens
    shutil.copytree(source / CACHE_NAMES['validation'], destination / CACHE_NAMES['validation'])
    meta = {'version': 1, 'original_num_rows': n, 'physical_rows': len(indices),
            'start_step': start_step, 'end_step': end_step, 'epoch': 0,
            'sampler_seed': seed, 'global_batch': batch, 'sequence_length': 1024,
            'export_stop_offset': stop, 'prefetch_rows': 4096,
            'index_semantics': 'Zero-based ORIGINAL packed-row IDs, not raw document IDs. '
                               'training_order.npy is the original epoch-zero permutation slice; '
                               'original_indices.npy maps sorted physical token rows to logical IDs.',
            'files': {}}
    for name in ('tokens.npy', 'original_indices.npy', 'training_order.npy'):
        path = destination / name
        meta['files'][name] = {'size_bytes': path.stat().st_size, 'sha256': digest(path)}
    # Write completion descriptor last. Interrupted exports cannot be loaded.
    updated = dict(manifest, compact_training=meta)
    (destination / 'transfer_manifest.json').write_text(json.dumps(updated, indent=2) + '\n')
    (destination / DESCRIPTOR).write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({'status': 'FINISHED', 'start_step': start_step, 'end_step': end_step,
                      'training_examples': (end_step-start_step)*batch,
                      'prefetch_rows': 4096, 'destination': str(destination),
                      'payload_gib': (sum(v['size_bytes'] for v in meta['files'].values())
                                      + manifest['datasets']['validation']['size_bytes']) / 2**30}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--destination', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', default='experiments/h100_transfer_manifest.json')
    p.add_argument('--end-step', type=int, default=5000)
    a = p.parse_args()
    export(a.source, a.destination, json.loads(Path(a.manifest).read_text()), a.checkpoint, a.end_step)
