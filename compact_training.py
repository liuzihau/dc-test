"""Physical subset with the ORIGINAL logical indices for deterministic DDP resume."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

DESCRIPTOR = 'compact_train.json'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def order_indices(n, start, end, seed=0, epoch=0):
    g = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(n, generator=g)[start:end].numpy().copy()


class CompactTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / DESCRIPTOR).read_text())
        m = self.meta
        if m['version'] != 1 or m['sequence_length'] != 1024:
            raise ValueError('Unsupported compact training format')
        self.indices = np.load(self.directory / 'original_indices.npy', mmap_mode='r', allow_pickle=False)
        self.tokens = np.load(self.directory / 'tokens.npy', mmap_mode='r', allow_pickle=False)
        if (self.indices.dtype != np.dtype('int64') or
                self.indices.shape != (m['physical_rows'],) or
                self.tokens.dtype != np.dtype('int32') or
                self.tokens.shape != (m['physical_rows'], 1024)):
            raise ValueError('Compact array shape/dtype mismatch')

    def __len__(self):
        # Essential: randperm must still operate on the ORIGINAL dataset size.
        return self.meta['original_num_rows']

    def __getitem__(self, index):
        physical = int(np.searchsorted(self.indices, index))
        if physical >= len(self.indices) or self.indices[physical] != index:
            raise IndexError(f'Original row {index} is outside the exported training window')
        return {'input_ids': torch.tensor(self.tokens[physical], dtype=torch.long),
                'attention_mask': torch.ones(1024, dtype=torch.float32)}

    def __getstate__(self):
        return {'directory': self.directory}

    def __setstate__(self, state):
        self.__init__(state['directory'])

    def validate_resume(self, *, epoch, counter, world_size, global_batch, max_steps, seed):
        m = self.meta
        offset = counter * world_size
        if (epoch != m['epoch'] or seed != m['sampler_seed'] or
                global_batch != m['global_batch'] or
                not m['start_step'] * global_batch <= offset < m['end_step'] * global_batch or
                max_steps > m['end_step']):
            raise ValueError('Resume sampler/config is outside the compact dataset coverage')


def verify_compact(directory, expected):
    dataset = CompactTrainingDataset(directory)
    m = dataset.meta
    if m != expected:
        raise ValueError('Compact descriptor differs from trusted transfer manifest')
    if set(m['files']) != {'tokens.npy', 'original_indices.npy', 'training_order.npy'}:
        raise ValueError('Invalid compact file inventory')
    for name, info in m['files'].items():
        path = Path(directory) / name
        if path.stat().st_size != info['size_bytes'] or digest(path) != info['sha256']:
            raise ValueError(f'Compact file integrity failure: {name}')
    order = np.load(Path(directory) / 'training_order.npy', allow_pickle=False)
    expected_order = order_indices(m['original_num_rows'], m['start_step'] * m['global_batch'],
                                  m['export_stop_offset'], m['sampler_seed'], m['epoch'])
    if (not np.array_equal(order, expected_order) or
            not np.array_equal(dataset.indices, np.sort(expected_order))):
        raise ValueError('Compact row indices do not reproduce the original sampler')
    return m


def reset_compact_fetcher(fetcher):
    fetcher.teardown()
    iter(fetcher)
