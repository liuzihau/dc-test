"""Full released training data, frozen benchmark-v2 holdouts, compact mmap storage.

No random training subset. Every distinct training puzzle is retained except
frozen validation identities and ALL official test identities. Original row IDs
and prompt fingerprints accompany each packed row. No source solver traces enter
the tensors. Publication is atomic; interrupted preparation never looks complete.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .benchmark import PROTOCOL, _puzzle_keys, convert_sudoku, convert_zebra
from .data import ReasoningDataset, _sha256, encode_record
from .tasks import task_identity


def record_key(record):
    if record['task'] == 'sudoku-benchmark':
        board = bytes(0 if x == '[MASK]' else int(x) for x in record['prompt'])
        return hashlib.sha256(board).digest()
    return hashlib.sha256(task_identity(record).encode()).digest()


def prepare_full(task, train_file, test_file, frozen_dir, output):
    from .zebra_official import load_source, convert_source_record
    frozen_dir, output = Path(frozen_dir), Path(output)
    if task not in ('zebra-benchmark', 'sudoku-benchmark'):
        raise ValueError('Only released Zebra and Sudoku-Puzzle are supported')
    if output.exists():
        raise FileExistsError('Refusing to replace full dataset: ' + str(output))
    heldout = {s: ReasoningDataset(frozen_dir, s) for s in ('validation', 'test')}
    old = heldout['validation'].manifest
    if old['task'] != task or old['benchmark_protocol'] != PROTOCOL:
        raise ValueError('Frozen benchmark task/protocol mismatch')
    if any(not len(d) for d in heldout.values()):
        raise ValueError('Frozen validation/test must be nonempty')
    sources = {}
    for split, path in (('train', train_file), ('test', test_file)):
        sha = _sha256(path)
        if sha != old['source']['source_files'][split]['sha256']:
            raise ValueError('Raw source differs from frozen holdout provenance: ' + split)
        sources[split] = dict(path=str(Path(path).resolve()), sha256=sha)
    if task == 'sudoku-benchmark':
        arrays = {s: np.load(p, mmap_mode='r', allow_pickle=False)
                  for s, p in (('train', train_file), ('test', test_file))}
        for a in arrays.values():
            if a.ndim != 2 or a.shape[1] != 325 or a.dtype.kind not in 'iu':
                raise ValueError('Expected released numeric Sudoku N x 325 source')
        test_keys = set(_puzzle_keys(arrays['test']))
        # Keys are computed in vectorized chunks; validate each retained solution.
        rows = ((i, key, row) for i, (key, row) in enumerate(
            zip(_puzzle_keys(arrays['train']), arrays['train'])))
        convert = lambda i, row: convert_sudoku(row, 'train', i)
    else:
        arrays = {'test': load_source(test_file), 'train': load_source(train_file)}
        test_keys = {record_key(convert_zebra(convert_source_record(raw, 'test', i)))
                     for i, raw in enumerate(arrays['test'])}
        def zebra_rows():
            for i, raw in enumerate(arrays['train']):
                record = convert_zebra(convert_source_record(raw, 'train', i))
                yield i, record_key(record), record
        rows = zebra_rows()
        convert = lambda i, row: row
    for split in sources:
        sources[split]['records'] = len(arrays[split])
    validation_keys = {record_key(r) for r in heldout['validation'].records}
    if validation_keys & test_keys:
        raise ValueError('Frozen validation leaks into the official test source')
    if not {record_key(r) for r in heldout['test'].records} <= test_keys:
        raise ValueError('Frozen test is not a subset of the official test source')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '.prepare-', dir=output.parent))
    counters = dict(raw_train=len(arrays['train']), excluded_test=0,
                    excluded_validation=0, duplicate_train=0, retained_train=0)
    files = {name: stage / ('train.' + name + '.bin')
             for name in ('tokens', 'layout', 'source_indices', 'identities')}
    seen, found_validation = set(), set()
    template = heldout['validation']
    with ExitStack() as stack:
        writers = {name: stack.enter_context(path.open('xb')) for name, path in files.items()}
        for index, key, row in rows:
            if key in test_keys:
                counters['excluded_test'] += 1
            elif key in validation_keys:
                counters['excluded_validation'] += 1
                found_validation.add(key)
            elif key in seen:
                counters['duplicate_train'] += 1
            else:
                record = convert(index, row)
                if record_key(record) != key:
                    raise ValueError('Packed identity does not match source identity')
                clean, prefix, used = encode_record(record, template.tokenizer, template.max_length)
                writers['tokens'].write(np.asarray(clean, dtype='u1').tobytes())
                writers['layout'].write(np.asarray([prefix, used], dtype='<u2').tobytes())
                writers['source_indices'].write(np.asarray([index], dtype='<i8').tobytes())
                writers['identities'].write(key)
                seen.add(key)
                counters['retained_train'] += 1
            if (index + 1) % 25000 == 0:
                print(json.dumps(dict(task=task, processed=index + 1, **counters)), flush=True)
    if found_validation != validation_keys:
        raise ValueError('Frozen validation puzzle absent from raw training source')
    if not counters['retained_train']:
        raise ValueError('No training puzzles remain')
    if sum(counters[k] for k in ('excluded_test', 'excluded_validation', 'duplicate_train', 'retained_train')) != counters['raw_train']:
        raise AssertionError('Source accounting failed')
    packed = dict(storage='packed_uint8_v1', records=counters['retained_train'], files={
        name: dict(filename=p.name, sha256=_sha256(p), size_bytes=p.stat().st_size)
        for name, p in files.items()})
    splits = {'train': packed}
    for split in ('validation', 'test'):
        entry = old['splits'][split]
        shutil.copyfile(frozen_dir / entry['filename'], stage / entry['filename'])
        if _sha256(stage / entry['filename']) != entry['sha256']:
            raise ValueError('Holdout copy checksum mismatch')
        splits[split] = entry
    manifest = {**old, 'schema_version': 2, 'splits': splits, 'source': dict(
        kind='full_released_training_unique_puzzles', source_files=sources,
        frozen_manifest_sha256=_sha256(frozen_dir / 'manifest.json'),
        split_policy='all unique source train minus frozen validation and ALL official test identities',
        row_order='source order; epoch shuffling belongs to trainer, seed=1',
        provenance='packed source_indices are zero-based original source row numbers; identities are SHA256 prompt fingerprints',
        benchmark_equivalence=False, protocol=PROTOCOL, counters=counters,
        solver_order_used=False, strategy_tokens_used=False)}
    (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    ReasoningDataset(stage, 'train')  # Verify sizes, hashes and ranges before publishing.
    stage.rename(output)
    print(json.dumps(dict(status='prepared', task=task, output=str(output), **counters)), flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('zebra-benchmark', 'sudoku-benchmark'), required=True)
    for name in ('train-file', 'test-file', 'frozen-dir', 'output'):
        parser.add_argument('--' + name, required=True)
    prepare_full(**vars(parser.parse_args(argv)))


if __name__ == '__main__':
    main()
