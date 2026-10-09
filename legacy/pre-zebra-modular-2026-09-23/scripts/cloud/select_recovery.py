#!/usr/bin/env python3
"""Choose a verified full checkpoint; stdout is path or DONE, diagnostics stderr."""
import argparse
import contextlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from recovery_training import committed_receipts, sha256
from scripts.cloud.check_h100_resume import checkpoint_snapshot


def select(directory, source, manifest, target):
    directory, source = Path(directory), Path(source)
    candidates = []
    managed = set()
    for meta, _ in committed_receipts(directory):
        path = directory / meta['file']
        managed.add(path.resolve())
        try:
            if path.stat().st_size != meta['size_bytes'] or sha256(path) != meta['sha256']:
                raise ValueError('size/hash mismatch')
            candidates.append((path, meta))
        except (OSError, ValueError, KeyError) as exc:
            print(f'Ignoring invalid recovery checkpoint {path}: {exc}', file=sys.stderr)
    # Legacy periodic checkpoints are allowed, but NEVER bypass a managed receipt
    # by following last.ckpt to an uncommitted or corrupt recovery file.
    legacy = (list(directory.glob('0-*.ckpt')) + list(directory.glob('legacy-last-*.ckpt'))
              + [directory / 'last.ckpt', source])
    seen = set()
    for path in legacy:
        resolved = path.resolve()
        if (resolved in managed or resolved in seen or
                resolved.name.startswith('recovery-') or not path.is_file()):
            continue
        seen.add(resolved)
        candidates.append((path, None))
    verified = []
    for path, meta in candidates:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                snap = checkpoint_snapshot(path, manifest['checkpoint'],
                    max_steps=2**63-1, allow_descendant=True)
            if meta is not None and snap['global_step'] != meta['step']:
                raise ValueError('receipt/optimizer step mismatch')
            verified.append((snap['global_step'], path.stat().st_mtime_ns, path.resolve(), meta))
        except Exception as exc:
            print(f'Ignoring incompatible checkpoint {path}: {exc}', file=sys.stderr)
    if not verified:
        raise ValueError('No valid full checkpoint remains; refusing to start from scratch')
    step, _, path, meta = max(verified, key=lambda item: item[:2])
    if step >= target:
        if meta is not None and not meta['completed']:
            raise ValueError('Target-step recovery checkpoint is not marked training/validation complete')
        print(f'Training target {target} already reached at {step}.', file=sys.stderr)
        return 'DONE'
    print(f'Recovery selected optimizer step {step}: {path}', file=sys.stderr)
    return str(path)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--target', type=int, required=True)
    a = p.parse_args()
    try:
        print(select(a.directory, a.source, json.loads(Path(a.manifest).read_text()), a.target))
    except Exception as exc:
        print(f'RECOVERY_FATAL: {exc}', file=sys.stderr)
        sys.exit(2)
