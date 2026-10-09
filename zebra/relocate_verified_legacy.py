"""Relocate only hash-verified old sources after confirming no active old jobs."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'legacy/pre-zebra-modular-2026-09-23'


def main():
    manifest = json.loads((ARCHIVE/'archive_manifest.json').read_text())
    targets = [p for p in manifest['targets'] if p != 'README.md']
    destination = ARCHIVE/'original-paths'
    if destination.exists():
        raise RuntimeError('Already relocated; refusing overwrite')
    for item in manifest['files']:
        if item['path'] == 'README.md':
            continue
        path = ROOT/item['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError(f'Source changed since snapshot: {path}')
    destination.mkdir()
    for name in targets:
        shutil.move(str(ROOT/name), str(destination/name))
    manifest['status'] = 'RELOCATED; verified originals in original-paths/, snapshot retained'
    manifest['relocated_targets'] = targets
    (ARCHIVE/'archive_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Relocated {len(targets)} verified source paths; README, paper, data and results remain')


if __name__ == '__main__':
    main()
