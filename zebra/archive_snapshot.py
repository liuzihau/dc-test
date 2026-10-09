"""Non-destructive legacy snapshot while source relocation awaits approval."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'legacy/pre-zebra-modular-2026-09-23'
TARGETS = [
    'configs', 'models', 'reasoning', 'scripts', 'tests', 'docs', 'experiments', 'ssd-lm',
    'checkpoint_resume.py', 'compact_training.py', 'dataloader.py', 'diffusion.py',
    'main.py', 'metrics.py', 'metrics_history.py', 'neighbor_prediction.py',
    'noise_schedule.py', 'push_to_hf.py', 'recovery_training.py',
    'recurrent_gradients.py', 'rollout_utils.py', 'utils.py', 'README.md',
]


def main():
    if DEST.exists():
        raise RuntimeError(f'Snapshot exists: {DEST}')
    DEST.mkdir(parents=True)
    files = []
    for name in TARGETS:
        source = ROOT/name
        paths = [source] if source.is_file() else sorted(source.rglob('*'))
        for path in paths:
            if path.is_file() and '__pycache__' not in path.parts:
                relative = path.relative_to(ROOT)
                target = DEST / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f'Copy mismatch: {relative}')
                files.append(dict(path=str(relative), size=path.stat().st_size, sha256=digest))
    (DEST/'archive_manifest.json').write_text(json.dumps(dict(
        status='COPIED; original source paths retained pending relocation approval',
        targets=TARGETS, files=files), indent=2)+'\n')
    print(f'Copied and verified {len(files)} source files to {DEST}. No source paths removed.')


if __name__ == '__main__':
    main()
