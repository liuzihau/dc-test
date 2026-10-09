"""Keep revised plots current while the already-running queue uses old code."""
import argparse
import json
import os
import time
from pathlib import Path

from zebra.schedule import refresh


def snapshot(root):
    return tuple(sorted((str(path), path.stat().st_mtime_ns)
        for path in root.glob('*/generation/epoch-*/complete.json')))


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    root = args.root.resolve()
    previous = None
    while True:
        current = json.loads((root/'current.json').read_text())
        observed = snapshot(root)
        if observed != previous:
            refresh(root)
            print(f'Refreshed {len(observed)} generation points', flush=True)
            previous = observed
        if current.get('stage') == 'complete' or not alive(int(current['queue_pid'])):
            refresh(root)
            print('Queue ended; final plots refreshed.', flush=True)
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
