#!/usr/bin/env python
"""CPU-only reporting entry point; keep plotting/runtime files inside the repo."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
runtime = ROOT / '.cache/runtime/reasoning'
for name, suffix in (('TMPDIR', 'tmp'), ('TMP', 'tmp'), ('TEMP', 'tmp'), ('MPLCONFIGDIR', 'matplotlib')):
    directory = runtime / suffix
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(directory)
sys.path.insert(0, str(ROOT))

from reasoning.reporting import main

if __name__ == '__main__':
    main()
