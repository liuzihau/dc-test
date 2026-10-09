#!/usr/bin/env python3
"""CPU official-source Zebra report; no source pickle or model loading."""
from pathlib import Path
import os
import sys

root = Path(__file__).resolve().parents[2]
runtime = root / '.cache/runtime/reasoning'
for variable, subdirectory in (('TMPDIR', 'tmp'), ('TMP', 'tmp'), ('TEMP', 'tmp'), ('MPLCONFIGDIR', 'matplotlib')):
    if not os.environ.get(variable) or os.environ[variable] == '/tmp' or os.environ[variable].startswith('/tmp/'):
        os.environ[variable] = str(runtime / subdirectory)
    Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root))
from reasoning.zebra_official_report import main

if __name__ == '__main__':
    main()
