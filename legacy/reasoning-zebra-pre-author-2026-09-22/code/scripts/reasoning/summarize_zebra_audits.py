#!/usr/bin/env python3
"""CPU-only report, with plotting cache kept out of /tmp."""
from pathlib import Path
import os
import sys

root = Path(__file__).resolve().parents[2]
runtime = root / '.cache/runtime/reasoning'
for key, subdirectory in (('TMPDIR', 'tmp'), ('TMP', 'tmp'), ('TEMP', 'tmp'), ('MPLCONFIGDIR', 'matplotlib')):
    if not os.environ.get(key) or os.environ[key] == '/tmp' or os.environ[key].startswith('/tmp/'):
        os.environ[key] = str(runtime / subdirectory)
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root))
from reasoning.zebra_audit_report import main

if __name__ == '__main__':
    main()
