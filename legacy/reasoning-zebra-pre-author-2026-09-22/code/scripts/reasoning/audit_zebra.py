#!/usr/bin/env python3
"""Repository-root-safe, non-training Zebra checkpoint audit."""
from pathlib import Path
import os
import sys

root = Path(__file__).resolve().parents[2]
runtime = root / '.cache/runtime/reasoning'
for variable, subdir in (('TMPDIR', 'tmp'), ('TMP', 'tmp'), ('TEMP', 'tmp'),
                         ('MPLCONFIGDIR', 'matplotlib'),
                         ('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('TRITON_CACHE_DIR', 'triton')):
    value = os.environ.get(variable)
    if not value or value == '/tmp' or value.startswith('/tmp/'):
        os.environ[variable] = str(runtime / subdir)
    Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root))

from reasoning.prompt_audit import main

if __name__ == '__main__':
    main()
