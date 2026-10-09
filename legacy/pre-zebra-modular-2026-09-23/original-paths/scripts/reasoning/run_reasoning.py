#!/usr/bin/env python3
"""Repository-root-safe entry point; all commands are explicit (no downloads)."""
from pathlib import Path
import os
import sys

root = Path(__file__).resolve().parents[2]
# Configure before importing torch/matplotlib/tempfile: direct Python commands
# have the same persistent runtime paths as the shell launcher.
runtime = root / '.cache/runtime/reasoning'
for variable, subdir in (('TMPDIR', 'tmp'), ('TMP', 'tmp'), ('TEMP', 'tmp'),
                         ('MPLCONFIGDIR', 'matplotlib'),
                         ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                         ('TRITON_CACHE_DIR', 'triton')):
    configured = os.environ.get(variable)
    if not configured or configured == '/tmp' or configured.startswith('/tmp/'):
        os.environ[variable] = str(runtime / subdir)
    Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root))
from reasoning.runner import main

if __name__ == '__main__':
    main()
