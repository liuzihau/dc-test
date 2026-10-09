#!/usr/bin/env python3
"""Foreground overnight reasoning coordinator, with project-local runtime files."""
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
runtime = root / '.cache/runtime/reasoning'
for variable, subdirectory in (('TMPDIR', 'tmp'), ('TMP', 'tmp'), ('TEMP', 'tmp'),
                              ('MPLCONFIGDIR', 'matplotlib'), ('CUDA_CACHE_PATH', 'cuda'),
                              ('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor')):
    os.environ[variable] = str(runtime / subdirectory)
    Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root))
from reasoning.overnight_queue import main

if __name__ == '__main__':
    main()
