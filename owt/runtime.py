"""Select the BD3 module tree before importing its absolute module names."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/bd3'


def install():
    for name in ('diffusion', 'dataloader', 'metrics', 'models', 'noise_schedule', 'utils'):
        module = sys.modules.get(name)
        if module is not None and not str(getattr(module, '__file__', '')).startswith(str(UPSTREAM)):
            raise RuntimeError(f'{name} already imported from another backbone; use an isolated OWT process')
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(UPSTREAM))
