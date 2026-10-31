"""Task-specific startup guard for `python -m qacpa.run_same80_bits_r100`."""
from pathlib import Path
import sys

_project = Path(__file__).resolve().parent
_workspace = _project.parent
_ai = _workspace / ".ai"
for _path in (str(_workspace), str(_ai)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cpa_same80_guard import enforce_if_disabled

enforce_if_disabled()
