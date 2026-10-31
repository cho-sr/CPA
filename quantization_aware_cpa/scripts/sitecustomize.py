"""Task-specific startup guard for direct same80 orchestration script launches."""
from pathlib import Path
import sys

_scripts = Path(__file__).resolve().parent
_project = _scripts.parent
_workspace = _project.parent
_ai = _workspace / ".ai"
for _path in (str(_workspace), str(_ai)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cpa_same80_guard import enforce_if_disabled

enforce_if_disabled()
