"""Task-local one-shot wrapper for the authorized combination-ICA run."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "quantization_aware_cpa"))
sys.path.insert(0, str(ROOT))

from qacpa.signed_combination_ica_experiment import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
