from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / ".ai/execution-results/CPA_SCREEN3_SLIDE_PNG_5_20260918/render_int5_slide_pngs.py"
)
SPEC = importlib.util.spec_from_file_location("screen3_int5_slide_adapter", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_int5_selection_and_output_contract() -> None:
    assert MODULE.SELECTION == [5, 7, 6]
    assert MODULE.EXPECTED_DIMENSIONS == (3840, 2160)
    assert MODULE.EXPECTED_DPI == 240


def test_int5_source_and_role_bindings_are_fixed() -> None:
    assert MODULE.EVALUATION_DIR.name == "evaluation_recovery_20260918"
    assert "signed_int5_randomized8_screen3_fia" in MODULE.EVALUATION_DIR.parent.name
    assert MODULE.OUTPUT_ROOT.name == "slide_png_5_20260918"
