from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / ".ai/execution-results/CPA_SCREEN3_SLIDE_PNG_876_20260918/render_slide_pngs.py"
)
SPEC = importlib.util.spec_from_file_location("screen3_slide_renderer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_detail_pages_cover_all_rows_in_order() -> None:
    assert MODULE.page_ranges() == [(0, 4), (4, 8), (8, 12), (12, 16)]
    assert [end - start for start, end in MODULE.page_ranges()] == [4, 4, 4, 4]


def test_final_roles_keep_truth_baseline_and_original_member_ids() -> None:
    selection = [1, 7, 5]
    assert MODULE.role_names(selection) == [
        "Ground truth",
        "Naive dequantization baseline",
        "Public rank 1 | member 01",
        "Public rank 2 | member 07",
        "Public rank 3 | member 05",
    ]
    assert MODULE.candidate_names(selection) == [
        "center_25000",
        "rank1_member_01_25000",
        "rank2_member_07_25000",
        "rank3_member_05_25000",
    ]


def test_export_contract_is_slide_sized_and_non_research() -> None:
    assert MODULE.EXPECTED_DIMENSIONS == (3840, 2160)
    assert MODULE.EXPECTED_DPI == 240
    assert MODULE.STEPS == 25000
    assert MODULE.FOOTER.startswith("Rows matched by pixel error")
