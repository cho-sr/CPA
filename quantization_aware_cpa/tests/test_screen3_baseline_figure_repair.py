from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
RENDERER = ROOT / ".ai/execution-results/CPA_SCREEN3_BASELINE_FIGURE_REPAIR_20260918/render_baseline_figures.py"
spec = importlib.util.spec_from_file_location("screen3_baseline_renderer", RENDERER)
renderer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(renderer)


class BaselineFigureRepairTests(unittest.TestCase):
    def test_layout_has_truth_baseline_and_original_member_ids(self):
        layout = renderer.build_layout(6, [1, 3, 4])
        self.assertEqual(len(layout["final_names"]), 4)
        self.assertEqual(len(layout["final_labels"]), 5)
        self.assertEqual(layout["final_labels"][0], "Ground truth")
        self.assertIn("baseline", layout["final_labels"][1].lower())
        self.assertIn("member 01", layout["final_labels"][2])
        self.assertIn("member 03", layout["final_labels"][3])
        self.assertIn("member 04", layout["final_labels"][4])
        self.assertEqual(len(layout["screen_names"]), 9)
        self.assertEqual(len(layout["screen_labels"]), 10)
        self.assertEqual(layout["screen_names"][1:], [f"member_{i:02d}_5000" for i in range(8)])

    def test_render_reuses_correspondence_and_keeps_all_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            truth = torch.zeros((16, 3, 8, 8), dtype=torch.float32)
            baseline = torch.full_like(truth, 0.1)
            members = {f"member_{i:02d}_5000": torch.full_like(truth, 0.2 + i / 100)
                       for i in range(8)}
            candidates = {"center_5000": baseline, **members}
            rows = [{"candidate_index": (i + 1) % 16, "truth_index": i} for i in range(16)]
            metrics = {name: {"per_image": rows} for name in candidates}
            labels = ["Ground truth", "baseline"] + [f"member {i:02d}" for i in range(8)]
            result = renderer.render_sheet(Path(temp) / "screen.png", labels,
                                           list(candidates), candidates, truth, metrics, "test")
            self.assertEqual(result["column_count"], 10)
            self.assertEqual(result["rows"], 16)
            self.assertTrue(result["correspondence_reused"])
            self.assertFalse(result["metrics_recomputed"])
            self.assertTrue(Path(result["path"]).is_file())

    def test_aligned_candidate_uses_stored_mapping_without_matching(self):
        candidate = torch.arange(16, dtype=torch.float32).view(16, 1, 1, 1).expand(16, 3, 2, 2)
        rows = [{"candidate_index": 15 - i, "truth_index": i} for i in range(16)]
        aligned = renderer.aligned_candidate(candidate, rows)
        self.assertEqual(float(aligned[0, 0, 0, 0]), 15.0)
        self.assertEqual(float(aligned[15, 0, 0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
