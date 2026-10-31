"""Focused contracts for the new bitwidth branches; synthetic data only."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("two_arm_task", Path(__file__).with_name("run_experiment.py"))
task = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = task
spec.loader.exec_module(task)
torch = task.torch


class BranchTests(unittest.TestCase):
    def setUp(self):
        task.base.configure_runtime(42, threads=1)

    def test_fp32_reader_keeps_small_nonzero_values_without_quantization(self):
        matrix = torch.zeros((4096, 25088), dtype=torch.float32)
        matrix[0, :4] = torch.tensor([1.0, 1e-5, -1e-7, 0.0])
        payload = {"updates": {"classifier.0.weight": matrix}}
        with patch.object(torch, "load", return_value=payload), patch.object(task.int8, "read_bundle") as reader:
            actual = task.load_observation("fp32")
        self.assertIs(actual, matrix)
        self.assertNotEqual(float(actual[0, 2]), 0.0)
        reader.assert_not_called()

    def test_int8_reader_consumes_only_decoded_public_codes_and_scale(self):
        q = torch.zeros((4096, 25088), dtype=torch.int8)
        q[0, :3] = torch.tensor([-127, 1, 127], dtype=torch.int8)
        scale = torch.tensor(0.001, dtype=torch.float32)
        with patch.object(task.int8, "read_bundle", return_value=({}, {"classifier.0.weight": (q, scale)})), patch.object(torch, "load") as raw_load:
            actual = task.load_observation("int8")
        self.assertTrue(torch.equal(actual[0, :3], q[0, :3].float()*scale))
        self.assertEqual(actual.dtype, torch.float32)
        raw_load.assert_not_called()

    def test_int8_uses_scalar_absmax_and_even_tie_rounding(self):
        value = torch.tensor([[127., .5, 1.5], [-127., -.5, -1.5]])
        q, scale = task.int8.encode(value)
        self.assertEqual(scale.ndim, 0)
        self.assertEqual(float(scale), 1.0)
        self.assertTrue(torch.equal(q, torch.tensor([[127, 0, 2], [-127, 0, -2]], dtype=torch.int8)))

    def test_public_gate_rejects_old_early_stopped_count_before_pixel_loading(self):
        result = {"status": "COMPLETED", "bits": 8, "fast_iterations": 84, "gd_iterations": 4000, "fia_iterations": 25000}
        with patch.object(task, "provenance_gate"), patch.object(task, "read", return_value=result), patch.object(torch, "load") as private_or_pixels:
            with self.assertRaises(ValueError):
                task.public_gate("int8")
        private_or_pixels.assert_not_called()

    def test_unknown_bitwidth_is_rejected(self):
        with self.assertRaises(ValueError):
            task.load_observation("int4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
