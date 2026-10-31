"""Focused synthetic validation; no victim data or model inference."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("experiment", Path(__file__).with_name("run_experiment.py"))
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
torch, core = experiment.torch, experiment.core


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        experiment.configure_runtime(42, threads=1)
        self.whitening = core.WhitenedObservation(torch.zeros(2, 1), torch.eye(2),
                                                   torch.tensor([[1., 0., -1., 0.], [0., 1., 0., -1.]]), (2, 4), 2)

    def test_fastica_continues_after_immediate_convergence_for_exact_4000(self):
        with patch.object(core, "_fastica_update", side_effect=lambda w, x: (w.clone(), torch.zeros(2))) as update:
            fast, log, first = experiment.fixed_fastica(self.whitening)
        self.assertEqual(update.call_count, 4000)
        self.assertEqual((fast.iterations, len(log), first), (4000, 4000, 1))

    def test_gd_honors_4000_and_abs_target(self):
        fast = core.FastICAResult(torch.eye(2), True, 4000, 0., 4000, 1e-6)
        refined = core.refine_from_fastica(self.whitening, fast, experiment.CONFIG)
        self.assertEqual((refined.iterations, len(refined.log)), (4000, 4000))
        self.assertTrue(torch.equal(refined.target_z, refined.S_hat.abs()))
        self.assertTrue(torch.isfinite(refined.target_z).all())

    def test_nonfinite_fastica_aborts(self):
        with patch.object(core, "_fastica_update", return_value=(torch.full((2, 2), float("nan")), torch.zeros(2))):
            with self.assertRaises(core.ICAError):
                experiment.fixed_fastica(self.whitening)

    def test_public_gate_rejects_shortened_run_before_loading_pixels(self):
        registration = {"input_hashes": {}, "code_hashes": {}}
        result = {"status": "COMPLETED", "fast_iterations": 84, "gd_iterations": 4000, "fia_iterations": 25000}
        with patch.object(experiment, "read", side_effect=[registration, result]), \
             patch.object(experiment, "input_hashes", return_value={}), \
             patch.object(experiment, "code_hashes", return_value={}), patch.object(torch, "load") as load:
            with self.assertRaises(ValueError):
                experiment.public_gate()
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
