from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qacpa import signed_int8_screen3_fia as fia
from qacpa import signed_int8_screen3_repair_evaluation as evaluation
from qacpa import signed_int8_screen3_repair_runner as repair


class _Scheduler:
    def __init__(self):
        self.calls = 0

    def state_dict(self):
        return {"T_max": fia.TOTAL_STEPS}


class _SyntheticAttack:
    def __init__(self):
        self.device = torch.device("cpu")
        self.S_hat = torch.zeros((16, 3, 2, 2), dtype=torch.float32, requires_grad=True)
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.ones(1, 3, 1, 1)
        self.model = lambda x, return_z=False: (None, x.flatten(1)[:, :4])
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-10)
        self.sch = _Scheduler()
        self.steps = 0

    def step(self):
        self.steps += 1
        with torch.no_grad():
            self.S_hat.add_(0.000001).clamp_(0, 1)
        return {"loss": 0.0, "loss_fi": 0.0, "loss_gm": 0.0, "loss_tv": 0.0}

    def get_rec(self):
        return self.S_hat.detach().clamp(0, 1)

    def get_attack_state(self):
        return {
            "S_hat": self.S_hat.detach().clone(),
            "opt": {"steps": self.steps},
            "sch": {"last_epoch": self.steps, "T_max": fia.TOTAL_STEPS},
        }

    def set_attack_state(self, state):
        with torch.no_grad():
            self.S_hat.copy_(state["S_hat"])
        self.steps = int(state["opt"]["steps"])

    def total_variation(self, x):
        return (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean() + (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()


def _fake_construct(target, *, output, seed=42, device="cpu"):
    return _SyntheticAttack(), {
        "seed": seed, "device": str(device), "initial_image_sha256": fia.tensor_sha256(torch.zeros(16, 3, 2, 2)),
    }


class Screen3RepairTests(unittest.TestCase):
    def test_actual_writer_fresh_prefix_fresh_total_and_child(self):
        target = torch.ones((16, 4), dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(fia, "construct_attack", side_effect=_fake_construct), \
                 patch.object(fia, "configure_runtime"):
                prefix = fia.run_trajectory(
                    target, output=root / "fresh_prefix", source_sha256="source",
                    arm="member_00_5000", end_steps=fia.PREFIX_STEPS, device="cpu",
                )
                fresh = fia.run_trajectory(
                    target, output=root / "fresh_total", source_sha256="source",
                    arm="center_25000", end_steps=fia.TOTAL_STEPS, device="cpu",
                )
                child = fia.run_trajectory(
                    target, output=root / "child_total", source_sha256="source",
                    arm="member_07_25000", end_steps=fia.TOTAL_STEPS,
                    parent_checkpoint=root / "fresh_prefix/checkpoint_5000.pt",
                    parent_role="MEMBER_PREFIX_5000", device="cpu",
                )
            self.assertEqual(prefix["completed_iterations"], 5000)
            self.assertIsNone(prefix["prefix_checkpoint"])
            self.assertIsNotNone(fresh["prefix_checkpoint"])
            self.assertIsNone(child["prefix_checkpoint"])
            self.assertEqual(child["parent_checkpoint_sha256"],
                             fia.file_sha256(root / "fresh_prefix/checkpoint_5000.pt"))
            self.assertTrue((root / "child_total/checkpoint_25000.pt").is_file())

    def test_zero_step_export_restores_completed_checkpoint_without_step(self):
        target = torch.ones((16, 4), dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "completed.pt"
            attack = _SyntheticAttack()
            payload = fia._checkpoint_payload(
                attack, completed=fia.TOTAL_STEPS, log=[],
                parent_checkpoint_sha256=None,
                branch_meta={"arm": "member_01_25000", "target_z_sha256": fia.tensor_sha256(target),
                             "source_sha256": "source"},
                wall_seconds=0.0,
            )
            torch.save(payload, source)
            with patch.object(fia, "construct_attack", side_effect=_fake_construct), \
                 patch.object(fia, "configure_runtime"):
                result = fia.export_completed_checkpoint(
                    source, output=root / "exported", target_z=target,
                    source_sha256="source", arm="member_01_25000", device="cpu",
                )
            self.assertEqual(result["optimizer_steps_executed_by_recovery"], 0)
            self.assertTrue(result["checkpoint_bytes_unchanged"])
            self.assertEqual(result["completed_iterations"], fia.TOTAL_STEPS)
            self.assertFalse((root / "exported/step_called").exists())

    def test_zero_step_export_rejects_mutated_completed_checkpoint(self):
        target = torch.ones((16, 4), dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "bad.pt"
            attack = _SyntheticAttack()
            payload = fia._checkpoint_payload(
                attack, completed=fia.TOTAL_STEPS - 1, log=[],
                parent_checkpoint_sha256=None,
                branch_meta={"arm": "member_01_25000", "target_z_sha256": fia.tensor_sha256(target),
                             "source_sha256": "source"},
                wall_seconds=0.0,
            )
            torch.save(payload, source)
            with self.assertRaises(fia.StagedFIAError):
                fia.export_completed_checkpoint(
                    source, output=root / "exported", target_z=target,
                    source_sha256="source", arm="member_01_25000", device="cpu",
                )
            self.assertFalse((root / "exported").exists())

    def test_parent_source_snapshot_matches_registered_parent_revision(self):
        sources = repair._ensure_parent_source_snapshot()
        self.assertEqual(repair.sha(sources["fia"]), repair.PARENT_FIA_SHA256)
        self.assertEqual(repair.sha(sources["runner"]), repair.PARENT_RUNNER_SHA256)

    def test_inherited_candidate_manifest_is_exactly_the_nine_public_endpoints(self):
        candidates = repair._all_inherited_candidates()
        expected = {"center_5000", *[f"member_{i:02d}_5000" for i in range(8)], "center_25000"}
        self.assertEqual(set(candidates), expected)
        self.assertEqual(candidates["center_5000"]["step"], 5000)
        self.assertEqual(candidates["center_25000"]["step"], 25000)

    def test_private_adapter_accepts_registered_z_bundle_but_consumes_x_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truth.pt"
            x = torch.zeros((16, 3, 224, 224), dtype=torch.float32)
            torch.save({"x": x, "z": torch.zeros((16, 25088)), "labels": torch.zeros(16, dtype=torch.int64)}, path)
            with patch.object(evaluation, "PRIVATE_TRUTH", path), \
                 patch.object(evaluation, "TRUTH_SHA256", repair.sha(path)):
                loaded, meta = evaluation._private_x(path)
            self.assertTrue(torch.equal(loaded, x))
            self.assertEqual(meta["consumed_field"], "x only")
            self.assertFalse(meta["labels_consumed"])


if __name__ == "__main__":
    unittest.main()
