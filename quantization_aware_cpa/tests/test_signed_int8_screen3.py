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

from qacpa import signed_int8_observation as observation
from qacpa import signed_int8_screen3_runner as runner
from qacpa.signed_int8_screen3_fia import score_attack
from types import SimpleNamespace


class _FakeModel(torch.nn.Module):
    def forward(self, x, return_z=False):
        return None, x.flatten(1)[:, :4]


class _FakeAttack:
    def __init__(self):
        self.device = torch.device("cpu")
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.ones(1, 3, 1, 1)
        self.S_hat = torch.nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) / 10)
        self.model = _FakeModel()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-10)

    def total_variation(self, x):
        return (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean() + (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()


class SignedINT8Screen3Tests(unittest.TestCase):
    def test_sender_dispatch_does_not_pass_subparser_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            paths = [base / name for name in ("raw.pt", "sender.json", "truth.pt", "public", "private")]
            with patch.object(observation, "collect", return_value={"status": "COMPLETED"}) as collect:
                self.assertEqual(observation.main([
                    "sender", "--raw-updates", str(paths[0]),
                    "--archived-sender-validation", str(paths[1]), "--evaluation-truth", str(paths[2]),
                    "--output", str(paths[3]), "--private-output", str(paths[4]),
                ]), 0)
            kwargs = collect.call_args.kwargs
            self.assertNotIn("command", kwargs)
            self.assertEqual(set(kwargs), {"raw_updates", "archived_sender_validation", "evaluation_truth", "output", "private_output"})

    def test_alignment_preserves_native_member_row_magnitudes(self):
        center = torch.tensor([[1., 0., 0., 0.], [0., 2., 0., 0.]])
        member = torch.tensor([[0., -6., 0., 0.], [3., 0., 0., 0.]])
        aligned, meta = runner._align_preserve_norm(center, member, 0)
        self.assertTrue(torch.equal(aligned.abs(), torch.tensor([[3., 0., 0., 0.], [0., 6., 0., 0.]])))
        self.assertFalse(meta["row_norm_rescaling"])
        self.assertAlmostEqual(meta["native_to_reference_norm_ratios"][0], 3.0)

    def test_public_selection_is_eight_way_deterministic_with_index_ties(self):
        entries = []
        for index in range(8):
            entries.append({"member_index": index, "score": {"score": 1.0 if index < 4 else 2.0,
                                                               "feature_cosine_distance_mean": 1.0,
                                                               "tv_normalized": 0.0,
                                                               "raw_state_unchanged": True},
                            "result": {"checkpoint_sha256": str(index), "candidate_sha256": str(index)},
                            "target_z_sha256": str(index), "source_file_sha256": str(index)})
        selection = runner.freeze_public_selection(list(reversed(entries)))
        self.assertEqual(selection["selected_member_indices"], [0, 1, 2])
        self.assertEqual([row["member_index"] for row in selection["ranking"]], list(range(8)))
        with self.assertRaises(runner.Screen3Error):
            runner.freeze_public_selection(entries[:7])

    def test_public_score_does_not_mutate_raw_image(self):
        attack = _FakeAttack()
        target = attack.S_hat.detach().flatten(1)[:, :4].clone()
        before = attack.S_hat.detach().clone()
        score = score_attack(attack, target)
        self.assertTrue(score["raw_state_unchanged"])
        self.assertTrue(torch.equal(attack.S_hat.detach(), before))
        self.assertTrue(torch.isfinite(torch.tensor(score["score"])))

    def test_checkpoint_state_is_cpu_serializable_and_preserves_prefix_schema(self):
        from qacpa.signed_int8_screen3_fia import _checkpoint_payload
        attack = _FakeAttack()
        attack.get_attack_state = lambda: {"S_hat": attack.S_hat, "opt": {"step": torch.tensor(5)}}
        payload = _checkpoint_payload(attack, completed=5000, log=[{"iter": 1}],
                                      parent_checkpoint_sha256=None,
                                      branch_meta={"checkpoint_role": "MEMBER_PREFIX_5000"}, wall_seconds=1.0)
        self.assertEqual(payload["schema"], "CPA_SCREEN3_DIRECT_CHECKPOINT_V1")
        self.assertEqual(payload["completed_iterations"], 5000)
        self.assertEqual(payload["attack_state"]["S_hat"].device.type, "cpu")
        self.assertEqual(payload["scheduler_horizon"], 25000)

    def test_mocked_screen3_separation_writes_eight_bound_member_targets(self):
        with tempfile.TemporaryDirectory(dir=runner.REPO_ROOT) as tmp:
            base = Path(tmp)
            observation_folder = base / "observation"
            observation_folder.mkdir()
            (observation_folder / "manifest.json").write_text("{}\n", encoding="utf-8")
            q = torch.tensor([[127, -127, 0, 1], [2, -2, 126, -126]], dtype=torch.int8)
            scale = torch.tensor(1.0 / 127.0, dtype=torch.float32)
            calls = []

            def fake_separate(value, source_count, *, config, device):
                calls.append(value.clone())
                ref = SimpleNamespace(log=[{"iteration": 1, "total": 0.0}], final_losses={"total": 0.0})
                metadata = {"fastica": {"converged": True, "iterations": 1, "residual": 0.0},
                            "refinement": {"iterations": config.refinement_iterations},
                            "whitening": {"shared_between_stages": True}}
                return SimpleNamespace(S_hat=value.clone(), refinement=ref, metadata=metadata)

            output = base / "separation"
            fake_manifest = {"layers": {"classifier.0.weight": {"file": "fc_00.sgn8"}}}
            with patch.object(runner, "SOURCE_COUNT", 2), patch.object(runner, "FEATURE_DIM", 4), \
                 patch.object(runner, "OBSERVATION_ROWS", 2), patch.object(runner, "MEMBER_SEEDS", list(range(8))), \
                 patch.object(runner, "read_bundle", return_value=(fake_manifest, {"classifier.0.weight": (q, scale)})), \
                 patch.object(runner, "separate_observation", side_effect=fake_separate):
                result = runner.separate(observation_folder, output)
            self.assertEqual(result["separation_calls"], 9)
            self.assertEqual(len(calls), 9)
            self.assertFalse(result["alignment"]["feature_average"])
            self.assertFalse(result["alignment"]["row_norm_rescaling"])
            member = torch.load(output / "member_00/sources.pt", map_location="cpu", weights_only=True)
            self.assertEqual(set(member), {"S_hat_native", "S_hat_aligned", "target_z"})
            self.assertTrue(torch.equal(member["target_z"], member["S_hat_aligned"].abs()))


if __name__ == "__main__":
    unittest.main()
