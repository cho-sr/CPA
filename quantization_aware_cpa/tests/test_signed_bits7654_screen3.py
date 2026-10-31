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

from qacpa import signed_bits_observation as codec
from qacpa import signed_bits_randomized as randomized
from qacpa import signed_bits_screen3_runner as runner


class Bits7654Tests(unittest.TestCase):
    def test_each_bit_codec_roundtrip_and_logical_physical_accounting(self):
        value = torch.tensor([-1., -.25, 0., .25, 1.], dtype=torch.float32)
        for bits in (7, 6, 5, 4):
            q, scale = codec.encode(value, bits)
            q2, scale2 = codec.unpack(codec.pack(q, scale, bits), tuple(value.shape), bits)
            self.assertTrue(torch.equal(q, q2))
            self.assertTrue(torch.equal(scale, scale2))
            self.assertEqual(int(q.min()), -(2 ** (bits - 1) - 1))
            self.assertEqual(int(q.max()), 2 ** (bits - 1) - 1)
            self.assertEqual(codec.codec_metadata(bits)["logical_code_bits"], bits)

    def test_each_bit_cell_boundary_sampler_and_anchor(self):
        for bits in (7, 6, 5, 4):
            qmax = codec.qmax_for_bits(bits)
            scale = torch.tensor(1.0 / qmax, dtype=torch.float32)
            q = torch.tensor([[qmax, -qmax, 0, 1]], dtype=torch.int8)
            first, meta = randomized.sample_matrix(q, scale, bits=bits, seed=1234, chunk_rows=1)
            second, meta2 = randomized.sample_matrix(q, scale, bits=bits, seed=1234, chunk_rows=1)
            self.assertTrue(torch.equal(first, second))
            self.assertEqual(meta["tensor_sha256"], meta2["tensor_sha256"])
            self.assertTrue(meta["scale_reproduced"])

    def test_mocked_separation_keeps_each_member_native_and_does_nine_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            observation = base / "observation"
            observation.mkdir()
            (observation / "manifest.json").write_text("{}\n", encoding="utf-8")
            calls = []
            q = torch.tensor([[63, -63, 0, 1], [2, -2, 62, -62]], dtype=torch.int8)
            scale = torch.tensor(1.0 / 63.0, dtype=torch.float32)

            def fake_separate(value, source_count, *, config, device):
                calls.append(value.clone())
                source = value.clone()
                result = type("R", (), {})()
                result.S_hat = source
                result.metadata = {"fastica": {"converged": True}, "refinement": {"iterations": 4000},
                                   "whitening": {"shared_between_stages": True}}
                result.refinement = type("F", (), {"final_losses": {"total": 0.0}})()
                return result

            with patch.object(runner, "SOURCE_COUNT", 2), patch.object(runner, "FEATURE_DIM", 4), \
                 patch.object(runner, "OBSERVATION_ROWS", 2), \
                 patch.object(runner, "read_bundle", return_value=({"format": "x"}, {"classifier.0.weight": (q, scale)})), \
                 patch.object(runner, "separate_observation", side_effect=fake_separate):
                result = runner.separate(7, observation, base / "separation")
            self.assertEqual(result["separation_calls"], 9)
            self.assertEqual(len(calls), 9)
            self.assertFalse(result["alignment"]["feature_average"])
            member = torch.load(base / "separation/member_00/sources.pt", map_location="cpu", weights_only=True)
            self.assertTrue(torch.equal(member["target_z"], member["S_hat_aligned"].abs()))

    def test_mocked_run_public_executes_center_plus_eight_prefixes_plus_frozen_top3(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            public, private = base / "public", base / "private"
            runner.register_bit(7, public, private)
            target = torch.ones((16, 4), dtype=torch.float32)

            def fake_collect(bits, output, private_output, **kwargs):
                output.mkdir()
                return {"status": "COMPLETED", "public_manifest_sha256": "manifest"}

            def fake_separate(bits, observation, output):
                output.mkdir()
                (output / "result.json").write_text('{"status":"COMPLETED","separation_calls":9}\n', encoding="utf-8")
                return {"status": "COMPLETED", "separation_calls": 9}

            def fake_source(*args):
                return target, "source", {}

            def fake_trajectory(target_z, *, output, arm, end_steps, **kwargs):
                calls.append((arm, end_steps))
                output.mkdir()
                pixels = torch.zeros((16, 3, 224, 224), dtype=torch.float32)
                torch.save({"pixels": pixels}, output / "candidate.pt")
                result = {"status": "COMPLETED", "arm": arm, "completed_iterations": end_steps,
                          "candidate_sha256": runner.file_sha256(output / "candidate.pt"),
                          "checkpoint_sha256": arm, "score": {"score": float(int(arm.split("_")[1]) if "member" in arm else 0),
                          "feature_cosine_distance_mean": 0.0, "tv_normalized": 0.0,
                          "raw_state_unchanged": True}}
                if arm == "center_25000":
                    torch.save({"pixels": pixels}, output / "candidate_5000.pt")
                    result["prefix_candidate_sha256"] = runner.file_sha256(output / "candidate_5000.pt")
                runner.write_json(output / "result.json", result)
                return result

            with patch.object(runner, "collect", side_effect=fake_collect), \
                 patch.object(runner, "separate", side_effect=fake_separate), \
                 patch.object(runner, "_source_entry", side_effect=fake_source), \
                 patch.object(runner, "_public_input_hashes", return_value={}), \
                 patch.object(runner, "run_trajectory", side_effect=fake_trajectory):
                result = runner.run_public(7, public, private)
            self.assertEqual(result["status"], "PUBLIC_COMPLETED")
            self.assertEqual(len(calls), 12)
            self.assertEqual(calls[0], ("center_25000", 25000))
            self.assertEqual(calls[1:9], [(f"member_{i:02d}_5000", 5000) for i in range(8)])
            self.assertEqual([x[0] for x in calls[9:]], ["member_00_25000", "member_01_25000", "member_02_25000"])


if __name__ == "__main__":
    unittest.main()
