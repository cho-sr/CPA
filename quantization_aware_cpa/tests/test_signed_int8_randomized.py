from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qacpa.signed_int8_observation import CODEC_METADATA, FC_SHAPES, HEADER, pack, unpack, encode
from qacpa.signed_int8_randomized import align_and_average, cell_bounds_table, sample_matrix
from qacpa import signed_int8_randomized_runner as runner


class SignedINT8RandomizedTests(unittest.TestCase):
    def test_signed255_packet_zero_roundtrip_and_byte_accounting(self):
        value = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32)
        q, scale = encode(value)
        self.assertEqual(int(q.min()), -127)
        self.assertEqual(int(q.max()), 127)
        packet = pack(q, scale)
        self.assertEqual(len(packet), HEADER.size + value.numel())
        q2, scale2 = unpack(packet, tuple(value.shape))
        self.assertTrue(torch.equal(q, q2))
        self.assertTrue(torch.equal(scale, scale2))
        zero_q, zero_s = encode(torch.zeros(7, dtype=torch.float32))
        self.assertTrue(torch.equal(zero_q, torch.zeros_like(zero_q)))
        self.assertEqual(float(zero_s), 1.0)
        self.assertEqual(CODEC_METADATA["qmax"], 127)

    def test_exact_cells_endpoints_and_reproducible_public_sampler(self):
        scale = torch.tensor(1.0 / 127.0, dtype=torch.float32)
        lower, upper = cell_bounds_table(scale)
        codes = torch.arange(-127, 128, dtype=torch.int8)
        self.assertTrue(torch.equal(torch.round(lower / scale).clamp(-127, 127).to(torch.int8), codes))
        self.assertTrue(torch.equal(torch.round(upper / scale).clamp(-127, 127).to(torch.int8), codes))
        q = torch.tensor([[127, -127, 0, 1], [2, -2, 126, -126]], dtype=torch.int8)
        first, first_meta = sample_matrix(q, scale, seed=1234, chunk_rows=1)
        second, second_meta = sample_matrix(q, scale, seed=1234, chunk_rows=1)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_meta["tensor_sha256"], second_meta["tensor_sha256"])
        self.assertTrue(first_meta["scale_reproduced"])
        self.assertEqual(first_meta["anchor"]["flat_index"], 0)

    def test_blind_permutation_sign_scale_alignment_and_average(self):
        generator = torch.Generator().manual_seed(7)
        center = torch.randn((16, 32), generator=generator, dtype=torch.float32)
        permutation = torch.randperm(16, generator=generator)
        signs = torch.where(torch.arange(16) % 2 == 0, 1.0, -1.0)
        member = signs[:, None] * center[permutation]
        aggregate, meta = align_and_average(center, [member.clone() for _ in range(8)])
        self.assertTrue(torch.equal(aggregate, center))
        self.assertTrue(meta["target_z_exact_abs"])
        self.assertEqual(meta["member_count"], 8)

    def test_historical_int4_contract_is_separate(self):
        from quantized_update_cpa.cell_utils import CODEC_METADATA as old
        self.assertEqual(old["bits"], 4)
        self.assertEqual(old["qmax"], 7)
        self.assertEqual(FC_SHAPES["classifier.0.weight"], (4096, 25088))

    def test_registration_collision_and_public_private_boundary(self):
        if not runner.CHECKPOINT.is_file():
            self.skipTest("registered checkpoint is unavailable")
        with tempfile.TemporaryDirectory(dir=runner.REPO_ROOT) as temporary:
            base = Path(temporary)
            public = base / "public"
            private = base / "private"
            value = runner.register(public, private)
            self.assertEqual(value["status"], "REGISTERED")
            self.assertTrue((public / "registration.json").is_file())
            with self.assertRaises(FileExistsError):
                runner.register(public, private)

    def test_mocked_nine_call_separation_and_aggregate_binding(self):
        """Exercise the staged writer without invoking the expensive separator."""
        with tempfile.TemporaryDirectory(dir=runner.REPO_ROOT) as temporary:
            base = Path(temporary)
            observation = base / "observation"
            observation.mkdir()
            (observation / "manifest.json").write_text("{}\n", encoding="utf-8")
            q = torch.tensor([[127, -127, 0, 1], [2, -2, 126, -126]], dtype=torch.int8)
            scale = torch.tensor(1.0 / 127.0, dtype=torch.float32)
            fake_manifest = {"layers": {"classifier.0.weight": {"file": "fc_00.sgn8"}}}
            calls = []

            def fake_separate(X, source_count, *, config, device):
                calls.append(X.clone())
                source = X.clone().contiguous()
                ref = SimpleNamespace(log=[{"iteration": 1, "total": 0.0}], final_losses={"total": 0.0})
                metadata = {"fastica": {"converged": True, "iterations": 1, "residual": 0.0},
                            "refinement": {"iterations": config.refinement_iterations},
                            "whitening": {"shared_between_stages": True}}
                return SimpleNamespace(S_hat=source, refinement=ref, metadata=metadata)

            output = base / "separation"
            with patch.object(runner, "SOURCE_COUNT", 2), patch.object(runner, "FEATURE_DIM", 4), patch.object(runner, "OBSERVATION_ROWS", 2), \
                 patch.object(runner, "read_bundle", return_value=(fake_manifest, {"classifier.0.weight": (q, scale)})), \
                 patch.object(runner, "separate_observation", side_effect=fake_separate):
                result = runner.separate(observation, output)
            self.assertEqual(result["separation_calls"], 9)
            self.assertEqual(len(calls), 9)
            self.assertTrue((output / "aggregate/sources.pt").is_file())
            self.assertTrue(result["target_z_exact_abs_S_hat"])


if __name__ == "__main__":
    unittest.main()
