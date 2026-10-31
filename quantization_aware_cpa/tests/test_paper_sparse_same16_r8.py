import json
from pathlib import Path

import pytest

from qacpa import report_paper_sparse_same16_r8 as report
from qacpa import run_paper_sparse_same16_r8 as same16


PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("percent", [50, 30, 20])
def test_same16_config_is_authorized(percent):
    path = PROJECT / "configs" / f"cpa_fia_random_topk{percent}_same16_R8_S5k_K3_F25k.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    baseline = same16.validate_config(cfg)
    assert baseline["n_images"] == 80
    assert cfg["n_images"] == 16
    assert cfg["batch_indices"] == [0]
    assert cfg["candidate_ids"] == list(range(8))
    assert cfg["candidate_count"] == 8
    assert cfg["plan_version"] == 3
    assert cfg["amendment_id"] == same16.AMENDMENT_ID
    assert cfg["local_training"]["independent_client_updates"] == 1


def test_same16_constants_and_budget():
    assert same16.N_BATCHES == 1
    assert same16.N_IMAGES == 16
    assert same16.BATCH_INDICES == (0,)
    assert same16.CANDIDATES == tuple(range(8))
    assert same16.RANDOMIZATION["screen_iterations"] == 5000
    assert same16.RANDOMIZATION["advance_top_k"] == 3
    assert same16.RANDOMIZATION["final_iterations"] == 25000
    assert same16.CANDIDATE_COUNT * 25000 == 200000
    assert same16.CANDIDATE_COUNT * 5000 + 3 * 20000 == 100000


def test_same16_input_hash_and_scope_are_fixed():
    assert same16.SAME16_INPUT_SHA256 == "24995487511a3e9d8d4665fdcd704f7a0f792249a9ba79880638d13bc4740141"
    assert same16.BATCH_INDICES == (0,)
    assert "same16" in same16.SAME16_TAG


def test_same16_output_names_do_not_reuse_all_image_artifact_names():
    source = (PROJECT / "qacpa" / "report_paper_sparse_same16_r8.py").read_text(encoding="utf-8")
    assert "all16_rec.pt" in source
    assert "naive_all16_rec.pt" in source
    assert "all16_original_vs_naive_vs_random.png" in source
    assert "all80_original_vs_naive_vs_random.png" not in source


def test_same16_migration_scope_rejects_later_batches(tmp_path):
    manifest = {
        "task_id": same16.TASK_ID,
        "plan_version": same16.PLAN_VERSION,
        "identity": {"sentinel": True},
        "candidate_ids": list(range(8)),
        "batch_indices": [0, 1],
        "n_images": 16,
        "v1_selection_reused": False,
        "v2_selection_reused": False,
        "old_candidates_8_19_used": False,
        "batches01_04_used": False,
        "records": [],
    }
    (tmp_path / "migration_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="batch00 only"):
        report._validate_migration(tmp_path, {"identity": {"sentinel": True}})
