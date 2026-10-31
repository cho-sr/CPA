"""Focused contract checks for the user-authorized v2/R8 migration."""
from copy import deepcopy
import inspect
from pathlib import Path

import pytest

from qacpa import run_paper_sparse_same80_r8 as runner


CONFIGS = {
    percent: runner.PROJECT / f"configs/cpa_fia_random_topk{percent}_same80_R8_S5k_K3_F25k.json"
    for percent in (50, 30, 20)
}


@pytest.mark.parametrize("percent", [50, 30, 20])
def test_r8_config_is_exactly_candidate_ids_zero_through_seven(percent):
    cfg = runner.read_json(CONFIGS[percent])
    baseline = runner.validate_config(cfg)
    assert cfg["candidate_count"] == 8
    assert cfg["candidate_ids"] == list(range(8))
    assert cfg["randomization"]["candidates"] == 8
    assert cfg["keep_percent"] == percent
    assert cfg["baseline_output"] == baseline["output_dir"]


def test_r8_rejects_count_or_candidate_list_mutations():
    cfg = runner.read_json(CONFIGS[30])
    for key, value in (("candidate_count", 20), ("candidate_ids", list(range(20))),
                       ("plan_version", 1), ("keep_percent", 10)):
        changed = deepcopy(cfg)
        changed[key] = value
        with pytest.raises(ValueError):
            runner.validate_config(changed)


@pytest.mark.parametrize("percent", [50, 30, 20])
def test_migration_manifest_has_exact_scope_and_truthful_reuse(percent):
    path = runner.resolve(f"results/cpa_fia_random_topk{percent}_same80_R8_S5k_K3_F25k/migration_manifest.json")
    migration = runner.read_json(path)
    assert migration["plan_version"] == 2
    assert migration["candidate_ids"] == list(range(8))
    assert migration["v1_selection_reused"] is False
    assert migration["old_candidates_8_19_used"] is False
    assert all(0 <= int(row["candidate"]) < 8 for row in migration["records"])


def test_keep50_imports_only_authenticated_v1_candidates():
    migration = runner.read_json(runner.resolve("results/cpa_fia_random_topk50_same80_R8_S5k_K3_F25k/migration_manifest.json"))
    imported = [row for row in migration["records"] if row["status"] != "fresh_no_v1_artifact"]
    assert [(row["batch_index"], row["candidate"], row["status"]) for row in imported] == [
        (0, 0, "imported_cpa_and_completed_screen"),
        (0, 1, "imported_cpa_and_completed_screen"),
        (0, 2, "imported_cpa_and_partial_screen_checkpoint"),
    ]


def test_r8_selection_and_attack_boundaries_are_truth_free():
    source = inspect.getsource(runner.run_candidate)
    assert "inputs.pt" not in source and "Z_evaluation_only" not in source
    batch_source = inspect.getsource(runner.run_batch)
    assert batch_source.index('"final_selection.json"') < batch_source.index("inputs = torch.load")
    assert batch_source.index("inputs = torch.load") < batch_source.index("evaluate_after_fia(")
    assert "range(20)" not in Path(runner.__file__).read_text(encoding="utf-8")


def test_r8_budget_is_8_screen_plus_3_final():
    assert runner.CANDIDATES == tuple(range(8))
    assert runner.CANDIDATE_COUNT * 25000 == 200000
    assert runner.CANDIDATE_COUNT * 5000 + 3 * 20000 == 100000
    assert runner.CANDIDATE_COUNT + 3 == 11
