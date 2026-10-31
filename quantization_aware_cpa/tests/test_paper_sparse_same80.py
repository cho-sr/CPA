"""Focused gates for the authorized paper sparse same80 task-local runner."""
from copy import deepcopy
import inspect
from pathlib import Path

import pytest

from qacpa import run_paper_sparse_same80 as runner
from qacpa import random_fill


CONFIGS = {
    percent: runner.PROJECT / f"configs/cpa_fia_random_topk{percent}_same80_R20_S5k_K3_F25k.json"
    for percent in (50, 30, 20)
}


@pytest.mark.parametrize("percent", [50, 30, 20])
def test_registered_fraction_config_keeps_the_matched_baseline(percent):
    config = runner.read_json(CONFIGS[percent])
    baseline = runner.validate_config(config)
    assert baseline["run_tag"] == f"cpa_fia_topk{percent}_same80_B16"
    assert config["keep_percent"] == percent
    assert config["baseline_output"] == baseline["output_dir"]
    assert config["randomization"] == runner.RANDOMIZATION
    assert config["expected_packet_bytes"] == runner.expected_packet_bytes(percent)


def test_fraction_and_selection_mutations_are_rejected():
    config = runner.read_json(CONFIGS[30])
    for key, value in (("keep_percent", 10), ("n", 10), ("T", 1),
                       ("source_codec", "changed_codec")):
        changed = deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError):
            runner.validate_config(changed)
    changed = deepcopy(config)
    changed["randomization"]["advance_top_k"] = 5
    with pytest.raises(ValueError):
        runner.validate_config(changed)


def test_packet_schema_audit_manifest_records_historical_50_equivalence():
    manifest = runner.read_json(
        runner.resolve("results/cpa_fia_random_topk50_same80_R20_S5k_K3_F25k/input_manifest.json")
    )
    audits = manifest["packet_audits"]
    assert len(audits) == 5
    assert all(audit["historical_50_schema_checked"] for audit in audits)
    assert all(audit["generic_recompression_equal"] for audit in audits)
    assert all(audit["packet_bytes"] == runner.expected_packet_bytes(50) for audit in audits)


def test_candidate_seed_and_tie_ranking_are_frozen():
    assert len({random_fill.candidate_seed(batch, candidate)
                 for batch in range(5) for candidate in range(20)}) == 100
    rows = [{"candidate": candidate, "relative_l2": 1.0}
            for candidate in reversed(range(20))]
    assert [row["candidate"] for row in random_fill.rank_candidates(rows, list(range(20)))[:3]] == [0, 1, 2]


def test_attack_boundary_loads_private_inputs_after_final_selection():
    for function in (runner.run_candidate, random_fill.random_fill_packet,
                     random_fill.consistency_score):
        source = inspect.getsource(function)
        assert "inputs.pt" not in source
        assert "Z_evaluation_only" not in source
        assert "evaluate_images(" not in source
    source = inspect.getsource(runner.run_batch)
    assert source.index('"final_selection.json"') < source.index("inputs = torch.load")
    assert source.index("inputs = torch.load") < source.index("evaluate_after_fia(")


def test_task_local_wrapper_lists_only_three_authorized_fractions():
    wrapper = (runner.PROJECT / "scripts/run_paper_sparse_same80.py").read_text(encoding="utf-8")
    assert "FRACTIONS = (50, 30, 20)" in wrapper
    assert "--phase\", \"prepare\"" in wrapper
    assert "--phase\", \"run\"" in wrapper
