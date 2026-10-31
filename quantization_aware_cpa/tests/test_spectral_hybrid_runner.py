"""Pre-execution guards for the bounded public three-arm reconstruction."""
import json
from pathlib import Path

import pytest

from qacpa.run_spectral_hybrid import require_new_output, validate_plan


def plan():
    return json.loads((Path(__file__).parents[1] / "configs/spectral_hybrid_B16_T10.json").read_text(encoding="utf-8"))


def test_registered_plan_is_accepted():
    validate_plan(plan())


@pytest.mark.parametrize("change", [
    {"branch_iterations": 51}, {"seed": 43}, {"epsilon": 0.0},
    {"arms": ["identity", "flat", "mp", "flat_seed2"]},
    {"private_truth": "must_not_be_an_attack_input.pt"},
    {"all_arms_use_bin_and_radius": False},
    {"expected_common_initial_sha256": "wrong"},
])
def test_unregistered_execution_scope_is_rejected(change):
    candidate = plan()
    candidate.update(change)
    with pytest.raises(ValueError):
        validate_plan(candidate)


def test_original_output_is_preserved(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "result.txt"
    sentinel.write_bytes(b"prior result")
    with pytest.raises(FileExistsError):
        require_new_output(output, ())
    assert sentinel.read_bytes() == b"prior result"


def test_output_cannot_overlap_public_input(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="overlaps"):
        require_new_output(source / "nested_output", (source,))
    require_new_output(tmp_path / "new_run", (source,))
