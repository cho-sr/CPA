"""Small synthetic public-symbol fixtures; never read the research packet."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import linalg
import torch

from qacpa.run_public_spectrum_diagnostic import (
    PARTITIONS, analyze_partition, blocked_centered_covariance, compare_subspaces,
    partition_rows, run_diagnostic, validate_output, validate_plan,
)


def _codes():
    # Fixed literal uint4 symbols: no private model, images or observations.
    return torch.tensor([[0, 1, 3, 7, 15, 12, 6, 2],
                         [15, 2, 7, 3, 8, 0, 5, 12],
                         [2, 7, 9, 15, 4, 0, 13, 3],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [15, 8, 1, 5, 4, 3, 0, 14]], dtype=torch.uint8)


@pytest.mark.parametrize("block_cols", [1, 3, 8, 20])
def test_blocked_covariance_equals_direct_centering_and_preserves_symbols(block_cols):
    codes = _codes()
    original = codes.clone()
    rows = np.array([4, 0, 2])
    covariance, means = blocked_centered_covariance(codes, rows, block_cols)
    direct = codes[rows].double().numpy()
    centered = direct - direct.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(means, direct.mean(axis=1), atol=0, rtol=0)
    np.testing.assert_allclose(covariance, centered @ centered.T / 7, atol=2e-14, rtol=2e-15)
    assert torch.equal(codes, original)


def test_public_symbol_normalization_matches_ideal_decode_but_never_needs_radius_division():
    q = _codes()[[0, 1, 2]].double()
    radius = torch.tensor([1e-30, .017, 13.], dtype=torch.float64)
    ideal = radius[:, None] * (2 * q / 15 - 1)
    normalized = ideal / (radius[:, None] * (2 / 15))
    torch.testing.assert_close(normalized - normalized.mean(1, keepdim=True),
                               q - q.mean(1, keepdim=True), rtol=1e-14, atol=3e-15)


def test_partition_covariances_and_right_subspaces_match_direct_svd():
    codes = _codes()
    radius = torch.tensor([1., 2., 3., 0., .5], dtype=torch.float32)
    floor = 2. * (2 / 15)
    partitions = partition_rows(radius, floor)
    assert list(partitions) == list(PARTITIONS)
    assert partitions["positive_all"].tolist() == [0, 1, 2, 4]
    assert partitions["at_or_above_floor"].tolist() == [1, 2]
    assert partitions["below_floor"].tolist() == [0, 4]
    covariance, _ = blocked_centered_covariance(codes, partitions["positive_all"], 3)
    for selected in partitions.values():
        result = analyze_partition(codes, covariance, partitions["positive_all"], selected,
                                   top_k=2, block_cols=3)
        direct = codes[selected].double().numpy()
        direct -= direct.mean(1, keepdims=True)
        _, singular, vt = linalg.svd(direct, full_matrices=False)
        np.testing.assert_allclose(result["eigenvalues_desc"], singular ** 2 / 7, rtol=3e-14, atol=1e-14)
        measured = compare_subspaces(result["right_vectors"], vt[:2].T)
        np.testing.assert_allclose(measured["pairwise_principal_cosines"], [1., 1.], atol=2e-15)
        assert result["right_vectors"].shape == (codes.shape[1], 2)
        assert result["row_indices"].tolist() == selected.tolist()
        assert result["checks"]["numerical_checks_passed"]
        assert result["checks"]["orthogonality_error"] < 2e-14
        assert result["checks"]["eigen_residual"] < 2e-14
        assert result["checks"]["right_relation_residual"] < 2e-14


def test_zero_radius_empty_partition_and_constant_symbol_rows_remain_counted():
    codes = torch.tensor([[0] * 8, [0] * 8, [15] * 8], dtype=torch.uint8)
    radius = torch.tensor([0., 0., 1.], dtype=torch.float32)
    partitions = partition_rows(radius, 1.)
    assert len(partitions["positive_all"]) == 1
    assert len(partitions["at_or_above_floor"]) == 0
    assert len(partitions["below_floor"]) == 1
    covariance, _ = blocked_centered_covariance(codes, partitions["positive_all"], 3)
    full = analyze_partition(codes, covariance, partitions["positive_all"], partitions["positive_all"])
    assert full["row_indices"].tolist() == [2]
    assert full["eigenvalues_desc"].tolist() == [0.]
    assert full["right_vectors"].shape == (8, 0)
    empty = analyze_partition(codes, covariance, partitions["positive_all"], partitions["at_or_above_floor"])
    assert empty["checks"]["status"] == "empty_partition"
    assert empty["right_vectors"].shape == (8, 0)
    assert compare_subspaces(full["right_vectors"], empty["right_vectors"])["projector_overlap"] is None
    all_zero = partition_rows(torch.zeros(3), 1.)
    empty_cov, empty_means = blocked_centered_covariance(codes, all_zero["positive_all"], 3)
    assert empty_cov.shape == (0, 0) and empty_means.shape == (0,)


def test_subspace_comparison_is_basis_invariant_and_uses_original_feature_coordinates():
    left = np.eye(8)[:, :2]
    same = left @ np.array([[0., -3.], [2., 0.]])
    orthogonal = np.eye(8)[:, 2:4]
    assert compare_subspaces(left, same)["projector_overlap"] == pytest.approx(1.)
    assert compare_subspaces(left, orthogonal)["projector_overlap"] == 0.
    assert compare_subspaces(left, left[:, :1])["projector_overlap"] == pytest.approx(1.)
    with pytest.raises(ValueError):
        compare_subspaces(left, np.eye(3))


def test_bad_eigenvector_normalization_is_explicit_numerical_failure(monkeypatch):
    codes = _codes()
    rows = np.array([0, 1, 2])
    covariance, _ = blocked_centered_covariance(codes, rows, 3)
    real_eigh = linalg.eigh

    def wrong_normalization(*args, **kwargs):
        values, vectors = real_eigh(*args, **kwargs)
        return values, vectors * 2

    monkeypatch.setattr(linalg, "eigh", wrong_normalization)
    result = analyze_partition(codes, covariance, rows, rows, top_k=2, block_cols=3)
    assert result["checks"]["status"] == "numerical_check_failed"
    assert result["checks"]["numerical_checks_passed"] is False


def _plan():
    return json.loads((Path(__file__).parents[1] / "configs/public_spectrum_diagnostic_B16_T10.json").read_text(encoding="utf-8"))


def test_registered_plan_accepts_exact_settings_and_rejects_private_or_changed_fields():
    plan = _plan()
    validate_plan(plan)
    for key, value in (("private_truth_path", "secret.pt"), ("checkpoint", "public.pt"),
                       ("device", "cuda"), ("feature_chunk", 2048), ("expected_shape", [4, 8]),
                       ("exclude_top", 3), ("threads", True)):
        altered = {**plan, key: value}
        with pytest.raises(ValueError):
            validate_plan(altered)


def test_output_overwrite_and_source_overlap_rejected(tmp_path):
    observation = tmp_path / "observation"
    observation.mkdir()
    plan = tmp_path / "plan.json"
    plan.write_text("{}", encoding="utf-8")
    output = tmp_path / "new_output"
    assert validate_output(output, observation, plan) == output.resolve()
    with pytest.raises(ValueError):
        validate_output(observation / "nested", observation, plan)
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        validate_output(output, observation, plan)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_runner_rejects_unrecognized_config_before_reading_observation_or_creating_output(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({**_plan(), "private_truth": "never_read.pt"}), encoding="utf-8")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="registered public diagnostic fields"):
        run_diagnostic(observation=tmp_path / "missing_public", plan_path=plan_path, output=output)
    assert not output.exists()


@pytest.mark.parametrize("rows", [np.array([0, 0]), np.array([-1]), np.array([5]), np.array([.5])])
def test_invalid_row_selection_rejected(rows):
    with pytest.raises(ValueError):
        blocked_centered_covariance(_codes(), rows, 3)
