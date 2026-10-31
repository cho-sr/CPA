import json
from pathlib import Path

import pytest
import torch

from quantization_aware_cpa.qacpa.signed_stage_diagnostics import (
    DiagnosticError,
    assemble_rows,
    cosine_info,
    geometric_interval,
    normalize_pixels,
    sign_metrics,
    spearman,
    validate_bijection,
)


def _synthetic_measurement_inputs():
    d = 25088
    # Different nonnegative target_z rows keep the two arms independent.
    targets = {"naive": torch.stack([torch.ones(d) * (1 + i / 100) for i in range(16)]),
               "cell_rank": torch.stack([torch.ones(d) * (2 + i / 100) for i in range(16)])}
    signed = {arm: value.clone() for arm, value in targets.items()}
    signed["naive"][::2] *= -1
    signed["cell_rank"][1::2] *= -1
    features = {arm: value.clone() for arm, value in targets.items()}
    truth = torch.stack([torch.ones(d) * (1 + i / 100) for i in range(16)])
    image_maps = {"naive": {i: i for i in range(16)}, "cell_rank": {i: (i + 1) % 16 for i in range(16)}}
    embedding_maps = {"naive": {i: (i + 3) % 16 for i in range(16)}, "cell_rank": {i: (i + 5) % 16 for i in range(16)}}
    comp = {"arms": {}}
    for arm in ("naive", "cell_rank"):
        comp["arms"][arm] = {"image_metrics": {"per_image": [{"candidate_index": i, "truth_index": image_maps[arm][i], "mse": float(i), "psnr_db": float(i), "ssim": float(i), "lpips": float(i)} for i in range(16)]},
                              "embedding_oracle": {"per_source": [{"source_index": i, "truth_index": embedding_maps[arm][i]} for i in range(16)]}}
    return features, targets, signed, truth, comp, image_maps, embedding_maps


def test_normalize_pixels_is_exactly_once():
    x = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    x[:, 0] = 0.485
    x[:, 1] = 0.485
    x[:, 2] = 0.485
    y = normalize_pixels(x)
    assert torch.allclose(y[:, 0], torch.zeros_like(y[:, 0]))
    assert torch.allclose(y[:, 1], torch.full_like(y[:, 1], (0.485 - 0.456) / 0.224))


def test_normalize_rejects_out_of_range():
    with pytest.raises(DiagnosticError):
        normalize_pixels(torch.full((1, 3, 224, 224), 1.1))


def test_geometry_bounds_and_degenerate():
    result = geometric_interval(1.0, 1.0, 1.0)
    assert result["lower"] == result["upper"] == 1.0
    assert result["status"] == "COMPLETED"
    undefined = geometric_interval(None, 0.2, 0.3)
    assert undefined["status"] == "UNDEFINED" and undefined["lower"] is None
    violation = geometric_interval(1.0, 1.0, -1.0)
    assert violation["status"] == "VIOLATION"


def test_cosine_zero_and_nonfinite_reasons():
    assert cosine_info(torch.zeros(3), torch.ones(3))["reason"] == "ZERO_NORM"
    assert cosine_info(torch.tensor([float("nan")]), torch.ones(1))["reason"] == "NONFINITE_OR_SHAPE_MISMATCH"


def test_sign_metrics_global_sign_invariant():
    a = sign_metrics(torch.tensor([1.0, -2.0, 3.0]))
    b = sign_metrics(torch.tensor([-1.0, 2.0, -3.0]))
    assert a["minority_sign_energy_fraction"] == b["minority_sign_energy_fraction"]
    assert a["abs_vs_best_global_sign_cosine"] == b["abs_vs_best_global_sign_cosine"]
    assert sign_metrics(torch.zeros(4))["reason"] == "ZERO_SOURCE_NORM"


def test_bijection_accepts_permutation_and_rejects_duplicate():
    rows = [{"source_index": i, "truth_index": (i + 3) % 16} for i in range(16)]
    assert validate_bijection(rows, "source_index", "truth_index", "synthetic") == {i: (i + 3) % 16 for i in range(16)}
    rows[-1]["truth_index"] = rows[0]["truth_index"]
    with pytest.raises(DiagnosticError):
        validate_bijection(rows, "source_index", "truth_index", "synthetic")


def test_spearman_constant_and_tied_values():
    assert spearman([1.0, 1.0], [1.0, 2.0])["rho"] is None
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])["rho"] == pytest.approx(-1.0)


def test_corrected_targets_maps_and_sign_reversal():
    features, targets, signed, truth, comp, image_maps, embedding_maps = _synthetic_measurement_inputs()
    rows, paired_i, paired_e = assemble_rows(features, targets, signed, truth, comp, image_maps, embedding_maps)
    # Every corrected C_fit is positive even where signed S_hat is all-negative.
    assert all(row["C_fit"] > 0.999 for row in rows)
    assert all(row["naive_truth_index"] == row["truth_index"] for row in paired_i)
    assert all(row["cell_rank_truth_index"] == row["truth_index"] for row in paired_i)
    assert all(row["naive_truth_index"] == row["truth_index"] for row in paired_e)
    assert all(row["cell_rank_truth_index"] == row["truth_index"] for row in paired_e)
    reversed_rows, _, _ = assemble_rows(features, targets, {arm: -value for arm, value in signed.items()}, truth, comp, image_maps, embedding_maps)
    for left, right in zip(rows, reversed_rows):
        assert left["C_fit"] == pytest.approx(right["C_fit"])
        assert left["C_target_I"] == pytest.approx(right["C_target_I"])
        assert left["sign"]["minority_sign_energy_fraction"] == pytest.approx(right["sign"]["minority_sign_energy_fraction"])
        assert left["sign"]["negative_energy_fraction"] == pytest.approx(1 - right["sign"]["negative_energy_fraction"])
