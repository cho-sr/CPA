"""Small deterministic mathematical checks; no model, packet, or GPU access."""
import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from qacpa.mp_spectrum import (
    GRID_POINTS, _mp_cdf_grid, _mp_quantiles_on_grid,
    fit_mp_spectrum, mp_quantiles,
)


def reference_quantile(beta, probability):
    """Independent adaptive quadrature and root solve in the theta variable."""
    root = np.sqrt(beta)
    lower = (1 - root) ** 2

    def mass(theta):
        return quad(lambda angle: (2 / np.pi) * np.sin(angle) ** 2 /
                    (lower + 4 * root * np.sin(angle / 2) ** 2),
                    0, theta, epsabs=2e-12, epsrel=2e-12, limit=200)[0]

    theta = brentq(lambda angle: mass(angle) - probability, 0, np.pi, xtol=2e-13)
    return lower + 4 * root * np.sin(theta / 2) ** 2


@pytest.mark.parametrize("beta", [1e-6, 0.01, 0.164, 0.6, 0.99, 1 - 1e-12])
def test_mp_quantiles_normalization_support_monotonicity_and_refinement(beta):
    probabilities = np.array([0, .001, .1, .2, .5, .8, .9, .999, 1])
    quantiles = mp_quantiles(beta, probabilities)
    refined = _mp_quantiles_on_grid(beta, probabilities, 2 * GRID_POINTS - 1)
    _, cdf, mass = _mp_cdf_grid(beta)
    assert mass == pytest.approx(1.0, abs=1e-7)
    assert np.all(np.diff(cdf) > 0)
    assert np.all(np.diff(quantiles) > 0)
    assert quantiles[0] == (1 - np.sqrt(beta)) ** 2
    assert quantiles[-1] == (1 + np.sqrt(beta)) ** 2
    np.testing.assert_allclose(quantiles, refined, rtol=0, atol=3e-7)


@pytest.mark.parametrize("beta", [0.01, 0.164, 0.9, 1 - 1e-12])
def test_quantiles_agree_with_independent_adaptive_integral(beta):
    probabilities = [.05, .2, .5, .8, .95]
    expected = [reference_quantile(beta, p) for p in probabilities]
    np.testing.assert_allclose(mp_quantiles(beta, probabilities), expected, rtol=0, atol=3e-7)


def test_quantile_shape_empty_and_input_immutability():
    probabilities = np.array([[.1, .5], [.9, .2]])
    saved = probabilities.copy()
    assert mp_quantiles(.1, probabilities).shape == probabilities.shape
    assert mp_quantiles(.1, .5).shape == ()
    assert mp_quantiles(.1, []).shape == (0,)
    np.testing.assert_array_equal(probabilities, saved)


@pytest.mark.parametrize("beta", [0, 1, -1, np.nan, np.inf, True])
def test_invalid_beta(beta):
    with pytest.raises(ValueError):
        mp_quantiles(beta, [.5])


@pytest.mark.parametrize("probabilities", [[-.01], [1.01], [np.nan], [np.inf], [1j], [True]])
def test_invalid_probabilities(probabilities):
    with pytest.raises(ValueError):
        mp_quantiles(.1, probabilities)


def exact_spectrum(m=100, d=1001, scale=2.75):
    # Fixture uses independent adaptive integration, not the implementation CDF.
    probabilities = (np.arange(m) + .5) / m
    values = np.array([reference_quantile(m / (d - 1), p) for p in probabilities]) * scale
    return values[::-1].copy()


def test_scaled_mp_recovery_and_original_rank_mapping_after_top16_exclusion():
    m, d, scale = 80, 801, 2.75
    values = exact_spectrum(m, d, scale)
    values[:16] += 100.0  # Known top outliers must not change beta or fit ranks.
    original = values.copy()
    result = fit_mp_spectrum(values, m, d)
    primary = result["primary_fit"]
    assert result["status"] == "OK"
    assert result["beta"] == m / (d - 1)
    assert result["bulk_count"] == m - 16
    assert result["sigma2"] == pytest.approx(scale, abs=2e-7)
    assert result["edge"] == pytest.approx(scale * (1 + np.sqrt(m / (d - 1))) ** 2, abs=4e-7)
    assert result["lambda16_over_edge"] == pytest.approx(values[15] / result["edge"])
    assert result["count_above_edge"] == 16
    assert primary["relative_residual_rms"] < 1e-7
    assert primary["relative_residual_p90"] < 1e-7
    expected_indices = [j for j in range(m - 16) if .2 <= (j + .5) / m <= .8]
    assert primary["ascending_indices"] == expected_indices
    assert primary["probabilities"] == [(j + .5) / m for j in expected_indices]
    for fit in result["sensitivity_fits"].values():
        assert fit["status"] == "OK"
        assert fit["sigma2"] == pytest.approx(scale, abs=2e-7)
        assert max(fit["ascending_indices"]) < m - 16
    np.testing.assert_array_equal(values, original)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("m", [0, 5, 16, 47])
def test_empty_small_and_insufficient_splits(m):
    result = fit_mp_spectrum(np.zeros(m), m, 101)
    assert result["status"] == "INSUFFICIENT_BULK"
    assert result["edge"] is None
    assert result["bulk_count"] == max(m - 16, 0)
    assert result["beta"] == (m / 100 if m else None)
    json.dumps(result, allow_nan=False)


def test_noiseless_zero_and_mixed_zero_bulk_are_degenerate_without_fallback():
    for values in (np.zeros(80), np.r_[np.ones(32), np.zeros(48)]):
        result = fit_mp_spectrum(values, 80, 801)
        assert result["status"] == "DEGENERATE"
        assert result["primary_fit"]["positive_fraction"] < .9
        assert result["edge"] is None
        assert result["count_above_edge"] is None
        json.dumps(result, allow_nan=False)


def test_scale_below_numerical_tolerance_is_separate_degeneracy():
    values = np.r_[1e14, np.ones(79)]
    result = fit_mp_spectrum(values, 80, 801)
    assert result["status"] == "DEGENERATE"
    assert result["primary_fit"]["positive_fraction"] == 1.0
    assert "scale" in result["reason"]
    assert result["sigma2"] is None


def test_small_negatives_reported_unchanged_but_large_negative_fails():
    values = np.linspace(2, 1, 80)
    values[-1] = -1e-14
    result = fit_mp_spectrum(values, 80, 801)
    assert result["status"] == "OK"
    assert result["raw_eigenvalues_desc"][-1] == -1e-14
    assert result["small_negative_clipped_count"] == 1
    assert values[-1] == -1e-14
    values[-1] = -1e-3
    failed = fit_mp_spectrum(values, 80, 801)
    assert failed["status"] == "NUMERICAL_FAILURE"
    assert failed["edge"] is None


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_is_serializable_numerical_failure(bad):
    values = np.ones(80)
    values[9] = bad
    result = fit_mp_spectrum(values, 80, 801)
    assert result["status"] == "NUMERICAL_FAILURE"
    assert result["raw_eigenvalues_desc"][9] is None
    assert result["nonfinite_entries"] == [{"index": 9, "value": str(bad)}]
    assert result["edge"] is None
    json.dumps(result, allow_nan=False)


def test_residuals_are_explicit_relative_errors_without_model_acceptance_threshold():
    result = fit_mp_spectrum(np.linspace(10, 1, 80), 80, 801)
    fit = result["primary_fit"]
    indices = fit["descending_indices"]
    raw = np.asarray(result["raw_eigenvalues_desc"])[indices]
    residual = np.abs(raw / (fit["sigma2"] * np.array(fit["mp_unit_quantiles"])) - 1)
    assert fit["relative_residual_rms"] == pytest.approx(np.sqrt(np.mean(residual**2)))
    assert fit["relative_residual_p50"] == pytest.approx(np.quantile(residual, .5))
    assert fit["relative_residual_p90"] == pytest.approx(np.quantile(residual, .9))
    assert "not MP-model acceptance" in result["interpretation"]


@pytest.mark.parametrize("values,m,d,extra", [
    ([1, 2], 2, 10, {}), ([1], 2, 10, {}), ([[1]], 1, 10, {}),
    ([1j], 1, 10, {}), ([1], True, 10, {}), ([1], 1, 2, {}),
    ([1], 1, 10, {"exclude_top": -1}), ([1], 1, 10, {"min_bulk": 0}),
])
def test_structural_input_misuse_raises(values, m, d, extra):
    with pytest.raises(ValueError):
        fit_mp_spectrum(values, m, d, **extra)
