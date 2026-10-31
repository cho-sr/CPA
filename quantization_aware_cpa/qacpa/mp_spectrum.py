"""Public-spectrum MP diagnostics; no packet access or reconstruction decisions.

The reference law is the unit-mean covariance Marchenko--Pastur law for
0 < beta < 1, with density sqrt((b-x)*(x-a))/(2*pi*beta*x) and endpoints
a=(1-sqrt(beta))**2, b=(1+sqrt(beta))**2.  See Gavish and Donoho,
https://arxiv.org/html/1405.7511, Sections 2.2, 3 and 8.1, for the white-noise
framework and the use of empirical spectra to estimate noise scale.

Here scale fitting is only a declared descriptive heuristic: an OK fit does
not establish independent noise, a correct deterministic-quantization error
model, or a recoverable image count.  No goodness-of-fit rejection threshold
is imposed.  The top eigenvalues remain in the report and original ranks.

Quantiles use a 65,537-point monotone theta-grid CDF with trapezoidal
integration and linear inversion.  theta=pi*t**2 resolves the lower endpoint
as beta approaches one.  The substitution x=a+4*sqrt(beta)*sin(theta/2)**2
has density (2/pi)*sin(theta)**2/x in theta.  Numerical mass is normalized
after checking it; tests compare refined grids and independent quadrature.
This is a numerical approximation, not an exact CDF or an accuracy guarantee
for every representable beta/probability pair.
"""
from __future__ import annotations

from functools import lru_cache
from numbers import Integral

import numpy as np
from scipy.integrate import cumulative_trapezoid


GRID_POINTS = 65_537
FIT_WINDOWS = {"primary": (0.2, 0.8), "p10_p90": (0.1, 0.9), "p30_p70": (0.3, 0.7)}


def _validate_beta(beta):
    if isinstance(beta, (bool, np.bool_)) or not np.isscalar(beta):
        raise ValueError("beta must be a finite scalar with 0 < beta < 1")
    beta = float(beta)
    if not np.isfinite(beta) or not 0.0 < beta < 1.0:
        raise ValueError("beta must be a finite scalar with 0 < beta < 1")
    return beta


@lru_cache(maxsize=16)
def _mp_cdf_grid(beta, grid_points=GRID_POINTS):
    """Return read-only x/CDF arrays and pre-normalization integration mass."""
    beta = _validate_beta(beta)
    if not isinstance(grid_points, Integral) or grid_points < 3:
        raise ValueError("grid_points must be an integer >= 3")
    root = np.sqrt(beta)
    lower = (1.0 - root) ** 2
    theta = np.pi * np.linspace(0.0, 1.0, int(grid_points), dtype=np.float64) ** 2
    x = lower + 4.0 * root * np.sin(theta / 2.0) ** 2
    density_theta = (2.0 / np.pi) * np.sin(theta) ** 2 / x
    density_theta[[0, -1]] = 0.0
    cdf = cumulative_trapezoid(density_theta, theta, initial=0.0)
    mass = float(cdf[-1])
    if not np.isfinite(mass) or abs(mass - 1.0) > 1e-6:
        raise ArithmeticError("MP CDF grid failed its unit-mass numerical check")
    cdf /= mass
    # Floating-point accumulation can repeat CDF values at the upper edge.
    # Keep the final endpoint so probability one maps to the exact support.
    keep = np.r_[np.diff(cdf) > 0.0, True]
    x, cdf = x[keep], cdf[keep]
    x[0], x[-1] = lower, (1.0 + root) ** 2
    cdf[0], cdf[-1] = 0.0, 1.0
    x.setflags(write=False)
    cdf.setflags(write=False)
    return x, cdf, mass


def _mp_quantiles_on_grid(beta, probabilities, grid_points):
    beta = _validate_beta(beta)
    raw = np.asarray(probabilities)
    if raw.dtype.kind not in "iuf":
        raise ValueError("probabilities must be real numeric values")
    probabilities = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    x, cdf, _ = _mp_cdf_grid(beta, grid_points)
    return np.asarray(np.interp(probabilities.ravel(), cdf, x)).reshape(probabilities.shape)


def mp_quantiles(beta, probabilities):
    """Return FP64 unit-scale MP quantiles, preserving the input array shape.

    Endpoints 0 and 1 are accepted and map to the theoretical support limits.
    Neither global random state nor caller-owned arrays are changed.
    """
    return _mp_quantiles_on_grid(beta, probabilities, GRID_POINTS)


def _integer(name, value, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _empty_fit(window, status, reason):
    return {
        "status": status, "reason": reason, "probability_window": list(window),
        "ascending_indices": [], "descending_indices": [], "probabilities": [],
        "mp_unit_quantiles": [], "fit_count": 0, "positive_fraction": None,
        "sigma2": None, "edge": None, "relative_residual_rms": None,
        "relative_residual_p50": None, "relative_residual_p90": None,
        "lambda16_over_edge": None, "count_above_edge": None,
    }


def _fit_window(values_desc, beta, exclude_top, tau, window):
    m = values_desc.size
    indices = np.arange(m - exclude_top, dtype=np.int64)
    probabilities = (indices.astype(np.float64) + 0.5) / float(m)
    mask = (probabilities >= window[0]) & (probabilities <= window[1])
    indices, probabilities = indices[mask], probabilities[mask]
    result = _empty_fit(window, "INSUFFICIENT_BULK", "no original ranks in the fixed fit window")
    result.update(ascending_indices=indices.tolist(),
                  descending_indices=(m - 1 - indices).tolist(),
                  probabilities=probabilities.tolist(), fit_count=int(indices.size))
    if not indices.size:
        return result
    values = np.maximum(values_desc[::-1][indices], 0.0)
    positive_fraction = float(np.mean(values > 0.0))
    result["positive_fraction"] = positive_fraction
    if positive_fraction < 0.9:
        result.update(status="DEGENERATE", reason="fewer than 90% of fit eigenvalues are positive")
        return result
    try:
        unit = mp_quantiles(beta, probabilities)
    except ArithmeticError as error:
        result.update(status="NUMERICAL_FAILURE", reason=str(error))
        return result
    result["mp_unit_quantiles"] = unit.tolist()
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratios = np.sort(values / unit)
        midpoint = ratios.size // 2
        sigma2 = (float(ratios[midpoint]) if ratios.size % 2 else
                  float(ratios[midpoint - 1] / 2.0 + ratios[midpoint] / 2.0))
    if not np.isfinite(sigma2):
        result.update(status="NUMERICAL_FAILURE", reason="nonfinite fitted scale")
        return result
    if sigma2 <= tau:
        result.update(status="DEGENERATE", reason="fitted scale does not exceed the numerical tolerance")
        return result
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        predicted = sigma2 * unit
        relative = np.abs(values / predicted - 1.0)
        edge = sigma2 * (1.0 + np.sqrt(beta)) ** 2
        maximum = float(relative.max())
        rms = maximum * float(np.sqrt(np.mean((relative / maximum) ** 2))) if maximum else 0.0
        p50, p90 = np.quantile(relative, [0.5, 0.9])
        lambda16_ratio = float(values_desc[15] / edge) if m >= 16 else None
    numbers = [edge, rms, p50, p90] + ([lambda16_ratio] if lambda16_ratio is not None else [])
    if not np.isfinite(predicted).all() or not np.isfinite(numbers).all():
        result.update(status="NUMERICAL_FAILURE", reason="nonfinite edge or relative residual")
        return result
    result.update(status="OK", reason=None, sigma2=sigma2, edge=float(edge),
                  relative_residual_rms=float(rms), relative_residual_p50=float(p50),
                  relative_residual_p90=float(p90), lambda16_over_edge=lambda16_ratio,
                  count_above_edge=int(np.count_nonzero(values_desc > edge)))
    return result


def fit_mp_spectrum(eigenvalues_desc, m, d, exclude_top=16, min_bulk=32):
    """Fit fixed original-rank central quantiles to an MP scale.

    Input is the full descending spectrum of C=M M.T/(d-1), not an already
    truncated spectrum.  beta=m/(d-1) is never changed by excluding top values.
    Ascending zero-based rank j maps to p_j=(j+0.5)/m; retained ranks are NOT
    renormalized.  Primary p=[.2,.8] and sensitivity p=[.1,.9], [.3,.7] are
    inclusive intersections with those ranks after top exclusion.

    Residuals are abs(lambda/(sigma2*MP_quantile)-1); RMS, P50, P90 use exactly
    each window's fitted ranks.  Negative eigenvalues within tau are clipped
    only for fitting.  The original finite negative values remain reported.
    tau=100*eps64*max(m,d)*max(abs(lambda)).  Nonfinite input is a diagnostic
    NUMERICAL_FAILURE, with null plus an explicit string preserving its kind.
    Structural API misuse raises ValueError.  All reports permit strict JSON.
    """
    m, d = _integer("m", m, 0), _integer("d", d, 2)
    exclude_top = _integer("exclude_top", exclude_top, 0)
    min_bulk = _integer("min_bulk", min_bulk, 1)
    if m >= d - 1:
        raise ValueError("supported covariance aspect ratio requires m < d - 1")
    raw = np.asarray(eigenvalues_desc)
    if raw.ndim != 1 or raw.size != m or raw.dtype.kind not in "iuf":
        raise ValueError("eigenvalues_desc must be a real one-dimensional array of length m")
    values = np.array(raw, dtype=np.float64, copy=True)
    finite = np.isfinite(values)
    beta = float(m / (d - 1)) if m else None
    result = {
        "schema_version": "mp_spectrum_quantile_fit_v1", "status": None, "reason": None,
        "m": m, "d": d, "beta": beta, "covariance_denominator": d - 1,
        "exclude_top": exclude_top, "min_bulk": min_bulk, "bulk_count": max(0, m - exclude_top),
        "negative_tolerance": None, "small_negative_clipped_count": 0,
        "raw_eigenvalues_desc": [float(value) if ok else None for value, ok in zip(values, finite)],
        "nonfinite_entries": [{"index": int(i), "value": str(values[i])} for i in np.flatnonzero(~finite)],
        "quantile_rank_rule": "p_j=(j+0.5)/m; j is original zero-based ascending rank; no renormalization",
        "residual_rule": "abs(lambda/(sigma2*MP_quantile(p))-1), over each fit window",
        "primary_fit": None, "sensitivity_fits": {}, "sigma2": None, "edge": None,
        "lambda16_over_edge": None, "count_above_edge": None,
        "numerics": {"dtype": "float64", "cdf_grid_points": GRID_POINTS,
                     "cdf_grid": "theta=pi*t**2; cumulative trapezoid; linear inverse",
                     "maximum_cdf_mass_error": 1e-6},
        "interpretation": "OK means a finite heuristic fit, not MP-model acceptance or a recoverable image count",
    }

    def fail(status, reason):
        result.update(status=status, reason=reason)
        result["primary_fit"] = _empty_fit(FIT_WINDOWS["primary"], status, reason)
        result["sensitivity_fits"] = {
            name: _empty_fit(window, status, reason)
            for name, window in FIT_WINDOWS.items() if name != "primary"
        }
        return result

    if not finite.all():
        return fail("NUMERICAL_FAILURE", "nonfinite input eigenvalues")
    if np.any(values[:-1] < values[1:]):
        raise ValueError("eigenvalues_desc must be sorted in nonincreasing order")
    maximum = float(np.max(np.abs(values))) if m else 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        tau = 100.0 * np.finfo(np.float64).eps * max(m, d) * maximum
    if not np.isfinite(tau):
        return fail("NUMERICAL_FAILURE", "nonfinite numerical tolerance")
    result["negative_tolerance"] = float(tau)
    if np.any(values < -tau):
        return fail("NUMERICAL_FAILURE", "negative eigenvalue exceeds the numerical tolerance")
    result["small_negative_clipped_count"] = int(np.count_nonzero(values < 0.0))
    if m - exclude_top < min_bulk:
        return fail("INSUFFICIENT_BULK", "fewer than min_bulk eigenvalues remain after fixed top exclusion")
    primary = _fit_window(values, beta, exclude_top, tau, FIT_WINDOWS["primary"])
    result.update(status=primary["status"], reason=primary["reason"], primary_fit=primary)
    result["sensitivity_fits"] = {
        name: _fit_window(values, beta, exclude_top, tau, window)
        for name, window in FIT_WINDOWS.items() if name != "primary"
    }
    for key in ("sigma2", "edge", "lambda16_over_edge", "count_above_edge"):
        result[key] = primary[key]
    return result
