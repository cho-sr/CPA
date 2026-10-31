import numpy as np
import pytest

from qacpa.m7_shuffle_distribution import distribution_rows, shuffle_rows, threshold_from_controls


@pytest.mark.parametrize("n", [36, 40, 80])
def test_shuffle_preserves_distinct_endpoint_ids_and_does_not_rematch(n):
    gids = [i * 3 + 5 for i in range(n)]
    matrix = 1000 * np.arange(n)[:, None] + np.arange(n)[None, :]
    rows, summaries = shuffle_rows(matrix, gids, "test")
    assert len(rows) == 20 * n
    for seed, summary in enumerate(summaries):
        selected = [r for r in rows if r["seed"] == seed]
        permutation = np.random.default_rng(seed).permutation(n)
        assert [r["reconstruction_global_index"] for r in selected] == gids
        assert [r["target_global_index"] for r in selected] == [gids[j] for j in permutation]
        assert sum(r["fixed_point"] for r in selected) == summary["fixed_points"]
        for i, row in enumerate(selected):
            assert row["lpips"] == 1000 * i + permutation[i]
        values = matrix[np.arange(n), permutation]
        assert summary["mean"] == pytest.approx(values.mean())
        assert summary["std"] == pytest.approx(values.std(ddof=1))
        assert summary["q05"] == pytest.approx(np.quantile(values, .05, method="linear"))


def test_common_threshold_uses_control_quantile_and_strict_boundary():
    threshold = threshold_from_controls(np.arange(40, dtype=float))
    assert threshold == pytest.approx(1.95)
    rows = [{"global_index": i, "lpips": value} for i, value in enumerate([threshold - 1e-8, threshold, threshold + 1e-8])]
    result = distribution_rows("gia_4bit", rows, threshold)
    assert [row["success_strict_below"] for row in result] == [1, 0, 0]


@pytest.mark.parametrize("values", [np.arange(39), np.append(np.arange(39), np.nan)])
def test_threshold_requires_full_finite_reference(values):
    with pytest.raises(ValueError, match="all 40 finite"):
        threshold_from_controls(values)


def test_shuffle_rejects_duplicate_or_missing_axis_ids():
    with pytest.raises(ValueError):
        shuffle_rows(np.zeros((3, 3)), [1, 1, 2], "test")
    with pytest.raises(ValueError):
        shuffle_rows(np.zeros((3, 3)), [1, 2], "test")
