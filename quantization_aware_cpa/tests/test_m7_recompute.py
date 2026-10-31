import numpy as np
import pytest

from qacpa.m7_recompute import per_victim_batch_summary, shuffled_rows


def test_shuffled_preserves_pair_ids_and_fixed_points_across_precisions():
    # Each distance identifies both endpoints, so transposed indexing or
    # re-matching cannot accidentally pass this check.
    matrix = 100 * np.arange(40)[:, None] + np.arange(40)[None, :]
    fp_rows, fp_seeds = shuffled_rows(matrix, method="fp32")
    four_rows, four_seeds = shuffled_rows(matrix, method="naive")
    assert len(fp_rows) == len(four_rows) == 800
    assert len(fp_seeds) == len(four_seeds) == 20
    assert fp_seeds[0]["fixed_points"] == 2
    assert fp_seeds[14]["fixed_points"] == 5
    assert [row["permutation"] for row in fp_seeds] == [row["permutation"] for row in four_seeds]
    for seed in range(20):
        selected = [row for row in fp_rows if row["seed"] == seed]
        assert sorted(row["target_global_index"] for row in selected) == list(range(40))
        assert [row["reconstruction_global_index"] for row in selected] == list(range(40))
        assert sum(row["fixed_point"] for row in selected) == fp_seeds[seed]["fixed_points"]
        for row in selected:
            assert row["lpips"] == 100 * row["reconstruction_global_index"] + row["target_global_index"]


@pytest.mark.parametrize("shape", [(39, 40), (40, 39), (8, 8)])
def test_shuffled_rejects_incomplete_matrix(shape):
    with pytest.raises(ValueError, match="complete 40 x 40"):
        shuffled_rows(np.zeros(shape), method="fp32")


def test_shuffled_rejects_nonfinite_matrix():
    matrix = np.zeros((40, 40))
    matrix[3, 7] = np.nan
    with pytest.raises(ValueError, match="finite"):
        shuffled_rows(matrix, method="fp32")


def test_batch_summaries_follow_permuted_original_not_reconstruction():
    # LPIPS here is the original's index. Every seed must give batch means
    # 3.5, 11.5, ... regardless of the reconstruction endpoints.
    matrix = np.broadcast_to(np.arange(40), (40, 40))
    rows, _ = shuffled_rows(matrix, method="fp32")
    pooled = per_victim_batch_summary(rows)
    by_seed = per_victim_batch_summary(rows, per_seed=True)
    assert len(pooled) == 5 and len(by_seed) == 100
    for row in pooled + by_seed:
        assert row["lpips_mean"] == 8 * row["batch_index"] + 3.5
        assert row["n_target_images"] == 8
        assert row["n_values"] == (160 if row["seed"] == "all" else 8)
