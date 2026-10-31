"""CPU-only offline evaluation gates; truth never chooses attack artifacts."""
import json
import math

import pytest
import torch

from qacpa.full_update_evaluation import evaluate_candidate


MEAN = (.485, .456, .406)
STD = (.229, .224, .225)


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def normalized(pixels):
    mean = pixels.new_tensor(MEAN).view(1, 3, 1, 1)
    std = pixels.new_tensor(STD).view(1, 3, 1, 1)
    return (pixels - mean) / std


def denormalized(images):
    images = images.double()
    return (images * images.new_tensor(STD).view(1, 3, 1, 1)
            + images.new_tensor(MEAN).view(1, 3, 1, 1)).clamp(0, 1)


def save_pair(tmp_path, candidate=None, truth=None):
    if candidate is None:
        candidate = {"pixels": torch.full((1, 3, 224, 224), .5), "labels": torch.tensor([12])}
    if truth is None:
        truth = {"x": torch.zeros((1, 3, 224, 224)), "labels": torch.tensor([12])}
    candidate_path, truth_path = tmp_path / "candidate.pt", tmp_path / "truth.pt"
    torch.save(candidate, candidate_path)
    torch.save(truth, truth_path)
    return candidate_path, truth_path


def test_one_pixel_hungarian_permutation_used_by_every_metric_without_candidate_selection(tmp_path, monkeypatch):
    import scipy.optimize
    import torchmetrics.functional.image as image_metrics

    truth_pixels = torch.stack([torch.full((3, 224, 224), .2, dtype=torch.float64),
                                torch.full((3, 224, 224), .8, dtype=torch.float64)])
    truth_images = normalized(truth_pixels)
    reference = denormalized(truth_images)
    pixels = reference[[1, 0]] + torch.tensor([.01, .03], dtype=torch.float64).view(2, 1, 1, 1)
    # Labels deliberately favor the wrong assignment; matching must use pixels.
    candidate = {"pixels": pixels, "labels": torch.tensor([5, 7])}
    truth = {"x": truth_images, "labels": torch.tensor([5, 7])}
    paths = save_pair(tmp_path, candidate, truth)
    before = [path.read_bytes() for path in paths]
    # A better artifact next to the supplied candidate must never be selected.
    torch.save({"pixels": reference, "labels": truth["labels"]}, tmp_path / "perfect_candidate.pt")
    original_load = torch.load
    original_match = scipy.optimize.linear_sum_assignment
    original_ssim = image_metrics.structural_similarity_index_measure
    loads, matches, metric_pairs = [], [], []

    def load(path, **kwargs):
        loads.append(path)
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return original_load(path, **kwargs)

    def matching(costs):
        matches.append(costs.copy())
        return original_match(costs)

    def ssim(candidate_images, reference_images, **kwargs):
        assert candidate_images.device.type == reference_images.device.type == "cpu"
        metric_pairs.append((candidate_images.clone(), reference_images.clone()))
        return original_ssim(candidate_images, reference_images, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(scipy.optimize, "linear_sum_assignment", matching)
    monkeypatch.setattr(image_metrics, "structural_similarity_index_measure", ssim)
    report = evaluate_candidate(*paths)
    assert loads == list(paths) and len(matches) == 1 and len(metric_pairs) == 2
    assert report["matching"] == "Hungarian_pixel_MSE_once_for_all_metrics"
    assert report["candidate_selection"] == "none_single_supplied_artifact"
    assert report["all_images_included"] and report["n_images"] == 2
    assert [item["truth_index"] for item in report["per_image"]] == [1, 0]
    assert report["label_accuracy"] == 0.0
    for i, j in ((0, 1), (1, 0)):
        item = report["per_image"][i]
        expected_mse = float((pixels[i] - reference[j]).square().mean())
        assert item["mse"] == pytest.approx(expected_mse)
        assert item["psnr_db"] == pytest.approx(-10 * math.log10(expected_mse))
        torch.testing.assert_close(metric_pairs[i][0], pixels[i:i + 1])
        torch.testing.assert_close(metric_pairs[i][1], reference[j:j + 1])
        assert not item["label_correct"]
    assert [path.read_bytes() for path in paths] == before
    json.dumps(report, allow_nan=False)


def test_perfect_match_psnr_uses_explicit_json_null_and_metrics_are_finite(tmp_path):
    images = torch.zeros((1, 3, 224, 224), dtype=torch.float64)
    pixels = denormalized(images)
    paths = save_pair(tmp_path, {"pixels": pixels, "labels": torch.tensor([250], dtype=torch.uint8)},
                      {"x": images, "labels": torch.tensor([250], dtype=torch.int32)})
    report = evaluate_candidate(*paths)
    assert report["mean_mse"] == 0.0
    assert report["mean_psnr_db"] is None and report["per_image"][0]["psnr_db"] is None
    assert report["perfect_pixel_matches"] == 1
    assert report["per_image"][0]["perfect_pixel_match"]
    assert report["psnr_null_meaning"] == "positive_infinity_for_perfect_pixel_match"
    assert report["mean_ssim"] == pytest.approx(1.0)
    assert report["label_accuracy"] == 1.0
    assert report["lpips"] == "not_computed"
    assert report["reconstruction_success"] == "no_threshold_registered"
    json.dumps(report, allow_nan=False)


def test_mean_psnr_includes_perfect_image_as_infinite_not_finite_subset_average(tmp_path):
    images = torch.zeros((2, 3, 224, 224), dtype=torch.float64)
    images[1] = .5
    pixels = denormalized(images)
    pixels[1] += .02
    paths = save_pair(tmp_path, {"pixels": pixels, "labels": torch.tensor([1, 2])},
                      {"x": images, "labels": torch.tensor([1, 2])})
    report = evaluate_candidate(*paths)
    assert report["perfect_pixel_matches"] == 1
    assert report["mean_psnr_db"] is None
    assert report["per_image"][1]["psnr_db"] is not None
    assert report["mean_mse"] == pytest.approx(.02 ** 2 / 2)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normalization_roundoff_at_zero_and_one_is_accepted_but_reported(tmp_path, dtype):
    pixels = torch.zeros((1, 3, 224, 224), dtype=dtype)
    pixels[:, :, 112:, :] = 1.0
    images = normalized(pixels)
    paths = save_pair(tmp_path, {"pixels": pixels, "labels": torch.tensor([0])},
                      {"x": images, "labels": torch.tensor([0])})
    report = evaluate_candidate(*paths)
    assert report["truth_denormalization_tolerance_abs"] == 8 * torch.finfo(dtype).eps
    assert report["mean_mse"] < 1e-12
    assert report["truth_denormalization_clipped_values"] >= 0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("role", ["candidate", "truth"])
@pytest.mark.parametrize("labels", [torch.tensor([12.]), torch.tensor([12.5]),
                                    torch.tensor([True]), torch.tensor([[12]]),
                                    torch.tensor(12), torch.tensor([], dtype=torch.long),
                                    torch.tensor([12, 13]), torch.tensor([-1]),
                                    torch.tensor([1000]), torch.tensor([float("nan")]), [12]])
def test_malformed_labels_are_rejected_without_integer_coercion(tmp_path, role, labels):
    candidate = {"pixels": torch.full((1, 3, 224, 224), .5), "labels": torch.tensor([12])}
    truth = {"x": torch.zeros((1, 3, 224, 224)), "labels": torch.tensor([12])}
    (candidate if role == "candidate" else truth)["labels"] = labels
    with pytest.raises(ValueError, match="labels"):
        evaluate_candidate(*save_pair(tmp_path, candidate, truth))


@pytest.mark.parametrize("role", ["candidate", "truth"])
@pytest.mark.parametrize("bad", ["nan", "inf", "integer", "complex", "half", "empty", "wrong_shape", "list"])
def test_image_tensor_validation_rejects_malformed_inputs(tmp_path, role, bad):
    images = torch.full((1, 3, 224, 224), .5)
    if bad == "nan":
        images[0, 0, 0, 0] = float("nan")
    elif bad == "inf":
        images[0, 0, 0, 0] = float("inf")
    elif bad == "integer":
        images = images.long()
    elif bad == "complex":
        images = images.to(dtype=torch.complex64)
    elif bad == "half":
        images = images.half()
    elif bad == "empty":
        images = images[:0]
    elif bad == "wrong_shape":
        images = images[:, :, :223, :]
    else:
        images = [0.5]
    payload = {"pixels" if role == "candidate" else "x": images, "labels": torch.tensor([12])}
    paths = save_pair(tmp_path, candidate=payload) if role == "candidate" else save_pair(tmp_path, truth=payload)
    with pytest.raises(ValueError, match="images"):
        evaluate_candidate(*paths)


@pytest.mark.parametrize("role,value", [("candidate", -.00001), ("candidate", 1.00001),
                                       ("truth", -10.), ("truth", 10.)])
def test_out_of_range_pixels_or_normalized_truth_are_rejected_not_clamped(tmp_path, role, value):
    images = torch.full((1, 3, 224, 224), .5)
    images[0, 0, 0, 0] = value
    payload = {"pixels" if role == "candidate" else "x": images, "labels": torch.tensor([12])}
    paths = save_pair(tmp_path, candidate=payload) if role == "candidate" else save_pair(tmp_path, truth=payload)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        evaluate_candidate(*paths)


@pytest.mark.parametrize("role", ["candidate", "truth"])
@pytest.mark.parametrize("payload", [{}, {"labels": torch.tensor([12])}, [1, 2, 3]])
def test_payload_structure_is_validated(tmp_path, role, payload):
    paths = save_pair(tmp_path, candidate=payload) if role == "candidate" else save_pair(tmp_path, truth=payload)
    with pytest.raises(ValueError, match="payload"):
        evaluate_candidate(*paths)


def test_different_image_counts_rejected_before_matching(tmp_path):
    truth = {"x": torch.zeros((2, 3, 224, 224)), "labels": torch.tensor([12, 13])}
    with pytest.raises(ValueError, match="match exactly"):
        evaluate_candidate(*save_pair(tmp_path, truth=truth))


def test_nonfinite_ssim_fails_instead_of_serializing_nan(tmp_path, monkeypatch):
    import torchmetrics.functional.image as image_metrics
    monkeypatch.setattr(image_metrics, "structural_similarity_index_measure",
                        lambda *args, **kwargs: torch.tensor(float("nan")))
    with pytest.raises(ValueError, match="SSIM.*nonfinite"):
        evaluate_candidate(*save_pair(tmp_path))
