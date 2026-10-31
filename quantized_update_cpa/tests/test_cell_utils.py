"""Authorized CPU-only synthetic checks; no common.py/model/data imports."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260914


def load_file(name):
    spec = importlib.util.spec_from_file_location("signed_test_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = load_file("quantization")
cells = load_file("cell_utils")
metrics = load_file("trial_metrics")


def ref(G):
    return reference.quantize_tensor_symmetric(G, bits=4, rounding="nearest")


def from_bits(bits):
    return torch.tensor(bits, dtype=torch.int32).view(torch.float32)


def fixed_reference(G, s):
    # Set reference absmax to exactly 7, so it always divides by FP32 one.
    # Test the observed-s division independently before invoking its rounding.
    ratios = (G / s).clamp(-7, 7)
    return ref(torch.cat((ratios.flatten(), torch.tensor([-7., 7.]))))[0][:-2].reshape(G.shape)


def fixtures():
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    normal = torch.randn((4, 19), generator=generator, dtype=torch.float32)
    return {
        "normal": normal,
        "zero": torch.zeros((2, 5), dtype=torch.float32),
        "ties": torch.cat((torch.arange(-6.5, 7., dtype=torch.float32), torch.tensor([-7., 7., 0.]))),
        "tiny_normal": normal * torch.tensor(1e-30, dtype=torch.float32),
        "subnormal_positive_scale": from_bits([-2147483641, -2147483644, 0, 4, 7]),
        "scale_rounding": torch.tensor([-1.0000001192092896, 0., 0.7, 1.0000001192092896]),
        "max_finite": torch.tensor([-torch.finfo(torch.float32).max, 0., torch.finfo(torch.float32).max]),
        "scalar": torch.tensor(0.12345679),
    }


FIXTURES = fixtures()


@pytest.fixture(params=list(FIXTURES))
def sample(request):
    G = FIXTURES[request.param].clone()
    q, dq, s, _ = ref(G)
    L, U = cells.get_cells(q, s, dict(cells.CODEC_METADATA))
    return G, q, dq, s, L, U


def test_reference_exact(sample):
    G, q, dq, s, _, _ = sample
    actual_q, actual_s = cells.encode(G)
    assert actual_q.dtype == torch.int8 and actual_s.dtype == torch.float32
    assert torch.equal(actual_q, q)
    assert torch.equal(actual_s, s)
    assert torch.equal(cells.dequantize(q, s), dq)


def test_A_original_contained(sample):
    G, _, _, _, L, U = sample
    assert bool(((L <= G) & (G <= U)).all())


def candidates(G):
    return (torch.full_like(G, -torch.finfo(torch.float32).max),
            torch.full_like(G, torch.finfo(torch.float32).max),
            torch.zeros_like(G), -G)


def test_B_projection_contained(sample):
    G, _, _, _, L, U = sample
    for candidate in candidates(G):
        P = cells.project_to_cells(candidate, L, U)
        assert bool(((L <= P) & (P <= U)).all())


def test_C_projection_fixed_scale_codes(sample):
    G, q, _, s, L, U = sample
    for candidate in candidates(G):
        P = cells.project_to_cells(candidate, L, U)
        assert torch.equal(fixed_reference(P, s), q)
        assert cells.check_code_consistency(P, q, s)["all_equal"]


def test_D_reference_dequantization_contained(sample):
    _, _, dq, _, L, U = sample
    assert bool(((L <= dq) & (dq <= U)).all())


def test_E_all_endpoint_owners_and_neighbors(sample):
    _, _, _, s, _, _ = sample
    q = torch.arange(-7, 8, dtype=torch.int8)
    L, U = cells.get_cells(q, s, dict(cells.CODEC_METADATA))
    assert torch.equal(fixed_reference(L, s), q)
    assert torch.equal(fixed_reference(U, s), q)
    dq = cells.dequantize(q, s)
    assert bool(((L <= dq) & (dq <= U)).all())
    left = torch.nextafter(L, torch.full_like(L, -torch.inf))
    right = torch.nextafter(U, torch.full_like(U, torch.inf))
    assert bool((fixed_reference(left[1:], s) != q[1:]).all())
    assert bool((fixed_reference(right[:-1], s) != q[:-1]).all())
    cap = U[-1]
    assert torch.equal(cap / 7, s)
    next_cap = torch.nextafter(cap, torch.tensor(torch.inf))
    assert bool(next_cap / 7 > s)
    assert torch.equal(L, -U.flip(0))
    assert bool(torch.isfinite(L).all() & torch.isfinite(U).all())


def test_F_zero_reference_exact():
    G = torch.zeros(11, dtype=torch.float32)
    q, dq, s, _ = ref(G)
    actual_q, actual_s = cells.encode(G)
    assert torch.equal(q, torch.zeros(11, dtype=torch.int8))
    assert torch.equal(s, torch.tensor(1., dtype=torch.float32))
    assert torch.equal(actual_q, q) and torch.equal(actual_s, s)
    assert torch.equal(cells.dequantize(q, s), dq) and torch.equal(dq, G)


@pytest.mark.parametrize("tie", [k + 0.5 for k in range(-7, 7)])
def test_E_exact_tie_and_both_representable_neighbors(tie):
    center = torch.tensor(tie, dtype=torch.float32)
    G = torch.stack((torch.nextafter(center, torch.tensor(-torch.inf)), center,
                     torch.nextafter(center, torch.tensor(torch.inf))))
    expected = torch.tensor([int(tie - 0.5), round(tie), int(tie + 0.5)], dtype=torch.int8)
    s = torch.tensor(1., dtype=torch.float32)
    assert torch.equal(fixed_reference(G, s), expected)
    L, U = cells.get_cells(expected, s, dict(cells.CODEC_METADATA))
    assert bool(((L <= G) & (G <= U)).all())
    assert cells.check_code_consistency(G, expected, s)["all_equal"]


def test_clipping_is_fixed_scale_and_separate_from_scale_consistency():
    G = torch.tensor([-100., -7., 0., 7., 100.])
    q = torch.tensor([-7, -7, 0, 7, 7], dtype=torch.int8)
    s = torch.tensor(1.)
    assert torch.equal(fixed_reference(G, s), q)
    assert cells.check_code_consistency(G, q, s)["all_equal"]
    assert not torch.equal(cells.encode(G)[0], q)
    metric = cells.scale_consistency(G, s)
    assert not metric["exact_equal"] and metric["candidate_supported"]
    assert metric["candidate_scale"] == float(ref(G)[2])
    L, U = cells.get_cells(q, s, dict(cells.CODEC_METADATA))
    P = cells.project_to_cells(G, L, U)
    assert bool((P.abs() < 100).all())
    assert cells.check_code_consistency(P, q, s)["all_equal"]


def test_projection_does_not_install_scale_anchor():
    q = torch.tensor([0, 1, -1], dtype=torch.int8)
    s = torch.tensor(1.)
    L, U = cells.get_cells(q, s, dict(cells.CODEC_METADATA))
    G = cells.dequantize(q, s)
    P = cells.project_to_cells(G, L, U)
    assert torch.equal(P, G)
    assert cells.check_code_consistency(P, q, s)["all_equal"]
    assert not cells.scale_consistency(P, s)["exact_equal"]


def test_scale_rounding_cap_counterexample_to_naive_7s():
    # Deterministic FP32 bit fixtures; one fixed synthetic test, not a seed sweep.
    maxima = from_bits([0x3FE00006, 0x3FE0000B, 0x3FFFFFFF, 0x00800001])
    witnessed = False
    for maximum in maxima:
        G = torch.stack((-maximum, maximum))
        q, _, s, _ = ref(G)
        L, U = cells.get_cells(q, s, dict(cells.CODEC_METADATA))
        assert bool(((L <= G) & (G <= U)).all())
        if bool(7 * s < maximum):
            witnessed = True
            assert bool(U[1] > 7 * s)
    assert witnessed, "fixture must expose the naive 7*s cap failure"


@pytest.mark.parametrize("bits", [1, 2, 3])
def test_reference_limitation_nonzero_absmax_div7_underflow(bits):
    tiny = from_bits(bits)
    G = torch.stack((-tiny, torch.tensor(0.), tiny))
    q, dq, s, _ = ref(G)
    assert float(s) == 0.0
    assert torch.equal(q, torch.tensor([-7, 0, 7], dtype=torch.int8))
    assert torch.equal(dq, torch.zeros_like(G))
    assert bool(torch.isnan(G / s)[1])
    with pytest.raises(cells.UnsupportedCodecInput, match="underflow"):
        cells.encode(G)
    with pytest.raises(cells.UnsupportedCodecInput):
        cells.get_cells(q, s, dict(cells.CODEC_METADATA))
    metric = cells.scale_consistency(G, torch.tensor(1.))
    assert not metric["candidate_supported"]
    assert metric["reason"] == "NONZERO_ABSMAX_DIV7_UNDERFLOW"


@pytest.mark.parametrize("bad", [{}, {"codec": "uniform_affine_uint4_nearest_even_v1"},
                                  {**cells.CODEC_METADATA, "scale_scope": "per_row"}])
def test_reject_unknown_or_affine_metadata(bad):
    with pytest.raises(cells.UnsupportedCodecInput, match="metadata"):
        cells.get_cells(torch.tensor([0], dtype=torch.int8), torch.tensor(1.), bad)


@pytest.mark.parametrize("G", [torch.tensor([1.], dtype=torch.float64), torch.tensor([]),
                               torch.tensor([torch.nan]), torch.tensor([torch.inf])])
def test_explicit_input_support(G):
    with pytest.raises(cells.UnsupportedCodecInput):
        cells.encode(G)


def test_reject_invalid_scale_codes_and_shape():
    q = torch.tensor([0], dtype=torch.int8)
    for s in (torch.tensor(-1.), torch.tensor(0.), torch.tensor([1.]), torch.tensor(1e38)):
        with pytest.raises(cells.UnsupportedCodecInput):
            cells.get_cells(q, s, dict(cells.CODEC_METADATA))
    for bad in (torch.tensor([-8], dtype=torch.int8), torch.tensor([0], dtype=torch.uint8)):
        with pytest.raises(cells.UnsupportedCodecInput):
            cells.get_cells(bad, torch.tensor(1.), dict(cells.CODEC_METADATA))
    with pytest.raises(ValueError, match="shapes"):
        cells.check_code_consistency(torch.ones(2), q, torch.tensor(1.))


def test_raw_metric_schema_keeps_uncomputed_and_oracles_distinct():
    record = metrics.new_trial_metrics()
    assert set(record["raw_metrics"]) == set(metrics.RAW_METRICS)
    for value in record["raw_metrics"].values():
        assert value["value"] is None and value["status"] == "NOT_COMPUTED" and value["reason"]
    assert set(record["evaluation_metrics"]) == {"embedding_oracle", "image_oracle"}
    assert set(record["evaluation_metrics"]["image_oracle"]) == {"lpips", "psnr", "ssim"}
    q, s = cells.encode(torch.tensor([1., -1.]))
    diagnostic = cells.check_code_consistency(torch.tensor([1., -1.]), q, s)
    record["raw_metrics"]["quantization_consistency"] = metrics.metric_record(
        diagnostic, status="COMPUTED", definition="Exact code comparison at observed s")
    encoded = json.dumps(record, allow_nan=False)
    assert json.loads(encoded) == record
    assert '"mi": {"value": null' in encoded


def test_raw_metric_missing_reason_definition_and_nan_rejected():
    for kwargs in ({"value": 0}, {"reason": ""}, {"value": 1, "status": "COMPUTED"},
                   {"value": float("nan"), "status": "COMPUTED", "definition": "test"},
                   {"value": {"nested": float("inf")}, "status": "COMPUTED", "definition": "test"}):
        with pytest.raises(ValueError):
            metrics.metric_record(**kwargs)
