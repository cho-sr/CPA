"""Offline, forward-only diagnostics for the completed signed CPA/FIA arms.

The module deliberately keeps provenance validation, private truth loading, feature
extraction, and descriptive statistics separate.  It never selects an image, flips
a sign, or feeds a diagnostic result back into CPA/FIA.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch

try:
    from .signed_baseline_observation import CHECKPOINT_SHA
    from . import signed_cell_rank_evaluation as _baseline_evaluator
except Exception:  # pragma: no cover - direct utility import
    CHECKPOINT_SHA = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
    _baseline_evaluator = None

PLAN_VERSION = 2
TOLERANCE = 1e-10
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_COUNT = 16
FEATURE_DIM = 25088
EXPECTED_PLAN_HASH = "d24c271ae8461bdad3061c957b53f247cf852c7ba1bbf1d63933205511f0d14d"


class DiagnosticError(ValueError):
    """A provenance, numerical, or budget guard failed."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _finite_tensor(value: Any, shape: tuple[int, ...], label: str,
                   *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.device.type != "cpu":
        raise DiagnosticError(f"{label} must be a CPU strided tensor")
    if value.dtype != dtype or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
        raise DiagnosticError(f"{label} has invalid shape or non-finite values")
    return value.contiguous()


def normalize_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization exactly once, preserving FP32."""
    if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4 or pixels.shape[1:] != (3, 224, 224):
        raise DiagnosticError("pixels must have shape [N,3,224,224]")
    if pixels.dtype != torch.float32 or not bool(torch.isfinite(pixels).all()):
        raise DiagnosticError("pixels must be finite FP32")
    if bool((pixels < 0).any()) or bool((pixels > 1).any()):
        raise DiagnosticError("pixels must be in [0,1]")
    mean = pixels.new_tensor(MEAN).reshape(1, 3, 1, 1)
    std = pixels.new_tensor(STD).reshape(1, 3, 1, 1)
    return (pixels - mean) / std


def cosine_info(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    """Cosine with explicit undefined reasons and no silent epsilon."""
    a = left.reshape(-1).to(torch.float64)
    b = right.reshape(-1).to(torch.float64)
    if a.numel() != b.numel() or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        return {"value": None, "reason": "NONFINITE_OR_SHAPE_MISMATCH", "defined": False}
    na, nb = float(torch.linalg.vector_norm(a)), float(torch.linalg.vector_norm(b))
    if na == 0.0 or nb == 0.0:
        return {"value": None, "reason": "ZERO_NORM", "defined": False, "left_norm": na, "right_norm": nb}
    value = float(torch.dot(a, b) / (na * nb))
    if not math.isfinite(value):
        return {"value": None, "reason": "NONFINITE_COSINE", "defined": False, "left_norm": na, "right_norm": nb}
    return {"value": max(-1.0, min(1.0, value)), "reason": None, "defined": True,
            "left_norm": na, "right_norm": nb}


def geometric_interval(a: float | None, b: float | None, c: float | None, *, tol: float = TOLERANCE) -> dict[str, Any]:
    result: dict[str, Any] = {"fit_cosine": a, "target_cosine": b, "image_cosine": c}
    if a is None or b is None or c is None:
        result.update(status="UNDEFINED", lower=None, upper=None, width=None,
                      violation=None, oracle_cosine_change=None, reason="UNDEFINED_COSINE")
        return result
    if any(abs(x) > 1.0 + tol for x in (a, b, c)):
        result.update(status="INVALID", lower=None, upper=None, width=None,
                      violation=None, oracle_cosine_change=None, reason="COSINE_OUT_OF_RANGE")
        return result
    a, b, c = (max(-1.0, min(1.0, float(x))) for x in (a, b, c))
    radius = math.sqrt(max(0.0, (1.0 - a * a) * (1.0 - b * b)))
    lower, upper = a * b - radius, a * b + radius
    violation = max(lower - c, c - upper, 0.0)
    result.update(status="COMPLETED" if violation <= tol else "VIOLATION",
                  lower=lower, upper=upper, width=upper - lower, violation=violation,
                  oracle_cosine_change=c - b, radius=radius,
                  reason=None if violation <= tol else "GEOMETRIC_INTERVAL_VIOLATION")
    return result


def sign_metrics(source: torch.Tensor) -> dict[str, Any]:
    s = source.reshape(-1).to(torch.float64)
    if not bool(torch.isfinite(s).all()):
        raise DiagnosticError("S_hat contains non-finite values")
    pos = float((s[s > 0] ** 2).sum())
    neg = float((s[s < 0] ** 2).sum())
    zero = int((s == 0).sum())
    total = pos + neg
    common: dict[str, Any] = {"positive_count": int((s > 0).sum()), "negative_count": int((s < 0).sum()),
                              "zero_count": zero, "positive_energy": pos, "negative_energy": neg,
                              "total_energy": total}
    if total == 0.0:
        common.update(negative_energy_fraction=None, minority_sign_energy_fraction=None,
                      abs_vs_best_global_sign_cosine=None, abs_vs_best_global_sign_relative_l2=None,
                      reason="ZERO_SOURCE_NORM")
    else:
        minority = min(pos, neg) / total
        common.update(negative_energy_fraction=neg / total,
                      minority_sign_energy_fraction=minority,
                      abs_vs_best_global_sign_cosine=abs(pos - neg) / total,
                      abs_vs_best_global_sign_relative_l2=2.0 * math.sqrt(minority), reason=None)
    return common


def validate_bijection(rows: Iterable[dict[str, Any]], source_key: str, truth_key: str, label: str) -> dict[int, int]:
    rows = list(rows)
    if len(rows) != IMAGE_COUNT:
        raise DiagnosticError(f"{label} must contain {IMAGE_COUNT} rows")
    mapping: dict[int, int] = {}
    for row in rows:
        if type(row.get(source_key)) is not int or type(row.get(truth_key)) is not int:
            raise DiagnosticError(f"{label} has non-integer mapping fields")
        source, truth = row[source_key], row[truth_key]
        if source in mapping or source not in range(IMAGE_COUNT) or truth not in range(IMAGE_COUNT):
            raise DiagnosticError(f"{label} is not a source bijection")
        mapping[source] = truth
    if set(mapping.values()) != set(range(IMAGE_COUNT)):
        raise DiagnosticError(f"{label} truth indices are not a bijection")
    return mapping


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DiagnosticError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"expected JSON object: {path}")
    return value


def validate_public_provenance(naive: Path, cell: Path, comparison: Path, public_observation: Path,
                               private_observation: Path, checkpoint: Path, output: Path,
                               method_plan: Path, expected_plan_sha256: str) -> dict[str, Any]:
    """Perform every binding check before deserializing evaluation_truth.pt."""
    if output.exists():
        raise FileExistsError(output)
    if sha256_file(method_plan) != expected_plan_sha256:
        raise DiagnosticError("method plan SHA256 mismatch")
    if sha256_file(checkpoint) != CHECKPOINT_SHA:
        raise DiagnosticError("checkpoint SHA256 mismatch")
    if _baseline_evaluator is None:
        raise DiagnosticError("signed-cell provenance validator is unavailable")
    # This is the completed evaluator's strict validator.  It hashes the private
    # bundles and raw update without deserializing either tensor bundle, and is
    # intentionally run before the diagnostic creates its output directory.
    try:
        baseline = _baseline_evaluator.validate_provenance(
            naive=naive, cell_rank=cell, private_observation=private_observation,
            public_observation=public_observation, output=output)
    except TypeError:
        # Keep compatibility with the evaluator's historical parameter name.
        baseline = _baseline_evaluator.validate_provenance(
            naive=naive, cell_rank=cell, private_observation=private_observation,
            public_observation=public_observation, output=output)
    except Exception as exc:
        raise DiagnosticError(f"baseline provenance validation failed: {exc}") from exc
    public_manifest = public_observation / "manifest.json"
    if not public_manifest.is_file():
        raise DiagnosticError("public manifest is missing")
    manifest_sha = sha256_file(public_manifest)
    sender = _load_json(private_observation / "sender_validation.json")
    truth_path = private_observation / "evaluation_truth.pt"
    truth_sha = sha256_file(truth_path)
    if sender.get("status") != "COMPLETED" or sender.get("evaluation_truth_sha256") != truth_sha:
        raise DiagnosticError("private sender/truth provenance mismatch")
    validated_results = baseline.get("results", {})
    for arm, folder, expected in (("naive", naive, "INT4_naive"), ("cell_rank", cell, "INT4_cell_rank")):
        result = _load_json(folder / "result.json")
        if result.get("status") != "COMPLETED" or result.get("cpa_iterations") != 25000 or result.get("fia_iterations") != 25000:
            raise DiagnosticError(f"{arm} is not a completed CPA25k/FIA25k arm")
        if arm == "cell_rank" and result.get("arm") != expected:
            raise DiagnosticError("cell arm identity mismatch")
        if result.get("input", {}).get("public_manifest_sha256") != manifest_sha:
            raise DiagnosticError(f"{arm} public manifest binding mismatch")
        if result.get("checkpoint_sha256") != CHECKPOINT_SHA and result.get("input", {}).get("checkpoint_sha256") != CHECKPOINT_SHA:
            raise DiagnosticError(f"{arm} checkpoint binding mismatch")
        for name, key in (("candidate.pt", "candidate_sha256"), ("sources.pt", "sources_sha256")):
            p = folder / name
            if not p.is_file() or sha256_file(p) != result.get(key):
                raise DiagnosticError(f"{arm} {name} hash mismatch")
    comp = _load_json(comparison)
    if comp.get("status") != "COMPLETED" or comp.get("evaluation_status") != "COMPLETED" or comp.get("labels_consumed") is not False:
        raise DiagnosticError("comparison is not a completed label-free result")
    if comp.get("format") != "signed_cell_rank_label_free_offline_comparison_v1":
        raise DiagnosticError("comparison format is not the completed label-free evaluator")
    comp_prov = comp.get("provenance", {})
    if (comp_prov.get("public_manifest_sha256") != manifest_sha
            or comp_prov.get("checkpoint_sha256") != CHECKPOINT_SHA
            or comp_prov.get("evaluation_truth_sha256") != truth_sha):
        raise DiagnosticError("comparison provenance does not bind supplied observation/checkpoint/truth")
    if comp.get("truth_sha256") != truth_sha:
        raise DiagnosticError("comparison truth SHA does not bind supplied truth")
    # The comparison has mutable output-path/status fields, so bind only its
    # finalized arm identities and artifact hashes to the strict validator.
    for arm in ("naive", "cell_rank"):
        final_result = comp.get("arms", {}).get(arm, {}).get("final_result", {})
        expected_result = validated_results.get(arm, {})
        for key in ("candidate_sha256", "sources_sha256", "status", "cpa_iterations", "fia_iterations"):
            if final_result.get(key) != expected_result.get(key):
                raise DiagnosticError(f"comparison {arm} final_result.{key} binding mismatch")
    image_maps = {}
    embedding_maps = {}
    for arm in ("naive", "cell_rank"):
        data = comp.get("arms", {}).get(arm, {})
        image_maps[arm] = validate_bijection(data.get("image_metrics", {}).get("per_image", []), "candidate_index", "truth_index", f"{arm} image map")
        embedding_maps[arm] = validate_bijection(data.get("embedding_oracle", {}).get("per_source", []), "source_index", "truth_index", f"{arm} embedding map")
    return {"manifest_sha256": manifest_sha, "checkpoint_sha256": CHECKPOINT_SHA, "truth_sha256": truth_sha,
            "comparison_sha256": sha256_file(comparison), "method_plan_sha256": expected_plan_sha256,
            "image_maps": image_maps, "embedding_maps": embedding_maps, "sender": sender, "comparison": comp,
            "baseline_validation": {k: v for k, v in baseline.items() if k != "results"},
            "validated_results": validated_results}


def load_private_z_after_validation(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise DiagnosticError("evaluation_truth.pt must contain a dictionary")
    z = _finite_tensor(payload.get("z"), (IMAGE_COUNT, FEATURE_DIM), "evaluation truth z")
    return z.float(), {"private_loaded": True, "labels_consumed": False,
                       "bundled_fields": sorted(payload.keys()), "x_or_labels_may_materialize": True}


def _load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from official_cpa.src.vgg16 import vgg16
    model = vgg16(weights=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise DiagnosticError("checkpoint state dict is invalid")
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


def forward_features(model: torch.nn.Module, pixels: torch.Tensor, device: torch.device, microbatch: int,
                     max_seconds: float, max_gib: float, *, deadline: float | None = None,
                     global_start: float | None = None, start_alloc: int = 0) -> tuple[torch.Tensor, dict[str, Any]]:
    if microbatch != 1:
        raise DiagnosticError("v2 requires evaluation microbatch 1")
    start = time.perf_counter()
    deadline = deadline if deadline is not None else start + max_seconds
    global_start = global_start if global_start is not None else start
    peak = 0
    if device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(device))
    outputs = []
    with torch.inference_mode():
        for index in range(pixels.shape[0]):
            if time.perf_counter() > deadline:
                raise DiagnosticError("diagnostic wall-time budget exceeded")
            x = normalize_pixels(pixels[index:index + 1]).to(device, non_blocking=False)
            out = model.features(x)
            out = model.avgpool(out)
            out = torch.flatten(out, 1).to("cpu", dtype=torch.float32)
            if out.shape != (1, FEATURE_DIM) or not bool(torch.isfinite(out).all()):
                raise DiagnosticError("VGG feature output has invalid shape or values")
            if bool((out < -TOLERANCE).any()):
                raise DiagnosticError("VGG ReLU feature output contains a negative value")
            outputs.append(out)
            if device.type == "cuda":
                peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
                if peak > max_gib * (1024 ** 3):
                    raise DiagnosticError("CUDA allocated-memory budget exceeded")
    if device.type == "cuda":
        peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
    return torch.cat(outputs, dim=0), {"wall_seconds": time.perf_counter() - start,
                                      "global_wall_seconds": time.perf_counter() - global_start,
                                      "start_cuda_allocated_bytes": start_alloc,
                                      "peak_cuda_allocated_bytes": peak,
                                      "delta_cuda_allocated_bytes": max(0, peak - start_alloc),
                                      "max_evaluation_seconds": max_seconds,
                                      "max_cuda_allocated_gib": max_gib,
                                      "microbatch": microbatch}


def _rank(values: list[float | None]) -> list[float | None]:
    valid = [(i, v) for i, v in enumerate(values) if v is not None and math.isfinite(v)]
    out: list[float | None] = [None] * len(values)
    ordered = sorted(valid, key=lambda item: item[1])
    j = 0
    while j < len(ordered):
        k = j + 1
        while k < len(ordered) and ordered[k][1] == ordered[j][1]: k += 1
        rank = (j + 1 + k) / 2.0
        for i, _ in ordered[j:k]: out[i] = rank
        j = k
    return out


def spearman(x: list[float | None], y: list[float | None]) -> dict[str, Any]:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 2: return {"rho": None, "n": len(pairs), "reason": "INSUFFICIENT_DEFINED_VALUES"}
    rx, ry = _rank([a for a, _ in pairs]), _rank([b for _, b in pairs])
    ax, ay = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - ax) * (b - ay) for a, b in zip(rx, ry)); dx = math.sqrt(sum((a - ax) ** 2 for a in rx)); dy = math.sqrt(sum((b - ay) ** 2 for b in ry))
    if dx == 0 or dy == 0: return {"rho": None, "n": len(pairs), "reason": "CONSTANT_RANK"}
    return {"rho": num / (dx * dy), "n": len(pairs), "reason": None}


def _metric_rows(comp: dict[str, Any], arm: str) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    data = comp["arms"][arm]
    return ({int(r["candidate_index"]): r for r in data["image_metrics"]["per_image"]},
            {int(r["source_index"]): r for r in data["embedding_oracle"]["per_source"]})


def assemble_rows(features_by_arm: dict[str, torch.Tensor], targets: dict[str, torch.Tensor],
                  signed_sources: dict[str, torch.Tensor], truth_z: torch.Tensor,
                  comp: dict[str, Any], image_maps: dict[str, dict[int, int]],
                  embedding_maps: dict[str, dict[int, int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build corrected rows and same-truth I/E pair tables from supplied vectors."""
    all_rows: list[dict[str, Any]] = []
    for arm in ("naive", "cell_rank"):
        features = features_by_arm[arm]
        target = targets[arm]
        signed = signed_sources[arm]
        _finite_tensor(features, (IMAGE_COUNT, FEATURE_DIM), f"{arm} features")
        _finite_tensor(target, (IMAGE_COUNT, FEATURE_DIM), f"{arm} target_z")
        _finite_tensor(signed, (IMAGE_COUNT, FEATURE_DIM), f"{arm} S_hat")
        image_rows, _ = _metric_rows(comp, arm)
        imap, emap = image_maps[arm], embedding_maps[arm]
        for i in range(IMAGE_COUNT):
            fit = cosine_info(features[i], target[i])
            # A ReLU encoder and nonnegative target_z cannot produce a valid
            # negative fit cosine.  Keep zero/undefined behavior explicit.
            if fit["value"] is not None and fit["value"] < -TOLERANCE:
                raise DiagnosticError(f"{arm} C_fit is negative for a ReLU/nonnegative target pair")
            ti, te = truth_z[imap[i]], truth_z[emap[i]]
            ci, ce = cosine_info(features[i], ti), cosine_info(features[i], te)
            b_i, b_e = cosine_info(target[i], ti), cosine_info(target[i], te)
            row = {"arm": arm, "candidate_index": i, "image_truth_index": imap[i],
                   "embedding_truth_index": emap[i], "C_fit": fit["value"],
                   "C_fit_reason": fit.get("reason"), "C_target_I": b_i["value"],
                   "C_image_I": ci["value"], "C_target_E": b_e["value"],
                   "C_image_E": ce["value"], "C_target_I_reason": b_i.get("reason"),
                   "C_image_I_reason": ci.get("reason"), "C_target_E_reason": b_e.get("reason"),
                   "C_image_E_reason": ce.get("reason"),
                   "geometry_I": geometric_interval(fit["value"], b_i["value"], ci["value"]),
                   "geometry_E": geometric_interval(fit["value"], b_e["value"], ce["value"]),
                   "sign": sign_metrics(signed[i]), "selection_used": False}
            row.update({k: image_rows[i].get(k) for k in ("mse", "psnr_db", "ssim", "lpips")})
            all_rows.append(row)
    by_key = {(r["arm"], r["candidate_index"]): r for r in all_rows}
    paired_i, paired_e = [], []
    for truth_index in range(IMAGE_COUNT):
        ni = next(i for i in range(IMAGE_COUNT) if image_maps["naive"][i] == truth_index)
        ci = next(i for i in range(IMAGE_COUNT) if image_maps["cell_rank"][i] == truth_index)
        nr, cr = by_key[("naive", ni)], by_key[("cell_rank", ci)]
        paired_i.append({"truth_index": truth_index, "naive_candidate_index": ni,
                         "naive_truth_index": nr["image_truth_index"],
                         "cell_rank_candidate_index": ci, "cell_rank_truth_index": cr["image_truth_index"],
                         **{f"delta_{k}": (cr.get(k) - nr.get(k)) if isinstance(cr.get(k), (int, float)) and isinstance(nr.get(k), (int, float)) else None
                            for k in ("mse", "psnr_db", "ssim", "lpips", "C_fit", "C_target_I", "C_image_I")}})
        ne = next(i for i in range(IMAGE_COUNT) if embedding_maps["naive"][i] == truth_index)
        ce = next(i for i in range(IMAGE_COUNT) if embedding_maps["cell_rank"][i] == truth_index)
        nr, cr = by_key[("naive", ne)], by_key[("cell_rank", ce)]
        paired_e.append({"truth_index": truth_index, "naive_candidate_index": ne,
                         "naive_truth_index": nr["embedding_truth_index"],
                         "cell_rank_candidate_index": ce, "cell_rank_truth_index": cr["embedding_truth_index"],
                         **{f"delta_{k}": (cr.get(k) - nr.get(k)) if isinstance(cr.get(k), (int, float)) and isinstance(nr.get(k), (int, float)) else None
                            for k in ("C_fit", "C_target_E", "C_image_E")}})
    return all_rows, paired_i, paired_e


def _summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def _summaries(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metrics = ("C_fit", "C_target_I", "C_image_I", "C_target_E", "C_image_E", "mse", "psnr_db", "ssim", "lpips")
    summary = {arm: {key: _summary([r for r in rows if r["arm"] == arm], key) for key in metrics} for arm in ("naive", "cell_rank")}
    associations: dict[str, Any] = {}
    sign_summary: dict[str, Any] = {}
    for arm in ("naive", "cell_rank"):
        rs = [r for r in rows if r["arm"] == arm]
        associations[arm] = {metric: {
            "C_fit_vs_metric": spearman([r["C_fit"] for r in rs], [r[metric] for r in rs]),
            "C_image_I_vs_metric": spearman([r["C_image_I"] for r in rs], [r[metric] for r in rs]),
        } for metric in ("mse", "psnr_db", "ssim", "lpips")}
        sign_summary[arm] = {"mean_negative_energy_fraction": statistics.mean(r["sign"]["negative_energy_fraction"] for r in rs if r["sign"]["negative_energy_fraction"] is not None),
                             "mean_minority_sign_energy_fraction": statistics.mean(r["sign"]["minority_sign_energy_fraction"] for r in rs if r["sign"]["minority_sign_energy_fraction"] is not None),
                             "mean_abs_vs_best_global_sign_cosine": statistics.mean(r["sign"]["abs_vs_best_global_sign_cosine"] for r in rs if r["sign"]["abs_vs_best_global_sign_cosine"] is not None),
                             "mean_abs_vs_best_global_sign_relative_l2": statistics.mean(r["sign"]["abs_vs_best_global_sign_relative_l2"] for r in rs if r["sign"]["abs_vs_best_global_sign_relative_l2"] is not None),
                             "undefined": sum(r["sign"]["reason"] is not None for r in rs)}
    return summary, associations, sign_summary


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    naive, cell, comparison = map(lambda x: Path(x).resolve(), (args.naive, args.cell_rank, args.comparison))
    public_observation, private_observation, checkpoint, method_plan = map(lambda x: Path(x).resolve(), (args.public_observation, args.private_observation, args.checkpoint, args.method_plan))
    provenance = validate_public_provenance(naive, cell, comparison, public_observation, private_observation, checkpoint, output, method_plan, args.expected_plan_sha256)
    output.mkdir(parents=True)
    registered = {"format": "signed_stage_diagnostic_v2_repaired", "implementation_revision": "repair_20260915_r1",
                  "supersedes": "stage_diagnostic_v2 (signed-target and cross-arm-map metrics)", "plan_version": PLAN_VERSION,
                  "method_plan_sha256": args.expected_plan_sha256, "config": vars(args),
                  "normalization": {"mean": MEAN, "std": STD, "exactly_once": True},
                  "encoder": "official_cpa/src/vgg16.py:vgg16 features->avgpool->flatten; frozen FP32",
                  "offline_only": True, "labels_consumed": False, "selection_used": False,
                  "input_hashes": {"naive_candidate": sha256_file(naive/'candidate.pt'), "cell_candidate": sha256_file(cell/'candidate.pt'),
                                   "naive_sources": sha256_file(naive/'sources.pt'), "cell_sources": sha256_file(cell/'sources.pt'),
                                   "comparison": provenance["comparison_sha256"], "checkpoint": provenance["checkpoint_sha256"],
                                   "public_manifest": provenance["manifest_sha256"], "evaluation_truth": provenance["truth_sha256"]}}
    json_write(output / "registered_plan.json", registered)
    candidates, signed_sources, targets = {}, {}, {}
    for arm, folder in (("naive", naive), ("cell_rank", cell)):
        cand = torch.load(folder / "candidate.pt", map_location="cpu", weights_only=True)
        src = torch.load(folder / "sources.pt", map_location="cpu", weights_only=True)
        if set(cand) != {"pixels"} or set(src) != {"S_hat", "target_z"}:
            raise DiagnosticError(f"{arm} artifact schema mismatch")
        candidates[arm] = _finite_tensor(cand["pixels"], (IMAGE_COUNT, 3, 224, 224), f"{arm} pixels")
        if bool((candidates[arm] < 0).any()) or bool((candidates[arm] > 1).any()): raise DiagnosticError(f"{arm} pixels outside [0,1]")
        signed_sources[arm] = _finite_tensor(src["S_hat"], (IMAGE_COUNT, FEATURE_DIM), f"{arm} S_hat")
        targets[arm] = _finite_tensor(src["target_z"], (IMAGE_COUNT, FEATURE_DIM), f"{arm} target_z")
        if not torch.equal(targets[arm], signed_sources[arm].abs()): raise DiagnosticError(f"{arm} target_z is not abs(S_hat)")
    truth_z, private_meta = load_private_z_after_validation(private_observation / "evaluation_truth.pt")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise DiagnosticError("CUDA requested but unavailable")
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = _load_model(checkpoint, device)
    global_start = time.perf_counter()
    deadline = global_start + args.max_evaluation_seconds
    start_alloc = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    features_by_arm: dict[str, torch.Tensor] = {}; runtime = {}
    for arm in ("naive", "cell_rank"):
        features_by_arm[arm], runtime[arm] = forward_features(model, candidates[arm], device, args.eval_microbatch, args.max_evaluation_seconds, args.max_cuda_allocated_gib, deadline=deadline, global_start=global_start, start_alloc=start_alloc)
    all_rows, paired_i, paired_e = assemble_rows(features_by_arm, targets, signed_sources, truth_z, provenance["comparison"], provenance["image_maps"], provenance["embedding_maps"])
    torch.save({"naive_h": features_by_arm["naive"], "cell_rank_h": features_by_arm["cell_rank"], "dtype": "torch.float32", "provenance": "evaluation-only; corrected v1"}, output / "features_h.pt")
    with (output / "all32_per_candidate.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["arm", "candidate_index", "image_truth_index", "embedding_truth_index", "C_fit", "C_target_I", "C_image_I", "C_target_E", "C_image_E", "mse", "psnr_db", "ssim", "lpips", "selection_used"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in all_rows: w.writerow({k: r.get(k) for k in fields})
    with (output / "paired16_by_truth.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(paired_i[0])); w.writeheader(); w.writerows(paired_i)
    with (output / "paired16_by_embedding_truth.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(paired_e[0])); w.writeheader(); w.writerows(paired_e)
    agreement = {arm: {"count": sum(provenance["image_maps"][arm][i] == provenance["embedding_maps"][arm][i] for i in range(IMAGE_COUNT)),
                       "fraction": sum(provenance["image_maps"][arm][i] == provenance["embedding_maps"][arm][i] for i in range(IMAGE_COUNT)) / IMAGE_COUNT,
                       "disagreement_ids": [i for i in range(IMAGE_COUNT) if provenance["image_maps"][arm][i] != provenance["embedding_maps"][arm][i]]} for arm in ("naive", "cell_rank")}
    geometry = {arm: {view: {"defined": sum(r[f"geometry_{view}"]["status"] == "COMPLETED" for r in all_rows if r["arm"] == arm),
                              "undefined": sum(r[f"geometry_{view}"]["status"] == "UNDEFINED" for r in all_rows if r["arm"] == arm),
                              "invalid": sum(r[f"geometry_{view}"]["status"] == "INVALID" for r in all_rows if r["arm"] == arm),
                              "violations": sum(r[f"geometry_{view}"]["status"] == "VIOLATION" for r in all_rows if r["arm"] == arm),
                              "mean_c_minus_b": statistics.mean([r[f"geometry_{view}"]["oracle_cosine_change"] for r in all_rows if r["arm"] == arm and r[f"geometry_{view}"]["oracle_cosine_change"] is not None]) if any(r[f"geometry_{view}"]["oracle_cosine_change"] is not None for r in all_rows if r["arm"] == arm) else None} for view in ("I", "E")} for arm in ("naive", "cell_rank")}
    summaries, associations, sign_summary = _summaries(all_rows)
    runtime["total_wall_seconds"] = time.perf_counter() - global_start
    runtime["global_deadline_seconds"] = args.max_evaluation_seconds
    runtime["peak_cuda_allocated_bytes_total"] = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    diagnostic = {"status": "COMPLETED", "format": "signed_stage_diagnostic_v2_repaired", "implementation_revision": "repair_20260915_r1", "rows": all_rows, "paired_rows_by_image_truth": paired_i, "paired_rows_by_embedding_truth": paired_e,
                  "map_agreement_within_arm": agreement, "geometry": geometry, "summaries": summaries, "descriptive_associations": associations, "sign_summary": sign_summary,
                  "runtime": runtime, "private_truth": private_meta, "provenance": provenance,
                  "historical_comparison": provenance["comparison"].get("cell_rank_minus_naive"), "selection_used": False,
                  "superseded_metrics": ["signed-target C_fit/C_target values from stage_diagnostic_v2", "cross-arm mapping agreement", "cross-map paired C_image_E deltas"]}
    json_write(output / "diagnostic.json", diagnostic)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for arm, color in (("naive", "tab:blue"), ("cell_rank", "tab:orange")):
            rs = [r for r in all_rows if r["arm"] == arm]
            ax.scatter([r["C_fit"] for r in rs], [r["mse"] for r in rs], label=arm, alpha=.8, color=color)
        ax.set(xlabel="C_fit = cos(h,target_z)", ylabel="pixel MSE"); ax.legend(); fig.tight_layout(); fig.savefig(output / "stage_diagnostic_v2_repaired.png", dpi=160); plt.close(fig)
        diagnostic["plot_status"] = "COMPLETED"
    except Exception as exc: diagnostic["plot_status"] = "FAILED"; diagnostic["plot_error"] = str(exc)
    json_write(output / "diagnostic.json", diagnostic)
    return diagnostic


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("naive", "cell-rank", "comparison", "public-observation", "private-observation", "checkpoint", "method-plan", "output"): p.add_argument("--" + name, required=True)
    p.add_argument("--expected-plan-sha256", required=True); p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--eval-microbatch", type=int, default=1); p.add_argument("--max-evaluation-seconds", type=float, default=600.0); p.add_argument("--max-cuda-allocated-gib", type=float, default=6.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_diagnostic(args)
    except Exception as exc:
        print(f"signed_stage_diagnostics: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "COMPLETED", "output": str(Path(args.output).resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
