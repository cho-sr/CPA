"""Offline two-set evaluation for the frozen ICA-1000/FIA and ICA-4000/FIA.

The public provenance gate runs before the private truth bundle is opened.
Only its ``x`` field is consumed.  Matching and all image metrics use the
unchanged pixel-MSE Hungarian correspondence for each supplied candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from .deterministic_hybrid_setup import configure_runtime
from .full_update_evaluation import evaluate_candidate
from .report_full_update_iterations import _make_lpips_metric
from . import signed_ica4000_fia_runner as runner


TASK_ID = runner.TASK_ID
PLAN_VERSION = runner.PLAN_VERSION
ROOT = runner.REPO_ROOT
PRIVATE_TRUTH = ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt"
REVIEWED_COMPARISON = runner.OLD_PRIVATE_REVIEW
PRIVATE_ROOT = ROOT / "quantization_aware_cpa/results_private/signed_ica4000_fia_B16_T10_20260917"
CONTROL_NAME = "ica_gd1000_direct_fia25000"
NEW_NAME = "ica_gd4000_direct_fia25000"
CANDIDATE_NAMES = (CONTROL_NAME, NEW_NAME)
METRIC_NAMES = ("mean_mse", "mean_psnr_db", "mean_ssim", "mean_lpips")
CONTROL_TOLERANCE = 2e-8


class EvaluationError(RuntimeError):
    """A public provenance, metric, or private-data contract failed."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _must_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise EvaluationError(f"required regular file is missing: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    _must_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"invalid UTF-8 JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvaluationError(f"expected a JSON object: {path}")
    return value


def _norm_map(values: Mapping[str, str]) -> dict[str, str]:
    return {str(key).replace("\\", "/").casefold(): str(value) for key, value in values.items()}


def _code_hashes() -> dict[str, str]:
    paths = [Path(runner.__file__), Path(__file__),
             ROOT / "quantization_aware_cpa/qacpa/signed_combination_ica.py",
             ROOT / "quantization_aware_cpa/qacpa/upstream_naive.py"]
    return {path.resolve().as_posix(): file_sha256(path) for path in paths}


def _load_reviewed() -> dict[str, Any]:
    if file_sha256(REVIEWED_COMPARISON) != runner.REVIEWED_COMPARISON_SHA:
        raise EvaluationError("reviewed 20260917 comparison hash changed")
    value = _read_json(REVIEWED_COMPARISON)
    row = value.get("mean_metrics", {}).get("combination_separator_direct")
    if not isinstance(row, dict) or any(name not in row for name in METRIC_NAMES):
        raise EvaluationError("reviewed comparison lacks the canonical direct control row")
    return value


def _validate_old_control() -> dict[str, Any]:
    try:
        authenticated = runner._validate_old_control()
    except Exception as error:
        raise EvaluationError(str(error)) from error
    pixels = runner._validate_candidate(runner.OLD_DIRECT_CANDIDATE,
                                        runner.OLD_ARTIFACT_SHA["direct_fia/candidate.pt"])
    if tuple(pixels.shape) != (runner.SOURCE_COUNT, 3, 224, 224):
        raise EvaluationError("archived 1000-step candidate shape is invalid")
    reviewed = _load_reviewed()
    return {"artifact_hashes": authenticated["artifact_hashes"],
            "candidate_path": runner.OLD_DIRECT_CANDIDATE.resolve().as_posix(),
            "candidate_sha256": file_sha256(runner.OLD_DIRECT_CANDIDATE),
            "reviewed_comparison": REVIEWED_COMPARISON.resolve().as_posix(),
            "reviewed_comparison_sha256": runner.REVIEWED_COMPARISON_SHA,
            "reviewed_control_row": reviewed["mean_metrics"]["combination_separator_direct"]}


def _validate_new_public(public_output: Path, registration_path: Path) -> dict[str, Any]:
    registration = _read_json(registration_path)
    if (registration.get("task_id") != TASK_ID
            or registration.get("status") not in {"PUBLIC_COMPLETED", "COMPLETED"}
            or registration.get("stage_order") != ["SEPARATE", "INVERT"]
            or registration.get("public_output_root") != public_output.resolve().as_posix()
            or registration.get("private_data_policy", {}).get("load_after_public_gate") is not True):
        raise EvaluationError("outer experiment registration identity or policy is invalid")
    if registration.get("separator_config") != runner.new_config():
        raise EvaluationError("outer registration does not bind refinement_iterations=4000")
    if registration.get("fia_config") != runner.FIA_CONFIG:
        raise EvaluationError("outer registration does not bind FIA25000")
    if _norm_map(registration.get("public_input_hashes", {})) != _norm_map(runner.public_input_hashes()):
        raise EvaluationError("registered public input hashes changed")
    expected_codes = registration.get("code_hashes")
    if not isinstance(expected_codes, Mapping):
        raise EvaluationError("outer code hash registration is missing")
    if _norm_map(expected_codes) != _norm_map(_code_hashes()):
        raise EvaluationError("registered runner/evaluator code hashes changed")
    if public_output.resolve() != Path(registration["public_output_root"]).resolve():
        raise EvaluationError("public output binding changed")
    source_registration, source, target = runner._load_source(public_output)
    runner._validate_public_binding(source_registration)
    direct_root = public_output / "direct_fia"
    direct_registration = _read_json(direct_root / "registration.json")
    direct_result = _read_json(direct_root / "result.json")
    candidate_path = direct_root / "candidate.pt"
    candidate = runner._validate_candidate(candidate_path, direct_registration.get("candidate_sha256"))
    if (direct_registration.get("task_id") != TASK_ID
            or direct_registration.get("status") != "COMPLETED"
            or direct_registration.get("stage") != "INVERT"
            or direct_registration.get("fia_config") != runner.FIA_CONFIG
            or direct_registration.get("source_artifact_sha256") != file_sha256(public_output / "sources.pt")
            or direct_registration.get("target_z_sha256") != runner.tensor_sha256(target)
            or direct_result.get("status") != "COMPLETED"
            or direct_result.get("source_artifact_sha256") != file_sha256(public_output / "sources.pt")
            or direct_result.get("target_z_sha256") != runner.tensor_sha256(target)
            or direct_result.get("target_from_4000_source") is not True
            or direct_result.get("fia_config") != runner.FIA_CONFIG
            or direct_result.get("candidate_sha256") != file_sha256(candidate_path)):
        raise EvaluationError("new 4000 target -> FIA -> candidate binding is invalid")
    if direct_result.get("direct_fia", {}).get("settings", {}).get("n_iter") != 25000:
        raise EvaluationError("new Direct FIA did not run the registered 25000 steps")
    before_protected = registration.get("integrity_before", {}).get("protected_source_hashes", {})
    before_shared = registration.get("integrity_before", {}).get("shared_source_hashes", {})
    if _norm_map(before_protected) != _norm_map(runner.protected_integrity()):
        raise EvaluationError("protected source hashes changed")
    if _norm_map(before_shared) != _norm_map(runner.shared_integrity()):
        raise EvaluationError("shared source hashes changed")
    return {
        "registration": registration, "source_registration": source_registration,
        "source_sha256": file_sha256(public_output / "sources.pt"),
        "target_z_sha256": runner.tensor_sha256(target),
        "candidate_path": candidate_path.resolve().as_posix(),
        "candidate_sha256": file_sha256(candidate_path),
        "candidate_shape": list(candidate.shape),
        "result": direct_result,
        "source_stats": {"S_hat": runner.tensor_stats(source), "target_z": runner.tensor_stats(target)},
    }


def validate_public_gate(*, registration_path: Path, public_output: Path) -> dict[str, Any]:
    """Authenticate all public and archived artifacts without loading private x."""
    if not registration_path.resolve().is_file():
        raise EvaluationError("outer registration is missing")
    controls = _validate_old_control()
    new_public = _validate_new_public(public_output.resolve(), registration_path.resolve())
    reviewed = _load_reviewed()
    return {
        "task_id": TASK_ID, "status": "PUBLIC_GATE_PASSED", "private_loaded": False,
        "public_output": public_output.resolve().as_posix(), "registration": new_public["registration"],
        "archived_control": controls, "new_public": new_public,
        "reviewed_control_row": reviewed["mean_metrics"]["combination_separator_direct"],
        "integrity": {"protected_source_hashes": runner.protected_integrity(),
                       "shared_source_hashes": runner.shared_integrity(),
                       "public_input_hashes": runner.public_input_hashes(), "unchanged": True},
    }


def _load_private_x(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    if path.resolve() != PRIVATE_TRUTH.resolve():
        raise EvaluationError("private truth path is outside the registered baseline bundle")
    _must_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "x" not in payload:
        raise EvaluationError("private truth bundle lacks x")
    x = payload["x"]
    if (not isinstance(x, torch.Tensor) or x.dtype != torch.float32
            or tuple(x.shape) != (runner.SOURCE_COUNT, 3, 224, 224)
            or not bool(torch.isfinite(x).all())):
        raise EvaluationError("private x is not finite FP32 [16,3,224,224]")
    return x.detach().contiguous(), {
        "path": path.resolve().as_posix(), "sha256": file_sha256(path),
        "keys_materialized": sorted(payload), "consumed_field": "x only",
        "labels_consumed": False, "true_updates_consumed": False,
    }


def _lpips(candidates: Mapping[str, torch.Tensor], truth_x: torch.Tensor,
           reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metric, provenance = _make_lpips_metric()
    mean = truth_x.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = truth_x.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    originals = (truth_x * std + mean).clamp(0, 1)
    output: dict[str, Any] = {}
    for name, report in reports.items():
        values = []
        for pair in report["per_image"]:
            i, j = pair["candidate_index"], pair["truth_index"]
            metric.reset()
            score = metric((candidates[name][i:i + 1] - .5) * 2,
                           (originals[j:j + 1] - .5) * 2)
            if score.numel() != 1 or not bool(torch.isfinite(score).all()):
                raise EvaluationError(f"nonfinite LPIPS score for {name}/{i}")
            pair["lpips"] = float(score.detach())
            values.append(pair["lpips"])
        report["mean_lpips"] = sum(values) / len(values)
        report["lpips_provenance"] = provenance
        output[name] = {"mean_lpips": report["mean_lpips"], "per_image": values}
    return {"metrics": output, "provenance": provenance}


def _means(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
    means = {}
    for name, report in reports.items():
        rows = report["per_image"]
        psnr = [row["psnr_db"] for row in rows if row["psnr_db"] is not None]
        means[name] = {
            "mean_mse": sum(row["mse"] for row in rows) / len(rows),
            "mean_psnr_db": sum(psnr) / len(rows) if len(psnr) == len(rows) else None,
            "mean_ssim": sum(row["ssim"] for row in rows) / len(rows),
            "mean_lpips": sum(row["lpips"] for row in rows) / len(rows),
        }
    return means


def _control_reproduction(means: Mapping[str, Mapping[str, float | None]],
                          reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    reviewed = _load_reviewed()
    expected = reviewed["mean_metrics"]["combination_separator_direct"]
    actual = means[CONTROL_NAME]
    differences = {metric: (None if expected.get(metric) is None or actual.get(metric) is None
                            else actual[metric] - expected[metric]) for metric in METRIC_NAMES}
    within = {metric: (differences[metric] is not None
                       and abs(float(differences[metric])) <= CONTROL_TOLERANCE)
              for metric in METRIC_NAMES}
    prior_rows = reviewed.get("image_metrics", {}).get("combination_separator_direct", {}).get("per_image", [])
    current_rows = reports[CONTROL_NAME]["per_image"]
    per_image = []
    for index, current in enumerate(current_rows):
        prior = prior_rows[index] if index < len(prior_rows) else None
        if prior is None:
            per_image.append({"index": index, "prior": None, "current": current, "difference": None})
            continue
        per_image.append({
            "index": index,
            "matching_same": (current.get("candidate_index") == prior.get("candidate_index")
                               and current.get("truth_index") == prior.get("truth_index")),
            "prior": {metric: prior.get(metric) for metric in ("mse", "psnr_db", "ssim", "lpips")},
            "current": {metric: current.get(metric) for metric in ("mse", "psnr_db", "ssim", "lpips")},
            "difference": {metric: (None if prior.get(metric) is None or current.get(metric) is None
                                    else current[metric] - prior[metric])
                           for metric in ("mse", "psnr_db", "ssim", "lpips")},
        })
    return {"status": "PASSED" if all(within.values()) else "NEEDS_REVIEW",
            "tolerance": {metric: CONTROL_TOLERANCE for metric in METRIC_NAMES},
            "reviewed_row": expected, "current_row": actual, "difference": differences,
            "within_tolerance": within, "per_image_residuals": per_image,
            "scientific_conclusion_allowed": all(within.values())}


def _contrasts(means: Mapping[str, Mapping[str, float | None]]) -> dict[str, Any]:
    return {
        "differences_are_new_minus_old": {
            metric: (None if means[NEW_NAME][metric] is None or means[CONTROL_NAME][metric] is None
                     else means[NEW_NAME][metric] - means[CONTROL_NAME][metric])
            for metric in METRIC_NAMES
        },
        "favorable_directions": {"mean_mse": "lower", "mean_psnr_db": "higher",
                                 "mean_ssim": "higher", "mean_lpips": "lower"},
        "interpretation": "paired descriptive fixed-batch contrast; no generalization claim",
    }


def _image_improvement(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    old_rows = {row["truth_index"]: row for row in reports[CONTROL_NAME]["per_image"]}
    new_rows = {row["truth_index"]: row for row in reports[NEW_NAME]["per_image"]}
    out = {}
    for metric, lower_is_better in (("mse", True), ("psnr_db", False),
                                    ("ssim", False), ("lpips", True)):
        differences = []
        improved = worsened = tied = 0
        for index in sorted(old_rows):
            delta = new_rows[index][metric] - old_rows[index][metric]
            differences.append({"truth_index": index, "new_minus_old": delta})
            favorable = delta < 0 if lower_is_better else delta > 0
            unfavorable = delta > 0 if lower_is_better else delta < 0
            improved += int(favorable)
            worsened += int(unfavorable)
            tied += int(not favorable and not unfavorable)
        out[metric] = {"improved": improved, "worsened": worsened, "tied": tied,
                       "per_image": differences}
    return out


def _write_figure(output: Path, candidates: Mapping[str, torch.Tensor],
                  reports: Mapping[str, Mapping[str, Any]], truth_x: torch.Tensor) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mean = truth_x.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = truth_x.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    originals = (truth_x * std + mean).clamp(0, 1)
    aligned = {}
    for name, report in reports.items():
        ordered = torch.empty_like(candidates[name])
        for row in report["per_image"]:
            ordered[row["truth_index"]] = candidates[name][row["candidate_index"]]
        aligned[name] = ordered
    names = ["truth", CONTROL_NAME, NEW_NAME]
    columns = [originals, aligned[CONTROL_NAME], aligned[NEW_NAME]]
    figure, axes = plt.subplots(runner.SOURCE_COUNT, len(names), figsize=(9, 2.4 * runner.SOURCE_COUNT))
    axes = axes.reshape(runner.SOURCE_COUNT, len(names))
    try:
        for row in range(runner.SOURCE_COUNT):
            for col, values in enumerate(columns):
                axes[row, col].imshow(values[row].permute(1, 2, 0).clamp(0, 1))
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(names[col])
        figure.tight_layout()
        figure.savefig(output / "all16_comparison.png", dpi=110)
    finally:
        plt.close(figure)
    return "all16_comparison.png"


def evaluate(*, registration_path: Path, public_output: Path,
             private_truth: Path = PRIVATE_TRUTH, output: Path = PRIVATE_ROOT) -> dict[str, Any]:
    started = time.perf_counter()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing preexisting private output: {output}")
    gate = validate_public_gate(registration_path=registration_path.resolve(),
                                public_output=public_output.resolve())
    # This is the single private phase boundary.  Canonical runtime setup is
    # recorded before the private x-only load and before metric computation.
    configure_runtime(42, threads=1)
    runtime = runner._runtime("cpu", "PRIVATE_METRICS_CANONICAL_RUNTIME")
    truth_x, private_meta = _load_private_x(private_truth.resolve())
    output.mkdir(parents=True, exist_ok=False)
    paths = {CONTROL_NAME: runner.OLD_DIRECT_CANDIDATE,
             NEW_NAME: public_output.resolve() / "direct_fia/candidate.pt"}
    candidates: dict[str, torch.Tensor] = {}
    reports: dict[str, dict[str, Any]] = {}
    candidate_records = {}
    for name in CANDIDATE_NAMES:
        candidate = runner._validate_candidate(paths[name])
        candidates[name] = candidate
        candidate_records[name] = {"path": paths[name].resolve().as_posix(),
                                   "sha256": file_sha256(paths[name]),
                                   "shape": list(candidate.shape)}
        reports[name] = evaluate_candidate(paths[name], private_truth.resolve(), include_labels=False)
    lpips = _lpips(candidates, truth_x, reports)
    means = _means(reports)
    reproduction = _control_reproduction(means, reports)
    figure = _write_figure(output, candidates, reports, truth_x)
    integrity = {
        "protected_source_hashes": runner.protected_integrity(),
        "shared_source_hashes": runner.shared_integrity(),
        "public_input_hashes": runner.public_input_hashes(), "unchanged": True,
    }
    registered = gate["registration"]
    if (_norm_map(registered.get("public_input_hashes", {})) != _norm_map(integrity["public_input_hashes"])
            or _norm_map(gate["integrity"]["protected_source_hashes"]) != _norm_map(integrity["protected_source_hashes"])
            or _norm_map(gate["integrity"]["shared_source_hashes"]) != _norm_map(integrity["shared_source_hashes"])
            or file_sha256(REVIEWED_COMPARISON) != runner.REVIEWED_COMPARISON_SHA):
        raise EvaluationError("public or reviewed source integrity changed during private evaluation")
    status = "COMPLETED" if reproduction["status"] == "PASSED" else "NEEDS_REVIEW"
    comparison = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": status,
        "evaluation_status": status, "variant": "ICA_GD4000_vs_GD1000_after_same_FIA25000",
        "public_output": public_output.resolve().as_posix(),
        "private_truth_path": private_truth.resolve().as_posix(), "private_output": output.as_posix(),
        "labels_consumed": False, "true_updates_consumed": False,
        "private_truth_metadata": private_meta, "canonical_runtime": runtime,
        "matching": "Hungarian_pixel_MSE_once_per_candidate_reused_for_SSIM_LPIPS_and_aligned_figure",
        "all_images_included": True, "candidate_order": list(CANDIDATE_NAMES),
        "candidates": candidate_records, "image_metrics": reports, "mean_metrics": means,
        "lpips": lpips, "control_metric_reproduction": reproduction,
        "contrasts": _contrasts(means), "per_image_improvement": _image_improvement(reports),
        "public_gate": gate, "figure_artifacts": [figure], "source_integrity": integrity,
        "exact_cell_feasibility": False,
        "scientific_quality": "EVALUATED" if status == "COMPLETED" else "NEEDS_REVIEW",
        "wall_seconds": time.perf_counter() - started,
        "limitations": ["one fixed B16/T10 batch and seed", "comparison changes only the ICA refinement budget",
                        "FIA is fixed at 25000 steps", "no actual-image cell feasibility claim"],
    }
    runner.write_json(output / "comparison.json", comparison)
    runner.write_json(output / "results.json", comparison)
    lines = [f"# {TASK_ID}", "", f"Status: **{status}**.", "",
             "The public provenance gate passed before the private truth bundle was opened. "
             "The evaluator consumed x only; labels and true updates were not consumed.", "",
             "| Set | MSE | PSNR dB | SSIM | LPIPS |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in CANDIDATE_NAMES:
        row = means[name]
        lines.append(f"| {name} | {row['mean_mse']:.12g} | {row['mean_psnr_db']:.12g} | "
                     f"{row['mean_ssim']:.12g} | {row['mean_lpips']:.12g} |")
    lines += ["", f"Control reproduction: **{reproduction['status']}** "
              f"(absolute tolerance {CONTROL_TOLERANCE:g}).",
              "Comparison values are new 4000-step ICA minus archived 1000-step ICA after the same FIA25000.",
              "Exact actual-image cell feasibility is false by task scope and was not executed.",
              "Aligned figure uses each set's one pixel-MSE Hungarian correspondence."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", dest="registration_path", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--private-truth", type=Path, default=PRIVATE_TRUTH)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = evaluate(**vars(args))
    except Exception as error:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        runner.write_json(output / "failure.json", {"task_id": TASK_ID, "status": "FAILED",
                         "error_type": type(error).__name__, "error": str(error)})
        print(f"{TASK_ID}: {type(error).__name__}: {error}", flush=True)
        return 1
    print(json.dumps({"task_id": TASK_ID, "status": result["status"],
                      "scientific_quality": result["scientific_quality"],
                      "wall_seconds": result["wall_seconds"]}, allow_nan=False), flush=True)
    return 0 if result["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
