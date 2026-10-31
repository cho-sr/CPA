"""CPU-only renderer for corrected screen3 figures with the true baseline column.

This module reads an already completed comparison.json, authenticates the
public candidate files named by its public gate, and reuses the stored
per-image Hungarian correspondence.  It never computes metrics or matching.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


TASK_ID = "CPA_SCREEN3_BASELINE_FIGURE_REPAIR_20260918"
REPO_ROOT = Path(__file__).resolve().parents[3]
PRIVATE_TRUTH = REPO_ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt"
PRIVATE_TRUTH_SHA = "724bbacbe316cc28dac086db9cfaf6507f4c1af31144fd610cc480985aab72e2"


class FigureRepairError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FigureRepairError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def build_layout(bits: int, selection: list[int]) -> dict[str, Any]:
    if len(selection) != 3 or len(set(selection)) != 3 or any(int(x) not in range(8) for x in selection):
        raise FigureRepairError("frozen public selection must contain three distinct member IDs")
    final_names = ["center_25000"] + [f"rank{rank}_member_{int(member):02d}_25000"
                                      for rank, member in enumerate(selection, 1)]
    final_labels = [f"Ground truth",
                    f"Naive dequantization baseline | {bits}-bit | center | 25000 steps"]
    final_labels += [f"Public rank {rank} | member {int(member):02d} | {bits}-bit | 25000 steps"
                     for rank, member in enumerate(selection, 1)]
    screen_names = ["center_5000"] + [f"member_{index:02d}_5000" for index in range(8)]
    screen_labels = [f"Ground truth", f"Naive dequantization baseline | {bits}-bit | center | 5000 steps"]
    screen_labels += [f"member {index:02d} | {bits}-bit | 5000 steps" for index in range(8)]
    return {"final_names": final_names, "final_labels": final_labels,
            "screen_names": screen_names, "screen_labels": screen_labels,
            "bits": bits, "final_steps": 25000, "screen_steps": 5000}


def _public_roots_from_paths(paths: dict[str, Path]) -> list[Path]:
    roots = []
    for path in paths.values():
        current = path.resolve()
        for parent in (current, *current.parents):
            if (parent / "registration.json").is_file():
                roots.append(parent)
                break
    unique = sorted(set(roots), key=lambda path: path.as_posix())
    if not unique:
        raise FigureRepairError("candidate paths do not resolve to a public registration root")
    return unique


def _candidate_step_binding(name: str, path: Path) -> None:
    if name == "center_5000":
        expected = "fia_center_25000/candidate_5000.pt"
    elif name == "center_25000":
        expected = "fia_center_25000/candidate.pt"
    elif name.startswith("member_") and name.endswith("_5000"):
        member = name[len("member_"):-len("_5000")]
        expected = f"fia_member_{int(member):02d}_5000/candidate.pt"
    elif name.startswith("rank") and name.endswith("_25000"):
        member = name.split("_member_", 1)[1][:-len("_25000")]
        expected = f"fia_member_{int(member):02d}_25000/candidate.pt"
    else:
        raise FigureRepairError(f"unrecognized candidate name: {name}")
    normalized = path.as_posix().lower()
    if not normalized.endswith(expected.lower()):
        raise FigureRepairError(f"candidate step/name binding mismatch: {name} -> {path}")


def authenticate_comparison(comparison_path: Path, bits: int) -> tuple[dict[str, Any], dict[str, Path], Path]:
    comparison_path = Path(comparison_path).resolve()
    comparison = read_json(comparison_path)
    if (comparison.get("status") != "COMPLETED" or comparison.get("evaluation_status") != "COMPLETED"
            or comparison.get("scientific_quality") != "EVALUATED"
            or comparison.get("private_x_only") is not True
            or comparison.get("labels_consumed") is not False):
        raise FigureRepairError(f"comparison is not a completed x-only evaluation: {comparison_path}")
    gate = comparison.get("public_gate")
    if (not isinstance(gate, dict) or gate.get("status") != "PUBLIC_GATE_PASSED"
            or gate.get("private_loaded") is not False or gate.get("candidate_count") != 13):
        raise FigureRepairError("stored public gate is not PUBLIC_GATE_PASSED/private_loaded=false")
    paths_raw = comparison.get("candidate_paths")
    hashes = comparison.get("candidate_hashes")
    gate_candidates = gate.get("candidates")
    if not isinstance(paths_raw, dict) or not isinstance(hashes, dict) or not isinstance(gate_candidates, dict):
        raise FigureRepairError("comparison lacks complete candidate/gate bindings")
    if set(paths_raw) != set(hashes) or set(paths_raw) != set(gate_candidates) or len(paths_raw) != 13:
        raise FigureRepairError("candidate/gate bindings are not the same 13-set")
    paths: dict[str, Path] = {}
    for name, raw_path in paths_raw.items():
        path = Path(raw_path).resolve()
        record = gate_candidates[name]
        record_path = Path(record.get("path", "")).resolve()
        record_hash = record.get("sha256", record.get("candidate_sha256"))
        if record_path != path or not path.is_file() or sha256(path) != hashes[name] or sha256(path) != record_hash:
            raise FigureRepairError(f"public candidate authentication failed: {name}")
        _candidate_step_binding(name, path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(payload, dict) or set(payload) != {"pixels"}
                or not isinstance(payload["pixels"], torch.Tensor)
                or payload["pixels"].dtype != torch.float32
                or tuple(payload["pixels"].shape) != (16, 3, 224, 224)
                or not bool(torch.isfinite(payload["pixels"]).all())):
            raise FigureRepairError(f"public candidate schema failed: {name}")
        paths[name] = path
    public_roots = _public_roots_from_paths(paths)
    registration_statuses = {}
    for public_root in public_roots:
        registration = read_json(public_root / "registration.json")
        if registration.get("status") not in {"REGISTERED", "PUBLIC_COMPLETED"}:
            raise FigureRepairError(f"public registration status is not authenticated: {public_root}")
        inherited_sha = gate.get("inherited_registration_sha256")
        if registration.get("status") == "REGISTERED" and inherited_sha and sha256(public_root / "registration.json") != inherited_sha:
            raise FigureRepairError(f"inherited registration hash mismatch: {public_root}")
        registration_statuses[public_root.as_posix()] = registration.get("status")
    metrics = comparison.get("image_metrics")
    means = comparison.get("mean_metrics")
    if not isinstance(metrics, dict) or not isinstance(means, dict) or set(metrics) != set(paths):
        raise FigureRepairError("stored image metrics do not cover exactly the authenticated candidates")
    for name in paths:
        rows = metrics[name].get("per_image")
        if (not isinstance(rows, list) or len(rows) != 16
                or {row.get("truth_index") for row in rows} != set(range(16))
                or {row.get("candidate_index") for row in rows} != set(range(16))):
            raise FigureRepairError(f"stored correspondence is not a 16-image permutation: {name}")
    if "Hungarian" not in str(comparison.get("matching", "")) or "reused" not in str(comparison.get("matching", "")):
        raise FigureRepairError("comparison does not declare reused Hungarian correspondence")
    selection = comparison.get("frozen_selection")
    layout = build_layout(bits, [int(x) for x in selection])
    required = set(layout["screen_names"] + layout["final_names"])
    if required != set(paths):
        raise FigureRepairError("stored candidates do not match screen/final layout")
    if sha256(comparison_path) != comparison.get("comparison_sha256", sha256(comparison_path)):
        raise FigureRepairError("comparison hash self-binding failed")
    comparison["_renderer_registration_statuses"] = registration_statuses
    return comparison, paths, public_roots


def load_private_x(comparison: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    private = comparison.get("private_truth", {})
    path = Path(private.get("path", "")).resolve()
    if path != PRIVATE_TRUTH.resolve() or sha256(path) != PRIVATE_TRUTH_SHA:
        raise FigureRepairError("private truth path/hash is not the registered evaluation bundle")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "x" not in payload:
        raise FigureRepairError("registered private bundle lacks x")
    truth_x = payload["x"]
    if (not isinstance(truth_x, torch.Tensor) or truth_x.dtype != torch.float32
            or tuple(truth_x.shape) != (16, 3, 224, 224) or not bool(torch.isfinite(truth_x).all())):
        raise FigureRepairError("private x schema failed")
    return truth_x.detach().contiguous(), {"path": path.as_posix(), "sha256": PRIVATE_TRUTH_SHA,
        "bundle_keys_present": sorted(payload), "consumed_field": "x only", "labels_consumed": False,
        "true_updates_consumed": False}


def aligned_candidate(candidate: torch.Tensor, rows: list[dict[str, Any]]) -> torch.Tensor:
    aligned = torch.empty_like(candidate)
    for row in rows:
        aligned[int(row["truth_index"])] = candidate[int(row["candidate_index"])]
    return aligned


def render_sheet(output: Path, labels: list[str], names: list[str], candidates: dict[str, torch.Tensor],
                 truth_x: torch.Tensor, metrics: dict[str, Any], title: str) -> dict[str, Any]:
    if len(labels) != len(names) + 1:
        raise FigureRepairError("column label/name contract mismatch")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mean = truth_x.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = truth_x.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    truth = (truth_x * std + mean).clamp(0, 1)
    displayed = [truth] + [aligned_candidate(candidates[name], metrics[name]["per_image"]) for name in names]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(16, len(displayed), figsize=(3.4 * len(displayed), 38), squeeze=False)
    try:
        for row in range(16):
            for column, images in enumerate(displayed):
                axes[row, column].imshow(images[row].permute(1, 2, 0).clamp(0, 1))
                axes[row, column].axis("off")
                if row == 0:
                    axes[row, column].set_title(labels[column], fontsize=9, wrap=True)
        figure.suptitle(title, fontsize=15)
        figure.tight_layout(rect=(0, 0, 1, 0.985))
        figure.savefig(output, dpi=110)
    finally:
        plt.close(figure)
    return {"path": output.as_posix(), "sha256": sha256(output), "columns": labels,
            "column_count": len(labels), "rows": 16, "candidate_names": names,
            "correspondence_reused": True, "metrics_recomputed": False}


def metric_table(comparison: dict[str, Any], bits: int, selection: list[int]) -> str:
    names = ["center_25000"] + [f"rank{rank}_member_{int(member):02d}_25000"
                                for rank, member in enumerate(selection, 1)]
    lines = [f"# Existing metrics | {bits}-bit screen3", "",
             "These values are copied from the frozen comparison.json; no metric or correspondence was recomputed.",
             "Baseline: same-bit naive dequantization → FastICA/GD ICA4000 → Direct FIA, center target, 25000 steps.",
             "", "| Set | MSE | PSNR dB | SSIM | LPIPS |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in names:
        row = comparison["mean_metrics"][name]
        lines.append(f"| {name} | {row['mean_mse']:.12g} | {row['mean_psnr_db']:.12g} | {row['mean_ssim']:.12g} | {row['mean_lpips']:.12g} |")
    return "\n".join(lines) + "\n"


def render_evaluation(bits: int, evaluation_dir: Path) -> dict[str, Any]:
    evaluation_dir = Path(evaluation_dir).resolve()
    comparison_path = evaluation_dir / "comparison.json"
    comparison, paths, public_roots = authenticate_comparison(comparison_path, bits)
    truth_x, private_meta = load_private_x(comparison)
    selection = [int(x) for x in comparison["frozen_selection"]]
    layout = build_layout(bits, selection)
    candidates = {name: torch.load(path, map_location="cpu", weights_only=True)["pixels"].detach().contiguous()
                  for name, path in paths.items()}
    output = evaluation_dir / "figures_with_baseline_20260918"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated repair output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    final = render_sheet(output / f"contact_sheet_b{bits}_25000_with_baseline.png",
                         layout["final_labels"], layout["final_names"], candidates, truth_x,
                         comparison["image_metrics"],
                         f"{bits}-bit screen3 | 25000 steps | truth + baseline + public ranks")
    screen = render_sheet(output / f"contact_sheet_b{bits}_5000_with_baseline.png",
                          layout["screen_labels"], layout["screen_names"], candidates, truth_x,
                          comparison["image_metrics"],
                          f"{bits}-bit screen3 | 5000 steps | truth + baseline + all members")
    table = metric_table(comparison, bits, selection)
    (output / "existing_metrics.md").write_text(table, encoding="utf-8", newline="\n")
    manifest = {"task_id": TASK_ID, "status": "COMPLETED", "bits": bits,
        "evaluation_directory": evaluation_dir.as_posix(), "comparison_json": comparison_path.as_posix(),
        "comparison_sha256": sha256(comparison_path), "public_roots": [root.as_posix() for root in public_roots],
        "public_gate_status": comparison["public_gate"]["status"], "private_truth": private_meta,
        "selection": selection, "column_contract": {"final": layout["final_labels"], "screen": layout["screen_labels"]},
        "metric_source": "comparison.json mean_metrics and image_metrics.per_image",
        "matching_recomputed": False, "metrics_recomputed": False, "optimizer_steps": 0,
        "figures": {"final": final, "screen": screen}, "old_figures_preserved": True,
        "scientific_quality": "REPORTING_ONLY"}
    write_json(output / "figure_manifest.json", manifest)
    return manifest


def discover_evaluation(bits: int) -> Path | None:
    if bits == 8:
        candidate = REPO_ROOT / "quantization_aware_cpa/results_private/signed_int8_screen3_repair_B16_T10_20260918/evaluation_final_20260918"
        return candidate if (candidate / "comparison.json").is_file() else None
    root = REPO_ROOT / f"quantization_aware_cpa/results_private/signed_int{bits}_randomized8_screen3_fia_B16_T10_20260918_sweep"
    matches = []
    for path in root.rglob("comparison.json") if root.is_dir() else []:
        try:
            comparison = read_json(path)
        except Exception:
            continue
        if comparison.get("evaluation_status") == "COMPLETED":
            matches.append(path.parent)
    return sorted(matches)[-1] if matches else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 7, 6])
    parser.add_argument("--evaluation-dir", type=Path, action="append")
    args = parser.parse_args(argv)
    try:
        targets = {bits: path.resolve() for bits, path in zip(args.bits, args.evaluation_dir or [])}
        manifests = []
        for bits in args.bits:
            evaluation = targets.get(bits) or discover_evaluation(bits)
            if evaluation is None:
                continue
            manifests.append(render_evaluation(bits, evaluation))
        if not manifests:
            raise FigureRepairError("no completed comparison evaluation is available")
        result_root = REPO_ROOT / ".ai/execution-results" / TASK_ID
        write_json(result_root / "available_manifest.json", {"task_id": TASK_ID, "figures": manifests,
                   "pending_bits": [bits for bits in (8, 7, 6, 5, 4) if bits not in {x["bits"] for x in manifests}]})
    except Exception as error:
        print(f"{TASK_ID}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps({"task_id": TASK_ID, "status": "COMPLETED", "bits": [item["bits"] for item in manifests]},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
