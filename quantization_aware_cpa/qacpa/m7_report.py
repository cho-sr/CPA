"""M7 controls report; inference is fixed by M7_CONTROL_PREREGISTRATION.json.

This module never reconstructs images, chooses thresholds from CPA results, or
changes historical method keys. Missing controls remain explicitly pending.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .result_validity import require_valid_no_cpa_control


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT.parent / "codex" / "docs" / "M7_CONTROL_PREREGISTRATION.json"
EXPECTED_IDS = set(range(40))
LABELS = {
    "fp32_cpa": "fp32 CPA (36 ICA + 4 truth-fill)",
    "4bit_cpa": "4-bit CPA (unmodified) + naive dequant",
    "fp32_no_cpa": "fp32 no-CPA (naive control)",
    "4bit_no_cpa": "4-bit no-CPA (naive control)",
    "prior_only": "prior-only FIA",
    "fp32_shuffled": "fp32 shuffled (20 seeds)",
    "4bit_shuffled": "4-bit shuffled (20 seeds)",
    "dataset_mean": "dataset mean image",
    "gray": "gray (0.5)",
    "noise": "uniform random image noise",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def complete_rows(rows: list[dict[str, Any]]) -> bool:
    """Validate complete identity coverage, rather than trusting row count."""
    if len(rows) != 40:
        return False
    gids = [int(row["global_index"]) for row in rows]
    if len(set(gids)) != 40 or set(gids) != EXPECTED_IDS:
        return False
    for row in rows:
        gid = int(row["global_index"])
        if int(row["batch_index"]) != gid // 8 or not np.isfinite(float(row["lpips"])):
            return False
    return True


def load_completed_control(folder: Path, *, protocol_hash: str) -> list[dict[str, str]]:
    """Only the runner's final integrity marker makes a control reportable."""
    from .m7_protocol import file_sha256

    marker = folder / "summary.json"
    if not marker.is_file():
        return []
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("identity", {}).get("protocol_sha256") != protocol_hash:
        raise ValueError(f"Control protocol hash mismatch: {folder}")
    hashes = payload.get("output_hashes", {})
    if not {"rec.pt", "rec_raw.pt", "per_image_metrics.csv", "fia_log.json"}.issubset(hashes):
        raise ValueError(f"Incomplete control integrity manifest: {folder}")
    for filename, digest in hashes.items():
        artifact = folder / filename
        if not artifact.is_file() or file_sha256(artifact) != digest:
            raise ValueError(f"Control artifact hash mismatch: {artifact}")
    rows = read_csv(folder / "per_image_metrics.csv")
    if payload.get("n_images") != 40 or not complete_rows(rows):
        raise ValueError(f"Final control marker has incomplete image coverage: {folder}")
    return rows


def exact_sign_flip(batch_effects: list[float] | np.ndarray) -> dict[str, Any]:
    """One-sided mean sign randomization, including the observed sign and ties."""
    effects = np.asarray(batch_effects, dtype=np.float64)
    if effects.shape != (5,) or not np.isfinite(effects).all():
        raise ValueError("The preregistered test requires five finite victim-batch effects")
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=5)))
    null = (signs * effects).mean(axis=1)
    observed = float(effects.mean())
    tolerance = 16 * np.finfo(float).eps * max(1.0, float(np.abs(effects).max()))
    extreme = int(np.count_nonzero(null >= observed - tolerance))
    return {
        "batch_effects_control_minus_cpa": effects.tolist(),
        "mean_effect": observed,
        "p_one_sided": extreme / 32,
        "extreme_sign_patterns": extreme,
        "total_sign_patterns": 32,
    }


def group_comparison(
    cpa: list[dict[str, Any]], no_cpa: list[dict[str, Any]],
    prior: list[dict[str, Any]], *, ids: set[int] | None = None,
) -> dict[str, Any]:
    if not all(complete_rows(rows) for rows in (cpa, no_cpa, prior)):
        raise ValueError("Complete 40-image rows are required before group inference")
    selected = EXPECTED_IDS if ids is None else ids
    if not selected or not selected.issubset(EXPECTED_IDS):
        raise ValueError("Sensitivity image IDs must be a nonempty subset of the fixed evaluation set")
    indexed = [{int(row["global_index"]): float(row["lpips"]) for row in rows} for rows in (cpa, no_cpa, prior)]
    results: dict[str, Any] = {"n_images": len(selected), "n_batches": 5, "global_indices": sorted(selected)}
    results["mean_lpips_by_condition"] = {
        name: float(np.mean([values[gid] for gid in sorted(selected)]))
        for name, values in zip(("cpa", "no_cpa", "prior_only"), indexed)
    }
    for name, control in zip(("no_cpa", "prior_only"), indexed[1:]):
        by_batch = []
        for batch in range(5):
            batch_ids = sorted(gid for gid in selected if gid // 8 == batch)
            if not batch_ids:
                raise ValueError("Every victim batch must be represented in the sign-flip test")
            by_batch.append(float(np.mean([control[gid] - indexed[0][gid] for gid in batch_ids])))
        results[name] = exact_sign_flip(by_batch)
    results["iut_p"] = max(results["no_cpa"]["p_one_sided"], results["prior_only"]["p_one_sided"])
    results["both_mean_effects_positive"] = all(results[name]["mean_effect"] > 0 for name in ("no_cpa", "prior_only"))
    results["passes_preregistered_rule"] = bool(results["both_mean_effects_positive"] and results["iut_p"] <= 0.05)
    return results


def calibrate_thresholds(
    no_cpa: dict[str, list[dict[str, Any]]], prior: list[dict[str, Any]], *, prereg_sha256: str,
) -> dict[str, Any]:
    """Read controls only. Callers must serialize this before evaluating candidates."""
    if set(no_cpa) != {"fp32", "4bit"} or not all(complete_rows(rows) for rows in [*no_cpa.values(), prior]):
        raise ValueError("Both no-CPA precisions and prior-only require all 40 images; no partial calibration")
    thresholds: dict[str, Any] = {
        "protocol_id": "m7_controls_v1", "preregistration_sha256": prereg_sha256,
        "alpha": 0.05, "reference_count_per_control": 32,
        "rule": "max((1 + count(control LPIPS <= candidate LPIPS)) / 33) <= 0.05",
        "equivalent_rule": "candidate LPIPS strictly below both leave-batch-out control minima",
        "interpretation": "control-calibrated lower-tail count; not per-image statistical significance",
        "calibration_uses_candidate_values": False, "by_precision": {},
    }
    for precision in ("fp32", "4bit"):
        batches = []
        for batch in range(5):
            refs = {
                "no_cpa": sorted(float(row["lpips"]) for row in no_cpa[precision] if int(row["batch_index"]) != batch),
                "prior_only": sorted(float(row["lpips"]) for row in prior if int(row["batch_index"]) != batch),
            }
            batches.append({
                "held_out_batch": batch, "n_reference_each": 32,
                "no_cpa_min": refs["no_cpa"][0], "prior_only_min": refs["prior_only"][0],
                "strict_lpips_upper_bound": min(refs["no_cpa"][0], refs["prior_only"][0]),
                "reference_lpips": refs,
            })
        thresholds["by_precision"][precision] = batches
    return thresholds


def candidate_success(
    rows: list[dict[str, Any]], *, method: str, precision: str, thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    if not complete_rows(rows):
        raise ValueError("Do not count success for incomplete candidate rows")
    out = []
    for row in sorted(rows, key=lambda item: int(item["global_index"])):
        lpips = float(row["lpips"])
        calibration = thresholds["by_precision"][precision][int(row["batch_index"])]
        refs = calibration["reference_lpips"]
        p_no = (1 + sum(value <= lpips for value in refs["no_cpa"])) / 33
        p_prior = (1 + sum(value <= lpips for value in refs["prior_only"])) / 33
        out.append({
            "method": method, "precision": precision, "global_index": int(row["global_index"]),
            "batch_index": int(row["batch_index"]), "lpips": lpips,
            "strict_lpips_upper_bound": calibration["strict_lpips_upper_bound"],
            "rank_p_no_cpa": p_no, "rank_p_prior_only": p_prior,
            "success_control_calibrated": int(max(p_no, p_prior) <= thresholds["alpha"]),
        })
    return out


def histogram(groups: dict[str, list[float]], path: Path, *, pending: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = {key: np.asarray(values, dtype=float) for key, values in groups.items() if values}
    finite = {key: values[np.isfinite(values)] for key, values in finite.items()}
    finite = {key: values for key, values in finite.items() if len(values)}
    fig, ax = plt.subplots(figsize=(13, 7), dpi=170)
    if finite:
        combined = np.concatenate(list(finite.values()))
        lower = min(0.0, float(combined.min()))
        upper = max(1.0, float(combined.max()) * 1.03)
        bins = np.linspace(lower, upper, 41)
        colors = ["#2563eb", "#dc2626", "#0891b2", "#ea580c", "#7c3aed", "#64748b", "#a16207", "#16a34a", "#374151", "#c026d3"]
        for index, (key, values) in enumerate(finite.items()):
            linestyle = "--" if "shuffled" in key else (":" if key in {"gray", "noise", "dataset_mean"} else "-")
            ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.8,
                    linestyle=linestyle, color=colors[index % len(colors)], label=f"{LABELS.get(key, key)} (n={len(values)})")
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    else:
        ax.text(0.5, 0.5, "PENDING: no metric rows available", ha="center", transform=ax.transAxes)
    ax.set_xlabel("Per-image LPIPS (lower is better)")
    ax.set_ylabel("Density within each condition")
    ax.set_title("M7: CPA, no-CPA, matching, and prior controls" + (" — PENDING" if pending else ""))
    ax.grid(axis="y", alpha=0.18)
    note = "Common bins; shuffled pairs reuse the same 40 images and are not 800 independent victims."
    if pending:
        note += "\nPending: " + ", ".join(pending)
    fig.text(0.07, 0.015, note, fontsize=8)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def random_grid(no_cpa_dir: Path, output: Path, *, prereg: dict[str, Any]) -> dict[str, Any]:
    import torch
    from PIL import Image, ImageDraw, ImageFont

    seed, count = prereg["figures"]["grid_seed"], prereg["figures"]["grid_size"]
    ids = np.random.default_rng(seed).choice(40, count, replace=False).tolist()
    records: dict[str, Any] = {"seed": seed, "global_indices": ids, "selection": "not cherry-picked", "depends_on_lpips": False}
    tensors = {}
    for key, path in (("original", no_cpa_dir / "truth.pt"), ("fp32 no-CPA", no_cpa_dir / "fp32" / "rec.pt"), ("4-bit no-CPA", no_cpa_dir / "4bit" / "rec.pt")):
        if path.is_file() and (key == "original" or (path.parent / "summary.json").is_file()):
            tensor = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or tensor.shape[0] != 40 or tensor.shape[1] != 3:
                raise ValueError(f"Expected 40 RGB images in {path}")
            tensors[key] = tensor.detach().cpu().float()
    tile, gap, metadata, heading = 144, 12, 25, 100
    columns = ("original", "fp32 no-CPA", "4-bit no-CPA")
    block_width = tile * 3 + gap * 2
    width = block_width * 3 + gap * 4
    height = heading + ((count + 2) // 3) * (tile + metadata + gap) + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    label_font = ImageFont.truetype(str(font_path), 14) if font_path.is_file() else font
    title_font = ImageFont.truetype(str(font_path), 25) if font_path.is_file() else font
    draw.text((gap, 10), "M7 no-CPA reconstructions — not cherry-picked", fill="#111827", font=title_font)
    draw.text((gap, 44), f"12 of 40 originals; random seed={seed}; selection fixed independently of LPIPS", fill="#374151", font=font)
    for idx, gid in enumerate(ids):
        bx = gap + (idx % 3) * (block_width + gap)
        by = heading + (idx // 3) * (tile + metadata + gap)
        for col, label in enumerate(columns):
            x = bx + col * (tile + gap)
            draw.text((x, by - metadata), f"g{gid:02d} {label}", fill="#111827", font=label_font)
            if label in tensors:
                array = tensors[label][gid].clamp(0, 1).permute(1, 2, 0).numpy()
                thumb = Image.fromarray(np.round(array * 255).astype(np.uint8)).resize((tile, tile), Image.Resampling.LANCZOS)
                canvas.paste(thumb, (x, by))
            else:
                draw.rectangle((x, by, x + tile, by + tile), fill="#f1f5f9")
                draw.text((x + 18, by + 58), "PENDING", fill="#64748b", font=font)
    canvas.save(output)
    records["available_columns"] = list(tensors)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cpa-dir", type=Path, default=ROOT / "results" / "m7_no_cpa_control")
    parser.add_argument("--prior-dir", type=Path, default=ROOT / "results" / "m7_prior_only")
    parser.add_argument("--recompute-dir", type=Path, default=ROOT / "results" / "m7_recompute_controls")
    parser.add_argument("--preregistration", type=Path, default=PREREG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_report(args.no_cpa_dir, args.prior_dir, args.recompute_dir, args.preregistration)


def generate_report(no_cpa_dir: Path, prior_dir: Path, recompute_dir: Path, prereg_path: Path = PREREG) -> dict[str, Any]:
    """Render currently available artifacts. Call again after queued FIA finishes."""
    require_valid_no_cpa_control()
    no_cpa_dir, prior_dir, recompute_dir = map(Path, (no_cpa_dir, prior_dir, recompute_dir))
    no_cpa_dir.mkdir(parents=True, exist_ok=True)
    prereg_bytes = Path(prereg_path).read_bytes()
    prereg = json.loads(prereg_bytes)
    prereg_hash = hashlib.sha256(prereg_bytes).hexdigest()
    lock_path = Path(prereg_path).with_name("M7_CONTROL_PREREGISTRATION.lock.json")
    if not lock_path.is_file() or json.loads(lock_path.read_text(encoding="utf-8-sig"))["sha256"] != prereg_hash:
        raise ValueError("M7 preregistration is not locked or changed after locking")
    if prereg.get("protocol_id") != "m7_controls_v1" or prereg["success"]["alpha"] != 0.05:
        raise ValueError("Unsupported preregistration; do not silently change the inference rule")
    historical_path = ROOT / prereg["historical"]["csv"]
    historical = read_csv(historical_path)
    cpa = {
        "fp32": [row for row in historical if row["method"] == "fp32"],
        "4bit": [row for row in historical if row["method"] == "naive"],
    }
    for precision, key in (("fp32", "fp32_requested_mean"), ("4bit", "4bit_requested_mean")):
        if not complete_rows(cpa[precision]):
            raise ValueError(f"Historical {precision} row lacks the fixed 40-image coverage")
        mean = float(np.mean([float(row["lpips"]) for row in cpa[precision]]))
        if abs(mean - prereg["historical"][key]) > 1e-8:
            raise ValueError(f"Historical {precision} mean changed: {mean}")
    no_cpa = {precision: load_completed_control(no_cpa_dir / precision, protocol_hash=prereg_hash) for precision in ("fp32", "4bit")}
    prior = load_completed_control(prior_dir / "prior_only", protocol_hash=prereg_hash)
    ready = all(complete_rows(rows) for rows in [*no_cpa.values(), prior])
    pending = [f"{precision} no-CPA" for precision in ("fp32", "4bit") if not complete_rows(no_cpa[precision])]
    if not complete_rows(prior):
        pending.append("prior-only")
    success_rows: list[dict[str, Any]] = []
    groups: dict[str, Any] = {}
    threshold_path = no_cpa_dir / "thresholds.json"
    if ready:
        thresholds = calibrate_thresholds(no_cpa, prior, prereg_sha256=prereg_hash)
        if threshold_path.is_file() and json.loads(threshold_path.read_text(encoding="utf-8")) != thresholds:
            raise ValueError("Serialized thresholds disagree with current controls/protocol; refusing to replace a frozen threshold")
        # This write precedes every candidate count and must not move below it.
        write_json(threshold_path, thresholds)
        frozen = json.loads(threshold_path.read_text(encoding="utf-8"))
        for precision in ("fp32", "4bit"):
            for kind, rows in (("cpa", cpa[precision]), ("no_cpa", no_cpa[precision])):
                success_rows.extend(candidate_success(rows, method=f"{precision}_{kind}", precision=precision, thresholds=frozen))
            groups[precision] = group_comparison(cpa[precision], no_cpa[precision], prior)
        fill_ids = set(prereg["historical"]["fp32_truth_fill_global_indices"])
        groups["fp32_same36_sensitivity"] = group_comparison(cpa["fp32"], no_cpa["fp32"], prior, ids=EXPECTED_IDS - fill_ids)
        groups["fp32_same36_sensitivity"]["excluded_oracle_fill_global_indices"] = sorted(fill_ids)
    fields = ["method", "precision", "global_index", "batch_index", "lpips", "strict_lpips_upper_bound", "rank_p_no_cpa", "rank_p_prior_only", "success_control_calibrated"]
    write_csv(no_cpa_dir / "image_success.csv", success_rows, fields)
    table = []
    for precision in ("fp32", "4bit"):
        for kind, rows in (("cpa", cpa[precision]), ("no_cpa", no_cpa[precision])):
            method = f"{precision}_{kind}"
            complete = complete_rows(rows)
            table.append({
                "precision": precision, "cpa": "present" if kind == "cpa" else "absent",
                "display_label": LABELS[method], "n_images": len(rows),
                "lpips_mean": float(np.mean([float(row["lpips"]) for row in rows])) if complete else "PENDING",
                "success_count": sum(row["success_control_calibrated"] for row in success_rows if row["method"] == method) if ready else "PENDING",
                "success_denominator": 40, "status": "complete" if complete and ready else "pending",
                "metric_source": "reused_historical" if kind == "cpa" else "new_FIA_control",
            })
    write_csv(no_cpa_dir / "grid_2x2.csv", table)
    distributions = {f"{precision}_cpa": [float(row["lpips"]) for row in cpa[precision]] for precision in ("fp32", "4bit")}
    distributions.update({f"{precision}_no_cpa": [float(row["lpips"]) for row in no_cpa[precision]] for precision in ("fp32", "4bit")})
    distributions["prior_only"] = [float(row["lpips"]) for row in prior]
    recompute = load_recompute_distributions(recompute_dir, protocol_hash=prereg_hash)
    distributions.update(recompute)
    for key in ("fp32_shuffled", "4bit_shuffled", "dataset_mean", "gray", "noise"):
        expected = 800 if "shuffled" in key else 40
        if len(recompute.get(key, [])) != expected:
            pending.append(LABELS[key])
    histogram(distributions, no_cpa_dir / "lpips_histogram_overlay.png", pending=pending)
    grid = random_grid(no_cpa_dir, no_cpa_dir / "no_cpa_random_grid.png", prereg=prereg)
    write_json(no_cpa_dir / "no_cpa_random_grid_selection.json", grid)
    status = "complete" if ready and not pending and len(grid["available_columns"]) == 3 else "pending"
    summary = {
        "schema_version": 1, "run_tag": "m7_no_cpa_control", "report_status": status,
        "preregistration": str(prereg_path), "preregistration_sha256": prereg_hash,
        "controls_complete_for_inference": ready, "pending_conditions": pending,
        "grid_2x2": table, "group_inference": groups,
        "primary_claim_supported": groups.get("4bit", {}).get("passes_preregistered_rule") if ready else None,
        "success_count_definition": prereg["success"], "grid_selection": grid,
        "fp32_caveat": "Requested historical mean 0.4448 includes four exact-source FIA fills (22,24,32,36); fp32 is a hybrid row, not pure CPA.",
        "label_correction": "Historical raw key naive = CPA (미수정) + naive dequant; human-facing naive is reserved for no-CPA controls.",
        "shuffled_seed_distribution": shuffled_seed_distribution(recompute_dir) if recompute else {},
        "positional_control_means": {
            name: float(np.mean([float(row["lpips_positional"]) for row in rows]))
            for name, rows in {"fp32_no_cpa": no_cpa["fp32"], "4bit_no_cpa": no_cpa["4bit"], "prior_only": prior}.items()
            if complete_rows(rows) and all(row.get("lpips_positional") is not None for row in rows)
        },
    }
    write_json(no_cpa_dir / "summary.json", summary)
    write_markdown(no_cpa_dir / "report.md", summary, distributions)
    execution_path = no_cpa_dir / "execution_status.json"
    if status == "complete" and execution_path.is_file():
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        if execution.get("state") == "fia_complete":
            write_json(execution_path, {
                **execution, "state": "complete", "report_path": str(no_cpa_dir / "report.md"),
                "report_completed_at_utc": datetime.now(timezone.utc).isoformat(),
            })
    print(f"M7 report: {status}; {no_cpa_dir / 'report.md'}", flush=True)
    return summary


def load_recompute_distributions(directory: Path, *, protocol_hash: str | None = None) -> dict[str, list[float]]:
    """Read the recompute-only agent's per-image artifacts (no LPIPS forwards)."""
    marker = directory / "summary.json"
    if not marker.is_file():
        return {}
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if protocol_hash is not None and metadata.get("spec_sha256") != protocol_hash:
        raise ValueError("Recompute controls used a different protocol hash")
    csv_hash = metadata.get("per_victim_batch_source", {}).get("source_csv_sha256")
    if csv_hash is not None:
        from .m7_protocol import file_sha256
        if file_sha256(directory / "per_image_metrics.csv") != csv_hash:
            raise ValueError("Recompute per-image artifact hash mismatch")
    groups: dict[str, list[float]] = {}
    mapping = {"fp32_CPA": "fp32_shuffled", "four_bit_CPA": "4bit_shuffled",
               "dataset_mean": "dataset_mean", "gray_0.5": "gray", "uniform_noise": "noise"}
    for row in read_csv(directory / "per_image_metrics.csv"):
        raw = row["method"] if row["control"] == "shuffled" else row["control"]
        if raw not in mapping:
            raise ValueError(f"Unexpected recompute control: {raw}")
        key = mapping[raw]
        groups.setdefault(key, []).append(float(row["lpips"]))
    return groups


def shuffled_seed_distribution(directory: Path) -> dict[str, Any]:
    rows = read_csv(directory / "shuffled_seed_summary.csv")
    results = {}
    for precision, raw in (("fp32", "fp32_CPA"), ("4bit", "four_bit_CPA")):
        selected = [row for row in rows if row["method"] == raw]
        if len(selected) != 20 or {int(row["seed"]) for row in selected} != set(range(20)):
            raise ValueError("Shuffled summary requires all 20 distinct preregistered seeds")
        means = [float(row["lpips_mean"]) for row in selected]
        results[precision] = {"n_seeds": 20, "mean": float(np.mean(means)),
                              "std_across_seed_means": float(np.std(means, ddof=1)),
                              "min_seed_mean": min(means), "max_seed_mean": max(means),
                              "per_seed": selected}
    return results


def write_markdown(path: Path, summary: dict[str, Any], distributions: dict[str, list[float]]) -> None:
    def metric(value: Any) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    cells = {(row["precision"], row["cpa"]): row for row in summary["grid_2x2"]}
    lines = [
        "# M7: CPA contribution controls", "",
        f"Status: **{summary['report_status'].upper()}**. B=8, five fixed victim batches, 40 original images, victim seed=0, round-0 VGG16.", "",
        "| Observation | CPA present: LPIPS / success count | No-CPA: LPIPS / success count |",
        "|---|---:|---:|",
    ]
    for precision in ("fp32", "4bit"):
        present, absent = cells[(precision, "present")], cells[(precision, "absent")]
        def cell(row: dict[str, Any]) -> str:
            count = f"{row['success_count']}/40" if isinstance(row["success_count"], int) else "PENDING"
            return f"{metric(row['lpips_mean'])} / {count}"
        lines.append(f"| {precision} | {cell(present)} | {cell(absent)} |")
    lines += [
        "", "The reused 4-bit row is **CPA (미수정) + naive dequant**. `naive` labels only the no-CPA controls; historical raw keys and filenames remain unchanged.",
        "", summary["fp32_caveat"],
        "", "No-CPA uses seeded observed matrix rows, with only row L2 normalization. FIA uses the unchanged 25,000 iterations, TV=0.1, Adam lr=0.1, cosine schedule, FIA seed=42, and original chunk boundaries. Permutation-only matching is evaluation after FIA; positional LPIPS is also retained.",
        "", "Success counts follow the preregistered control-calibrated lower-tail rule. Each image excludes its entire victim batch; it must have LPIPS strictly below both minima from the 32 remaining precision-matched no-CPA images and 32 prior-only images. `thresholds.json` is frozen from controls before counting candidate successes. These counts are descriptive, not per-image significance declarations.",
    ]
    if summary["controls_complete_for_inference"]:
        lines += ["", "| Comparison | Mean no-CPA minus CPA | p vs no-CPA | Mean prior minus CPA | p vs prior | IUT p | Rule passes |", "|---|---:|---:|---:|---:|---:|---|"]
        for key, group in summary["group_inference"].items():
            lines.append(f"| {key} | {group['no_cpa']['mean_effect']:.4f} | {group['no_cpa']['p_one_sided']:.5f} | {group['prior_only']['mean_effect']:.4f} | {group['prior_only']['p_one_sided']:.5f} | {group['iut_p']:.5f} | {group['passes_preregistered_rule']} |")
        sensitivity = summary["group_inference"]["fp32_same36_sensitivity"]["mean_lpips_by_condition"]
        lines += ["", f"On the same 36 images excluding exact-source fills, fp32 CPA LPIPS={sensitivity['cpa']:.4f}, no-CPA={sensitivity['no_cpa']:.4f}, and prior-only={sensitivity['prior_only']:.4f}."]
        supported = summary["primary_claim_supported"]
        lines += ["", "The primary 4-bit claim " + ("passes" if supported else "does not pass") + " the fixed rule: positive mean effects against both controls and maximum one-sided p <= 0.05. Each test enumerates all 32 signs over five paired victim-batch means and includes ties. Batch sign symmetry is assumed; fp32 comparisons are secondary."]
    else:
        lines += ["", "**Success counts and group inference are PENDING until both no-CPA precisions and prior-only have all 40 images.** Available LPIPS values are not interpreted as reconstruction success."]
    lines += ["", "| Auxiliary distribution | Per-image / pair mean LPIPS | Rows |", "|---|---:|---:|"]
    for key in ("prior_only", "fp32_shuffled", "4bit_shuffled", "dataset_mean", "gray", "noise"):
        values = distributions.get(key, [])
        lines.append(f"| {LABELS[key]} | {float(np.mean(values)):.4f} | {len(values)} |" if values else f"| {LABELS[key]} | PENDING | 0 available |")
    lines += [
        "", "Shuffled controls use 20 fixed permutations of the same 40 existing reconstructions; the 800 pair metrics per precision are dependent and are not 800 independent victims. Constants establish reference LPIPS values, not a mathematical lower bound. The dataset-average image uses the full transformed validation split.",
    ]
    if summary["shuffled_seed_distribution"]:
        lines += ["", "| Shuffled precision | Mean across 20 seeds | SD of seed means | Min–max seed mean |", "|---|---:|---:|---:|"]
        for precision, values in summary["shuffled_seed_distribution"].items():
            lines.append(f"| {precision} | {values['mean']:.4f} | {values['std_across_seed_means']:.4f} | {values['min_seed_mean']:.4f}–{values['max_seed_mean']:.4f} |")
    lines += [
        "", "![Common-axis per-image LPIPS histograms](lpips_histogram_overlay.png)",
        "", "![Random no-CPA reconstructions: not cherry-picked](no_cpa_random_grid.png)",
        "", "Grid indices: " + ", ".join(map(str, summary["grid_selection"]["global_indices"])) + "; NumPy seed=42, no replacement. **not cherry-picked**.",
        "", f"Preregistration SHA-256: `{summary['preregistration_sha256']}`. No Stage 2, `--refine-A`, or 4-bit GMA.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
