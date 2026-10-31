"""Offline comparison of completed three- and 100-iteration full-update runs.

This tool never resumes an attack, selects an intermediate candidate, or polls
an incomplete run. Private truth is used only for final-artifact evaluation and
an optional grid containing all sixteen examples in their original truth order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ARMS = {
    "cell_3": ("pilot_cell", "cell", 3),
    "center_3": ("pilot_center", "center", 3),
    "cell_100": ("attack_cell_I100", "cell", 100),
    "center_100": ("attack_center_I100", "center", 100),
}
FINGERPRINTS = ("implementation_sha256", "runtime", "observation_manifest_sha256",
                "contract_sha256", "checkpoint_sha256", "method")
HARD_FIT_KEYS = (
    "tolerance_steps", "closure_fit_within_tolerance", "bin_violation_rate",
    "max_bin_violation_steps", "mean_bin_violation_steps", "max_radius_error_steps",
    "max_zero_scale_error_abs", "zero_marginal_probability_count", "exact_joint_rng_match",
)
EVALUATION_RTOL = 1e-10
EVALUATION_ATOL = 1e-12
LPIPS_BACKBONE_SHA256 = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
LPIPS_CALIBRATION_SHA256 = "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868"


class ReportError(ValueError):
    """Required completed artifacts do not form the registered comparison."""


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_arms(base: Path) -> dict:
    # Check completion for all arms before opening any candidate or truth.
    arms = {}
    for key, (folder, mode, iterations) in ARMS.items():
        directory = base / folder
        required = ("registered_design.json", "result.json", "optimization_log.json", "evaluation.json", "candidate.pt")
        if any(not (directory / name).is_file() for name in required):
            raise ReportError(f"Arm {key} is incomplete; all final artifacts are required")
        if (directory / "failure.json").exists():
            raise ReportError(f"Arm {key} contains a failure artifact")
        result = read_json(directory / "result.json")
        design = read_json(directory / "registered_design.json")
        logs = read_json(directory / "optimization_log.json")
        evaluation = read_json(directory / "evaluation.json")
        if (result.get("status") != "candidate_generated" or result.get("iterations") != iterations
                or result.get("mode") != mode or design.get("mode") != mode
                or design.get("probe_only") is not False
                or design.get("attack", {}).get("iterations") != iterations):
            raise ReportError(f"Arm {key} completion/budget/mode does not match its registered role")
        if not isinstance(result.get("hard_label_observation_fit"), dict):
            raise ReportError(f"Arm {key} lacks final hard-label observation diagnostics")
        if not isinstance(logs, list) or [row.get("iteration") for row in logs] != list(range(1, iterations + 1)):
            raise ReportError(f"Arm {key} must contain every registered iteration exactly once")
        for row in logs:
            if any(not isinstance(row.get(field), (int, float)) or not math.isfinite(row[field])
                   for field in ("loss", "observation_loss", "tv", "image_gradient_norm", "elapsed_sec")):
                raise ReportError(f"Arm {key} contains missing/nonfinite optimization diagnostics")
            if row["loss"] <= 0:
                raise ReportError("Normalized logarithmic convergence plot requires positive soft objectives")
        arms[key] = {"path": directory, "design": design, "result": result, "logs": logs,
                     "evaluation": evaluation, "mode": mode, "iterations": iterations}
    return arms


def _check_provenance(arms: dict, manifest: dict, manifest_sha256: str) -> dict:
    reference = arms["cell_3"]["design"]
    common_attack = {key: value for key, value in reference["attack"].items() if key != "iterations"}
    for key, arm in arms.items():
        design = arm["design"]
        for field in FINGERPRINTS:
            if field not in design or design[field] != reference.get(field):
                raise ReportError(f"Arm {key} differs in registered {field}")
        if {name: value for name, value in design["attack"].items() if name != "iterations"} != common_attack:
            raise ReportError(f"Arm {key} changes attack settings beyond iterations and mode")
        if design["observation_manifest_sha256"] != manifest_sha256:
            raise ReportError(f"Arm {key} is bound to a different observation manifest")
        if any(design[field] != manifest.get(field) for field in ("checkpoint_sha256", "contract_sha256")):
            raise ReportError(f"Arm {key} model/contract binding disagrees with the observation")
    if manifest.get("contract", {}).get("image_shape") != [16, 3, 224, 224]:
        raise ReportError("This comparison requires all sixteen original 224x224 RGB images")
    for mode in ("cell", "center"):
        pilot = arms[f"{mode}_3"]["logs"]
        extended = arms[f"{mode}_100"]["logs"][:3]
        numerical = lambda rows: [{key: value for key, value in row.items() if key != "elapsed_sec"} for row in rows]
        if numerical(pilot) != numerical(extended):
            raise ReportError(f"Arm {mode} first three numerical logs differ from its seeded pilot")
    return {"only_budget_change_from_pilot": {"iterations": [3, 100]},
            "only_arm_change_at_each_budget": "mode: cell/center", "first_three_numeric_logs_exact": True,
            "common_recorded_fingerprints": {key: reference[key] for key in FINGERPRINTS},
            "common_attack_settings": common_attack,
            "initialization": "fresh identical seeded starts; 100 total iterations, not resumed +100",
            "fingerprint_scope_note": "Pilot recorded full_update_*.py; CLI and communication_codec hashes were not recorded there"}


def _verify_evaluations(arms: dict, truth: Path, manifest_sha256: str) -> None:
    import torch
    from .full_update_evaluation import evaluate_candidate
    # Recalculate the same fixed evaluation only to authenticate existing
    # metrics/mappings against these exact files. Any difference is rejected;
    # no alternative candidate or new assignment is adopted for the report.
    for key, arm in arms.items():
        candidate_path = arm["path"] / "candidate.pt"
        candidate = torch.load(candidate_path, weights_only=True, map_location="cpu")
        if candidate.get("observation_manifest_sha256") != manifest_sha256:
            raise ReportError(f"Candidate {key} is bound to a different observation")
        if tuple(candidate["pixels"].shape) != (16, 3, 224, 224):
            raise ReportError(f"Candidate {key} does not contain all sixteen registered images")
        del candidate
        evaluation = arm["evaluation"]
        if (evaluation.get("n_images") != 16 or evaluation.get("all_images_included") is not True
                or evaluation.get("matching") != "Hungarian_pixel_MSE_once_for_all_metrics"
                or evaluation.get("candidate_selection") != "none_single_supplied_artifact"):
            raise ReportError(f"Evaluation {key} changed the complete final-artifact matching rule")
        rows = evaluation.get("per_image", [])
        if (sorted(row["truth_index"] for row in rows) != list(range(16)) or
                sorted(row["candidate_index"] for row in rows) != list(range(16))):
            raise ReportError(f"Evaluation {key} lacks a complete one-to-one image assignment")
        if not _evaluation_matches(evaluate_candidate(candidate_path, truth), evaluation):
            raise ReportError(f"Existing evaluation {key} does not match the exact supplied candidate/truth files")


def _evaluation_matches(actual, recorded) -> bool:
    """Keep assignments/metadata exact; permit only FP64 reduction roundoff."""
    if isinstance(actual, dict):
        return (isinstance(recorded, dict) and actual.keys() == recorded.keys()
                and all(_evaluation_matches(value, recorded[key]) for key, value in actual.items()))
    if isinstance(actual, list):
        return (isinstance(recorded, list) and len(actual) == len(recorded)
                and all(_evaluation_matches(a, b) for a, b in zip(actual, recorded)))
    if isinstance(actual, float):
        return (isinstance(recorded, float) and math.isfinite(actual) and math.isfinite(recorded)
                and math.isclose(actual, recorded, rel_tol=EVALUATION_RTOL, abs_tol=EVALUATION_ATOL))
    return type(actual) is type(recorded) and actual == recorded


def _verified_lpips_weights() -> dict:
    """Verify both local weight files before a metric can be instantiated."""
    import importlib
    import torch
    import torchvision
    import torchmetrics
    from torchvision.models import VGG16_Weights
    if (VGG16_Weights.DEFAULT != VGG16_Weights.IMAGENET1K_V1 or
            not VGG16_Weights.DEFAULT.url.endswith("/vgg16-397923af.pth")):
        raise ReportError("LPIPS VGG default backbone changed; refusing an implicit weight substitution")
    implementation = importlib.import_module("torchmetrics.functional.image.lpips")
    backbone = Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"
    calibration = Path(implementation.__file__).parent / "lpips_models/vgg.pth"
    files = {}
    for name, path, expected in (("backbone", backbone, LPIPS_BACKBONE_SHA256),
                                  ("calibration", calibration, LPIPS_CALIBRATION_SHA256)):
        if not path.is_file():
            raise ReportError(f"Required local LPIPS {name} weights missing; downloads are disabled: {path}")
        actual = digest(path)
        if actual != expected:
            raise ReportError(f"Local LPIPS {name} SHA256 mismatch; refusing different weights")
        files[name] = {"path": str(path), "sha256": actual}
    return {"implementation": "torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity",
            "net_type": "vgg", "normalize_argument": False, "device": "cpu", "dtype": "float32",
            "versions": {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                         "torchmetrics": str(torchmetrics.__version__)},
            "input_normalization": "FP32 RGB [0,1] to [-1,1] using (pixels - 0.5) * 2",
            "weights": files, "downloads_allowed": False}


def _make_lpips_metric():
    from unittest.mock import patch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    provenance = _verified_lpips_weights()
    # File verification precedes construction. Also block the torchvision hub
    # download path in case a cache entry disappears between check and load.
    with patch("torch.hub.download_url_to_file", side_effect=ReportError("LPIPS weight downloads are disabled")):
        metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to("cpu").eval()
    return metric, provenance


def _lpips_supplement(arms: dict, truth: Path) -> dict:
    """Descriptive LPIPS for existing fixed Hungarian pairs, one pair at a time."""
    import torch
    metric, provenance = _make_lpips_metric()
    images = torch.load(truth, weights_only=True, map_location="cpu")["x"].float()
    reference = (images * images.new_tensor((.229, .224, .225))[None, :, None, None]
                 + images.new_tensor((.485, .456, .406))[None, :, None, None]).clamp(0, 1)
    result = {"provenance": provenance, "matching": "existing_Hungarian_pixel_MSE_pairs_unchanged",
              "all_images_included": True, "threshold": "none_registered_or_applied",
              "interpretation": "descriptive perceptual distance only; no reconstruction success classification",
              "arms": {}}
    with torch.no_grad():
        for key, arm in arms.items():
            pixels = torch.load(arm["path"] / "candidate.pt", weights_only=True, map_location="cpu")["pixels"].float()
            values = []
            for pair in sorted(arm["evaluation"]["per_image"], key=lambda row: row["truth_index"]):
                i, j = pair["candidate_index"], pair["truth_index"]
                metric.reset()
                score = metric((pixels[i:i+1] - 0.5) * 2, (reference[j:j+1] - 0.5) * 2)
                if score.numel() != 1 or not bool(torch.isfinite(score).all()):
                    raise ReportError(f"Nonfinite/nonscalar LPIPS for {key}, pair ({i}, {j})")
                values.append({"candidate_index": i, "truth_index": j, "lpips": float(score.detach().cpu())})
            if len(values) != 16:
                raise ReportError("Supplemental LPIPS requires all sixteen existing matched pairs")
            result["arms"][key] = {"n_images": len(values), "mean_lpips": sum(row["lpips"] for row in values) / len(values),
                                    "per_image": values}
    result["deltas_100_minus_3"] = {mode: result["arms"][f"{mode}_100"]["mean_lpips"]
                                         - result["arms"][f"{mode}_3"]["mean_lpips"] for mode in ("cell", "center")}
    return result


def _plots(arms: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for mode, color in (("cell", "#2166ac"), ("center", "#b2182b")):
        rows = arms[f"{mode}_100"]["logs"]
        initial = rows[0]["loss"]
        axis.plot([row["iteration"] for row in rows], [row["loss"] / initial for row in rows],
                  label=f"{mode}: 100 iterations", color=color)
        pilot = arms[f"{mode}_3"]["logs"]
        axis.scatter([3], [pilot[-1]["loss"] / pilot[0]["loss"]],
                     label=f"{mode}: 3-iteration pilot log", color=color, marker="s", zorder=5)
    axis.set_yscale("log")
    axis.set_xlabel("Outer iteration (objective measured before that update)")
    axis.set_ylabel("Soft-label objective / own initial objective")
    axis.set_title("Convergence of different objectives; ratios are not reconstruction scores")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(output / "convergence.png", dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for index, (budget, color) in enumerate(((3, "#969696"), (100, "#238b45"))):
        values = [arms[f"{mode}_{budget}"]["evaluation"]["mean_psnr_db"] for mode in ("cell", "center")]
        if any(value is None or not math.isfinite(value) for value in values):
            raise ReportError("Finite PSNR values are required for this fixed comparison plot")
        positions = [position + (index - 0.5) * 0.34 for position in range(2)]
        bars = axis.bar(positions, values, width=0.34, label=f"{budget} iterations", color=color)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axis.set_xticks([0, 1], ["Cell", "Center"])
    axis.set_ylabel("Mean PSNR (dB), all 16 images")
    axis.set_title("Final hard-label candidates: fixed offline Hungarian matching")
    axis.legend()
    figure.savefig(output / "final_psnr.png", dpi=180)
    plt.close(figure)


def _image_grid(arms: dict, truth: Path, output: Path) -> None:
    import torch
    import matplotlib.pyplot as plt
    reference = torch.load(truth, weights_only=True, map_location="cpu")["x"].double()
    reference = (reference * reference.new_tensor((.229, .224, .225))[None, :, None, None]
                 + reference.new_tensor((.485, .456, .406))[None, :, None, None]).clamp(0, 1)
    columns = [reference]
    for arm in arms.values():
        pixels = torch.load(arm["path"] / "candidate.pt", weights_only=True, map_location="cpu")["pixels"]
        indices = {row["truth_index"]: row["candidate_index"] for row in arm["evaluation"]["per_image"]}
        columns.append(pixels[[indices[index] for index in range(16)]])
    figure, axes = plt.subplots(16, 5, figsize=(12.5, 40), constrained_layout=True)
    titles = ["Truth", "Cell / 3", "Center / 3", "Cell / 100", "Center / 100"]
    for row in range(16):
        for column in range(5):
            axes[row, column].imshow(columns[column][row].permute(1, 2, 0).numpy())
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(titles[column])
            if column == 0:
                axes[row, column].set_ylabel(f"Truth {row}")
    figure.savefig(output / "all16_by_truth.png", dpi=120)
    plt.close(figure)


def build_report(*, base: Path, truth: Path, output: Path, image_grid: bool = False, lpips: bool = False) -> dict:
    base, truth, output = Path(base).resolve(), Path(truth).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite iteration report: {output}")
    for protected in (base / "observation", truth.parent, *(base / item[0] for item in ARMS.values())):
        if protected == output or protected in output.parents or output in protected.parents:
            raise ReportError("Report must use a separate new directory outside the input artifact trees")
    arms = _completed_arms(base)
    manifest_path = base / "observation/manifest.json"
    manifest, manifest_sha256 = read_json(manifest_path), digest(manifest_path)
    provenance = _check_provenance(arms, manifest, manifest_sha256)
    import torch
    torch.set_num_threads(1)
    _verify_evaluations(arms, truth, manifest_sha256)
    report = {"schema_version": 1, "scope": "same_batch00_B16_T10_three_vs_100_outer_iterations",
              "method": "direct_image_optimization_without_CPA", "comparison_checks": provenance,
              "truth_sha256": digest(truth), "reporter_sha256": digest(Path(__file__)),
              "existing_evaluations_revalidated": True,
              "evaluation_revalidation": {"assignments_and_metadata": "exact", "float_rtol": EVALUATION_RTOL,
                                          "float_atol": EVALUATION_ATOL},
              "arms": {}, "deltas_100_minus_3": {},
              "interpretation": "Single batch, fixed final candidates; no success threshold or statistical superiority established.",
              "log_semantics": "Last logged soft objective is pre-update; saved final candidate and hard-label fit are post-update."}
    for key, arm in arms.items():
        result, evaluation = arm["result"], arm["evaluation"]
        report["arms"][key] = {
            "status": result["status"], "mode": arm["mode"], "iterations": arm["iterations"],
            "artifact_sha256": {name: digest(arm["path"] / name) for name in
                ("registered_design.json", "result.json", "optimization_log.json", "candidate.pt", "evaluation.json")},
            "optimization_time_sec": result["optimization_time_sec"], "wall_time_sec": result["wall_time_sec"],
            "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
            "hard_label_observation_fit": {name: result["hard_label_observation_fit"][name] for name in HARD_FIT_KEYS},
            "final_metrics": {name: evaluation[name] for name in
                ("mean_mse", "mean_psnr_db", "mean_ssim", "label_accuracy", "matching", "n_images")},
            "first_pre_update_soft_objective": arm["logs"][0]["loss"],
            "last_pre_update_soft_objective": arm["logs"][-1]["loss"],
        }
    for mode in ("cell", "center"):
        report["deltas_100_minus_3"][mode] = {
            metric: arms[f"{mode}_100"]["evaluation"][metric] - arms[f"{mode}_3"]["evaluation"][metric]
            for metric in ("mean_mse", "mean_psnr_db", "mean_ssim", "label_accuracy")}
    if lpips:
        report["supplemental_lpips"] = _lpips_supplement(arms, truth)
    lines = ["Same batch00, VGG16 Conv+FC, B16/T10, 4bit observations; only outer iterations increase from 3 to 100.",
             "Fresh identical seeded runs; no pilot optimizer state is resumed. All 16 final images are evaluated.", "",
             "| Arm | Iterations | PSNR (dB) | SSIM | Label accuracy | Hard closure fit | Bin violation rate | Time (s) |",
             "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for key, arm in report["arms"].items():
        metrics, fit = arm["final_metrics"], arm["hard_label_observation_fit"]
        lines.append(f"| {arm['mode']} | {arm['iterations']} | {metrics['mean_psnr_db']:.5f} | {metrics['mean_ssim']:.6f} | "
                     f"{metrics['label_accuracy']:.3f} | {fit['closure_fit_within_tolerance']} | {fit['bin_violation_rate']:.6f} | {arm['wall_time_sec']:.2f} |")
    lines += ["", "The recorded fingerprints match, and the first three numerical logs match each pilot exactly.",
              "Convergence plots normalize each different soft objective by its own initial value; points are pre-update.",
              "Final hard-label fit and offline image metrics refer to the saved post-update candidate.",
              "No reconstruction-success threshold or statistical superiority is established by this single-batch comparison."]
    if lpips:
        lines += ["", "Descriptive VGG LPIPS uses the existing Hungarian pairs and all 16 images; no success cutoff is applied.",
                  "", "| Arm | Mean VGG LPIPS |", "| --- | ---: |"]
        lines += [f"| {key} | {values['mean_lpips']:.6f} |" for key, values in report["supplemental_lpips"]["arms"].items()]
    output.mkdir(parents=True)
    _plots(arms, output)
    if image_grid:
        _image_grid(arms, truth, output)
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["generated_artifact_sha256"] = {path.name: digest(path) for path in output.iterdir() if path.is_file()}
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "truth", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--image-grid", action="store_true", help="Render all sixteen examples using existing evaluation mappings")
    parser.add_argument("--lpips", action="store_true", help="Add descriptive CPU VGG LPIPS; cached weights only, no success threshold")
    args = parser.parse_args(argv)
    build_report(**vars(args))
    print(f"Wrote completed-run comparison to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
