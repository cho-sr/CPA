"""Offline all-16 evaluation of completed identity/flat/MP hybrid arms.

Only this reporting entry point accepts private truth. It never imports an
attack runner, resumes optimization, or selects an intermediate candidate.
All three arms contain the same bin/radius and label-gradient mechanism; their
FIA feature metric differs. Identity is therefore the hybrid identity baseline.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import torch

from .full_update_evaluation import evaluate_candidate
from .full_update_pipeline import denormalize_images, file_digest
from .report_full_update_iterations import _make_lpips_metric


ARMS = ("identity", "flat", "mp")
BINDINGS = ("observation_manifest_sha256", "cpa_target_sha256", "spectrum_npz_sha256",
            "plan_sha256", "common_initial_sha256")
FC_NAMES = {f"classifier.{layer}.{kind}" for layer in (0, 3, 6) for kind in ("weight", "bias")}
METRICS = ("mean_psnr_db", "mean_ssim", "mean_lpips")


class ReportError(ValueError):
    """Completed final artifacts do not satisfy the registered comparison."""


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _completed_arms(base: Path) -> tuple[dict, dict]:
    """Validate every arm and public binding before opening any private file."""
    summary = _read_json(base / "summary.json")
    if (summary.get("status") != "completed" or summary.get("input_and_source_hashes_unchanged") is not True
            or not isinstance(summary.get("arms"), dict) or set(summary["arms"]) != set(ARMS)):
        raise ReportError("All three registered arms must be completed with unchanged input/source hashes")
    if any(not _sha(summary.get(key)) for key in BINDINGS):
        raise ReportError("Summary lacks valid public input and common-start hashes")
    plan = base / "registered_plan.json"
    if not plan.is_file() or file_digest(plan) != summary["plan_sha256"]:
        raise ReportError("Registered plan file does not match its recorded hash")
    common = base / "common_initial.pt"
    if not common.is_file() or file_digest(common) != summary["common_initial_sha256"]:
        raise ReportError("Common initial artifact does not match its recorded hash")
    arms = {}
    for arm in ARMS:
        directory = base / arm
        if (directory / "failure.json").exists():
            raise ReportError(f"Arm {arm} has a failure artifact")
        if any(not (directory / name).is_file() for name in ("result.json", "candidate.pt", "attack_state.pt")):
            raise ReportError(f"Arm {arm} is missing a completed final artifact")
        result = _read_json(directory / "result.json")
        if (result.get("status") != "completed" or result.get("arm") != arm
                or type(result.get("iterations")) is not int or result["iterations"] != 50):
            raise ReportError(f"Arm {arm} must complete its registered 50 iterations")
        if result != summary["arms"][arm]:
            raise ReportError(f"Arm {arm} result disagrees with the completed summary")
        if any(result.get(key) != summary[key] for key in BINDINGS):
            raise ReportError(f"Arm {arm} differs in public inputs, plan or common initialization")
        if (not _sha(result.get("candidate_sha256"))
                or file_digest(directory / "candidate.pt") != result["candidate_sha256"]):
            raise ReportError(f"Arm {arm} candidate file does not match its recorded hash")
        if (result.get("observation_term_enabled") is not True
                or result.get("label_update_rule") != "observation_gradient"):
            raise ReportError(f"Arm {arm} must retain the same bin/radius and label-gradient treatment")
        fits = result.get("fits")
        if not isinstance(fits, dict) or set(fits) != {"soft", "hard"}:
            raise ReportError(f"Arm {arm} lacks final same-candidate soft/hard fits")
        for kind in ("soft", "hard"):
            fit = fits[kind]
            if (not isinstance(fit, dict) or set(fit.get("per_tensor", {})) != FC_NAMES
                    or not isinstance(fit.get("objective_at_same_post_step_candidate"), (int, float))
                    or not math.isfinite(fit["objective_at_same_post_step_candidate"])):
                raise ReportError(f"Arm {arm}/{kind} must retain every FC tensor and a finite final objective")
        arms[arm] = {"path": directory, "result": result}
    return summary, arms


def _load_candidates(arms: dict, manifest_hash: str) -> None:
    for arm, data in arms.items():
        candidate_path = data["path"] / "candidate.pt"
        if file_digest(candidate_path) != data["result"]["candidate_sha256"]:
            raise ReportError(f"Candidate {arm} changed before loading")
        candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
        if not isinstance(candidate, dict) or candidate.get("observation_manifest_sha256") != manifest_hash:
            raise ReportError(f"Candidate {arm} belongs to another observation")
        if any(candidate.get(key) != data["result"][key] for key in BINDINGS):
            raise ReportError(f"Candidate {arm} public input, plan or common-start binding differs from its result")
        pixels, labels = candidate.get("pixels"), candidate.get("labels")
        if (not isinstance(pixels, torch.Tensor) or tuple(pixels.shape) != (16, 3, 224, 224)
                or pixels.dtype not in (torch.float32, torch.float64)
                or not bool(torch.isfinite(pixels).all()) or bool(((pixels < 0) | (pixels > 1)).any())):
            raise ReportError(f"Candidate {arm} must contain all sixteen finite RGB images in [0,1]")
        if (not isinstance(labels, torch.Tensor) or tuple(labels.shape) != (16,)
                or labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
                or bool(((labels.long() < 0) | (labels.long() >= 1000)).any())):
            raise ReportError(f"Candidate {arm} has invalid final hard labels")
        data["candidate"] = candidate


def compare_evaluations(candidate: dict, reference: dict) -> dict:
    """Descriptive signs only; strictly better in all three required metrics.

    The existing evaluator's null PSNR denotes positive infinity for a perfect
    pixel match. Such differences are represented with a note rather than a
    non-JSON infinity or a fabricated finite PSNR.
    """
    deltas, directions, notes = {}, {}, {}
    for metric in METRICS:
        if metric not in candidate or metric not in reference:
            raise ReportError(f"Missing final evaluation metric: {metric}")
        new, old = candidate.get(metric), reference.get(metric)
        if any(not (metric == "mean_psnr_db" and value is None)
               and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value))
               for value in (new, old)):
            raise ReportError(f"Invalid final evaluation metric: {metric}")
        if metric == "mean_psnr_db":
            new_value = math.inf if new is None else new
            old_value = math.inf if old is None else old
        else:
            new_value, old_value = new, old
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or (not math.isfinite(value) and not (metric == "mean_psnr_db" and value == math.inf))
               for value in (new_value, old_value)):
            raise ReportError(f"Invalid final evaluation metric: {metric}")
        if not math.isfinite(new_value) or not math.isfinite(old_value):
            deltas[metric] = None
            notes[metric] = "At least one mean is positive infinity for a perfect pixel match; compare directions directly"
        else:
            deltas[metric] = new_value - old_value
        desired_sign = -1 if metric == "mean_lpips" else 1
        order = (new_value > old_value) - (new_value < old_value)
        directions[metric] = "improved" if order * desired_sign > 0 else "worsened" if order * desired_sign < 0 else "unchanged"
    improved = all(value == "improved" for value in directions.values())
    mixed = "improved" in directions.values() and "worsened" in directions.values()
    return {"differences_candidate_minus_reference": deltas, "metric_directions": directions,
            "primary_metric": "mean_lpips_lower_is_better",
            "interpretation": "directional_improvement_in_this_pilot" if improved else "mixed" if mixed else "no_established_improvement",
            "all_three_metrics_strictly_improved": improved,
            "statistical_significance": "not_tested", "success_threshold": None,
            "magnitude_and_robustness": "Sign-only description; even three favorable tiny differences do not establish practical or robust improvement",
            "notes": notes}


def _all16_grid(outputs: dict, reference: torch.Tensor, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(4, 16, figsize=(27, 7.4))
    names = ("reference", "identity", "flat", "mp")
    for truth_index in range(16):
        row, start = truth_index // 4, (truth_index % 4) * 4
        axes[row, start].imshow(reference[truth_index].permute(1, 2, 0))
        for offset, arm in enumerate(ARMS, 1):
            data = outputs[arm]
            pair = next(item for item in data["evaluation"]["per_image"] if item["truth_index"] == truth_index)
            axes[row, start + offset].imshow(data["candidate"]["pixels"][pair["candidate_index"]].permute(1, 2, 0))
        for offset, name in enumerate(names):
            axes[row, start + offset].set_title(f"{truth_index + 1}: {name}", fontsize=7)
            axes[row, start + offset].axis("off")
    figure.suptitle("Final all-16 comparison: identity, flat and MP all include the same hybrid bin/radius and label gradients", fontsize=11)
    figure.tight_layout(pad=.4, rect=(0, 0, 1, .96))
    figure.savefig(output / "all16_comparison.png", dpi=120)
    plt.close(figure)


def _number(value, digits=6) -> str:
    return "∞" if value is None else f"{value:.{digits}f}"


def _report_text(outputs: dict, comparisons: dict) -> str:
    lines = ["Completed MP feature-metric comparison: frozen Conv, all FC learn, deterministic uint4, single client, B16/T10.",
             "Identity, flat and MP all use the same hybrid bin/radius objective and private-label update rule. Only the FIA metric treatment differs.",
             "All arms start from the same saved public initialization and run 50 iterations. No additional candidate or iteration is selected using truth.",
             "Identity uses lambda_F=1. Flat's initial image-gradient norm determines one fixed lambda_F shared by flat and MP. Only flat is calibrated to identity's initial feature-gradient norm; MP's remaining norm difference is measured, not independently removed.", "",
             "| Hybrid metric | PSNR dB ↑ | SSIM ↑ | VGG LPIPS ↓ | Matched label accuracy | Label overlap /16 | Optimization seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, data in outputs.items():
        ev, run = data["evaluation"], data["result"]
        lines.append(f"| {arm} | {_number(ev['mean_psnr_db'], 5)} | {_number(ev['mean_ssim'])} | {_number(ev['mean_lpips'])} | {ev['label_accuracy']:.4f} | {ev['label_multiset_overlap_count']} | {run['optimization_time_sec']:.2f} |")
    lines += ["", "| Comparison | Δ PSNR | Δ SSIM | Δ LPIPS | Descriptive conclusion |",
              "| --- | ---: | ---: | ---: | --- |"]
    for name, comparison in comparisons.items():
        delta = comparison["differences_candidate_minus_reference"]
        values = ["undefined (infinite PSNR)" if delta[key] is None else f"{delta[key]:+.8f}" for key in METRICS]
        lines.append(f"| {name} | {' | '.join(values)} | {comparison['interpretation']} |")
    lines += ["", "| Hybrid metric | Final soft objective | Final hard objective | Hard positive-radius bin violation rate | Exact codes | Exact radius | Exact packet | Closure fit |",
              "| --- | ---: | ---: | ---: | --- | --- | --- | --- |"]
    for arm, data in outputs.items():
        soft, hard = data["result"]["fits"]["soft"], data["result"]["fits"]["hard"]
        lines.append(f"| {arm} | {soft['objective_at_same_post_step_candidate']:.6f} | {hard['objective_at_same_post_step_candidate']:.6f} | {hard['bin_violation_rate']:.6%} | {hard['exact_code_match']} | {hard['exact_radius_match']} | {hard['exact_packet_match']} | {hard['closure_fit_within_tolerance']} |")
    lines += ["", "Image matching: one pixel-MSE Hungarian assignment per arm, fixed for PSNR, SSIM, LPIPS and label accuracy. Every final image and all six FC observation tensors remain included.",
              "Directional improvement is reported only when LPIPS decreases and both PSNR and SSIM increase. Mixed signs do not establish improvement. These are descriptive results from one fixed pilot, without statistical significance or a reconstruction-success cutoff.",
              "The directional label reports signs only. Small favorable differences, even in all three metrics, do not establish a practically meaningful or robust improvement.",
              "Final soft and hard objectives use each arm's same saved post-step images. They are distinct from pre-step optimization logs. Changing soft probabilities can matter even when hard-label argmax stays unchanged.",
              "The MP weights remain a public spectral heuristic; reconstruction differences do not validate an iid-noise model or establish superiority over other spectral methods.",
              "", "![All sixteen final images with offline references](all16_comparison.png)"]
    return "\n".join(lines) + "\n"


def report(base: Path, truth: Path, output: Path) -> dict:
    base, truth, output = Path(base).resolve(), Path(truth).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output == base or output in base.parents or output == truth or output in truth.parents:
        raise ReportError("Report output must not overlap or replace its input artifacts")
    if any(base / arm == output or base / arm in output.parents for arm in ARMS):
        raise ReportError("Report must not be written inside an existing arm directory")
    started = time.monotonic()
    summary, outputs = _completed_arms(base)
    tracked = {"summary": base / "summary.json", "common_initial": base / "common_initial.pt",
               "registered_plan": base / "registered_plan.json",
               **{f"{arm}/{name}": base / arm / name for arm in ARMS
                  for name in ("result.json", "candidate.pt", "attack_state.pt")},
               "report_source": Path(__file__),
               "evaluation_source": Path(__file__).parent / "full_update_evaluation.py",
               "lpips_loader_source": Path(__file__).parent / "report_full_update_iterations.py"}
    before = {name: file_digest(path) for name, path in tracked.items()}
    _load_candidates(outputs, summary["observation_manifest_sha256"])
    # This is the first access to private truth, after all public completion and
    # candidate-shape checks have succeeded. Evaluation never feeds the attack.
    truth_hash = file_digest(truth)
    for arm, data in outputs.items():
        evaluation = evaluate_candidate(data["path"] / "candidate.pt", truth)
        if (evaluation.get("n_images") != 16 or evaluation.get("all_images_included") is not True
                or sorted(row["candidate_index"] for row in evaluation["per_image"]) != list(range(16))
                or sorted(row["truth_index"] for row in evaluation["per_image"]) != list(range(16))):
            raise ReportError(f"Evaluation {arm} did not retain every final image exactly once")
        data["evaluation"] = evaluation
    private = torch.load(truth, map_location="cpu", weights_only=True)
    reference = denormalize_images(private["x"].float()).clamp(0, 1)
    metric, lpips_provenance = _make_lpips_metric()
    with torch.no_grad():
        for arm, data in outputs.items():
            values = []
            pixels = data["candidate"]["pixels"].float()
            for pair in data["evaluation"]["per_image"]:
                i, j = pair["candidate_index"], pair["truth_index"]
                metric.reset()
                value = metric((pixels[i:i + 1] - .5) * 2, (reference[j:j + 1] - .5) * 2)
                if value.numel() != 1 or not bool(torch.isfinite(value).all()):
                    raise ReportError(f"Invalid LPIPS for {arm}")
                pair["lpips"] = float(value)
                values.append(float(value))
            data["evaluation"]["mean_lpips"] = sum(values) / 16
            data["evaluation"]["lpips"] = "vgg_same_MSE_Hungarian_pairs"
            overlap = Counter(data["candidate"]["labels"].tolist()) & Counter(private["labels"].tolist())
            data["evaluation"]["label_multiset_overlap_count"] = sum(overlap.values())
    del metric
    comparisons = {f"{a}_minus_{b}": compare_evaluations(outputs[a]["evaluation"], outputs[b]["evaluation"])
                   for a, b in (("mp", "identity"), ("flat", "identity"), ("mp", "flat"))}
    after = {name: file_digest(path) for name, path in tracked.items()}
    if before != after or file_digest(truth) != truth_hash:
        raise ReportError("A final artifact, evaluation source or private reference changed during reporting")
    output.mkdir(parents=True, exist_ok=False)
    _all16_grid(outputs, reference, output)
    text = _report_text(outputs, comparisons)
    serializable_arms = {arm: {"result": data["result"], "evaluation": data["evaluation"],
                               "candidate_sha256": before[f"{arm}/candidate.pt"]}
                         for arm, data in outputs.items()}
    result = {"status": "completed", "arms": serializable_arms, "comparisons": comparisons,
              "lpips_provenance": lpips_provenance, "truth_file_sha256": truth_hash,
              "public_bindings": {key: summary[key] for key in BINDINGS},
              "hashes_before": before, "hashes_after": after, "input_and_source_hashes_unchanged": True,
              "all_images_included": True, "success_threshold": None,
              "runtime_seconds": time.monotonic() - started,
              "runtime_scope": "Completion validation, offline all-image metrics, LPIPS, hash checks and image grid; excludes final JSON/markdown writes",
              "decision_rule": "LPIPS strictly decreases and PSNR/SSIM strictly increase; descriptive single-pilot direction only"}
    with (output / "comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    (output / "report.md").write_text(text, encoding="utf-8")
    print(json.dumps({"status": "completed", "comparisons": comparisons, "output": str(output)}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    report(args.base, args.truth, args.output)


if __name__ == "__main__":
    main()
