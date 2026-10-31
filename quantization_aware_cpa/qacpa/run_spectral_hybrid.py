"""Fixed public-input three-arm spectral FIA comparison within the FC hybrid."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from .deterministic_hybrid_setup import configure_runtime, read_observation
from .deterministic_observation import observation_loss, summarize_observation_fit
from .fc_replay import fc_recomputed_gradients, fc_sgd_updates
from .full_update_pipeline import file_digest, load_starting_vgg16, normalize_pixels, total_variation, write_json
from .run_deterministic_hybrid import balanced_weight, cosine_loss, cpu_tree
from .spectral_fia_metric import metric_weights, spectral_cosine_loss


ARMS = ("identity", "flat", "mp")


def validate_plan(plan):
    required = {
        "format": "spectral_hybrid_three_arm_v1", "model": "vgg16",
        "training_scope": "frozen_conv_all_fc_train", "observation_scope": "all_fc_weights_and_biases",
        "quantizer": "affine_uint4_16levels_nearest_even_with_public_fp32_row_radius",
        "batch": 16, "local_steps": 10, "client_sgd_lr": .001, "seed": 42,
        "branch_iterations": 50, "image_lr": .01, "label_lr": .03, "lr_final_fraction": .1,
        "epsilon": .1, "rank": 16, "partition": "positive_all", "arms": list(ARMS),
        "lambda_TV": 1.1169759484351267, "lambda_q_and_radius": .0010545856074191738,
        "label_update_rule": "observation_gradient", "all_arms_use_bin_and_radius": True,
    }
    descriptions = {"user_request", "weight_rule", "common_initialization", "lambda_F_rule",
                    "evaluation", "decision_rule", "budget_rule", "baseline_reproduction"}
    hashes = {"expected_observation_manifest_sha256", "expected_cpa_target_sha256",
              "expected_checkpoint_sha256", "expected_spectrum_plan_sha256",
              "expected_common_initial_sha256", "expected_previous_calibration_sha256",
              "expected_previous_registration_sha256", "expected_spectrum_npz_sha256",
              "expected_spectrum_report_sha256"}
    if not isinstance(plan, dict) or set(plan) != set(required) | descriptions | hashes:
        raise ValueError("Unexpected or missing registered plan fields")
    for key, value in required.items():
        if type(plan[key]) is not type(value) or plan[key] != value:
            raise ValueError(f"Registered setting changed: {key}")
    if any(not isinstance(plan[k], str) or not plan[k].strip() for k in descriptions):
        raise ValueError("Missing plan description")
    if any(not isinstance(plan[k], str) or len(plan[k]) != 64 or
           any(c not in "0123456789abcdef" for c in plan[k]) for k in hashes):
        raise ValueError("Invalid expected input digest")


def require_new_output(output, protected):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    for source in protected:
        source = Path(source).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Output overlaps a protected input")


def run_attack(*, previous: Path, checkpoint: Path, spectrum: Path,
               plan_path: Path, output: Path):
    """Accept public artifacts only. Truth is exclusively an offline-report input."""
    previous, spectrum, checkpoint, plan_path, output = [Path(p).resolve() for p in
        (previous, spectrum, checkpoint, plan_path, output)]
    require_new_output(output, (previous, spectrum, checkpoint, plan_path))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    observation = previous / "observation"
    inputs = {"plan": plan_path, "observation_manifest": observation / "manifest.json",
              "cpa_target": previous / "cpa/target.pt", "checkpoint": checkpoint,
              "common_initial": previous / "attack/common_initial.pt",
              "previous_calibration": previous / "attack/calibration.json",
              "previous_registration": previous / "attack/registered_run.json",
              "spectrum_npz": spectrum / "positive_all.npz", "spectrum_report": spectrum / "report.json"}
    for key in tuple(inputs):
        expected = plan.get("expected_" + key + "_sha256")
        if expected is not None and file_digest(inputs[key]) != expected:
            raise ValueError("Input hash mismatch: " + key)
    prior_run = json.loads(inputs["previous_registration"].read_text(encoding="utf-8"))
    old_calibration = json.loads(inputs["previous_calibration"].read_text(encoding="utf-8"))
    for name, expected in prior_run["source_sha256"].items():
        path = Path(__file__).parent / name
        if file_digest(path) != expected:
            raise ValueError("Original source changed: " + name)
        inputs["source/" + name] = path
    for name in ("run_spectral_hybrid.py", "spectral_fia_metric.py"):
        inputs["source/" + name] = Path(__file__).parent / name
    for path in sorted(observation.glob("packet_*.bin")):
        inputs["packet/" + path.name] = path
    configure_runtime(plan["seed"])
    manifest, observations = read_observation(observation, checkpoint)
    spectrum_report = json.loads(inputs["spectrum_report"].read_text(encoding="utf-8"))
    if (spectrum_report["status"] != "completed" or not spectrum_report["input_and_source_hashes_unchanged"]
            or spectrum_report["input_manifest_sha256"] != plan["expected_observation_manifest_sha256"]
            or spectrum_report["hashes_before"]["plan"] != plan["expected_spectrum_plan_sha256"]):
        raise ValueError("Spectrum provenance mismatch")
    partition = spectrum_report["partitions"]["positive_all"]
    mp_fit = partition["mp_fit"]
    if mp_fit["status"] != "OK":
        raise ValueError("No finite registered MP fit")
    with np.load(inputs["spectrum_npz"], allow_pickle=False) as arrays:
        eigenvalues = arrays["eigenvalues_desc"].copy()
        vectors_array = arrays["right_vectors"].copy()
    raw_report = np.asarray(mp_fit["raw_eigenvalues_desc"], dtype=np.float64)
    if (eigenvalues.shape != (4002,) or vectors_array.shape != (25088, 16)
            or not np.array_equal(eigenvalues, raw_report) or not np.isfinite(vectors_array).all()):
        raise ValueError("Saved spectrum arrays disagree with report or dimensions")
    if not np.allclose(vectors_array.T @ vectors_array, np.eye(16), atol=1e-10, rtol=0):
        raise ValueError("Saved feature directions are not orthonormal")
    eig = torch.from_numpy(eigenvalues[:16].copy())
    weights64 = metric_weights(eig, mp_fit["edge"])
    scale_floor = float(torch.cat([obs.radius[obs.radius > 0].double() * (2/15)
                                 for obs in observations.values()]).median())
    if scale_floor != old_calibration["scale_floor"]:
        raise ValueError("Public scale floor differs from original calibration")
    for key in ("lambda_TV", "lambda_q_and_radius"):
        if plan[key] != old_calibration[key]:
            raise ValueError("Changed original hybrid loss weight")
    target_payload = torch.load(inputs["cpa_target"], map_location="cpu", weights_only=True)
    common = torch.load(inputs["common_initial"], map_location="cpu", weights_only=True)
    if target_payload["observation_manifest_sha256"] != plan["expected_observation_manifest_sha256"]:
        raise ValueError("CPA target binding mismatch")
    for name, tensor, shape in (("target", target_payload["target_z"], (16,25088)),
                                ("pixels", common["pixels"], (16,3,224,224)),
                                ("logits", common["label_logits"], (16,1000))):
        if tensor.shape != shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError("Invalid public " + name)
    if (common["pixels"] < 0).any() or (common["pixels"] > 1).any() or (target_payload["target_z"].norm(dim=1) == 0).any():
        raise ValueError("Invalid initial pixels or zero target")
    hashes_before = {key: file_digest(path) for key, path in inputs.items()}
    binding = {"plan_sha256": hashes_before["plan"], "common_initial_sha256": hashes_before["common_initial"],
               "observation_manifest_sha256": hashes_before["observation_manifest"],
               "cpa_target_sha256": hashes_before["cpa_target"], "spectrum_npz_sha256": hashes_before["spectrum_npz"]}
    output.mkdir(parents=True)
    shutil.copyfile(inputs["common_initial"], output / "common_initial.pt")
    shutil.copyfile(plan_path, output / "registered_plan.json")
    provenance = {"plan": plan, **binding, "hashes_before": hashes_before,
                  "public_inputs_only": True, "input_paths": {k: str(p) for k,p in inputs.items()},
                  "log_timing": "losses/gradients pre-step; displacement post-step; final fits use saved post-step candidates"}
    write_json(output / "registered_run.json", provenance)
    dev = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Registered comparison requires CUDA")
    model = load_starting_vgg16(checkpoint, dev)
    model.features.requires_grad_(False)
    model.eval()
    target = target_payload["target_z"].to(dev)
    common_pixels = common["pixels"].to(dev)
    common_logits = common["label_logits"].to(dev)
    vectors = torch.from_numpy(vectors_array).to(device=dev, dtype=torch.float32)
    weights = weights64.to(device=dev, dtype=torch.float32)
    flat_weights = weights.mean().expand_as(weights)
    lambda_tv, lambda_q = plan["lambda_TV"], plan["lambda_q_and_radius"]

    def features(pixels):
        return model.avgpool(model.features(normalize_pixels(pixels))).flatten(1)

    def fia_loss(z, arm):
        if arm == "identity":
            return cosine_loss(z, target)
        return spectral_cosine_loss(z, target, vectors, flat_weights if arm == "flat" else weights, plan["epsilon"])

    def update_loss(updates):
        return observation_loss(updates, observations, radius_weight=1., scale_floor=scale_floor)

    calibration = {"scale_floor": scale_floor, "lambda_TV": lambda_tv, "lambda_q_and_radius": lambda_q,
                   "epsilon": plan["epsilon"], "weights_float64": weights64.tolist(),
                   "weights_float32": weights.cpu().tolist(), "flat_weight_float32": float(flat_weights[0]),
                   "edge": mp_fit["edge"], "eigenvalues_top16": eig.tolist(), "arms": {}}
    initial_gradients = {}
    for arm in ARMS:
        pixels = common_pixels.clone().requires_grad_(True)
        loss = fia_loss(features(pixels), arm)
        gradient, = torch.autograd.grad(loss, pixels)
        initial_gradients[arm] = gradient.detach()
        calibration["arms"][arm] = {"initial_raw_feature_loss": float(loss.detach()),
                                    "initial_raw_FIA_image_gradient_norm": float(gradient.norm())}
        del pixels, loss, gradient
    shared_spectral_weight = balanced_weight(initial_gradients["identity"], initial_gradients["flat"], 1.)
    # B/C use the same scalar, so their only objective difference is the metric.
    # Their initial gradients are reported separately, not asserted equal.
    lambda_f = {"identity": 1., "flat": shared_spectral_weight, "mp": shared_spectral_weight}
    for arm in ARMS:
        calibration["arms"][arm]["lambda_F"] = lambda_f[arm]
        calibration["arms"][arm]["initial_scaled_FIA_image_gradient_norm"] = float((lambda_f[arm]*initial_gradients[arm]).norm())
    calibration["baseline_gradient_norm_relative_difference_from_original"] = abs(
        float(initial_gradients["identity"].norm()) / old_calibration["initial_FIA_gradient_norm"] - 1)
    if calibration["baseline_gradient_norm_relative_difference_from_original"] > 1e-5:
        raise ValueError("Initial identity gradient differs from original public calibration")
    del initial_gradients
    write_json(output / "calibration.json", calibration)
    print(json.dumps({"stage": "calibration", **calibration}), flush=True)
    torch.cuda.empty_cache()
    results = {}
    for arm in ARMS:
        configure_runtime(plan["seed"])
        folder = output / arm
        folder.mkdir()
        pixels = torch.nn.Parameter(common_pixels.clone())
        logits = torch.nn.Parameter(common_logits.clone())
        optimizer = torch.optim.Adam([{"params": [pixels], "lr": plan["image_lr"]},
                                      {"params": [logits], "lr": plan["label_lr"]}])
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)
        started = time.perf_counter()
        logs = []
        for step in range(plan["branch_iterations"]):
            optimizer.zero_grad(set_to_none=True)
            z = features(pixels)
            raw_f, tv = fia_loss(z, arm), total_variation(pixels)
            raw_gf, = torch.autograd.grad(raw_f, pixels, retain_graph=True)
            gt, = torch.autograd.grad(tv, pixels, retain_graph=True)
            probabilities = logits.softmax(1)
            q_loss, dz, dp = fc_recomputed_gradients(model.classifier, z.detach(), probabilities.detach(),
                                                     10, .001, update_loss)
            gq, = torch.autograd.grad(z, pixels, grad_outputs=dz)
            gf = lambda_f[arm] * raw_gf
            gradient = gf + lambda_tv*gt + lambda_q*gq
            detached_p = probabilities.detach()
            logits.grad = lambda_q*detached_p*(dp-(detached_p*dp).sum(1, keepdim=True))
            if not torch.isfinite(gradient).all() or not torch.isfinite(logits.grad).all():
                raise ValueError("Nonfinite reconstruction gradient")
            pixels.grad = gradient
            before = pixels.detach().clone()
            decay = plan["lr_final_fraction"] + (1-plan["lr_final_fraction"])*(
                1+math.cos(math.pi*step/max(1,plan["branch_iterations"]-1)))/2
            optimizer.param_groups[0]["lr"] = plan["image_lr"]*decay
            optimizer.param_groups[1]["lr"] = plan["label_lr"]*decay
            optimizer.step()
            with torch.no_grad():
                clamp_fraction = float(((pixels<0)|(pixels>1)).float().mean())
                pixels.clamp_(0,1)
                displacement = float((pixels-before).norm())
                entropy = float(-(detached_p*detached_p.clamp_min(1e-30).log()).sum(1).mean())
            entry = {"iteration": step+1, "feature_loss": float(raw_f.detach()),
                     "scaled_feature_loss": lambda_f[arm]*float(raw_f.detach()),
                     "tv": float(tv.detach()), "observation_loss": float(q_loss),
                     "total_objective": lambda_f[arm]*float(raw_f.detach())+lambda_tv*float(tv.detach())+lambda_q*float(q_loss),
                     "raw_FIA_image_gradient_norm": float(raw_gf.norm()), "FIA_image_gradient_norm": float(gf.norm()),
                     "weighted_TV_image_gradient_norm": lambda_tv*float(gt.norm()),
                     "weighted_observation_image_gradient_norm": lambda_q*float(gq.norm()),
                     "combined_image_gradient_norm": float(gradient.norm()), "actual_pixel_step_l2": displacement,
                     "clamp_fraction": clamp_fraction, "label_entropy": entropy,
                     "image_lr": plan["image_lr"]*decay, "elapsed_sec": time.perf_counter()-started}
            logs.append(entry)
            del z, raw_f, tv, raw_gf, gt, probabilities, q_loss, dz, dp, gq, gf, gradient, detached_p, before
            if step == 0 or (step+1) % 10 == 0:
                write_json(folder / "optimization_log.json", logs)
                print(json.dumps({"stage": arm, **entry}), flush=True)
        torch.cuda.synchronize(dev)
        optimization_seconds = time.perf_counter()-started
        torch.save({"pixels": pixels.detach().cpu(), "labels": logits.detach().argmax(1).cpu(), **binding}, folder / "candidate.pt")
        torch.save({"pixels": pixels.detach().cpu(), "label_logits": logits.detach().cpu(),
                    "optimizer": cpu_tree(optimizer.state_dict()), "completed_iterations": len(logs),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "calibration": calibration, **binding}, folder / "attack_state.pt")
        with torch.no_grad():
            final_z = features(pixels.detach())
            final_identity_loss = float(cosine_loss(final_z, target))
            final_metric_loss = float(fia_loss(final_z, arm))
        fits = {}
        for label_kind, labels in (("soft",logits.detach().softmax(1)),("hard",logits.detach().argmax(1))):
            updates = fc_sgd_updates(model.classifier, final_z, labels, steps=10, lr=.001, create_graph=False)
            fit = summarize_observation_fit(updates, observations, tolerance_steps=1e-6)
            fit["objective_at_same_post_step_candidate"] = float(update_loss(updates))
            fits[label_kind] = fit
            del updates
        result = {"status": "completed", "arm": arm, "iterations": len(logs), **binding,
                  "observation_term_enabled": True, "label_update_rule": "observation_gradient",
                  "final_feature_loss": final_identity_loss, "final_metric_feature_loss": final_metric_loss,
                  "metric": {"epsilon": 1. if arm=="identity" else plan["epsilon"],
                             "lambda_F":lambda_f[arm], "partition":"positive_all", "rank":16},
                  "fits": fits, "optimization_time_sec": optimization_seconds,
                  "wall_time_sec": time.perf_counter()-started,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(dev),
                  "candidate_sha256":file_digest(folder / "candidate.pt"),
                  "reconstruction_success": "not_evaluated_no_threshold"}
        write_json(folder / "result.json", result)
        results[arm] = result
        print(json.dumps({"stage":"arm_complete","arm":arm,"seconds":result["wall_time_sec"]}),flush=True)
        del optimizer, pixels, logits, final_z
        torch.cuda.empty_cache()
    hashes_after = {key:file_digest(path) for key,path in inputs.items()}
    unchanged = hashes_before == hashes_after
    summary = {"status":"completed" if unchanged else "invalid_inputs_changed", "arms":results,
               **binding, "calibration":calibration, "input_and_source_hashes_unchanged":unchanged,
               "hashes_before":hashes_before, "hashes_after":hashes_after,
               "torch":torch.__version__, "cuda":torch.version.cuda, "gpu":torch.cuda.get_device_name(dev),
               "interpretation":"one fixed 50-step three-arm hybrid comparison; offline evaluation separate"}
    write_json(output / "summary.json",summary)
    if not unchanged:
        raise RuntimeError("Input/source changed during comparison; evaluation prohibited")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("previous","checkpoint","spectrum","plan","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    run_attack(previous=args.previous,checkpoint=args.checkpoint,spectrum=args.spectrum,
               plan_path=args.plan,output=args.output)


if __name__ == "__main__":
    main()
