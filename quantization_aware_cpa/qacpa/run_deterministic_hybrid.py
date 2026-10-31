"""Public-only frozen-Conv CPA/FIA pilot; evaluation is a separate command."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F

from .full_update_pipeline import (file_digest, load_starting_vgg16,
                                   normalize_pixels, total_variation, write_json)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def cosine_loss(features, target):
    return (1 - F.cosine_similarity(features, target, dim=1, eps=1e-8)).mean()


def balanced_weight(reference_gradient, auxiliary_gradient, ratio):
    reference = float(reference_gradient.norm())
    auxiliary = float(auxiliary_gradient.norm())
    if not (math.isfinite(reference) and reference > 0 and
            math.isfinite(auxiliary) and auxiliary > 0):
        raise ValueError("Gradient calibration requires finite, nonzero gradients")
    return ratio * reference / auxiliary


def run_attack(*, observation: Path, checkpoint: Path, cpa_target: Path,
               plan_path: Path, output: Path, device: str = "cuda"):
    # No true images, true labels, raw updates, or private file path is accepted.
    from .deterministic_hybrid_setup import configure_runtime, read_observation
    from .deterministic_observation import observation_loss, summarize_observation_fit
    from .fc_replay import fc_sgd_updates, fc_recomputed_gradients

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(output)
    configure_runtime(plan["seed"])
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError("This registered VGG pilot requires a CUDA device")
    manifest, observations = read_observation(observation, checkpoint)
    if (plan["batch"] != 16 or plan["local_steps"] != 10 or plan["client_sgd_lr"] != .001
            or manifest["image_shape"] != [16, 3, 224, 224]
            or manifest["training"]["steps"] != 10 or manifest["training"]["lr"] != .001):
        raise ValueError("Pilot plan and observation must agree on B16/T10/SGD0.001")
    manifest_hash = file_digest(observation / "manifest.json")
    payload = torch.load(cpa_target, map_location="cpu", weights_only=True)
    if payload["observation_manifest_sha256"] != manifest_hash:
        raise ValueError("CPA target belongs to another observation")
    target = payload["target_z"].to(dev)
    if target.shape != (16, 25088) or not torch.isfinite(target).all():
        raise ValueError("Invalid public CPA target")
    if (target.norm(dim=1) == 0).any():
        raise ValueError("Cannot use zero CPA components as cosine targets")
    model = load_starting_vgg16(checkpoint, dev)
    for parameter in model.features.parameters():
        parameter.requires_grad_(False)
    model.eval()

    def features(pixels):
        return model.avgpool(model.features(normalize_pixels(pixels))).flatten(1)

    output.mkdir(parents=True)
    provenance = {"plan": plan, "plan_sha256": file_digest(plan_path),
                  "observation_manifest_sha256": manifest_hash,
                  "cpa_target_sha256": file_digest(cpa_target),
                  "checkpoint_sha256": file_digest(checkpoint),
                  "torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(dev),
                  "log_timing": "losses, gradients and entropy are pre-step; displacement and clamp are post-step; final fits use the saved post-step pixels/logits",
                  "source_sha256": {name: file_digest(Path(__file__).parent / name) for name in (
                      "run_deterministic_hybrid.py", "deterministic_hybrid_setup.py",
                      "deterministic_observation.py", "fc_replay.py",
                      "full_update_replay.py", "full_update_adjoint.py", "full_update_pipeline.py")}}
    write_json(output / "registered_run.json", provenance)
    generator = torch.Generator(device=dev).manual_seed(plan["seed"])
    pixels = torch.nn.Parameter(torch.rand((16, 3, 224, 224), generator=generator, device=dev))
    optimizer = torch.optim.Adam([pixels], lr=plan["warmup_image_lr"])
    warmup_log = []
    started = time.perf_counter()
    for step in range(plan["warmup_iterations"]):
        optimizer.zero_grad(set_to_none=True)
        z = features(pixels)
        feature_loss, tv = cosine_loss(z, target), total_variation(pixels)
        if step == 0:
            gf, = torch.autograd.grad(feature_loss, pixels, retain_graph=True)
            gt, = torch.autograd.grad(tv, pixels, retain_graph=True)
            warmup_tv = balanced_weight(gf, gt, plan["tv_gradient_ratio"])
            del gf, gt
        loss = feature_loss + warmup_tv * tv
        gradient, = torch.autograd.grad(loss, pixels)
        if not torch.isfinite(gradient).all():
            raise ValueError("Nonfinite warmup gradient")
        pixels.grad = gradient
        fraction = step / max(1, plan["warmup_iterations"] - 1)
        optimizer.param_groups[0]["lr"] = plan["warmup_image_lr"] * (
            plan["lr_final_fraction"] + (1-plan["lr_final_fraction"]) * (1+math.cos(math.pi*fraction))/2)
        optimizer.step()
        with torch.no_grad():
            pixels.clamp_(0, 1)
        entry = {"iteration": step+1, "feature_loss": float(feature_loss.detach()),
                 "tv": float(tv.detach()), "elapsed_sec": time.perf_counter()-started}
        warmup_log.append(entry)
        del z, feature_loss, tv, loss, gradient
        if (step+1) % 100 == 0 or step == 0:
            print(json.dumps({"stage": "common_fia_warmup", **entry}), flush=True)
    common_pixels = pixels.detach().clone()
    with torch.no_grad():
        # Shared, public model prediction is an estimate, never a true label.
        common_logits = model.classifier(features(common_pixels)).detach()
    torch.save({"pixels": common_pixels.cpu(), "label_logits": common_logits.cpu(),
                "labels": common_logits.argmax(1).cpu()}, output / "common_initial.pt")
    write_json(output / "warmup_log.json", warmup_log)
    warmup_seconds = time.perf_counter()-started
    del optimizer, pixels

    positive_steps = torch.cat([obs.radius[obs.radius > 0].double() * (2/15)
                                for obs in observations.values()])
    scale_floor = float(positive_steps.median())

    def update_loss(updates):
        return observation_loss(updates, observations, radius_weight=1., scale_floor=scale_floor)

    # One public-only calibration, then the same fixed weights in both arms.
    pixels = common_pixels.clone().requires_grad_(True)
    z = features(pixels)
    f_loss, tv = cosine_loss(z, target), total_variation(pixels)
    gf, = torch.autograd.grad(f_loss, pixels, retain_graph=True)
    gt, = torch.autograd.grad(tv, pixels, retain_graph=True)
    q_loss, dz, dp = fc_recomputed_gradients(model.classifier, z.detach(),
        common_logits.softmax(1), 10, .001, update_loss)
    gq, = torch.autograd.grad(z, pixels, grad_outputs=dz)
    lambda_tv = balanced_weight(gf, gt, plan["tv_gradient_ratio"])
    lambda_q = balanced_weight(gf, gq, plan["observation_gradient_ratio"])
    calibration = {"method": "initial_public_image_gradient_norm_ratio_fixed_after_calibration",
                   "scale_floor": scale_floor, "floor_rule": "median_positive_public_quantization_step",
                   "lambda_F": 1., "lambda_TV": lambda_tv, "lambda_q_and_radius": lambda_q,
                   "warmup_lambda_TV": warmup_tv, "initial_FIA_gradient_norm": float(gf.norm()),
                   "initial_observation_gradient_norm": float(gq.norm()),
                   "initial_TV_gradient_norm": float(gt.norm()), "initial_observation_loss": float(q_loss)}
    write_json(output / "calibration.json", calibration)
    print(json.dumps({"stage": "calibration", **calibration}), flush=True)
    del pixels, z, f_loss, tv, gf, gt, gq, q_loss, dz, dp
    torch.cuda.empty_cache()

    results = {}
    for arm, use_observation in (("fia", False), ("hybrid", True)):
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
            f_loss, tv = cosine_loss(z, target), total_variation(pixels)
            gf, = torch.autograd.grad(f_loss, pixels, retain_graph=True)
            gt, = torch.autograd.grad(tv, pixels, retain_graph=True)
            raw_q_norm, q_value, weighted_q_norm = None, None, 0.
            probabilities = logits.softmax(1)
            if use_observation:
                q_loss, dz, dp = fc_recomputed_gradients(model.classifier, z.detach(),
                    probabilities.detach(), 10, .001, update_loss)
                gq, = torch.autograd.grad(z, pixels, grad_outputs=dz)
                raw_q_norm, q_value = float(gq.norm()), float(q_loss)
                weighted_q_norm = lambda_q * raw_q_norm
                gradient = gf + lambda_tv*gt + lambda_q*gq
                detached_p = probabilities.detach()
                logits.grad = lambda_q * detached_p * (dp-(detached_p*dp).sum(1, keepdim=True))
                del q_loss, dz, dp, gq, detached_p
            else:
                gradient = gf + lambda_tv*gt
            if not torch.isfinite(gradient).all() or (logits.grad is not None and not torch.isfinite(logits.grad).all()):
                raise ValueError("Nonfinite pilot gradient")
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
                entropy = float(-(probabilities.detach()*probabilities.detach().clamp_min(1e-30).log()).sum(1).mean())
            entry = {"iteration": step+1, "feature_loss": float(f_loss.detach()),
                     "tv": float(tv.detach()), "observation_loss": q_value,
                     "FIA_image_gradient_norm": float(gf.norm()),
                     "weighted_TV_image_gradient_norm": lambda_tv*float(gt.norm()),
                     "weighted_observation_image_gradient_norm": weighted_q_norm,
                     "raw_observation_image_gradient_norm": raw_q_norm,
                     "combined_image_gradient_norm": float(gradient.norm()),
                     "actual_pixel_step_l2": displacement, "clamp_fraction": clamp_fraction,
                     "label_entropy": entropy, "image_lr": plan["image_lr"]*decay,
                     "elapsed_sec": time.perf_counter()-started}
            logs.append(entry)
            del z, f_loss, tv, gf, gt, gradient, probabilities, before
            if step == 0 or (step+1) % 10 == 0 or step+1 == plan["branch_iterations"]:
                write_json(folder / "optimization_log.json", logs)
                print(json.dumps({"stage": arm, **entry}), flush=True)
        torch.cuda.synchronize(dev)
        optimization_seconds = time.perf_counter()-started
        torch.save({"pixels": pixels.detach().cpu(), "labels": logits.detach().argmax(1).cpu(),
                    "observation_manifest_sha256": manifest_hash}, folder / "candidate.pt")
        torch.save({"pixels": pixels.detach().cpu(), "label_logits": logits.detach().cpu(),
                    "optimizer": cpu_tree(optimizer.state_dict()), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(), "completed_iterations": len(logs),
                    "observation_manifest_sha256": manifest_hash,
                    "plan_sha256": provenance["plan_sha256"], "calibration": calibration}, folder / "attack_state.pt")
        with torch.no_grad():
            final_z = features(pixels.detach())
            final_feature_loss = float(cosine_loss(final_z, target))
        fits = {}
        for label_kind, final_labels in (("soft", logits.detach().softmax(1)),
                                          ("hard", logits.detach().argmax(1))):
            updates = fc_sgd_updates(model.classifier, final_z, final_labels,
                                      steps=10, lr=.001, create_graph=False)
            fit = summarize_observation_fit(updates, observations, tolerance_steps=1e-6)
            fit["objective_at_same_post_step_candidate"] = float(update_loss(updates))
            fits[label_kind] = fit
            del updates
        result = {"status": "completed", "arm": arm, "iterations": len(logs),
                  "observation_term_enabled": use_observation,
                  "label_update_rule": "observation_gradient" if use_observation else "shared_initial_public_model_estimate_fixed",
                  "final_feature_loss": final_feature_loss, "fits": fits,
                  "optimization_time_sec": optimization_seconds, "wall_time_sec": time.perf_counter()-started,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(dev),
                  "reconstruction_success": "not_evaluated_no_threshold"}
        write_json(folder / "result.json", result)
        results[arm] = result
        print(json.dumps({"stage": "arm_complete", "arm": arm,
                          "final_feature_loss": final_feature_loss, "time_sec": result["wall_time_sec"]}), flush=True)
        del optimizer, pixels, logits, final_z
        torch.cuda.empty_cache()
    write_json(output / "summary.json", {"arms": results, "warmup_time_sec": warmup_seconds,
               "calibration": calibration, "interpretation": "fixed_schedule_single_batch_pilot_bin_plus_radius_bundle"})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cpa-target", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_attack(observation=args.observation, checkpoint=args.checkpoint, cpa_target=args.cpa_target,
               plan_path=args.plan, output=args.output)


if __name__ == "__main__":
    main()
