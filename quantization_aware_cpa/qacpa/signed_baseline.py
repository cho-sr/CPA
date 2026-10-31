"""Authorized signed B16/T10 collector and isolated public/reference/offline CLIs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn import functional as F

from quantized_update_cpa import cell_utils
from quantized_update_cpa.quantization import quantize_tensor_symmetric
from quantized_update_cpa.trial_metrics import new_trial_metrics, metric_record
from . import upstream_naive
from .signed_baseline_observation import (CHECKPOINT_SHA, TRUTH_SHA, TRAINING, FC_SHAPES,
                                         read_bundle, write_bundle, digest)

CPA_CONFIG = {"cpa_ica_max_iter": 25000, "cpa_ica_lr": .001, "cpa_ica_optimizer": "adam",
              "cpa_ica_scheduler": "none", "cpa_ica_decor": 5.3, "cpa_ica_T": 7.7,
              "cpa_ica_ne": 1., "cpa_ica_nv": .13, "cpa_ica_l1": 5., "cpa_ica_tv": .1,
              "cpa_ica_log_every": 5000}
FIA_CONFIG = {"n_iter": 25000, "lr": .1, "tv": .1, "fi": 1., "optimizer": "adam",
              "scheduler": "cosine", "log_every": 5000}
SEED = 42
ATTACK_LAYER = "classifier.0.weight"


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def runtime():
    import numpy
    import torchvision
    return {"python": __import__("sys").version, "torch": torch.__version__, "numpy": numpy.__version__,
            "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "threads": torch.get_num_threads(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_peak_allocated": torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            "cuda_peak_reserved": torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0}


def configure():
    from .deterministic_hybrid_setup import configure_runtime
    configure_runtime(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("Approved CUDA GPU is unavailable")
    torch.cuda.reset_peak_memory_stats()


def verify_checkpoint(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if digest(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("starting checkpoint hash mismatch")
    # The upstream adapter uses this exact torchvision cache file.
    adapter_path = Path(torch.hub.get_dir())/"checkpoints"/"vgg16-397923af.pth"
    if adapter_path.resolve() != checkpoint or digest(adapter_path) != CHECKPOINT_SHA:
        raise ValueError("upstream IMAGENET1K_V1 adapter checkpoint differs from sender")


def separate(*folders):
    paths = [Path(p).resolve() for p in folders]
    for i, p in enumerate(paths):
        for q in paths[i+1:]:
            if p == q or p in q.parents or q in p.parents:
                raise ValueError("public/private/input output trees must be separate")


def quantization_fit(G, q, s):
    """Bounded-memory diagnostic only; no update replay or optimization."""
    if G.shape != q.shape:
        raise ValueError("fit shape mismatch")
    table_q = torch.arange(-7, 8, dtype=torch.int8)
    lower, upper = cell_utils.get_cells(table_q, s, dict(cell_utils.CODEC_METADATA))
    flat, code = G.flatten(), q.flatten()
    mismatches = violations = 0
    for start in range(0, G.numel(), 262144):
        values, codes = flat[start:start+262144], code[start:start+262144]
        mismatches += cell_utils.check_code_consistency(values, codes, s)["mismatch_count"]
        idx = codes.long()+7
        violations += int(((values < lower[idx]) | (values > upper[idx])).sum())
    return {"numel": G.numel(), "code_mismatch_count": mismatches,
            "code_mismatch_fraction": mismatches/G.numel(), "bounded_cell_violation_count": violations,
            "bounded_cell_violation_fraction": violations/G.numel(),
            "scale": cell_utils.scale_consistency(G, s),
            "scope": "supplied_matrix_only; no reconstructed_image_update_replay"}


def collect(*, truth_source, checkpoint, public_output, private_output):
    """Only sender subcommand can load original x/y and create raw Delta."""
    from .deterministic_hybrid_setup import _load_private_truth
    from .full_update_pipeline import load_starting_vgg16
    from .fc_replay import fc_sgd_updates
    public_output, private_output = Path(public_output), Path(private_output)
    separate(public_output, private_output, Path(truth_source).parent)
    if public_output.exists() or private_output.exists():
        raise FileExistsError("collector only writes new output folders")
    if digest(truth_source) != TRUTH_SHA:
        raise ValueError("source truth hash mismatch")
    verify_checkpoint(checkpoint)
    configure()
    images, labels = _load_private_truth(Path(truth_source))
    private_output.mkdir(parents=True)
    tick = time.perf_counter()
    model = load_starting_vgg16(Path(checkpoint), torch.device("cuda"))
    model.features.requires_grad_(False)
    model.classifier.requires_grad_(True)
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    with torch.no_grad():
        z = model.avgpool(model.features(images.cuda())).flatten(1).detach()
    oracle_z = z.cpu()
    reference = deepcopy(model.classifier)
    initial = {"classifier."+name: p.detach() for name, p in model.classifier.named_parameters()}
    optimizer = torch.optim.SGD(reference.parameters(), lr=.001, momentum=0., weight_decay=0., foreach=False)
    targets = labels.cuda()
    losses = []
    for step in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(reference(z), targets, reduction="mean")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        print(f"signed sender ordinary SGD {step+1}/10 loss={losses[-1]:.7g}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    replay = fc_sgd_updates(model.classifier, z, targets, steps=10, lr=.001,
                            create_graph=False, checkpoint_forward=False)
    updates, comparisons = {}, {}
    for name, parameter in reference.named_parameters():
        name = "classifier."+name
        delta = parameter.detach()-initial[name]
        equal = torch.equal(delta, replay[name])
        comparisons[name] = {"torch_equal": equal, "max_abs_error": float((delta-replay[name]).abs().max())}
        if not equal:
            raise RuntimeError(f"ordinary SGD vs replay mismatch: {name}")
        updates[name] = delta.cpu()
    del replay, reference, model, optimizer, initial, targets, parameter, delta, z, loss
    torch.cuda.empty_cache()
    manifest = write_bundle(public_output, updates)
    read_manifest, entries = read_bundle(public_output)
    codec_checks = {}
    for name, G in updates.items():
        q, s = entries[name]
        rq, rdq, rs, _ = quantize_tensor_symmetric(G, bits=4, rounding="nearest")
        exact = torch.equal(q, rq) and torch.equal(s, rs) and torch.equal(cell_utils.dequantize(q, s), rdq)
        if not exact:
            raise RuntimeError(f"signed reference mismatch: {name}")
        fit = quantization_fit(G, q, s)
        if fit["code_mismatch_count"] or fit["bounded_cell_violation_count"] or not fit["scale"]["exact_equal"]:
            raise RuntimeError(f"sender signed fit failed: {name}")
        codec_checks[name] = {"reference_q_s_dequantized_equal": exact, "fit": fit}
        del rq, rdq, rs
    torch.save({"updates": updates, "delta_convention": "theta_after10_minus_theta_start",
                "checkpoint_sha256": CHECKPOINT_SHA}, private_output/"raw_updates.pt")
    torch.save({"x": images, "labels": labels, "z": oracle_z}, private_output/"evaluation_truth.pt")
    result = {"status": "COMPLETED", "source_truth_sha256": TRUTH_SHA, "checkpoint_sha256": CHECKPOINT_SHA,
              "raw_updates_sha256": digest(private_output/"raw_updates.pt"),
              "evaluation_truth_sha256": digest(private_output/"evaluation_truth.pt"),
              "public_manifest_sha256": digest(public_output/"manifest.json"),
              "training": TRAINING, "training_losses": losses, "ordinary_runs": 1, "replay_runs": 1,
              "all_raw_deltas_torch_equal": True, "replay_comparisons": comparisons,
              "codec_checks": codec_checks, "wall_seconds": time.perf_counter()-tick, "runtime": runtime()}
    write_json(private_output/"sender_validation.json", result)
    print(json.dumps({"collector": "COMPLETED", "communication": manifest["communication"]}), flush=True)
    return result


@contextmanager
def telemetry(folder, kind):
    """Observe original steps/initialization without changing losses or RNG state."""
    upstream = upstream_naive.load_upstream_modules()
    cls = upstream.gradient_inversion.CocktailPartyAttack if kind == "cpa" else upstream.feature_inversion.Direct
    old_init, old_step = cls.__init__, cls.step
    tick = time.perf_counter()
    counter = 0
    def init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        value = self.W_hat if kind == "cpa" else self.S_hat
        write_json(folder/f"{kind}_initial.json", {"tensor_sha256": upstream_naive._tensor_hash(value),
                                                   "shape": list(value.shape), "runtime": runtime(), "seed": SEED})
    def step(self):
        nonlocal counter
        losses = old_step(self)
        counter += 1
        if not all(math.isfinite(float(x)) for x in losses.values()):
            raise FloatingPointError(f"nonfinite {kind} objective at iteration {counter}")
        if counter == 1 or counter % 100 == 0:
            elapsed = time.perf_counter()-tick
            write_json(folder/f"{kind}_progress.json", {"completed_this_process": counter, "total": 25000,
                       "elapsed_seconds_including_setup": elapsed, "iterations_per_second": counter/elapsed,
                       "estimated_remaining_seconds": elapsed/counter*(25000-counter),
                       "losses": losses, "time_utc": datetime.now(timezone.utc).isoformat()})
        return losses
    cls.__init__, cls.step = init, step
    try:
        yield
    finally:
        cls.__init__, cls.step = old_init, old_step


def source_metrics(S, cpa_log):
    record = new_trial_metrics()
    def computed(name, value, definition):
        record["raw_metrics"][name] = metric_record(value, status="COMPUTED", definition=definition)
    computed("cpa_objective", cpa_log[-1], "Original upstream last pre-update loss log; NE is overwritten to zero")
    record["raw_metrics"]["negentropy"] = metric_record(reason="Upstream overwrites loss_ne to zero; actual source negentropy not computed")
    computed("sparsity", {"mean_abs": float(S.abs().mean()), "exact_zero_fraction": float((S == 0).float().mean())},
             "Final signed S_hat raw L1 mean and exact zero fraction")
    positive = S.clamp_min(0).norm(dim=1)
    negative = (-S).clamp_min(0).norm(dim=1)
    computed("sign_consistency", {"minority_sign_l2_mean": float(torch.minimum(positive, negative).mean())},
             "Final signed S_hat min(positive L2, negative L2), per-row mean; no oracle")
    normalized = F.normalize(S, dim=1)
    cosine = normalized @ normalized.T
    off = cosine[~torch.eye(len(S), dtype=torch.bool)]
    computed("source_diversity", {"pairwise_cosine_mean": float(off.mean()), "pairwise_abs_cosine_mean": float(off.abs().mean()),
                                  "pairwise_abs_cosine_max": float(off.abs().max())},
             "Final signed S_hat off-diagonal row cosine; not mutual information")
    return record


def _run_arm(*, Y, output, checkpoint, input_info, observation_fit=None):
    """Common implementation receives observation matrix, never x/y/z truth."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("arm directory already exists; explicit same-setting recovery required")
    verify_checkpoint(checkpoint)
    configure()
    if Y.dtype != torch.float32 or tuple(Y.shape) != FC_SHAPES[ATTACK_LAYER]:
        raise ValueError("expected untransformed FP32 FC1 Delta/decoded Delta")
    output.mkdir(parents=True)
    source_before = upstream_naive.source_hashes()
    write_json(output/"registered_arm.json", {"input": input_info, "input_tensor_sha256": upstream_naive._tensor_hash(Y),
               "input_conversion": "identity_raw_Delta_or_raw_s_q; no sign/lr/T/B transformation",
               "checkpoint_sha256": CHECKPOINT_SHA, "seed": SEED, "cpa": CPA_CONFIG, "fia": FIA_CONFIG,
               "source_adapter": "abs(S_hat); estimated ordering; no oracle correction", "source_hashes": source_before})
    tick = time.perf_counter()
    with telemetry(output, "cpa"):
        cpa = upstream_naive.run_upstream_cpa(Y_hat=Y, n_components=16, device="cuda",
                                             config=deepcopy(CPA_CONFIG), output_dir=output/"cpa", seed=SEED)
    S = cpa.pop("S_hat")
    target = S.abs().contiguous()
    torch.save({"S_hat": S, "target_z": target}, output/"sources.pt")
    write_json(output/"cpa_result.json", cpa)
    cpa_runtime = runtime()
    del Y
    torch.cuda.empty_cache()
    raw_metrics = source_metrics(S, cpa["log"])
    if observation_fit is not None:
        raw_metrics["raw_metrics"]["quantization_consistency"] = metric_record(
            observation_fit, status="COMPUTED", definition="Decoded FC1 input at public fixed s; not reconstructed image update")
        raw_metrics["raw_metrics"]["scale_consistency"] = metric_record(
            observation_fit["scale"], status="COMPUTED", definition="Decoded FC1 reference scale versus observed s; diagnostic")
    with telemetry(output, "fia"):
        fia = upstream_naive.run_upstream_fia(target_z=target, device="cuda", config=deepcopy(FIA_CONFIG),
                                             output_dir=output/"fia", seed=SEED)
    pixels = fia.pop("rec")
    if tuple(pixels.shape) != (16,3,224,224) or not bool(torch.isfinite(pixels).all()) or bool(((pixels < 0)|(pixels > 1)).any()):
        raise ValueError("invalid final Direct FIA pixels")
    torch.save({"pixels": pixels}, output/"candidate.pt")
    write_json(output/"fia_result.json", fia)
    write_json(output/"raw_metrics.json", raw_metrics)
    if source_before != upstream_naive.source_hashes():
        raise RuntimeError("upstream source changed during arm")
    result = {"status": "COMPLETED", "input": input_info, "seed": SEED,
              "candidate_sha256": digest(output/"candidate.pt"), "sources_sha256": digest(output/"sources.pt"),
              "cpa_iterations": 25000, "fia_iterations": 25000, "image_count": 16,
              "selection": "final_iteration_only_no_oracle_selection", "labels_used": False,
              "cpa_runtime": cpa_runtime, "final_runtime": runtime(), "wall_seconds": time.perf_counter()-tick,
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output/"result.json", result)
    return result


def run_naive(*, public_observation, output, checkpoint):
    """Public q,s and checkpoint only. No truth/raw-update argument."""
    manifest, entries = read_bundle(Path(public_observation))
    q, s = entries[ATTACK_LAYER]
    Y = cell_utils.dequantize(q, s)
    fit = quantization_fit(Y, q, s)
    return _run_arm(Y=Y, output=output, checkpoint=checkpoint, observation_fit=fit,
                    input_info={"arm": "INT4_naive", "public_manifest_sha256": digest(Path(public_observation)/"manifest.json"),
                                "checkpoint_sha256": manifest["checkpoint_sha256"], "private_inputs_used": False})


def run_reference(*, raw_updates, output, checkpoint):
    """Privileged FP32 Reference: raw Delta only, no oracle images/features."""
    saved = torch.load(raw_updates, map_location="cpu", weights_only=True)
    if set(saved) != {"updates", "delta_convention", "checkpoint_sha256"} or saved["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("invalid reference raw input")
    if saved["delta_convention"] != "theta_after10_minus_theta_start":
        raise ValueError("wrong raw Delta convention")
    return _run_arm(Y=saved["updates"][ATTACK_LAYER], output=output, checkpoint=checkpoint,
                    input_info={"arm": "FP32 Reference", "raw_file_sha256": digest(raw_updates),
                                "private_raw_update_used": True, "oracle_truth_used": False})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    s = subs.add_parser("collect")
    for name in ("truth-source", "checkpoint", "public-output", "private-output"):
        s.add_argument("--"+name, type=Path, required=True)
    s = subs.add_parser("naive")
    for name in ("public-observation", "checkpoint", "output"):
        s.add_argument("--"+name, type=Path, required=True)
    s = subs.add_parser("reference")
    for name in ("raw-updates", "checkpoint", "output"):
        s.add_argument("--"+name, type=Path, required=True)
    s = subs.add_parser("evaluate")
    for name in ("naive", "reference", "private-observation", "public-observation", "output"):
        s.add_argument("--"+name, type=Path, required=True)
    return p


def main(argv=None):
    args = vars(parser().parse_args(argv))
    command = args.pop("command")
    if command == "evaluate":
        from .signed_baseline_evaluation import evaluate
        result = evaluate(**args)
    else:
        fn = {"collect": collect, "naive": run_naive, "reference": run_reference}[command]
        result = fn(**args)
    print(json.dumps({"command": command, "status": result["status"]}), flush=True)


if __name__ == "__main__":
    main()
