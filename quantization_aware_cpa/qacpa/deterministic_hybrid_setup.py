"""Isolated sender collection and public-only CPA initialization for frozen Conv.

Collection alone accepts private truth. The CPA entry point and observation
reader accept packets and the public checkpoint only. Historical packets,
features and attack outputs are never reused as observations or initializers.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from .deterministic_observation import (
    CODEC_ID, encode_updates, load_observations, packet_info, summarize_observation_fit,
)
from .fc_replay import fc_sgd_updates
from .full_update_pipeline import (
    ContractError, denormalize_images, file_digest, load_starting_vgg16,
    load_tensor_file, vgg16_shapes, write_json,
)
from .upstream_naive import run_upstream_cpa


FORMAT = "frozen_conv_all_fc_single_client_deterministic_uint4_v1"
IMAGE_SHAPE = [16, 3, 224, 224]
TRAINING = {"optimizer": "sgd", "steps": 10, "lr": .001, "momentum": 0.,
            "weight_decay": 0., "batch_order": [list(range(16)) for _ in range(10)]}
PUBLIC_CONTEXT = {
    "format": FORMAT, "codec": CODEC_ID, "model": "vgg16",
    "training_scope": "fc_only_conv_frozen", "observation_scope": "all_classifier_parameters",
    "observation_source": "single_client", "image_shape": IMAGE_SHAPE,
    "num_classes": 1000, "normalization": "imagenet", "dropout_probability": 0.,
    "bits": 4, "delta_convention": "final_minus_initial",
    "row_layout": "weights_output_rows_bias_single_row", "training": TRAINING,
    "permissions": {"packet_includes_public_radii": True, "starting_checkpoint_public": True,
                    "training_rules_public": True, "labels": "private",
                    "bandwidth_accounting": "4bit_codes_plus_reported_metadata"},
}


def configure_runtime(seed: int = 42, *, threads: int = 1) -> None:
    """Require deterministic FP32 execution; must precede CUDA work."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before CUDA initialization")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def _device(value) -> torch.device:
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _expected_shapes() -> dict[str, tuple[int, ...]]:
    return {name: shape for name, shape in vgg16_shapes().items() if name.startswith("classifier.")}


def _sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _separate(left: Path, right: Path) -> None:
    if left == right or left in right.parents or right in left.parents:
        raise ContractError("Public and private/source outputs require separate directory trees")


def _new_output(path: Path) -> Path:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    return path


def _counts(entries: dict) -> dict:
    result = {"elements": 0, "code_bytes": 0, "radius_bytes": 0, "packet_header_bytes": 0}
    for entry in entries.values():
        info = packet_info(entry["packet"])
        for key, source in (("elements", "elements"), ("code_bytes", "code_bytes"),
                            ("radius_bytes", "radius_bytes"), ("packet_header_bytes", "header_bytes")):
            result[key] += info[source]
    result["parameter_tensors"] = len(entries)
    result["layer_packet_bytes"] = sum(result[k] for k in ("code_bytes", "radius_bytes", "packet_header_bytes"))
    result["manifest_bytes"] = 0
    result["total_public_bytes"] = result["layer_packet_bytes"]
    return result


def _manifest_text(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _write_bundle(folder: Path, entries: dict, checkpoint_sha256: str) -> dict:
    layers = {}
    for index, name in enumerate(sorted(entries)):
        entry = entries[name]
        filename = f"packet_{index:02d}.bin"
        (folder / filename).write_bytes(entry["packet"])
        layers[name] = {"shape": entry["shape"], "packet": filename,
                        "sha256": hashlib.sha256(entry["packet"]).hexdigest(),
                        "packet_bytes": len(entry["packet"])}
    manifest = {**deepcopy(PUBLIC_CONTEXT), "checkpoint_sha256": checkpoint_sha256,
                "layers": layers, "communication": _counts(entries)}
    # Include the manifest's own bytes without an approximate metadata count.
    for _ in range(10):
        size = len(_manifest_text(manifest).encode("utf-8"))
        counts = manifest["communication"]
        if size == counts["manifest_bytes"]:
            break
        counts["manifest_bytes"] = size
        counts["total_public_bytes"] = counts["layer_packet_bytes"] + size
    else:
        raise RuntimeError("Manifest byte accounting did not converge")
    (folder / "manifest.json").write_bytes(_manifest_text(manifest).encode("utf-8"))
    return manifest


def read_observation(folder: Path, checkpoint: Path | None) -> tuple[dict, dict]:
    """Validate all six public FC packets; accepts no private truth path.

    A checkpoint path additionally verifies its bytes. CPA can use None because
    it never evaluates a model; its output remains bound to the manifest hash.
    """
    folder = Path(folder).resolve()
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != set(PUBLIC_CONTEXT) | {
            "checkpoint_sha256", "layers", "communication"}:
        raise ContractError("Unexpected public manifest fields; private data is forbidden")
    for name, expected in PUBLIC_CONTEXT.items():
        # Canonical JSON also distinguishes true from 1 and null from a missing grant.
        if json.dumps(manifest[name], sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ContractError(f"Public protocol mismatch: {name}")
    if not _sha(manifest["checkpoint_sha256"]):
        raise ContractError("Invalid public checkpoint digest")
    if checkpoint is not None and file_digest(Path(checkpoint)) != manifest["checkpoint_sha256"]:
        raise ContractError("Checkpoint does not match the public observation")
    shapes = _expected_shapes()
    layers = manifest["layers"]
    if not isinstance(layers, dict) or set(layers) != set(shapes):
        raise ContractError("All six classifier parameter names are required")
    entries = {}
    for index, name in enumerate(sorted(shapes)):
        layer = layers[name]
        if not isinstance(layer, dict) or set(layer) != {"shape", "packet", "sha256", "packet_bytes"}:
            raise ContractError("Invalid public layer descriptor")
        if layer["shape"] != list(shapes[name]) or layer["packet"] != f"packet_{index:02d}.bin":
            raise ContractError("Unexpected FC parameter shape or packet path")
        path = folder / layer["packet"]
        if path.resolve().parent != folder or path.is_symlink():
            raise ContractError("Packet must be a direct regular observation file")
        packet = path.read_bytes()
        if (type(layer["packet_bytes"]) is not int or len(packet) != layer["packet_bytes"]
                or hashlib.sha256(packet).hexdigest() != layer["sha256"]):
            raise ContractError(f"Packet integrity failure: {name}")
        entries[name] = {"shape": layer["shape"], "packet": packet}
    counts = _counts(entries)
    counts["manifest_bytes"] = manifest_path.stat().st_size
    counts["total_public_bytes"] = counts["layer_packet_bytes"] + counts["manifest_bytes"]
    if manifest["communication"] != counts:
        raise ContractError("Public communication byte accounting mismatch")
    return manifest, load_observations(entries)


def _load_private_truth(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    saved = load_tensor_file(path)
    if not isinstance(saved, dict) or set(saved) != {"x", "labels"}:
        raise ContractError("Sender truth must contain exactly x and labels")
    images, labels = saved["x"], saved["labels"]
    if (not isinstance(images, torch.Tensor) or images.dtype != torch.float32
            or list(images.shape) != IMAGE_SHAPE or not bool(torch.isfinite(images).all())):
        raise ContractError("Sender images must be finite FP32 [16,3,224,224]")
    if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64
            or list(labels.shape) != [16] or bool(((labels < 0) | (labels >= 1000)).any())):
        raise ContractError("Sender labels must be original int64 ImageNet indices")
    pixels = denormalize_images(images)
    if bool((pixels < -1e-5).any()) or bool((pixels > 1. + 1e-5).any()):
        raise ContractError("Sender images must retain their existing ImageNet normalization")
    return images, labels


def collect(truth_source: Path, checkpoint: Path, output: Path, private_output: Path,
            *, device="cuda", seed: int = 42) -> dict:
    """Sender only: new ordinary FC SGD, exact replay check, new deterministic packets."""
    output, private_output = _new_output(output), _new_output(private_output)
    truth_source, checkpoint = Path(truth_source).resolve(), Path(checkpoint).resolve()
    _separate(output, private_output)
    _separate(output, truth_source.parent)
    configure_runtime(seed)
    device = _device(device)
    images, labels = _load_private_truth(truth_source)
    checkpoint_sha256 = file_digest(checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    private_output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        model = load_starting_vgg16(checkpoint, device)
        model.features.requires_grad_(False)
        model.classifier.requires_grad_(True)
        model.eval()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        with torch.no_grad():
            features = model.avgpool(model.features(images.to(device))).flatten(1).detach()
        reference = deepcopy(model.classifier)
        initial = {"classifier." + name: p.detach() for name, p in model.classifier.named_parameters()}
        if {name: tuple(p.shape) for name, p in initial.items()} != _expected_shapes():
            raise ContractError("Loaded classifier does not match all six expected parameters")
        optimizer = torch.optim.SGD(reference.parameters(), lr=.001, momentum=0., weight_decay=0., foreach=False)
        targets = labels.to(device)
        losses = []
        for step in range(10):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(reference(features), targets, reduction="mean")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            print(f"FC sender local step {step + 1}/10 complete", flush=True)
        optimizer.zero_grad(set_to_none=True)
        replay = fc_sgd_updates(model.classifier, features, targets, steps=10, lr=.001,
                                create_graph=False, checkpoint_forward=False)
        updates, comparisons = {}, {}
        for name, parameter in reference.named_parameters():
            full_name = "classifier." + name
            delta = parameter.detach() - initial[full_name]
            exact = torch.equal(delta, replay[full_name])
            comparisons[full_name] = {"exact_torch_equal": exact,
                                      "max_absolute_error": float((delta - replay[full_name]).abs().max())}
            if not exact:
                raise RuntimeError(f"FC ordinary SGD/replay mismatch: {full_name}")
            updates[full_name] = delta.cpu()
        del model, reference, optimizer, replay, initial, features, targets, loss, delta, parameter
        if device.type == "cuda":
            torch.cuda.empty_cache()
        entries = encode_updates(updates)
        fit = summarize_observation_fit(updates, load_observations(entries), tolerance_steps=1e-6)
        if fit["exact_packet_match"] is not True or fit["closure_fit_within_tolerance"] is not True:
            raise RuntimeError("Sender raw updates do not exactly reencode to their deterministic packets")
        manifest = _write_bundle(output, entries, checkpoint_sha256)
        # Reader validation runs before publishing any completed private report.
        read_observation(output, checkpoint)
        torch.save({"x": images.detach().cpu(), "labels": labels.detach().cpu()},
                   private_output / "evaluation_truth.pt")
        report = {"status": "complete", "source_truth_sha256": file_digest(truth_source),
                  "checkpoint_sha256": checkpoint_sha256,
                  "observation_manifest_sha256": file_digest(output / "manifest.json"),
                  "truth_sha256": file_digest(private_output / "evaluation_truth.pt"),
                  "seed": seed, "device": str(device), "torch_version": str(torch.__version__),
                  "training_losses": losses, "training_steps": 10,
                  "frozen_conv_trainable_parameters": 0, "classifier_parameter_tensors": len(updates),
                  "replay_create_graph": False, "replay_checkpoint_forward": False,
                  "all_raw_deltas_torch_equal": True, "replay_comparison": comparisons,
                  "sender_codec_fit": fit, "raw_updates_saved": False, "features_saved": False,
                  "wall_time_sec": time.perf_counter() - started}
        write_json(private_output / "sender_validation.json", report)
        return manifest
    except Exception as error:
        write_json(private_output / "failure.json", {"status": "failed", "error_type": type(error).__name__,
                                                    "error_message": str(error)})
        raise


def _cpa_matrix(observation, lr: float) -> torch.Tensor:
    """Decode FP32 centers in bounded CPU row chunks; then preserve legacy Y sign."""
    result = torch.empty(observation.shape, dtype=torch.float32)
    for start in range(0, result.shape[0], 64):
        end = min(start + 64, result.shape[0])
        radius = observation.radius[start:end].double()
        codes = observation.codes[start:end].double()
        decoded = (radius[:, None] * (2. * codes / 15. - 1.)).float()
        result[start:end] = -decoded / lr
    return result


def prepare_cpa(observation: Path, output: Path, *, checkpoint: Path | None = None,
                iterations: int = 25000, seed: int = 42, device="cuda") -> dict:
    """Public deterministic packets -> original CPA -> fixed abs-source FIA target."""
    output = _new_output(output)
    observation = Path(observation).resolve()
    _separate(output, observation)
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    configure_runtime(seed)
    device = _device(device)
    manifest, observations = read_observation(observation, checkpoint)
    observed = observations["classifier.0.weight"]
    Y_hat = _cpa_matrix(observed, manifest["training"]["lr"])
    output.mkdir(parents=True, exist_ok=False)
    config = {"cpa_ica_max_iter": iterations, "cpa_ica_lr": .001,
              "cpa_ica_optimizer": "adam", "cpa_ica_scheduler": "none",
              "cpa_ica_decor": 5.3, "cpa_ica_T": 7.7, "cpa_ica_ne": 1.,
              "cpa_ica_nv": .13, "cpa_ica_l1": 5., "cpa_ica_tv": .1,
              "cpa_ica_log_every": min(5000, iterations)}
    try:
        result = run_upstream_cpa(Y_hat=Y_hat, n_components=16, device=device,
                                  config=config, output_dir=output / "upstream", seed=seed)
    finally:
        # Upstream's seed helper enables warn_only; restore the strict caller contract.
        configure_runtime(seed)
    target_z = result["S_hat"].detach().cpu().abs().contiguous()
    if tuple(target_z.shape) != (16, observed.shape[1]) or not bool(torch.isfinite(target_z).all()):
        raise RuntimeError("CPA returned invalid source dimensions or nonfinite features")
    manifest_sha256 = file_digest(observation / "manifest.json")
    torch.save({"target_z": target_z, "observation_manifest_sha256": manifest_sha256}, output / "target.pt")
    report = {"status": "complete", "method": "official_CPA_public_deterministic_FC1",
              "observation_manifest_sha256": manifest_sha256,
              "checkpoint_sha256": manifest["checkpoint_sha256"], "target_sha256": file_digest(output / "target.pt"),
              "target_shape": list(target_z.shape), "iterations": iterations, "seed": seed,
              "source_adapter": "abs(S_hat), estimated order, no truth scaling or matching",
              "input_conversion": "Y_hat = -decode(raw final-minus-initial FC1 packet) / public SGD lr",
              "private_inputs_used": False, "debias": False, "mp_truncation": False,
              "active_loss_terms": ["decorrelation", "negative_value_prior", "l1"],
              "upstream_negentropy_disabled_by_implementation": True,
              "upstream_tv_disabled_when_nv_positive": True,
              "settings": result["settings"], "source_hashes": result["source_hashes"],
              "wall_time_sec": result["wall_time_sec"],
              "setup_wall_time_sec": result["setup_wall_time_sec"],
              "optimizer_wall_time_sec": result["optimizer_wall_time_sec"]}
    write_json(output / "cpa_log.json", result["log"])
    write_json(output / "report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sender = commands.add_parser("collect")
    sender.add_argument("--truth-source", type=Path, required=True)
    sender.add_argument("--checkpoint", type=Path, required=True)
    sender.add_argument("--output", type=Path, required=True)
    sender.add_argument("--private-output", type=Path, required=True)
    cpa = commands.add_parser("cpa")
    cpa.add_argument("--observation", type=Path, required=True)
    cpa.add_argument("--checkpoint", type=Path)
    cpa.add_argument("--output", type=Path, required=True)
    cpa.add_argument("--iterations", type=int, default=25000)
    for command in (sender, cpa):
        command.add_argument("--device", default="cuda")
        command.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "collect":
        manifest = collect(args.truth_source, args.checkpoint, args.output, args.private_output,
                           device=args.device, seed=args.seed)
        result = {"status": "complete", "communication": manifest["communication"]}
    else:
        result = prepare_cpa(args.observation, args.output, checkpoint=args.checkpoint,
                             iterations=args.iterations, seed=args.seed, device=args.device)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
