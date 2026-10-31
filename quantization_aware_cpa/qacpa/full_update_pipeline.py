"""Explicit public contracts for full VGG16, single-client update inversion.

The sender and evaluator may read private images. The attack entry point only
receives packets, the declared public starting model, and a registered design.
Existing FC-only observations cannot be converted into full-model observations.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .communication_codec import CODEC_ID, packet_info
from .full_update_observation import (
    encode_updates, load_observations, observation_loss, summarize_observation_fit,
)
from .full_update_replay import SGDReplaySpec, differentiable_sgd


FORMAT = "full_vgg16_single_client_uint4_v1"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MANIFEST_KEYS = {
    "format", "contract", "contract_sha256", "checkpoint_sha256", "layers",
    "communication", "public_labels",
}


class ContractError(ValueError):
    """An observation or public-context contract is missing or inconsistent."""


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def load_tensor_file(path: Path) -> Any:
    """Support original torchvision legacy checkpoints without altering weights."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as error:
        if "mmap can only be used with files saved with" not in str(error):
            raise
        return torch.load(path, map_location="cpu", weights_only=True)


def validate_contract(contract: Mapping[str, Any]) -> SGDReplaySpec:
    if not isinstance(contract, Mapping):
        raise ContractError("Contract must be a JSON object")
    expected = {
        "schema_version": 1, "model": "vgg16", "training_scope": "conv_and_fc",
        "observation_scope": "all_parameters", "observation_source": "single_client",
        "bits": 4, "codec": CODEC_ID, "delta_convention": "final_minus_initial",
        "row_layout": "weights_output_rows_bias_single_row", "num_classes": 1000,
        "normalization": "imagenet", "dropout_probability": 0.0,
    }
    allowed = set(expected) | {"permissions", "image_shape", "training"}
    if set(contract) != allowed:
        raise ContractError(f"Contract fields must be exactly {sorted(allowed)}")
    for key, value in expected.items():
        if contract[key] != value:
            raise ContractError(f"{key} must be {value!r}; received {contract[key]!r}")
    permissions = contract["permissions"]
    required = {
        "packet_includes_public_radii": True,
        "bandwidth_accounting": "4bit_codes_plus_reported_metadata",
        "starting_checkpoint_public": True,
        "training_rules_public": True,
    }
    if not isinstance(permissions, dict) or set(permissions) != set(required) | {"labels"}:
        raise ContractError("Explicit public packet/model/training/label permissions are required")
    for key, value in required.items():
        if permissions[key] != value or (value is True and permissions[key] is not True):
            raise ContractError(f"Unconfirmed/unsupported public context: {key}={permissions[key]!r}")
    if permissions["labels"] not in {"private", "public"}:
        raise ContractError("labels visibility must explicitly be private or public")
    shape = contract["image_shape"]
    if (not isinstance(shape, list) or len(shape) != 4 or
            any(type(n) is not int for n in shape) or shape[0] < 1 or shape[1:] != [3, 224, 224]):
        raise ContractError("Full VGG16 images must have shape [N, 3, 224, 224]")
    training = contract["training"]
    if not isinstance(training, Mapping):
        raise ContractError("Training rules must be a JSON object")
    if set(training) != {"optimizer", "lr", "steps", "batch_order", "momentum", "weight_decay"}:
        raise ContractError("Training rules must explicitly specify SGD and every minibatch")
    if training["optimizer"] != "sgd":
        raise ContractError("Only the explicitly implemented SGD recurrence is supported")
    if type(training["steps"]) is not int or training["steps"] < 1:
        raise ContractError("steps must be a positive integer")
    order = training["batch_order"]
    if not isinstance(order, list) or len(order) != training["steps"]:
        raise ContractError("batch_order must contain exactly one index list per local step")
    for batch in order:
        if (not isinstance(batch, list) or not batch or
                any(type(i) is not int or i < 0 or i >= shape[0] for i in batch) or
                len(set(batch)) != len(batch)):
            raise ContractError("Invalid/duplicate sample index in a local minibatch")
    if set(i for batch in order for i in batch) != set(range(shape[0])):
        raise ContractError("All candidate samples must occur in the declared local training")
    for key in ("lr", "momentum", "weight_decay"):
        value = training[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ContractError(f"Invalid SGD {key}")
    if training["lr"] <= 0:
        raise ContractError("SGD learning rate must be positive")
    return SGDReplaySpec(lr=float(training["lr"]),
                         batch_order=tuple(tuple(batch) for batch in order),
                         momentum=float(training["momentum"]),
                         weight_decay=float(training["weight_decay"]))


def normalize_pixels(pixels: torch.Tensor) -> torch.Tensor:
    mean = pixels.new_tensor(MEAN).view(1, 3, 1, 1)
    std = pixels.new_tensor(STD).view(1, 3, 1, 1)
    return (pixels - mean) / std


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    return images * images.new_tensor(STD).view(1, 3, 1, 1) + images.new_tensor(MEAN).view(1, 3, 1, 1)


def vgg16_shapes() -> dict[str, tuple[int, ...]]:
    from torchvision.models import vgg16
    with torch.device("meta"):
        model = vgg16(weights=None, init_weights=False)
    return {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}


class VGGFixedSpatialPool(torch.nn.Module):
    """Exact 7x7 -> 7x7 adaptive-average identity for the fixed 224px contract.

    CUDA's general adaptive-average backward is flagged nondeterministic even
    for this identity case. Avoid that kernel without changing values or the
    training recurrence; reject sizes where the identity would not be valid.
    """
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[-2:] != (7, 7):
            raise ContractError("VGG fixed pooling requires the registered 7x7 feature map")
        return value


def load_starting_vgg16(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    """Load an explicit local state dict; never download/substitute a checkpoint."""
    from torchvision.models import vgg16
    with torch.device("meta"):
        model = vgg16(weights=None, init_weights=False)
    state = load_tensor_file(Path(checkpoint))
    model.load_state_dict(state, strict=True, assign=True)
    model.avgpool = VGGFixedSpatialPool()
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model = model.to(device).train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def validate_parameter_scope(entries: Mapping[str, Any], expected_shapes: Mapping[str, tuple]) -> None:
    if set(entries) != set(expected_shapes):
        raise ContractError("Observation must contain every Conv/FC weight and bias, exactly once")
    for name, expected in expected_shapes.items():
        actual = tuple(entries[name]["shape"])
        if actual != expected:
            raise ContractError(f"Observation shape mismatch for {name}: {actual} != {expected}")


def write_observation(folder: Path, entries: Mapping[str, dict], contract: dict,
                      checkpoint_sha256: str, labels: torch.Tensor | None = None) -> dict:
    validate_contract(contract)
    validate_parameter_scope(entries, vgg16_shapes())
    # Validate wire data before any output is committed.
    load_observations(entries)
    public = contract["permissions"]["labels"] == "public"
    if public != (labels is not None):
        raise ContractError("Public labels must be supplied iff explicitly declared public")
    if labels is not None:
        if (labels.dtype != torch.long or tuple(labels.shape) != (contract["image_shape"][0],) or
                bool(((labels < 0) | (labels >= 1000)).any())):
            raise ContractError("Invalid public labels")
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(f"Refusing to replace an observation: {folder}")
    folder.mkdir(parents=True)
    (folder / "layers").mkdir()
    layers, codes, radii, headers, total_elements = [], 0, 0, 0, 0
    for index, (name, entry) in enumerate(entries.items()):
        packet = entry["packet"]
        info = packet_info(packet)
        filename = f"layers/{index:03d}.bin"
        (folder / filename).write_bytes(packet)
        layers.append({"name": name, "shape": list(entry["shape"]), "file": filename,
                       "sha256": hashlib.sha256(packet).hexdigest(), "packet_bytes": len(packet)})
        codes += info["code_bytes"]
        radii += info["radius_bytes"]
        headers += info["header_bytes"]
        total_elements += info["elements"]
    manifest = {
        "format": FORMAT, "contract": contract, "contract_sha256": json_digest(contract),
        "checkpoint_sha256": checkpoint_sha256, "layers": layers,
        "communication": {"elements": total_elements, "code_bytes": codes,
                          "radius_bytes": radii, "packet_header_bytes": headers,
                          "layer_packet_bytes": codes + radii + headers,
                          "manifest_bytes_accounting": "count actual manifest.json bytes separately"},
        "public_labels": labels.cpu().tolist() if labels is not None else None,
    }
    write_json(folder / "manifest.json", manifest)
    return manifest


def read_observation(folder: Path, checkpoint: Path) -> tuple[dict, dict]:
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest) != MANIFEST_KEYS or manifest["format"] != FORMAT:
        raise ContractError("Invalid observation manifest; unexpected fields are not accepted")
    contract = manifest["contract"]
    validate_contract(contract)
    if json_digest(contract) != manifest["contract_sha256"]:
        raise ContractError("Contract digest mismatch")
    if file_digest(checkpoint) != manifest["checkpoint_sha256"]:
        raise ContractError("Checkpoint is not the model that generated this observation")
    if contract["permissions"]["labels"] == "private":
        if manifest["public_labels"] is not None:
            raise ContractError("Private labels leaked into receiver manifest")
    else:
        labels = manifest["public_labels"]
        if (not isinstance(labels, list) or len(labels) != contract["image_shape"][0] or
                any(type(i) is not int or i < 0 or i >= 1000 for i in labels)):
            raise ContractError("Invalid public labels")
    entries = {}
    counts = {"elements": 0, "code_bytes": 0, "radius_bytes": 0, "packet_header_bytes": 0}
    for index, layer in enumerate(manifest["layers"]):
        if set(layer) != {"name", "shape", "file", "sha256", "packet_bytes"}:
            raise ContractError("Invalid layer descriptor")
        if layer["file"] != f"layers/{index:03d}.bin" or layer["name"] in entries:
            raise ContractError("Unexpected packet path or duplicate parameter name")
        packet = (folder / layer["file"]).read_bytes()
        if len(packet) != layer["packet_bytes"] or hashlib.sha256(packet).hexdigest() != layer["sha256"]:
            raise ContractError(f"Packet integrity failure: {layer['name']}")
        info = packet_info(packet)
        if info["codec_id"] != CODEC_ID:
            raise ContractError("Only the 4bit stochastic codec is permitted")
        for key, source in (("elements", "elements"), ("code_bytes", "code_bytes"),
                            ("radius_bytes", "radius_bytes"), ("packet_header_bytes", "header_bytes")):
            counts[key] += info[source]
        entries[layer["name"]] = {"shape": layer["shape"], "packet": packet}
    validate_parameter_scope(entries, vgg16_shapes())
    for key, value in counts.items():
        if manifest["communication"].get(key) != value:
            raise ContractError("Communication accounting mismatch")
    total = sum(counts[key] for key in ("code_bytes", "radius_bytes", "packet_header_bytes"))
    if manifest["communication"].get("layer_packet_bytes") != total:
        raise ContractError("Total packet byte count mismatch")
    return manifest, load_observations(entries)


def collect_observation(*, inputs: Path, checkpoint: Path, contract: dict,
                        output: Path, private_output: Path, codec_seed: int,
                        device: torch.device) -> dict:
    """Sender only: train on private data and export an isolated public packet bundle."""
    spec = validate_contract(contract)
    output, private_output = Path(output).resolve(), Path(private_output).resolve()
    if (output == private_output or output in private_output.parents or private_output in output.parents):
        raise ContractError("Public packets and private evaluation data require separate directory trees")
    if output.exists() or private_output.exists():
        raise FileExistsError("Collection output already exists; choose a new run directory")
    saved = load_tensor_file(Path(inputs))
    images = saved["x"]
    if "labels" in saved:
        labels = saved["labels"]
    elif "y" in saved:
        labels = saved["y"]
    else:
        raw_labels = [row["label"] for row in saved["metadata"]]
        if any(type(label) is not int for label in raw_labels):
            raise ContractError("Original labels must be integers")
        labels = torch.tensor(raw_labels, dtype=torch.long)
    if not isinstance(labels, torch.Tensor) or labels.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ContractError("Original labels must be an integer tensor")
    labels = labels.long()
    if tuple(images.shape) != tuple(contract["image_shape"]) or images.dtype != torch.float32:
        raise ContractError("Private input shape/dtype differs from the registered contract")
    pixels = denormalize_images(images)
    if (not bool(torch.isfinite(images).all()) or bool((pixels < -1e-5).any()) or
            bool((pixels > 1.0 + 1e-5).any())):
        raise ContractError("Expected already ImageNet-normalized RGB images")
    if tuple(labels.shape) != (images.shape[0],) or bool(((labels < 0) | (labels >= 1000)).any()):
        raise ContractError("Invalid original ImageNet labels")
    checkpoint_sha256 = file_digest(checkpoint)
    model = load_starting_vgg16(checkpoint, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=spec.lr, momentum=spec.momentum,
                                weight_decay=spec.weight_decay, foreach=False)
    x, y = images.to(device), labels.to(device)
    started = time.perf_counter()
    training_losses = []
    for step, batch in enumerate(spec.batch_order):
        index = torch.tensor(batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[index]), y[index], reduction="mean")
        loss.backward()
        optimizer.step()
        training_losses.append(float(loss.detach().cpu()))
        print(f"sender local step {step + 1}/{len(spec.batch_order)} complete", flush=True)
    optimizer.zero_grad(set_to_none=True)
    initial = load_tensor_file(checkpoint)
    updates = {name: parameter.detach().cpu() - initial[name]
               for name, parameter in model.named_parameters()}
    del initial, model, optimizer, x, y, loss, pixels, saved
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not any(bool(delta.ne(0).any()) for name, delta in updates.items() if name.startswith("features.")):
        raise RuntimeError("No Conv parameter changed; refusing to report full-model training")
    entries = encode_updates(updates, seed=codec_seed)
    sender_fit = summarize_observation_fit(updates, load_observations(entries), tolerance_steps=1e-6)
    manifest = write_observation(output, entries, contract, checkpoint_sha256,
                                 labels if contract["permissions"]["labels"] == "public" else None)
    private_output.mkdir(parents=True)
    torch.save({"x": images.detach().cpu(), "labels": labels.cpu()}, private_output / "evaluation_truth.pt")
    report = {"source_inputs_sha256": file_digest(inputs), "checkpoint_sha256": checkpoint_sha256,
              "codec_seed_private": codec_seed, "training_losses": training_losses,
              "sender_codec_fit": sender_fit, "wall_time_sec": time.perf_counter() - started,
              "public_manifest_sha256": file_digest(output / "manifest.json"),
              "full_precision_update_saved": False}
    write_json(private_output / "sender_provenance.json", report)
    return {"public_observation": str(output), "communication": manifest["communication"],
            "manifest_bytes": (output / "manifest.json").stat().st_size,
            "training_steps": len(spec.batch_order), "wall_time_sec": report["wall_time_sec"]}


def validate_design(design: Mapping[str, Any]) -> None:
    if not isinstance(design, Mapping):
        raise ContractError("Attack design must be a JSON object")
    required = {"iterations", "seed", "image_lr", "label_lr", "tv_weight", "radius_weight",
                "checkpoint_forward", "saved_tensors_on_cpu", "log_every", "fit_tolerance_steps"}
    if set(design) - {"gradient_strategy"} != required:
        raise ContractError(f"Attack design fields must be exactly {sorted(required)}")
    if design.get("gradient_strategy", "unrolled") not in {"unrolled", "recompute_reverse"}:
        raise ContractError("Unknown gradient strategy")
    if design.get("gradient_strategy") == "recompute_reverse" and (
            not design["checkpoint_forward"] or design["saved_tensors_on_cpu"]):
        raise ContractError("Reverse recomputation requires checkpoint_forward=true and saved_tensors_on_cpu=false")
    for key in ("iterations", "log_every"):
        if type(design[key]) is not int or design[key] < 1:
            raise ContractError(f"{key} must be a positive integer")
    if type(design["seed"]) is not int or not 0 <= design["seed"] < 2**63:
        raise ContractError("Invalid attack seed")
    for key in ("image_lr", "label_lr", "tv_weight", "radius_weight", "fit_tolerance_steps"):
        if type(design[key]) not in (int, float) or not math.isfinite(design[key]) or design[key] < 0:
            raise ContractError(f"Invalid attack design value {key}")
    if design["image_lr"] == 0 or design["label_lr"] == 0:
        raise ContractError("Image and label learning rates must be positive")
    for key in ("checkpoint_forward", "saved_tensors_on_cpu"):
        if type(design[key]) is not bool:
            raise ContractError(f"{key} must be an explicit boolean")


def total_variation(pixels: torch.Tensor) -> torch.Tensor:
    return ((pixels[:, :, 1:, :] - pixels[:, :, :-1, :]).abs().mean() +
            (pixels[:, :, :, 1:] - pixels[:, :, :, :-1]).abs().mean())


def attack_observation(*, observation: Path, checkpoint: Path, design: dict,
                       output: Path, device: torch.device, mode: str = "cell",
                       probe: bool = False) -> dict:
    """Receiver only. No private image/update/label path is accepted by this API."""
    validate_design(design)
    design = {**design, "gradient_strategy": design.get("gradient_strategy", "unrolled")}
    if mode not in {"cell", "center"}:
        raise ContractError("Unsupported observation loss")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite attack output: {output}")
    manifest, observations = read_observation(observation, checkpoint)
    contract = manifest["contract"]
    spec = validate_contract(contract)
    model = load_starting_vgg16(checkpoint, device)
    generator = torch.Generator(device=device).manual_seed(design["seed"])
    pixels = torch.nn.Parameter(torch.rand(contract["image_shape"], generator=generator, device=device))
    public_labels = manifest["public_labels"]
    label_logits = None
    if public_labels is None:
        label_logits = torch.nn.Parameter(torch.randn((pixels.shape[0], 1000), device=device,
                                                       generator=generator) * .01)
        groups = [{"params": [pixels], "lr": design["image_lr"]},
                  {"params": [label_logits], "lr": design["label_lr"]}]
    else:
        public_labels = torch.tensor(public_labels, dtype=torch.long, device=device)
        groups = [{"params": [pixels], "lr": design["image_lr"]}]
    optimizer = torch.optim.Adam(groups)
    output.mkdir(parents=True)
    run_design = {"attack": design, "mode": mode, "probe_only": probe,
                  "method": "direct_image_optimization_without_CPA",
                  "implementation_sha256": {path.name: file_digest(path) for path in
                      sorted(Path(__file__).parent.glob("full_update_*.py"))},
                  "runtime": {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                      "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                      "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                      "cudnn_deterministic": torch.backends.cudnn.deterministic,
                      "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                      "spatial_pool": "checked_identity_for_7x7_to_7x7"},
                  "observation_manifest_sha256": file_digest(Path(observation) / "manifest.json"),
                  "contract_sha256": manifest["contract_sha256"],
                  "checkpoint_sha256": manifest["checkpoint_sha256"]}
    write_json(output / "registered_design.json", run_design)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    logs = []
    iterations = 1 if probe else design["iterations"]
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        targets = label_logits.softmax(dim=1) if label_logits is not None else public_labels
        context = torch.autograd.graph.save_on_cpu(pin_memory=False) if design["saved_tensors_on_cpu"] else nullcontext()
        with context:
            prior = total_variation(pixels)
            variables = [pixels] + ([label_logits] if label_logits is not None else [])
            if design["gradient_strategy"] == "recompute_reverse":
                from .full_update_adjoint import recomputed_image_gradients
                fit_loss, normalized_gradient, target_gradient = recomputed_image_gradients(
                    model, normalize_pixels(pixels), targets, spec,
                    lambda updates: observation_loss(updates, observations, mode=mode,
                                                     radius_weight=design["radius_weight"]))
                prior_gradient, = torch.autograd.grad(prior, pixels)
                pixel_gradient = normalized_gradient / pixels.new_tensor(STD).view(1, 3, 1, 1)
                gradients = [pixel_gradient + design["tv_weight"] * prior_gradient]
                if label_logits is not None:
                    probabilities = targets.detach()
                    gradients.append(probabilities * (target_gradient -
                        (probabilities * target_gradient).sum(dim=1, keepdim=True)))
                loss = fit_loss + design["tv_weight"] * prior.detach()
                del normalized_gradient, target_gradient, prior_gradient, pixel_gradient
            else:
                updates = differentiable_sgd(model, normalize_pixels(pixels), targets, spec,
                                             checkpoint_forward=design["checkpoint_forward"])
                fit_loss = observation_loss(updates, observations, mode=mode,
                                            radius_weight=design["radius_weight"])
                loss = fit_loss + design["tv_weight"] * prior
                gradients = torch.autograd.grad(loss, variables)
                del updates
        if not bool(torch.isfinite(loss)) or any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError("Nonfinite outer objective/gradient")
        for variable, gradient in zip(variables, gradients):
            variable.grad = gradient
        gradient_norm = float(gradients[0].norm().detach().cpu())
        entry = {"iteration": iteration + 1, "loss": float(loss.detach().cpu()),
                 "observation_loss": float(fit_loss.detach().cpu()), "tv": float(prior.detach().cpu()),
                 "image_gradient_norm": gradient_norm, "elapsed_sec": time.perf_counter() - started}
        logs.append(entry)
        del gradients, loss, fit_loss, prior, targets
        if not probe:
            optimizer.step()
            with torch.no_grad():
                pixels.clamp_(0.0, 1.0)
        if iteration == 0 or (iteration + 1) % design["log_every"] == 0 or iteration == iterations - 1:
            write_json(output / "optimization_log.json", logs)
            print(json.dumps(entry), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {"status": "outer_gradient_probe_complete" if probe else "candidate_generated",
              "iterations": iterations, "optimization_time_sec": time.perf_counter() - started,
              "image_gradient_nonzero": logs[-1]["image_gradient_norm"] > 0,
              "reconstruction_success": "not_evaluated", "mode": mode,
              "label_optimization": "soft_relaxation" if label_logits is not None else "public_hard_labels"}
    if not probe:
        hard_labels = label_logits.detach().argmax(dim=1) if label_logits is not None else public_labels
        # Only this hard-label replay can be used for final observation-fit diagnostics.
        hard_updates = differentiable_sgd(model, normalize_pixels(pixels.detach()), hard_labels, spec,
                                          create_graph=False, checkpoint_forward=design["checkpoint_forward"])
        result["hard_label_observation_fit"] = summarize_observation_fit(
            hard_updates, observations, tolerance_steps=design["fit_tolerance_steps"])
        torch.save({"pixels": pixels.detach().cpu(), "labels": hard_labels.cpu(),
                    "observation_manifest_sha256": run_design["observation_manifest_sha256"]}, output / "candidate.pt")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result.update(wall_time_sec=time.perf_counter() - started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None)
    if probe:
        result["registered_outer_iterations"] = design["iterations"]
        result["projected_optimization_time_sec"] = result["optimization_time_sec"] * design["iterations"]
        result["projection_excludes"] = ["initialization", "final_hard_label_validation"]
    write_json(output / "result.json", result)
    return result
