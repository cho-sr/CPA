"""Staged driver for the registered screen3 Direct FIA experiment.

The official ``Direct`` class is constructed unchanged with a 25000-step
cosine horizon.  This module only owns the outer stopping/checkpoint layer so
that a 5000-step public prefix can be resumed without rebuilding the
optimizer, scheduler, or image parameter from exported pixels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Callable

import torch

from .deterministic_hybrid_setup import configure_runtime
from .seed import seed_all
from .upstream_naive import (
    VGG16FeatureAdapter,
    _experiment,
    _runtime_environment,
    load_upstream_modules,
    source_hashes as upstream_source_hashes,
)


TOTAL_STEPS = 25_000
PREFIX_STEPS = 5_000
FIA_CONFIG = {
    "n_iter": TOTAL_STEPS,
    "lr": 0.1,
    "tv": 0.1,
    "fi": 1.0,
    "optimizer": "adam",
    "scheduler": "cosine",
    "log_every": 5000,
}


class StagedFIAError(RuntimeError):
    pass


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    array = value.numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def _cpu_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


def _device_state(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _device_state(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_device_state(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_device_state(item, device) for item in value)
    return value


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch_cpu": torch.get_rng_state().clone()}
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        tv=FIA_CONFIG["tv"], opt_fi=FIA_CONFIG["optimizer"], sch_fi=FIA_CONFIG["scheduler"],
        lr_fi=FIA_CONFIG["lr"], ds="imagenet", n_iter_fi=TOTAL_STEPS,
        use_labels=False, gm=0.0, fi=FIA_CONFIG["fi"],
    )


def _settings(seed: int, device: torch.device) -> dict[str, Any]:
    return {
        **FIA_CONFIG,
        "seed": seed,
        "device": str(device),
        "backend": "official_cpa/src/feature_inversion.py:Direct",
        "upstream_source_hashes": upstream_source_hashes(),
        "schedule_semantics": "constructed_with_T_max_25000_outer_stop_only",
    }


def construct_attack(target_z: torch.Tensor, *, output: Path, seed: int = 42,
                     device: str | torch.device = "cuda") -> tuple[Any, dict[str, Any]]:
    """Construct the unchanged official Direct attack and record its real init."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise StagedFIAError("registered Direct FIA requires CUDA")
    if target_z.dtype != torch.float32 or target_z.ndim != 2 or not bool(torch.isfinite(target_z).all()):
        raise StagedFIAError("target_z must be finite FP32 rank-2")
    settings = _settings(seed, device)
    output.mkdir(parents=True, exist_ok=True)
    _, cache_root, hashes = _experiment(output / "upstream_cache", "fia", target_z, settings)
    seed_all(seed)
    if device.type == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    upstream = load_upstream_modules()
    started = time.perf_counter()
    with _runtime_environment(upstream, device, cache_root):
        model = VGG16FeatureAdapter().to(device)
        # Match the existing adapter: model construction consumes no image RNG.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        attack = upstream.feature_inversion.Direct(
            target_z.detach().to(device=device, dtype=torch.float32), model, _args(), None, None,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    meta = {
        "seed": seed,
        "device": str(device),
        "target_z_sha256": tensor_sha256(target_z),
        "initial_image_sha256": tensor_sha256(attack.S_hat),
        "target_shape": list(target_z.shape),
        "settings": settings,
        "source_hashes": hashes,
        "setup_wall_seconds": time.perf_counter() - started,
        "schedule_T_max": TOTAL_STEPS,
    }
    return attack, meta


def _checkpoint_payload(attack: Any, *, completed: int, log: list[dict[str, Any]],
                        parent_checkpoint_sha256: str | None, branch_meta: dict[str, Any],
                        wall_seconds: float) -> dict[str, Any]:
    return {
        "schema": "CPA_SCREEN3_DIRECT_CHECKPOINT_V1",
        "completed_iterations": completed,
        "attack_state": _cpu_state(attack.get_attack_state()),
        "raw_image_sha256": tensor_sha256(attack.S_hat),
        "log": log,
        "rng_state": _cpu_state(_rng_state()),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "branch_meta": branch_meta,
        "fia_config": FIA_CONFIG,
        "scheduler_horizon": TOTAL_STEPS,
        "wall_seconds": wall_seconds,
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _boxed(attack: Any, raw: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(raw, (1 - attack.mean) / attack.std), -attack.mean / attack.std)


@torch.no_grad()
def score_attack(attack: Any, target_z: torch.Tensor) -> dict[str, Any]:
    """Score a cloned endpoint without changing raw image or optimizer state."""
    before = tensor_sha256(attack.S_hat)
    raw = attack.S_hat.detach().clone()
    boxed = _boxed(attack, raw.clone())
    _, z_hat = attack.model(boxed, return_z=True)
    distances = (1 - attack.cosine_similarity(target_z.to(attack.device), z_hat)).detach().cpu()
    tv = attack.total_variation(boxed).detach().cpu()
    result = {
        "feature_cosine_distances": [float(x) for x in distances],
        "feature_cosine_distance_mean": float(distances.mean()),
        "tv_normalized": float(tv),
        "score": float(distances.mean() + 0.1 * tv),
        "raw_image_sha256_before": before,
        "raw_image_sha256_after": tensor_sha256(attack.S_hat),
        "raw_state_unchanged": before == tensor_sha256(attack.S_hat),
        "score_semantics": "post_update_clone_feature_cosine_distance_plus_0.1_normalized_TV",
    }
    if not result["raw_state_unchanged"] or not all(torch.isfinite(distances).tolist()) or not torch.isfinite(tv):
        raise StagedFIAError("screen endpoint scoring mutated state or produced nonfinite values")
    return result


def _candidate(attack: Any) -> torch.Tensor:
    candidate = attack.get_rec().detach().cpu().contiguous()
    if candidate.dtype != torch.float32 or not bool(torch.isfinite(candidate).all()):
        raise StagedFIAError("Direct candidate is not finite FP32")
    if bool((candidate < -1e-5).any()) or bool((candidate > 1 + 1e-5).any()):
        raise StagedFIAError("Direct candidate is outside pixel box")
    return candidate.clamp(0, 1)


def _validate_resume(attack: Any, checkpoint: dict[str, Any], *, expected_parent: str,
                     target_hash: str, source_hash: str) -> None:
    if checkpoint.get("schema") != "CPA_SCREEN3_DIRECT_CHECKPOINT_V1":
        raise StagedFIAError("checkpoint schema mismatch")
    if int(checkpoint.get("completed_iterations", -1)) != PREFIX_STEPS:
        raise StagedFIAError("resume requires an authenticated 5000-step checkpoint")
    if checkpoint.get("fia_config") != FIA_CONFIG or checkpoint.get("scheduler_horizon") != TOTAL_STEPS:
        raise StagedFIAError("checkpoint schedule/config mismatch")
    meta = checkpoint.get("branch_meta", {})
    if meta.get("target_z_sha256") != target_hash or meta.get("source_sha256") != source_hash:
        raise StagedFIAError("checkpoint target/source binding mismatch")
    if expected_parent != meta.get("checkpoint_role"):
        raise StagedFIAError("checkpoint role mismatch")
    state = _device_state(checkpoint["attack_state"], attack.S_hat.device)
    attack.set_attack_state(state)
    if tensor_sha256(attack.S_hat) != checkpoint.get("raw_image_sha256"):
        raise StagedFIAError("raw image did not restore exactly")
    _restore_rng(_device_state(checkpoint["rng_state"], torch.device("cpu")))
    if attack.sch is None or attack.sch.state_dict().get("T_max") != TOTAL_STEPS:
        raise StagedFIAError("cosine scheduler horizon was not restored")


def run_trajectory(target_z: torch.Tensor, *, output: Path, source_sha256: str,
                   arm: str, end_steps: int, seed: int = 42,
                   parent_checkpoint: Path | None = None,
                   parent_role: str | None = None,
                   device: str | torch.device = "cuda",
                   checkpoint_at: int = PREFIX_STEPS,
                   progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Run a fresh prefix or a 20000-step child continuation."""
    if end_steps not in {PREFIX_STEPS, TOTAL_STEPS}:
        raise StagedFIAError("registered endpoints are 5000 and 25000")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing FIA branch output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    configure_runtime(42, threads=1)
    attack, construct_meta = construct_attack(target_z, output=output, seed=seed, device=device)
    target_hash = tensor_sha256(target_z)
    parent_hash = None
    completed = 0
    log: list[dict[str, Any]] = []
    if parent_checkpoint is not None:
        parent_checkpoint = Path(parent_checkpoint).resolve()
        checkpoint = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
        parent_hash = file_sha256(parent_checkpoint)
        _validate_resume(attack, checkpoint, expected_parent=parent_role or "MEMBER_PREFIX_5000",
                         target_hash=target_hash, source_hash=source_sha256)
        completed = PREFIX_STEPS
        log = list(checkpoint["log"])
    branch_meta = {
        "arm": arm, "target_z_sha256": target_hash, "source_sha256": source_sha256,
        "checkpoint_role": (("CENTER_PREFIX_5000" if arm == "center_25000" else "MEMBER_PREFIX_5000")
                            if parent_checkpoint is None and end_steps == PREFIX_STEPS else arm),
        "initial_image_sha256": construct_meta["initial_image_sha256"],
        "device": str(device), "fia_config": FIA_CONFIG,
    }
    started = time.perf_counter()
    prefix_checkpoint_path: Path | None = None
    prefix_candidate_path: Path | None = None
    for iteration in range(completed, end_steps):
        losses = attack.step()
        current = iteration + 1
        if current == 1 or current % FIA_CONFIG["log_every"] == 0 or current == end_steps:
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(torch.device(device))
            entry = {"iter": current, **losses, "time_sec": time.perf_counter() - started}
            log.append(entry)
            if progress is not None:
                progress({"arm": arm, "iter": current, "total": end_steps, **losses})
        if current == checkpoint_at and end_steps == TOTAL_STEPS and parent_checkpoint is None:
            prefix_checkpoint_path = output / "checkpoint_5000.pt"
            _save_checkpoint(prefix_checkpoint_path, _checkpoint_payload(
                attack, completed=PREFIX_STEPS, log=log, parent_checkpoint_sha256=None,
                branch_meta={**branch_meta, "checkpoint_role":
                             "CENTER_PREFIX_5000" if arm == "center_25000" else "MEMBER_PREFIX_5000"},
                wall_seconds=time.perf_counter() - started,
            ))
            prefix_candidate_path = output / "candidate_5000.pt"
            torch.save({"pixels": _candidate(attack)}, prefix_candidate_path)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))
    elapsed = time.perf_counter() - started
    if end_steps == PREFIX_STEPS and parent_checkpoint is None:
        checkpoint_path = output / "checkpoint_5000.pt"
        payload = _checkpoint_payload(
            attack, completed=PREFIX_STEPS, log=log, parent_checkpoint_sha256=None,
            branch_meta=branch_meta, wall_seconds=elapsed,
        )
        _save_checkpoint(checkpoint_path, payload)
    elif end_steps == TOTAL_STEPS:
        checkpoint_path = output / "checkpoint_25000.pt"
        payload = _checkpoint_payload(
            attack, completed=TOTAL_STEPS, log=log, parent_checkpoint_sha256=parent_hash,
            branch_meta=branch_meta, wall_seconds=elapsed,
        )
        _save_checkpoint(checkpoint_path, payload)
        if prefix_checkpoint_path is None:
            raise StagedFIAError("25000 trajectory did not produce its 5000 checkpoint")
    else:
        raise StagedFIAError("unregistered trajectory shape")
    candidate = _candidate(attack)
    torch.save({"pixels": candidate}, output / "candidate.pt")
    result = {
        "status": "COMPLETED", "arm": arm, "stage": "INVERT",
        "completed_iterations": end_steps, "total_schedule_iterations": TOTAL_STEPS,
        "checkpoint": checkpoint_path.as_posix(), "checkpoint_sha256": file_sha256(checkpoint_path),
        "prefix_checkpoint": prefix_checkpoint_path.as_posix() if prefix_checkpoint_path else None,
        "prefix_checkpoint_sha256": file_sha256(prefix_checkpoint_path) if prefix_checkpoint_path else None,
        "candidate": (output / "candidate.pt").as_posix(),
        "candidate_sha256": file_sha256(output / "candidate.pt"),
        "prefix_candidate": prefix_candidate_path.as_posix() if prefix_candidate_path else None,
        "prefix_candidate_sha256": file_sha256(prefix_candidate_path) if prefix_candidate_path else None,
        "candidate_tensor_sha256": tensor_sha256(candidate),
        "target_z_sha256": target_hash, "source_sha256": source_sha256,
        "initial_image_sha256": construct_meta["initial_image_sha256"],
        "parent_checkpoint_sha256": parent_hash,
        "log": log, "fia_config": FIA_CONFIG,
        "runtime": construct_meta, "wall_seconds": elapsed,
        "allocated_cuda_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
    }
    if end_steps == PREFIX_STEPS:
        result["score"] = score_attack(attack, target_z)
        result["score"]["checkpoint_sha256"] = result["checkpoint_sha256"]
        result["score"]["candidate_sha256"] = result["candidate_sha256"]
        result["score"]["completed_iterations"] = PREFIX_STEPS
    _json(output / "result.json", result)
    return result
