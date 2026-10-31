"""Pause/resume original Direct FIA with a fixed full-horizon cosine schedule.

The original objective and step/state methods are unmodified. The stop count
is deliberately excluded from the experiment identity; the final horizon is
always included. Checkpoints preserve Adam, scheduler and all RNG states.
"""
from __future__ import annotations

from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from .seed import seed_all
from .upstream_naive import (
    VGG16FeatureAdapter, _experiment, _runtime_environment, _save_checkpoint,
    _write_json, load_upstream_modules,
)


def optimize_until(attack, *, stop: int, horizon: int, log_every: int,
                   output: Path, device: torch.device) -> tuple[list[dict], float]:
    if not 0 < stop <= horizon or log_every <= 0:
        raise ValueError("Invalid staged iteration schedule")
    checkpoint = output / "fia_checkpoint.pt"
    completed, previous_runtime, log = 0, 0., []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        completed = int(state["completed_iterations"])
        if state["horizon"] != horizon or not 0 <= completed <= stop:
            raise RuntimeError("Cannot rewind FIA or change the cosine horizon")
        attack.set_attack_state(state["attack_state"])
        previous_runtime, log = float(state["wall_time_sec"]), state["log"]
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in state["rng_cuda"]])
        del state
    started = time.perf_counter()
    for iteration in range(completed, stop):
        losses = attack.step()
        if not all(np.isfinite(value) for value in losses.values()):
            raise RuntimeError("Nonfinite Direct FIA loss")
        if iteration == 0 or (iteration + 1) % log_every == 0 or iteration + 1 == stop:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = previous_runtime + time.perf_counter() - started
            log.append({"iter": iteration, **losses, "time_sec": elapsed})
            _save_checkpoint(checkpoint, {
                "completed_iterations": iteration + 1, "horizon": horizon,
                "attack_state": attack.get_attack_state(), "log": log, "wall_time_sec": elapsed,
                "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            })
            _write_json(output / "progress.json", {"completed_iterations": iteration + 1,
                        "stage_stop": stop, "horizon": horizon, "wall_time_sec": elapsed,
                        "losses": losses, "learning_rate": float(attack.opt.param_groups[0]["lr"])})
            print(f"staged FIA iter={iteration+1}/{horizon} stop={stop} loss={losses['loss']:.6g} time={elapsed:.1f}s", flush=True)
    return log, previous_runtime + time.perf_counter() - started


def run_staged_fia(*, target_z: torch.Tensor, device: torch.device, config: dict,
                    output_dir: Path, seed: int, stop: int) -> dict:
    device = torch.device(device)
    if target_z.ndim != 2 or target_z.shape[0] <= 0 or not torch.isfinite(target_z).all():
        raise ValueError("FIA requires finite feature targets")
    settings = {"seed": int(seed), "device": str(device), "n_iter": int(config["n_iter"]),
                "lr": float(config["lr"]), "tv": float(config["tv"]), "fi": float(config["fi"]),
                "optimizer": config["optimizer"], "scheduler": config["scheduler"],
                "log_every": int(config["log_every"]),
                "backend": "official_cpa/src/feature_inversion.py:Direct",
                "driver": "staged_fia_v1; fixed full cosine horizon; original Direct.step",
                "cudnn_benchmark": device.type == "cuda", "cudnn_deterministic": True}
    if settings["fi"] <= 0 or not 0 < stop <= settings["n_iter"]:
        raise ValueError("Invalid staged FIA config")
    output, cache_root, hashes = _experiment(output_dir, "fia", target_z, settings)
    seed_all(seed)
    if device.type == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    upstream = load_upstream_modules()
    args = SimpleNamespace(tv=settings["tv"], opt_fi=settings["optimizer"], sch_fi=settings["scheduler"],
                           lr_fi=settings["lr"], ds="imagenet", n_iter_fi=settings["n_iter"],
                           use_labels=False, gm=0., fi=settings["fi"])
    started = time.perf_counter()
    with _runtime_environment(upstream, device, cache_root):
        model = VGG16FeatureAdapter().to(device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        attack = upstream.feature_inversion.Direct(target_z.detach().to(device=device, dtype=torch.float32),
                                                    model, args, None, None)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    setup = time.perf_counter() - started
    log, runtime = optimize_until(attack, stop=stop, horizon=settings["n_iter"],
                                  log_every=settings["log_every"], output=output, device=device)
    rec = attack.get_rec().detach().cpu().contiguous()
    if not torch.isfinite(rec).all():
        raise RuntimeError("Nonfinite Direct FIA images")
    return {"rec": rec, "log": log, "completed_iterations": stop, "settings": settings,
            "optimizer_wall_time_sec": runtime, "setup_wall_time_sec": setup,
            "wall_time_sec": setup + runtime, "source_hashes": hashes}
