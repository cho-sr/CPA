#!/usr/bin/env python
"""Stage 0 runner for the official Cocktail Party Attack pipeline.

Design goal: keep the *upstream* scripts and their argparse defaults as the
single source of truth, so "use the repo hyperparameters as-is" is literally
true.  This runner therefore

  * installs the compatibility shims the upstream code needs on this machine
    (torchtext / wandb / matplotlib / DataLoader workers),
  * applies two *minimal, documented* fixes for upstream bugs (see
    ``stage0/notes/cpa_code_reading.md``),
  * turns a YAML config into the exact CLI argv the upstream README documents,
  * and then executes the upstream script unmodified via ``runpy``.

Usage
-----
    python stage0/run_cpa.py --config stage0/configs/fc2_tiny_imagenet.yaml --step train
    python stage0/run_cpa.py --config ... --step collect_grads
    python stage0/run_cpa.py --config ... --step attack --set batch_size=128

    # or fully manual, upstream argv passed through:
    python stage0/run_cpa.py --src official --step attack -- --model=fc2 --ds=tiny_imagenet ...
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SRC_VARIANTS = {
    # Byte-identical to facebookresearch/cocktail_party_attack @ main.
    # NOTE: in this variant the negentropy term is computed and then
    # overwritten with 0.0 (gradient_inversion.py:163).
    "official": REPO_ROOT / "official_cpa" / "src",
    # Local variant with the paper's log-cosh negentropy contrast restored
    # (properly standardized sources, numerically stable log cosh).
    "negentropy": REPO_ROOT / "cocktail_party_attack" / "src",
}


# --------------------------------------------------------------------------
# Compatibility shims
# --------------------------------------------------------------------------
SHIM_DIR = Path(__file__).resolve().parent / "shims"


def install_torchtext_shim() -> None:
    """Shadow the env's unusable ``torchtext`` via ``sys.path``.

    Upstream ``datasets.py`` imports torchtext at module scope for the
    out-of-scope IMDB pipeline, and the installed torchtext 0.18 fails to load
    its compiled extension under torch 2.13.

    A ``sys.modules`` stub is not sufficient: Windows ``spawn`` re-executes the
    DataLoader worker's ``__main__`` from its *file path*, so the child gets a
    fresh interpreter with no stub.  ``sys.path`` is part of the spawn
    preparation data and therefore inherited -- shadowing on the path fixes
    parent and workers alike.  See ``stage0/shims/torchtext/__init__.py``.
    """
    shim = str(SHIM_DIR)
    if shim not in sys.path:
        sys.path.insert(0, shim)


def install_wandb_stub() -> None:
    """Replace wandb with an inert stub.

    Upstream logging is wandb-first; ``AttackLog`` still writes every metric to
    the pickles we actually consume, so stubbing wandb loses nothing and avoids
    a network dependency / service process per run.
    """
    wandb = types.ModuleType("wandb")

    class _Run:
        id = "disabled"

        @staticmethod
        def get_url():
            return "disabled"

    class _Config:
        @staticmethod
        def update(*args, **kwargs):
            return None

    def init(*args, **kwargs):
        wandb.run = _Run()
        return wandb.run

    class Image:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    wandb.init = init
    wandb.log = lambda *a, **k: None
    wandb.Image = Image
    wandb.config = _Config()
    wandb.run = _Run()
    sys.modules["wandb"] = wandb


def install_matplotlib_compat() -> None:
    """``log_utils.get_pil_plot`` calls ``FigureCanvasAgg.tostring_rgb``,
    removed in Matplotlib 3.10.  Re-add it on top of ``buffer_rgba``."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except Exception:
        return

    if hasattr(FigureCanvasAgg, "tostring_rgb"):
        return

    def tostring_rgb(self):
        import numpy as np

        self.draw()
        arr = np.asarray(memoryview(self.buffer_rgba()))
        arr = arr.reshape(self.get_width_height()[::-1] + (4,))
        return arr[:, :, :3].tobytes()

    FigureCanvasAgg.tostring_rgb = tostring_rgb


def patch_dataloader_workers(num_workers: int) -> None:
    """Force ``get_dataloaders(num_workers=...)``.

    Upstream defaults to 8 workers.  On Windows those are ``spawn``ed and each
    child re-executes the *upstream* script as ``__mp_main__`` (runpy sets
    ``__main__.__file__`` to it), so runtime stubs installed here are invisible
    to the child.  Only the ``sys.path`` torchtext shim survives; everything
    else the child imports (wandb, matplotlib) is import-safe.
    """
    import datasets as ds_mod

    original = ds_mod.get_dataloaders

    def wrapper(*args, **kwargs):
        kwargs["num_workers"] = num_workers
        return original(*args, **kwargs)

    ds_mod.get_dataloaders = wrapper
    # ``train.py`` / ``collect_grads.py`` / ``attack.py`` do
    # ``from datasets import get_dataloaders`` at import time, i.e. after this
    # patch runs, so rebinding the module attribute is enough.


def patch_model_type_attr() -> None:
    """Minimal fix for an upstream bug in ``collect_grads.py``.

    ``collect_grads.py:104`` reads ``model.type``, but the models define
    ``model_type``.  ``nn.Module.type`` is a real bound method, so the
    comparison ``model_type == "conv"`` is always False and the penultimate
    embeddings ``z`` are never recorded -- which then makes the conv-model
    branch of the attack (``EmbEval(emb, ...)``) fail on an empty tensor.

    We shadow the instance attribute ``.type`` with the model-type string.  This
    is confined to the collector process; nothing in the collector calls
    ``Module.type()``.
    """
    import models as models_mod

    original = models_mod.get_model

    def wrapper(*args, **kwargs):
        model = original(*args, **kwargs)
        base = model.module if hasattr(model, "module") else model
        model_type = getattr(base, "model_type", None)
        if model_type is not None:
            object.__setattr__(model, "type", model_type)
        return model

    models_mod.get_model = wrapper


# The torchtext shim must be on sys.path before any upstream import, and must
# survive into spawned DataLoader workers -- hence import time, not main().
install_torchtext_shim()


def patch_seed(seed: int) -> None:
    """Add the ``--seed`` that upstream lacks.

    ``utils.setup()`` hardcodes ``set_seed(42)`` and there is no CLI seed, so
    upstream runs are not seed-varying.  We wrap ``setup`` and re-seed after it.
    ``--seed 42`` is a no-op, i.e. default behaviour is unchanged.
    """
    import utils as utils_mod

    original_setup = utils_mod.setup

    def wrapper(*args, **kwargs):
        logger = original_setup(*args, **kwargs)
        utils_mod.set_seed(seed)
        logger.info(f"[stage0] seed={seed}")
        return logger

    utils_mod.setup = wrapper


def patch_w_init(mode: str) -> None:
    """Make CPA's unmixing-matrix initialization selectable.

    Upstream initializes ``W_hat`` with ``torch.nn.init.eye_``
    (gradient_inversion.py:94) -- fully deterministic.  Re-running the attack on
    the same gradient therefore always gives the identical result, so the
    "10 independent initializations" that WSRR needs cannot be produced by
    upstream code at all.

    ``eye`` (default) is upstream behaviour, byte-for-byte.
    """
    if mode == "eye":
        return

    import torch

    import gradient_inversion as gi_mod

    base_cls = gi_mod.attack_dict["cp"]

    class CocktailPartyAttackInit(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            with torch.no_grad():
                if mode == "randn":
                    # rows are L2-normalized inside step(), so the scale is free
                    self.W_hat.normal_(mean=0.0, std=1.0)
                elif mode == "orthogonal":
                    torch.nn.init.orthogonal_(self.W_hat)
                elif mode == "eye_perturbed":
                    torch.nn.init.eye_(self.W_hat)
                    self.W_hat.add_(torch.randn_like(self.W_hat) * 0.01)
                else:
                    raise ValueError(f"unknown w_init: {mode}")

    gi_mod.attack_dict["cp"] = CocktailPartyAttackInit
    # ``get_gi`` resolves ``attack_dict`` from its module globals at call time,
    # and ``attack.py`` has not been executed yet, so this takes effect.


# --------------------------------------------------------------------------
# Config -> argv
# --------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def config_to_argv(cfg: dict, step: str, overrides: dict) -> list[str]:
    """Flatten ``common`` + ``<step>`` config sections into upstream argv.

    ``True`` becomes a bare ``--flag``; ``False``/``None`` is dropped.
    """
    merged: dict = {}
    merged.update(cfg.get("common", {}) or {})
    merged.update(cfg.get(step, {}) or {})
    merged.update(overrides)

    argv: list[str] = []
    for key, value in merged.items():
        if value is None or value is False:
            continue
        if value is True:
            argv.append(f"--{key}")
        else:
            argv.append(f"--{key}={value}")
    return argv


def parse_overrides(items: list[str]) -> dict:
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        if value.lower() in {"true", "false"}:
            out[key] = value.lower() == "true"
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--step",
        required=True,
        choices=["train", "collect_grads", "collect_updates", "attack"],
    )
    parser.add_argument(
        "--src",
        default=None,
        choices=sorted(SRC_VARIANTS),
        help="which CPA source tree to run (default: config `src`, else 'official')",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="42 = upstream's hardcoded value (no behavioural change)",
    )
    parser.add_argument(
        "--w_init",
        default="eye",
        choices=["eye", "randn", "orthogonal", "eye_perturbed"],
        help="CPA unmixing-matrix init; 'eye' = upstream (deterministic)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config entry (repeatable)",
    )
    parser.add_argument("--print_argv", action="store_true")
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="extra upstream args, after a bare --",
    )
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    src_name = args.src or cfg.get("src", "official")
    src_dir = SRC_VARIANTS[src_name]
    if not src_dir.is_dir():
        raise SystemExit(f"source tree not found: {src_dir}")

    argv = config_to_argv(cfg, args.step, parse_overrides(args.overrides))
    extra = args.passthrough
    if extra[:1] == ["--"]:
        extra = extra[1:]
    argv += extra

    script = src_dir / f"{args.step}.py"
    if not script.is_file():
        raise SystemExit(f"upstream script not found: {script}")

    print(f"[stage0] src      = {src_name} ({src_dir})", flush=True)
    print(f"[stage0] script   = {script.name}", flush=True)
    print(f"[stage0] argv     = {' '.join(argv)}", flush=True)
    if args.print_argv:
        return

    # cwd must be the repo root: upstream resolves ./exp and ./datasets relatively.
    os.chdir(REPO_ROOT)

    install_wandb_stub()
    install_matplotlib_compat()
    sys.path.insert(0, str(src_dir))

    patch_dataloader_workers(args.num_workers)
    patch_seed(args.seed)
    if args.step in {"collect_grads", "collect_updates"}:
        patch_model_type_attr()
    if args.step == "attack":
        patch_w_init(args.w_init)

    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
