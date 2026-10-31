#!/usr/bin/env python
"""Run the official CPA scripts with ImageNet-only optional stubs."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import types
from pathlib import Path


def install_torchtext_stub() -> None:
    """The official ImageNet path does not use torchtext, but imports it."""
    if "torchtext" in sys.modules:
        return

    torchtext = types.ModuleType("torchtext")
    data = types.ModuleType("torchtext.data")
    data_utils = types.ModuleType("torchtext.data.utils")
    data_functional = types.ModuleType("torchtext.data.functional")
    vocab = types.ModuleType("torchtext.vocab")

    def get_tokenizer(*args, **kwargs):
        raise RuntimeError("torchtext stub is only for non-text CPA runs")

    def to_map_style_dataset(*args, **kwargs):
        raise RuntimeError("torchtext stub is only for non-text CPA runs")

    class GloVe:  # noqa: D401 - stub
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torchtext stub is only for non-text CPA runs")

    data_utils.get_tokenizer = get_tokenizer
    data_functional.to_map_style_dataset = to_map_style_dataset
    vocab.GloVe = GloVe

    sys.modules["torchtext"] = torchtext
    sys.modules["torchtext.data"] = data
    sys.modules["torchtext.data.utils"] = data_utils
    sys.modules["torchtext.data.functional"] = data_functional
    sys.modules["torchtext.vocab"] = vocab


def install_wandb_stub() -> None:
    """Disable wandb service startup while keeping the official logging calls alive."""
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

    def log(*args, **kwargs):
        return None

    class Image:  # noqa: D401 - stub
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    wandb.init = init
    wandb.log = log
    wandb.Image = Image
    wandb.config = _Config()
    wandb.run = _Run()
    sys.modules["wandb"] = wandb


def install_matplotlib_compat() -> None:
    """Keep the official plotting code working on newer Matplotlib builds."""
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
        self.draw()
        rgba = memoryview(self.buffer_rgba())
        import numpy as np

        arr = np.asarray(rgba).reshape(self.get_width_height()[::-1] + (4,))
        return arr[:, :, :3].tobytes()

    FigureCanvasAgg.tostring_rgb = tostring_rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, choices=["attack", "collect_grads"])
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    official_src = repo_root / "official_cpa" / "src"
    install_torchtext_stub()
    install_wandb_stub()
    install_matplotlib_compat()
    sys.path.insert(0, str(official_src))
    script_args = args.script_args
    if script_args[:1] == ["--"]:
        script_args = script_args[1:]
    sys.argv = [str(official_src / f"{args.script}.py"), *script_args]
    runpy.run_path(str(official_src / f"{args.script}.py"), run_name="__main__")


if __name__ == "__main__":
    main()
