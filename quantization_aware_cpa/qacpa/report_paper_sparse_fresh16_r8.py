"""Regenerate the v4 fresh16 report from its own completed artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

from .run_m8_fedavg_naive import PROJECT, read_json
from .run_paper_sparse_fresh16_r8 import run_all, validate_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/cpa_fia_fresh16_R8_S5k_K3_F25k.json")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    cfg = read_json(config_path)
    validate_config(cfg, config_path)
    run_all(config_path, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
