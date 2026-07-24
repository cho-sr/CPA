#!/usr/bin/env python3
"""Deprecated randomized-dequant FIA postprocessor.

Use ``03_randomized_dequant_ensemble_cpa.py --run_attack --run_aggregate_fia``.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Deprecated entrypoint. Run quantized_update_cpa/03_randomized_dequant_ensemble_cpa.py "
        "--run_attack --run_aggregate_fia for blind ensemble alignment and separate Direct FIA."
    )


if __name__ == "__main__":
    main()
