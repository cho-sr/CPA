#!/usr/bin/env python3
"""Deprecated 04 entrypoint.

Use ``04_quant_aware_ica_objective_cpa.py`` for the corrected experiment 04.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Deprecated entrypoint. Run quantized_update_cpa/04_quant_aware_ica_objective_cpa.py "
        "for the corrected quantization-aware ICA objective."
    )


if __name__ == "__main__":
    main()
