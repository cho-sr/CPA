#!/usr/bin/env python3
"""Deprecated QGM entrypoint.

The old implementation compared a single-image gradient with a multi-step
FedAvg update.  Use ``05_quant_aware_fedavg_refinement_cpa.py`` for the
corrected update-matching refinement.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Deprecated QGM entrypoint. Run quantized_update_cpa/05_quant_aware_fedavg_refinement_cpa.py "
        "for CPA-initialized quantization-aware FedAvg update matching."
    )


if __name__ == "__main__":
    main()
