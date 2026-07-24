#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
Deprecated helper. Use the Windows PowerShell smoke/full scripts at repo root:

  ./run_01_03_smoke.ps1
  ./run_full_experiments.ps1

The corrected 03 implementation is:

  python quantized_update_cpa/03_randomized_dequant_ensemble_cpa.py --help
EOF
