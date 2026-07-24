#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
Deprecated helper. It belonged to the old randomized-dequant/QGM workflow and
called the modified fork directly.

Corrected replacements:

  python quantized_update_cpa/03_randomized_dequant_ensemble_cpa.py --run_attack --run_aggregate_fia
  python quantized_update_cpa/04_quant_aware_ica_objective_cpa.py --qica_ablation all
  python quantized_update_cpa/05_quant_aware_fedavg_refinement_cpa.py --help
EOF
