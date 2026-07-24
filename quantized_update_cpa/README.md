# Quantized-Update CPA Experiments

The corrected experiment definitions and commands are maintained in
`../README_EXPERIMENTS.md`.

Canonical entrypoints:

- `01_fp32_update_cpa.py`
- `02_naive_4bit_update_cpa.py`
- `03_randomized_dequant_ensemble_cpa.py`
- `04_quant_aware_ica_objective_cpa.py`
- `05_quant_aware_fedavg_refinement_cpa.py`

Historical logs, reports, and legacy helper scripts in this directory may refer
to the earlier QAS/QGM prototypes. Do not mix those outputs with the corrected
`outputs_v2/` results unless the manifest explicitly matches the corrected
definition.
