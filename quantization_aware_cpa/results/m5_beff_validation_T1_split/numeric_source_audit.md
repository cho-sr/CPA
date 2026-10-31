run_tag="m5_beff_validation_T1_split"

# Prediction Count Audit

Source CSV: `C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results\m5_uia_beff_validation\summary.csv`
Direct recount: 81/96 exact, MAE 0.218750, max error 2.

The authoritative prediction-summary CSV contains 96 conditions; 81 predictions are exact (MAE 0.218750). No alternative overall count is used.

Parsed prediction-summary candidates:
- `C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results\m5_uia_beff_smoke\summary.csv`: 0/1 exact, MAE 2.000000, run_tags=['m5_uia_beff_smoke']
- `C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results\m5_uia_beff_validation\summary.csv`: 81/96 exact, MAE 0.218750, run_tags=['m5_uia_beff_validation']
