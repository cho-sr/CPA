# CPA_SIGNED_INT4_GDONLY_SCREEN3_FIA_20260918

Status: **COMPLETED**.

The registered original upstream CPA gradient-only pipeline completed nine separations and 125,000 FIA updates. The public gate passed before private loading. The frozen public selection was `[2, 4, 1]`; the offline comparison evaluated 17 sets and reproduced all four archived controls with maximum absolute residual `0.0` under the unchanged `2e-8` tolerance.

The first reporting pass failed after the public computation because of a PIL array-axis error. That failure is preserved in `failure.json`. A reporting-only repair was recorded in `evaluation_repair_provenance.json`; it added zero optimizer steps and reran only the authorized saved-artifact evaluation/rendering in `evaluation_repair_20260918`.

Private evaluation consumed x only; labels and true updates were not consumed. Actual-image cell feasibility: **NOT_EVALUATED**.

Final comparison: `C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results_private\signed_int4_gdonly_randomized8_screen3_fia_B16_T10_20260918\evaluation_repair_20260918\comparison.json`
Final slides: `C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results_private\signed_int4_gdonly_randomized8_screen3_fia_B16_T10_20260918\evaluation_repair_20260918\slide_png_int4_gdonly_20260918.zip`
