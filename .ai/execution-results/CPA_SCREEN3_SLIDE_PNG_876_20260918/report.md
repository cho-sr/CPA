# CPA_SCREEN3_SLIDE_PNG_876_20260918

Status: COMPLETED.

Created an isolated CPU-only 16:9 presentation export from the authenticated frozen 25,000-step comparisons for INT8, INT7 and INT6. Each bit has one all-16 overview and four detail pages covering rows 01–04, 05–08, 09–12 and 13–16. Every page is exactly 3840x2160 PNG.

The renderer authenticated public candidates and registrations before loading the registered private x bundle, reused the stored per-image Hungarian correspondence, and did not recompute metrics or matching. No optimizer/model inference ran; optimizer_steps=0. Labels, z and true updates were not consumed.

## Output

- INT8 overview: `C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results_private\slide_png_8_7_6_20260918\int8_25000_overview.png`
- INT7 overview: `C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results_private\slide_png_8_7_6_20260918\int7_25000_overview.png`
- INT6 overview: `C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results_private\slide_png_8_7_6_20260918\int6_25000_overview.png`
- ZIP: `C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results_private\slide_png_8_7_6_20260918\slide_png_8_7_6_20260918.zip`
- Detail pages and index: `C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results_private\slide_png_8_7_6_20260918`

Selections preserve original member IDs: INT8 [1, 7, 5], INT7 [0, 5, 3], INT6 [1, 3, 4]. Every page uses Ground truth | same-bit naive baseline | public rank1 | rank2 | rank3. The footer states: Rows matched by pixel error; source identity is not verified.

## Validation

- Focused slide-layout regression: 3 passed.
- Actual PNG validation: 15 files, all 3840x2160.
- ZIP validation: exactly the 15 PNGs and index.md.
- Visual inspection: INT6 overview and INT6 detail page 01 show five columns, member IDs 01/03/04, and rows 01–16 / 01–04 respectively without overlap.
- Existing comparison, candidate, metric, figure, registration and active parent handoff bytes were preserved; hashes are recorded in integrity.json and slide_manifest.json.

Originating user thread: 01a0a7ab-fa58-7d22-ac52-3247b1dd2255.
