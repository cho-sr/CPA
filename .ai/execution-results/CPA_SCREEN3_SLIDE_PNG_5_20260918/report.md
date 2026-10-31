# CPA_SCREEN3_SLIDE_PNG_5_20260918

Status: COMPLETED.

Created the isolated INT5 slide export from the completed, evaluated and PUBLIC_GATE_PASSED frozen 25,000-step comparison. The output contains one all-16 overview and four detail pages covering 01–04, 05–08, 09–12 and 13–16. Every PNG is exactly 3840x2160.

The adapter reused the strict public authentication, private-x loading, stored per-image Hungarian alignment and proven 16:9 layout helpers. It did not invoke the hardcoded INT8/7/6 main, modify prior outputs, recompute metrics/matching, run model inference or take optimizer steps. Labels, z and true updates were not consumed.

## Output

- INT5 overview: `C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results_private\slide_png_5_20260918\int5_25000_overview.png`
- ZIP: `C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results_private\slide_png_5_20260918\slide_png_5_20260918.zip`
- Detail pages and index: `C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results_private\slide_png_5_20260918`

Selection is [5, 7, 6], shown as original member IDs 05, 07 and 06. Every page uses Ground truth | same-bit naive baseline | public rank1 | rank2 | rank3, with the existing interpretation footer.

## Validation

- Focused INT5 adapter regression: 2 passed.
- Actual PNG validation: 5 files, all 3840x2160.
- Row coverage: overview 01–16; detail pages 01–04, 05–08, 09–12 and 13–16.
- ZIP validation: exactly the five PNGs and index.md.
- Visual inspection: INT5 overview and detail page 01 show five readable columns, member IDs 05/07/06, and unclipped headings.
- Existing comparison, candidate, metric, registration and active parent handoff bytes were preserved; hashes are recorded in integrity.json and slide_manifest.json.

Originating user thread: 01a0a7ab-fa58-7d22-ac52-3247b1dd2255.
