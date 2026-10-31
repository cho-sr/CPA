# Publication figure archive: validity correction

The former five-figure package is withdrawn from current reporting because
figures 2, 3(b), and 5 depend on the invalid no-CPA control. The user confirmed
failure to control experimental variables on 2026-09-09.

Use only the requested same80 client-update comparison:
[PNG](../client_update_fp32_vs_4bit/client_update_fp32_vs_4bit.png) or
[ZIP](../client_update_fp32_vs_4bit.zip).
The FP32 data come from `m6_uia_fp32_B16_fia`, and the 4-bit data from
`m7_uia_4bit_B16`. Neither comes from `m7_no_cpa_control`.

| Archived figure | Current scientific-use status |
|---|---|
| 1: client-update reconstruction | Independent; current same80 comparison |
| 2: gradient-inversion controls | Withdrawn: invalid no-CPA comparison and IUT claim |
| 3: pairing and reference controls | Existing combined figure withdrawn; panel (a) raw pairing comparison remains independent |
| 4: prediction and leakage | Independent, but outside the user's current requested figure scope |
| 5: normalized LPIPS and cutoff counts | Withdrawn: both panels depend on no-CPA |

The original figures, CSVs and artifact manifest are retained as an archive.
The old manifest records the historical files; it is not a validity approval or
an up-to-date hash list after this correction. Captions 2/3/5 are withdrawn.
See [INVALIDATED.md](INVALIDATED.md), [VALIDITY.json](VALIDITY.json), and
[the central registry](../result_validity.json).

`render_reconstruction_metrics.py` now defaults to figure 1 only. Attempts to
regenerate the invalid figures or package the former five-figure bundle fail
before reading the invalid source or writing output. The current independent
bundle can be prepared with `../client_update_fp32_vs_4bit/prepare_bundle.py`.
