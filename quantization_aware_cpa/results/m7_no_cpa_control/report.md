# M7: CPA contribution controls

Status: **COMPLETE**. B=8, five fixed victim batches, 40 original images, victim seed=0, round-0 VGG16.

| Observation | CPA present: LPIPS / success count | No-CPA: LPIPS / success count |
|---|---:|---:|
| fp32 | 0.4448 / 2/40 | 0.7568 / 1/40 |
| 4bit | 0.5439 / 20/40 | 0.7538 / 1/40 |

The reused 4-bit row is **CPA (미수정) + naive dequant**. `naive` labels only the no-CPA controls; historical raw keys and filenames remain unchanged.

Requested historical mean 0.4448 includes four exact-source FIA fills (22,24,32,36); fp32 is a hybrid row, not pure CPA.

No-CPA uses seeded observed matrix rows, with only row L2 normalization. FIA uses the unchanged 25,000 iterations, TV=0.1, Adam lr=0.1, cosine schedule, FIA seed=42, and original chunk boundaries. Permutation-only matching is evaluation after FIA; positional LPIPS is also retained.

Success counts follow the preregistered control-calibrated lower-tail rule. Each image excludes its entire victim batch; it must have LPIPS strictly below both minima from the 32 remaining precision-matched no-CPA images and 32 prior-only images. `thresholds.json` is frozen from controls before counting candidate successes. These counts are descriptive, not per-image significance declarations.

| Comparison | Mean no-CPA minus CPA | p vs no-CPA | Mean prior minus CPA | p vs prior | IUT p | Rule passes |
|---|---:|---:|---:|---:|---:|---|
| fp32 | 0.3120 | 0.03125 | 0.3290 | 0.03125 | 0.03125 | True |
| 4bit | 0.2099 | 0.03125 | 0.2298 | 0.03125 | 0.03125 | True |
| fp32_same36_sensitivity | 0.3118 | 0.03125 | 0.3253 | 0.03125 | 0.03125 | True |

On the same 36 images excluding exact-source fills, fp32 CPA LPIPS=0.4465, no-CPA=0.7590, and prior-only=0.7730.

The primary 4-bit claim passes the fixed rule: positive mean effects against both controls and maximum one-sided p <= 0.05. Each test enumerates all 32 signs over five paired victim-batch means and includes ties. Batch sign symmetry is assumed; fp32 comparisons are secondary.

| Auxiliary distribution | Per-image / pair mean LPIPS | Rows |
|---|---:|---:|
| prior-only FIA | 0.7738 | 40 |
| fp32 shuffled (20 seeds) | 0.8090 | 800 |
| 4-bit shuffled (20 seeds) | 0.7989 | 800 |
| dataset mean image | 0.8055 | 40 |
| gray (0.5) | 0.8292 | 40 |
| uniform random image noise | 0.8566 | 40 |

Shuffled controls use 20 fixed permutations of the same 40 existing reconstructions; the 800 pair metrics per precision are dependent and are not 800 independent victims. Constants establish reference LPIPS values, not a mathematical lower bound. The dataset-average image uses the full transformed validation split.

| Shuffled precision | Mean across 20 seeds | SD of seed means | Min–max seed mean |
|---|---:|---:|---:|
| fp32 | 0.8090 | 0.0133 | 0.7678–0.8294 |
| 4bit | 0.7989 | 0.0124 | 0.7617–0.8128 |

![Common-axis per-image LPIPS histograms](lpips_histogram_overlay.png)

![Random no-CPA reconstructions: not cherry-picked](no_cpa_random_grid.png)

Grid indices: 14, 3, 23, 38, 20, 29, 2, 32, 37, 25, 39, 7; NumPy seed=42, no replacement. **not cherry-picked**.

Preregistration SHA-256: `41f402f5449deaadc92f37ec4a76909eafda329d89f11fb7d84b984580251539`. No Stage 2, `--refine-A`, or 4-bit GMA.
