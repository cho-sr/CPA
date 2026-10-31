#!/usr/bin/env python
"""Stage 0 evaluation: LPIPS / PSNR / SSIM / embedding-CS / BRR / WSRR -> CSV.

Inputs are the ``{batch_size}_rec.pkl`` files written by upstream
``log_utils.AttackLog.save_to_disk``, which hold, per data batch:

    inp      true images, already in [0, 1] (attack.py:155 -> normalize "ds")
    emb      true embeddings z                       (conv models only)
    rec_emb  CPA-recovered |z_hat|, oracle-reordered (conv models only)
    rec      final reconstruction; for the conv path this is the FIA output,
             normalize(method="ds") -> [0, 1], and only ``n_sample_fi`` rows

``inp`` is subsampled to ``n_sample_fi`` for the conv path (attack.py:121-128),
so ``inp`` and ``rec`` are row-aligned.

Why recompute instead of reading ``*_fi_summary.pkl``
-----------------------------------------------------
Upstream calls ``peak_signal_noise_ratio(rec.view(-1), inp.view(-1))`` with
``data_range=None``, so torchmetrics infers the range from each target's own
min/max.  That is *not* PSNR at a fixed data range, and the BRR/WSRR indicator
1[PSNR > 18] is sensitive to it.  We recompute with an explicit
``data_range=1.0`` and also report the upstream-style value for comparison.

Metric definitions used here
----------------------------
leak indicator   I(i) = 1[PSNR(i) > psnr_thresh AND LPIPS(i) < lpips_thresh]
BRR   (per run, per batch)  = mean_i I(i)
WSRR  (per batch, over runs) = mean_k 1[ exists i : I(i, k) = 1 ]
      i.e. "over K independent runs, the fraction in which at least one -- the
      most vulnerable -- sample in the batch was reconstructed".

All numbers are ORACLE-ALIGNED: upstream ``eval_utils.Eval.fix_order`` matches
recovered sources to ground truth, and for the fc path also picks the better
sign per sample.  Report them as an attacker upper bound, never as an
attacker-achievable score.  See stage0/notes/cpa_code_reading.md 4.2.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_pickle(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def normalize_lpips(x):
    """LPIPS expects [-1, 1]; inputs here are [0, 1]."""
    return (x - 0.5) * 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rec",
        action="append",
        required=True,
        metavar="RUN_ID=PATH",
        help="a *_rec.pkl to evaluate, labelled with a run id (repeatable)",
    )
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--psnr_thresh", type=float, default=18.0)
    parser.add_argument("--lpips_thresh", type=float, default=0.1)
    parser.add_argument("--lpips_net", default="vgg", choices=["vgg", "alex", "squeeze"])
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--label",
        default="",
        help="free-text tag copied into every CSV row (e.g. 'src=official,bs=256')",
    )
    args = parser.parse_args()

    import numpy as np
    import torch
    from torchmetrics.functional import (
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type=args.lpips_net).to(device)

    runs: list[tuple[str, Path]] = []
    for item in args.rec:
        if "=" not in item:
            raise SystemExit(f"--rec expects RUN_ID=PATH, got {item!r}")
        run_id, path = item.split("=", 1)
        runs.append((run_id, Path(path)))

    per_sample: list[dict] = []

    for run_id, path in runs:
        if not path.is_file():
            raise SystemExit(f"not found: {path}")
        rec_dict = read_pickle(path)
        n_batches = len(rec_dict["rec"])
        print(f"[eval] {run_id}: {path}  ({n_batches} batches)", flush=True)

        for b in range(n_batches):
            inp = torch.as_tensor(np.asarray(rec_dict["inp"][b]), dtype=torch.float32)
            rec = torch.as_tensor(np.asarray(rec_dict["rec"][b]), dtype=torch.float32)
            if rec.ndim != 4:
                raise SystemExit(
                    f"{run_id} batch {b}: expected NCHW reconstructions, got {tuple(rec.shape)}"
                )
            n = min(inp.shape[0], rec.shape[0])
            inp, rec = inp[:n].to(device), rec[:n].to(device)

            emb = np.asarray(rec_dict["emb"][b]) if len(rec_dict["emb"]) > b else None
            rec_emb = (
                np.asarray(rec_dict["rec_emb"][b])
                if len(rec_dict["rec_emb"]) > b
                else None
            )
            cs = None
            if emb is not None and rec_emb is not None and emb.size and rec_emb.size:
                e = torch.as_tensor(emb, dtype=torch.float32, device=device)
                r = torch.as_tensor(rec_emb, dtype=torch.float32, device=device)
                m = min(e.shape[0], r.shape[0])
                cs = torch.nn.functional.cosine_similarity(
                    e[:m], r[:m], dim=-1, eps=1e-20
                ).abs()

            for i in range(n):
                a = rec[i : i + 1]
                t = inp[i : i + 1]
                psnr_fixed = peak_signal_noise_ratio(a, t, data_range=1.0).item()
                psnr_upstream = peak_signal_noise_ratio(
                    a.reshape(-1), t.reshape(-1)
                ).item()
                ssim = structural_similarity_index_measure(a, t, data_range=1.0).item()
                with torch.no_grad():
                    lp = lpips_fn(normalize_lpips(a), normalize_lpips(t)).item()
                per_sample.append(
                    {
                        "label": args.label,
                        "run_id": run_id,
                        "batch": b,
                        "sample": i,
                        "psnr": psnr_fixed,
                        "psnr_upstream_datarange": psnr_upstream,
                        "ssim": ssim,
                        "lpips": lp,
                        "emb_cs": (
                            float(cs[i].item()) if cs is not None and i < len(cs) else ""
                        ),
                        "leak": int(
                            psnr_fixed > args.psnr_thresh and lp < args.lpips_thresh
                        ),
                    }
                )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # ---- per-sample ----
    ps_path = out_prefix.with_name(out_prefix.name + "_per_sample.csv")
    fields = list(per_sample[0].keys())
    with open(ps_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(per_sample)

    # ---- per (run, batch) ----
    import statistics as st

    groups: dict[tuple[str, int], list[dict]] = {}
    for row in per_sample:
        groups.setdefault((row["run_id"], row["batch"]), []).append(row)

    pb_path = out_prefix.with_name(out_prefix.name + "_per_batch.csv")
    with open(pb_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "label",
                "run_id",
                "batch",
                "n_samples",
                "lpips_mean",
                "psnr_mean",
                "ssim_mean",
                "emb_cs_mean",
                "brr",
                "any_leak",
            ]
        )
        for (run_id, b), rows in sorted(groups.items()):
            cs_vals = [float(r["emb_cs"]) for r in rows if r["emb_cs"] != ""]
            w.writerow(
                [
                    args.label,
                    run_id,
                    b,
                    len(rows),
                    st.fmean(r["lpips"] for r in rows),
                    st.fmean(r["psnr"] for r in rows),
                    st.fmean(r["ssim"] for r in rows),
                    st.fmean(cs_vals) if cs_vals else "",
                    st.fmean(r["leak"] for r in rows),
                    max(r["leak"] for r in rows),
                ]
            )

    # ---- aggregate: mean +/- std over batches, plus BRR / WSRR ----
    batch_ids = sorted({row["batch"] for row in per_sample})
    run_ids = sorted({row["run_id"] for row in per_sample})

    def agg(vals):
        vals = list(vals)
        if not vals:
            return "", ""
        if len(vals) == 1:
            return vals[0], 0.0
        return st.fmean(vals), st.stdev(vals)

    # per-batch means pooled over every (run, batch) cell
    lpips_cells = [st.fmean(r["lpips"] for r in rows) for rows in groups.values()]
    psnr_cells = [st.fmean(r["psnr"] for r in rows) for rows in groups.values()]
    ssim_cells = [st.fmean(r["ssim"] for r in rows) for rows in groups.values()]
    brr_cells = [st.fmean(r["leak"] for r in rows) for rows in groups.values()]
    cs_cells = [
        st.fmean(float(r["emb_cs"]) for r in rows if r["emb_cs"] != "")
        for rows in groups.values()
        if any(r["emb_cs"] != "" for r in rows)
    ]

    # WSRR: for each data batch, fraction of runs with >= 1 leaked sample
    wsrr_per_batch = []
    for b in batch_ids:
        flags = [
            max(r["leak"] for r in groups[(run_id, b)])
            for run_id in run_ids
            if (run_id, b) in groups
        ]
        if flags:
            wsrr_per_batch.append(st.fmean(flags))

    sm_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    with open(sm_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "metric", "mean", "std", "n_cells", "note"])
        note = f"oracle-aligned; leak=1[PSNR>{args.psnr_thresh} & LPIPS<{args.lpips_thresh}]"
        for name, cells in [
            ("lpips", lpips_cells),
            ("psnr", psnr_cells),
            ("ssim", ssim_cells),
            ("emb_cs", cs_cells),
            ("brr", brr_cells),
        ]:
            m, s = agg(cells)
            w.writerow([args.label, name, m, s, len(cells), note])
        m, s = agg(wsrr_per_batch)
        w.writerow(
            [
                args.label,
                "wsrr",
                m,
                s,
                len(wsrr_per_batch),
                f"over K={len(run_ids)} runs: frac of runs with >=1 leaked sample; "
                f"averaged over {len(wsrr_per_batch)} data batches",
            ]
        )

    print(f"[eval] wrote {ps_path}")
    print(f"[eval] wrote {pb_path}")
    print(f"[eval] wrote {sm_path}")
    m, s = agg(lpips_cells)
    print(f"[eval] LPIPS = {m:.4f} +/- {s:.4f}  over {len(lpips_cells)} (run,batch) cells")
    m, s = agg(brr_cells)
    print(f"[eval] BRR   = {m:.4f} +/- {s:.4f}")
    if wsrr_per_batch:
        m, s = agg(wsrr_per_batch)
        print(f"[eval] WSRR  = {m:.4f} +/- {s:.4f}  (K={len(run_ids)} runs)")


if __name__ == "__main__":
    sys.exit(main())
