"""All80 reporting after observation-only random-candidate selection is frozen."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw
import torch

from .run_m8_fedavg_naive import read_json, read_csv, write_json, write_csv, resolve, file_hash, utc_now
from .report_m10_packed4 import _append_run_once, _metrics, _font, _rgb


def render_slide(originals, recs, rows, path: Path):
    if len(rows) != 80 or originals.shape != recs.shape or recs.shape != (80, 3, 224, 224):
        raise RuntimeError("The slide must show exactly all80 original/reconstruction pairs")
    canvas = Image.new("RGB", (3840, 2160), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((60, 28), "TOP-K 10% | Uniform random fill | CPA + FIA", font=_font(52), fill="#0f172a")
    draw.text((60, 102), "Same 80 images | 20 candidates at FIA 5k, top 3 continued to 25k | final selection by update consistency", font=_font(30), fill="#475569")
    tile, gap, cell_width, cell_height, left, top = 174, 8, 374, 242, 54, 188
    for i, row in enumerate(rows):
        x, y = left + (i % 10) * cell_width, top + (i // 10) * cell_height
        draw.text((x, y), f"g{int(row['global_index']):02d} | candidate {int(row['selected_candidate']):02d}", font=_font(21), fill="#334155")
        for j, (name, tensor) in enumerate((("Original", originals[i]), ("Reconstruction", recs[i]))):
            xx = x + j * (tile + gap)
            draw.text((xx, y + 26), name, font=_font(19), fill="#64748b")
            image = Image.fromarray(_rgb(tensor)).resize((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(image, (xx, y + 53))
    canvas.save(path)


def summarize(cfg: dict, manifest: dict, output: Path) -> dict:
    from .run_random_fill_same80 import checked_marker, validate_config, RANDOMIZATION
    from .random_fill import rank_candidates
    validate_config(cfg)
    identity = manifest["identity"]
    rows, batches, originals, recs = [], [], [], []
    for batch in range(5):
        folder = output / f"batch{batch:02d}"
        done = checked_marker(folder / "summary.json", identity)
        if done is None or (done["n"], done["T"], done["n_images"], done["batch_index"]) != (16, 10, 16, batch):
            raise RuntimeError("All five complete original batches are required")
        screen = checked_marker(folder / "screen_selection.json", identity)
        final = checked_marker(folder / "final_selection.json", identity)
        screens = [checked_marker(folder / f"candidate{i:02d}/screen.json", identity) for i in range(20)]
        if any(row is None or row["completed_iterations"] != 5000 for row in screens):
            raise RuntimeError("All20 must finish screen5k")
        advanced = [row["candidate"] for row in rank_candidates(screens, list(range(20)))[:3]]
        if screen["advanced"] != advanced or done["advanced_candidates"] != advanced:
            raise RuntimeError("Screen top3 disagrees with observable score ordering")
        finals = [checked_marker(folder / f"candidate{i:02d}/final.json", identity) for i in advanced]
        if any(row is None or row["completed_iterations"] != 25000 for row in finals):
            raise RuntimeError("All three finalists must finish total25k")
        selected = rank_candidates(finals, advanced)[0]["candidate"]
        if final["selected_candidate"] != selected or done["selected_candidate"] != selected:
            raise RuntimeError("Final selection disagrees with observable score ordering")
        for i in range(20):
            if checked_marker(folder / f"candidate{i:02d}/cpa_result.json", identity) is None:
                raise RuntimeError("Missing completed CPA candidate")
        source_path = resolve(cfg["source_output"]) / f"batch{batch:02d}/inputs.pt"
        from .run_m8_fedavg_naive import PROJECT
        if file_hash(source_path) != manifest["inputs_sha256"][str(source_path.relative_to(PROJECT))]:
            raise RuntimeError("Original evaluation cache changed")
        inputs = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
        batch_rows = read_csv(folder / "per_image_metrics.csv")
        if [int(row["global_index"]) for row in batch_rows] != list(range(batch * 16, (batch + 1) * 16)):
            raise RuntimeError("Image coverage/order changed")
        for row, original in zip(batch_rows, inputs["metadata"]):
            for key in ("global_index", "batch_index", "source_index", "dataset_index", "label", "image_path"):
                if str(row[key]) != str(original[key]):
                    raise RuntimeError("Image identity changed")
            if int(row["selected_candidate"]) != selected:
                raise RuntimeError("Per-image cherry-picking is forbidden")
        measured = _metrics(batch_rows)
        if any(not np.isclose(measured[key], done[key], rtol=1e-9, atol=1e-11)
               for key in ("lpips_mean", "psnr_mean", "ssim_mean", "embedding_cos09")):
            raise RuntimeError("Batch means differ from saved per-image metrics")
        aligned = torch.load(folder / "rec.pt", map_location="cpu", weights_only=True)
        source_rec = torch.load(folder / f"candidate{selected:02d}/final_rec.pt", map_location="cpu", weights_only=True)
        order = torch.tensor([int(row["estimated_source_index"]) for row in batch_rows])
        if sorted(order.tolist()) != list(range(16)) or not torch.equal(aligned, source_rec.index_select(0, order)):
            raise RuntimeError("Evaluation must only permute the final selected images")
        x = inputs["x"]
        mean = x.new_tensor((.485, .456, .406))[None, :, None, None]
        std = x.new_tensor((.229, .224, .225))[None, :, None, None]
        originals.append((x * std + mean).clamp(0, 1))
        recs.append(aligned)
        rows.extend(batch_rows)
        batches.append({"batch_index": batch, "selected_candidate": selected, **measured,
                        "runtime_sec": done["runtime_sec"], "budget": done["budget"]})
    if len(rows) != 80 or len({row["image_path"] for row in rows}) != 80:
        raise RuntimeError("Need all80 distinct images")
    metrics = _metrics(rows)
    write_csv(output / "per_image_metrics.csv", rows)
    write_csv(output / "per_batch_metrics.csv", [dict(batch_index=row["batch_index"], selected_candidate=row["selected_candidate"],
               **{key: row[key] for key in metrics}) for row in batches])
    reconstructed = torch.cat(recs)
    torch.save(reconstructed, output / "all80_rec.pt")
    render_slide(torch.cat(originals), reconstructed, rows, output / "all80_original_vs_reconstruction_16x9.png")
    communications = manifest["communication_per_batch"]
    packet_bytes = sum(row["packet_bytes"] for row in communications)
    dense_bytes = sum(row["fp32_packet_bytes"] for row in communications)
    baseline = read_json(resolve(cfg["baseline_output"]) / "summary.json")
    comparison = [{"method": "Zero fill (1 candidate, FIA25k)", **baseline["metrics"]},
                  {"method": "Uniform random fill (20 at5k, top3 to25k)", **metrics}]
    write_csv(output / "comparison_metrics.csv", comparison)
    text = ["# Top-k10 random filling: same80 CPA + FIA", "",
            "The exact five packets previously attacked with zero fill are reused. Same original80 images, n16, round0 frozen-conv VGG16, all-classifier local SGD lr0.001 for10 steps. No new victim training or update collection.", "",
            "Only discarded coordinates are replaced by independent U(-tau,tau) samples, where tau is the minimum retained absolute value in the packet. Retained values remain bit-exact FP32. Each batch has20 fills, each with original CPA25k and FIA5k. The best3 observable update-consistency scores continue from the same Adam/cosine state to TOTAL25k; the cosine horizon is25k from the first step. CPA seeds and FIAseed42 match the baseline for every candidate.", "",
            "Ranking replays the same10-step classifier SGD on all16 reconstructed images, with labels predicted by the public round0 model, then applies the same FC1 Top-k codec. Relative L2 to the received decoded packet is minimized, with candidate index breaking ties. This observable proxy does not guarantee better image metrics. True labels/images/features are unavailable to construction and ranking. Final selection is one whole16-image candidate per batch, among the3 finalists, using the same proxy. Only then does the unchanged signed-source absolute-cosine Hungarian evaluation run.", "",
            "| Method | LPIPS lower | PSNR dB higher | SSIM higher |", "|---|---:|---:|---:|"]
    text += [f"| {row['method']} | {row['lpips_mean']:.6f} | {row['psnr_mean']:.4f} | {row['ssim_mean']:.6f} |" for row in comparison]
    text += ["", f"Same actual FC1 communication: {packet_bytes:,} bytes across5 packets, {100*(1-packet_bytes/dense_bytes):.4f}% fewer bytes than same-shape denseFP32. All100 fills and115 replay scores are attacker-local computation with zero extra transmitted bytes. No full-model traffic or training-utility claim.", "",
             "The user-requested attack budget differs: per batch the baseline has CPA25k+FIA25k, while random filling totals CPA500k+FIA160k and23 ranking replays. Consequently this comparison does not isolate filling from increased search compute. No extra zero-fill restart control was run.", "",
             "All3 final candidate reconstructions and all20 screen reconstructions remain saved per batch. The canonical all80 metrics/PNG use only the predeclared observable final selection, never the best LPIPS/SSIM/PSNR per image.", "",
             "![All80 original and reconstruction pairs](all80_original_vs_reconstruction_16x9.png)", ""]
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")
    names = ["per_image_metrics.csv", "per_batch_metrics.csv", "all80_rec.pt", "all80_original_vs_reconstruction_16x9.png",
             "comparison_metrics.csv", "report.md"]
    result = {"schema_version": 1, "run_tag": cfg["run_tag"], "status": "complete", "git_sha": manifest["git_sha"],
              "config": cfg, "identity": identity, "n_images": 80, "n": 16, "T": 10, "metrics": metrics,
              "per_image_metrics": rows, "per_victim_batch": batches, "randomization": RANDOMIZATION,
              "communication": {"scope": cfg["compression_scope"], "packet_bytes": packet_bytes,
                                "fp32_packet_bytes": dense_bytes, "n_packets": 5,
                                "packet_reduction_fraction": 1-packet_bytes/dense_bytes, "extra_bytes_for_random_fill": 0},
              "communication_per_batch": communications,
              "wall_time_sec": (datetime.fromisoformat(utc_now()) - datetime.fromisoformat(read_json(output / "run_clock.json")["first_started_at_utc"])).total_seconds(),
              "runtime_definition": "elapsed wall time since first launch; includes any resume downtime",
              "timed_compute_sec": sum(sum(value for key, value in row["runtime_sec"].items() if key != "current_invocation") for row in batches),
              "budget_per_batch": batches[0]["budget"], "oracle_fills": 0, "ls_A": False, "debias": False,
              "stage2": False, "mp_truncation": False, "new_training": False, "new_update_collection": False,
              "diagnostics_not_applicable": ["MP noise model", "Stage2 feasibility", "gamma", "LS mixing matrix"],
              "completed_at_utc": utc_now(), "output_hashes": {name: file_hash(output / name) for name in names}}
    _append_run_once(resolve(cfg["results_jsonl"]), result)
    write_json(output / "summary.json", result)
    return result
