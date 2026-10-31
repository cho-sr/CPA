"""Run the authorized 30/20/10 percent conditions sequentially, with CMD support."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .run_topk_ratios_same80 import (
    PROJECT, KEEP_PERCENTS, read_json, write_json, read_csv, write_csv, file_hash,
    utc_now, execution_lock, validate_config,
)

QUEUE_OUTPUT = PROJECT / "results/cpa_fia_topk30_20_10_same80_B16"


def input_audit(output: Path) -> dict:
    rows = []
    for percent in KEEP_PERCENTS:
        folder = PROJECT / f"results/cpa_fia_topk{percent}_same80_B16"
        for batch in range(5):
            path = folder / f"batch{batch:02d}"
            diag = read_json(path / "sparsification_diagnostics.json")
            comm = read_json(path / "communication.json")
            if comm["sha256"] != file_hash(path / "observation.bin"):
                raise RuntimeError("Input-audit packet hash mismatch")
            rows.append({"keep_percent": percent, "batch_index": batch,
                         "kept_elements": diag["kept_elements"],
                         "minimum_retained_abs": diag["minimum_retained_abs"],
                         "maximum_discarded_abs": diag["maximum_discarded_abs"],
                         "relative_l2_error": diag["relative_l2_error"],
                         "retained_energy_fraction": diag["retained_energy_fraction"],
                         "packet_bytes": comm["packet_bytes"], "saved_percent": comm["saved_percent"],
                         "packet_sha256": comm["sha256"],
                         "diagnostics_sha256": file_hash(path / "sparsification_diagnostics.json")})
    result = {"scope": "15 attacked FC1 packets; same five cached n16/T10 updates",
              "created_at_utc": utc_now(), "rows": rows}
    write_json(output / "sparsification_input_audit.json", result)
    write_csv(output / "sparsification_input_audit.csv", rows)
    return result


def comparison_report(output: Path) -> dict:
    """Read completed results, retaining all80 and original image correspondence."""
    rows, references, identities = [], [], None
    for percent in (50, *KEEP_PERCENTS):
        folder = PROJECT / f"results/cpa_fia_topk{percent}_same80_B16"
        result = read_json(folder / "summary.json")
        if result["status"] != "complete" or result["n_images"] != 80:
            raise RuntimeError(f"Top-k{percent} does not have all80 completed images")
        for name, expected in result["output_hashes"].items():
            if file_hash(folder / name) != expected:
                raise RuntimeError(f"Completed output changed: {folder / name}")
        images = read_csv(folder / "per_image_metrics.csv")
        fields = ("global_index", "batch_index", "source_index", "dataset_index", "label", "image_path")
        ids = [[str(row[key]) for key in fields] for row in images]
        if len(images) != 80 or [int(row["global_index"]) for row in images] != list(range(80)):
            raise RuntimeError("Comparison must cover exactly all80 in original order")
        if identities is not None and ids != identities:
            raise RuntimeError("Comparison image identities differ")
        identities = ids
        metrics = {f"{key}_mean": float(np.mean([float(row[key]) for row in images]))
                   for key in ("lpips", "ssim", "psnr")}
        if any(not np.isclose(value, result["metrics"][key], rtol=1e-10, atol=1e-12)
               for key, value in metrics.items()):
            raise RuntimeError("Comparison metrics do not match per-image means")
        comm = result["communication"]
        rows.append({"keep_percent": percent, "n_images": 80, **metrics,
                     "source_cos09_count": result["metrics"]["embedding_cos09"],
                     "packet_bytes_per_batch": comm["packet_bytes"] // 5,
                     "packet_bytes_total": comm["packet_bytes"],
                     "packet_saved_percent": 100 * comm["packet_reduction_fraction"],
                     "run_tag": result["run_tag"]})
        references.append({"summary": str(folder / "summary.json"), "sha256": file_hash(folder / "summary.json")})
    write_csv(output / "comparison_metrics.csv", rows)
    lines = ["# Top-k retention comparison: same80 CPA + FIA", "",
             "Same original80 images, five n16/T10 round0 FC-only SGD updates, seeds, CPA/FIA budgets and post-FIA matching. Only retained fraction changes. Top-k50 is the completed historical paired run; 30/20/10 are the requested new runs.", "",
             "| Retained | LPIPS (lower) | SSIM (higher) | PSNR dB (higher) | FC1 bytes/batch | Saved |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        percent = row["keep_percent"]
        lines.append(f"| {percent}% | {row['lpips_mean']:.6f} | {row['ssim_mean']:.6f} | {row['psnr_mean']:.4f} | {row['packet_bytes_per_batch']:,} | {row['packet_saved_percent']:.4f}% |")
    lines.extend(["", "Communication savings cover the actual attacked FC1 bitmap + FP32 packet relative to a same-shape dense packet. This does not measure full-model network traffic, latency or training utility. Source-cosine counts are embedding diagnostics, not image-success counts. Top-k50 was numerically lossless for these updates; see the new input audit for 30/20/10 distortion.", ""])
    for row in rows:
        percent = row["keep_percent"]
        lines.append(f"[Top-k {percent}%: all80 originals and reconstructions](../cpa_fia_topk{percent}_same80_B16/all80_reconstruction_landscape.png)")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {"status": "complete", "scope": "paired Top-k50/30/20/10; new attacks only30/20/10",
               "rows": rows, "references": references, "completed_at_utc": utc_now(),
               "output_hashes": {name: file_hash(output / name) for name in ("report.md", "comparison_metrics.csv")}}
    write_json(output / "summary.json", summary)
    return summary


def run_queue(phase: str, output: Path = QUEUE_OUTPUT) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with execution_lock(output):
        started = time.perf_counter()
        state = {"pid": os.getpid(), "phase": phase, "state": "starting",
                 "keep_percents": list(KEEP_PERCENTS), "completed_keep_percents": [],
                 "started_at_utc": utc_now(), "n_batches_total": 15, "n_distinct_images": 80}

        def save(**updates):
            state.update(updates, updated_at_utc=utc_now(), elapsed_sec=time.perf_counter() - started)
            write_json(output / "queue_status.json", state)

        try:
            save()
            if phase != "report":
                for percent in KEEP_PERCENTS:
                    config = PROJECT / f"configs/cpa_fia_topk{percent}_same80_B16.json"
                    cfg = read_json(config)
                    validate_config(cfg)
                    folder = PROJECT / cfg["output_dir"]
                    folder.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, "-u", "-m", "qacpa.run_topk_ratios_same80",
                               "--config", str(config), "--phase", phase]
                    print(f"START keep={percent}%, phase={phase}", flush=True)
                    with (folder / f"{phase}.stdout.log").open("ab") as stdout, (folder / f"{phase}.stderr.log").open("ab") as stderr:
                        process = subprocess.Popen(command, cwd=PROJECT, stdout=stdout, stderr=stderr)
                        save(state="running", active_keep_percent=percent, child_pid=process.pid, command=command)
                        returncode = process.wait()
                    if returncode != 0:
                        raise RuntimeError(f"Top-k{percent} {phase} exited with code {returncode}; see {folder}")
                    completed = read_json(folder / "execution_status.json")
                    expected = "complete" if phase == "run" else "packet_ready"
                    if completed["state"] != expected:
                        raise RuntimeError(f"Top-k{percent} exited without a {expected} marker")
                    state["completed_keep_percents"].append(percent)
                    save(child_pid=None)
                    print(f"COMPLETE keep={percent}%, phase={phase}", flush=True)
            input_audit(output)
            if phase in ("run", "report"):
                comparison_report(output)
            save(state="complete" if phase != "packet" else "packet_ready", active_keep_percent=None,
                 child_pid=None, completed_at_utc=utc_now())
        except BaseException as exc:
            save(state="failed", error=f"{type(exc).__name__}: {exc}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("packet", "run", "report"), default="run")
    run_queue(parser.parse_args().phase)


if __name__ == "__main__":
    main()
