from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reduced Milestone-4 Phase A manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("results/v10_m4_phase_plan/phase_A_manifest.csv"))
    parser.add_argument("--grid-root", type=Path, default=Path("results/v10_m4_grid"))
    parser.add_argument("--status-csv", type=Path, default=Path("results/v10_m4_grid/phase_A_status.csv"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.manifest)
    args.status_csv.parent.mkdir(parents=True, exist_ok=True)
    status_rows: list[dict[str, Any]] = []
    for row in rows:
        run_tag = str(row["run_tag"])
        output_dir = Path(args.grid_root) / run_tag
        summary_path = output_dir / "summary.json"
        start = time.perf_counter()
        status = "started"
        returncode = 0
        if summary_path.is_file() and not bool(args.force):
            status = "skipped_existing"
            wall = 0.0
            print(f"[{int(row['job_index']) + 1}/{len(rows)}] skip existing {run_tag}", flush=True)
        else:
            print(f"[{int(row['job_index']) + 1}/{len(rows)}] run {run_tag}", flush=True)
            completed = subprocess.run(str(row["command"]), cwd=Path.cwd(), shell=True)
            returncode = int(completed.returncode)
            wall = float(time.perf_counter() - start)
            status = "completed" if returncode == 0 and summary_path.is_file() else "failed"
            print(f"[{int(row['job_index']) + 1}/{len(rows)}] {status} {run_tag} wall={wall:.1f}s", flush=True)
            if status == "failed" and bool(args.stop_on_error):
                status_rows.append(status_row(row, status, returncode, wall, summary_path))
                write_csv(args.status_csv, status_rows)
                raise SystemExit(returncode or 1)
        status_rows.append(status_row(row, status, returncode, wall, summary_path))
        write_csv(args.status_csv, status_rows)


def status_row(row: dict[str, Any], status: str, returncode: int, wall: float, summary_path: Path) -> dict[str, Any]:
    return {
        "job_index": int(row["job_index"]),
        "run_tag": str(row["run_tag"]),
        "job_kind": str(row["job_kind"]),
        "B": str(row["B"]),
        "bits": str(row["bits"]),
        "rounding": str(row["rounding"]),
        "seed": str(row["seed"]),
        "scale_granularity": str(row["scale_granularity"]),
        "require_distinct_labels": str(row["require_distinct_labels"]),
        "status": status,
        "returncode": int(returncode),
        "wall_time_sec": float(wall),
        "summary_json": str(summary_path),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (path.with_suffix(".json")).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
