from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PYTHON = r"C:\miniconda\envs\DLG\python.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the Milestone-4 grid manifest and cmd runner.")
    parser.add_argument("--config", type=Path, default=Path("configs/spike.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m4_grid_plan"))
    parser.add_argument("--grid-root", type=Path, default=Path("results/m4_grid"))
    parser.add_argument("--bits", default="2,3,4,6,8,fp32")
    parser.add_argument("--batch-sizes", default="32,64,128,256,512,1024")
    parser.add_argument("--roundings", default="stochastic,deterministic")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--scale-granularities", default="per_tensor,per_row,per_bucket")
    parser.add_argument("--bucket-size", type=int, default=256)
    parser.add_argument("--batch-indices", default="0")
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--microbatch-size", type=int, default=32)
    parser.add_argument("--n-iter-fi", type=int, default=25000)
    parser.add_argument("--n-log-fi", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    manifest_path = output_dir / "m4_grid_manifest.csv"
    write_csv(manifest_path, jobs)
    cmd_path = output_dir / "run_m4_grid.cmd"
    write_cmd(cmd_path, jobs)
    summary = {
        "n_jobs": len(jobs),
        "n_fp32_jobs": sum(1 for job in jobs if job["bits"] == "fp32"),
        "n_quantized_jobs": sum(1 for job in jobs if job["bits"] != "fp32"),
        "manifest": str(manifest_path),
        "cmd": str(cmd_path),
        "grid_root": str(args.grid_root),
        "notes": [
            "fp32 is emitted once per B/seed instead of once per rounding/granularity.",
            "Quantized jobs run naive,A,A_oracle. Merge fp32 rows during aggregation.",
            "B=1024 uses data.require_distinct_labels=false because ImageNet has 1000 labels.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "wrote {n} M4 jobs ({fp32} fp32, {quant} quantized) to {path}".format(
            n=summary["n_jobs"],
            fp32=summary["n_fp32_jobs"],
            quant=summary["n_quantized_jobs"],
            path=manifest_path,
        ),
        flush=True,
    )


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    bits_values = [item.strip() for item in str(args.bits).split(",") if item.strip()]
    batch_sizes = [int(item.strip()) for item in str(args.batch_sizes).split(",") if item.strip()]
    roundings = [item.strip() for item in str(args.roundings).split(",") if item.strip()]
    seeds = [int(item.strip()) for item in str(args.seeds).split(",") if item.strip()]
    granularities = [item.strip() for item in str(args.scale_granularities).split(",") if item.strip()]
    jobs: list[dict[str, Any]] = []
    job_index = 0

    for B in batch_sizes:
        for seed in seeds:
            tag = f"m4_B{B}_bitsfp32_seed{seed}"
            jobs.append(
                make_job(
                    job_index=job_index,
                    run_tag=tag,
                    job_kind="fp32_truth_fia",
                    bits="fp32",
                    B=B,
                    rounding="none",
                    seed=seed,
                    scale_granularity="none",
                    bucket_size="",
                    command=fp32_command(args=args, B=B, seed=seed, run_tag=tag),
                )
            )
            job_index += 1

    for bits in bits_values:
        if bits == "fp32":
            continue
        for B in batch_sizes:
            for rounding in roundings:
                for seed in seeds:
                    for granularity in granularities:
                        tag = f"m4_B{B}_bits{bits}_{short_rounding(rounding)}_seed{seed}_scale{granularity}"
                        jobs.append(
                            make_job(
                                job_index=job_index,
                                run_tag=tag,
                                job_kind="quantized_m3_fia",
                                bits=bits,
                                B=B,
                                rounding=rounding,
                                seed=seed,
                                scale_granularity=granularity,
                                bucket_size=args.bucket_size if granularity == "per_bucket" else "",
                                command=quantized_command(
                                    args=args,
                                    B=B,
                                    bits=int(bits),
                                    rounding=rounding,
                                    seed=seed,
                                    scale_granularity=granularity,
                                    run_tag=tag,
                                ),
                            )
                        )
                        job_index += 1
    return jobs


def make_job(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)


def fp32_command(*, args: argparse.Namespace, B: int, seed: int, run_tag: str) -> str:
    indices = ",".join(str(i) for i in range(int(args.max_images)))
    return join_cmd(
        [
            PYTHON,
            "-m",
            "qacpa.run_m3_fp32_truth_fia",
            "--config",
            str(args.config),
            "--run-tag",
            run_tag,
            "--global-indices",
            indices,
            "--n-iter-fi",
            str(args.n_iter_fi),
            "--n-log-fi",
            str(args.n_log_fi),
            "--output-dir",
            str(Path(args.grid_root) / run_tag),
            "--set",
            f"seed={seed}",
            "--set",
            f"data.batch_size={B}",
            "--set",
            "data.n_batches=1",
            "--set",
            f"stage1.n_components={B}",
            "--set",
            f"data.microbatch_size={int(args.microbatch_size)}",
            "--set",
            f"data.require_distinct_labels={str(B <= 1000).lower()}",
        ]
    )


def quantized_command(
    *,
    args: argparse.Namespace,
    B: int,
    bits: int,
    rounding: str,
    seed: int,
    scale_granularity: str,
    run_tag: str,
) -> str:
    return join_cmd(
        [
            PYTHON,
            "-m",
            "qacpa.run_m3_fia_4bit",
            "--config",
            str(args.config),
            "--run-tag",
            run_tag,
            "--methods",
            "naive,A,A_oracle",
            "--batch-indices",
            str(args.batch_indices),
            "--max-images",
            str(args.max_images),
            "--n-iter-fi",
            str(args.n_iter_fi),
            "--n-log-fi",
            str(args.n_log_fi),
            "--output-dir",
            str(Path(args.grid_root) / run_tag),
            "--set",
            f"seed={seed}",
            "--set",
            f"data.batch_size={B}",
            "--set",
            "data.n_batches=1",
            "--set",
            f"stage1.n_components={B}",
            "--set",
            f"data.microbatch_size={int(args.microbatch_size)}",
            "--set",
            f"data.require_distinct_labels={str(B <= 1000).lower()}",
            "--set",
            f"quantization.bits={bits}",
            "--set",
            f"quantization.rounding={rounding}",
            "--set",
            f"quantization.scale_granularity={scale_granularity}",
            "--set",
            f"quantization.bucket_size={int(args.bucket_size)}",
        ]
    )


def join_cmd(parts: list[str]) -> str:
    return " ".join(quote_cmd(part) for part in parts)


def quote_cmd(value: str) -> str:
    text = str(value)
    if not text:
        return '""'
    if any(ch.isspace() for ch in text) or any(ch in text for ch in "&()[]{}^=;!'+,`~"):
        return '"' + text.replace('"', '""') + '"'
    return text


def short_rounding(rounding: str) -> str:
    return "stoch" if rounding == "stochastic" else "det"


def write_cmd(path: Path, jobs: list[dict[str, Any]]) -> None:
    lines = [
        "@echo off",
        "setlocal enabledelayedexpansion",
        "cd /d %~dp0\\..\\..",
        "echo Running Milestone 4 grid jobs",
    ]
    for job in jobs:
        lines.extend(
            [
                f"echo [{int(job['job_index']) + 1}/{len(jobs)}] {job['run_tag']}",
                str(job["command"]),
                "if errorlevel 1 exit /b %errorlevel%",
            ]
        )
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
