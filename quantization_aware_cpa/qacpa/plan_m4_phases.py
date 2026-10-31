from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PYTHON = r"C:\miniconda\envs\DLG\python.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the reduced two-phase Milestone-4 plan.")
    parser.add_argument("--config", type=Path, default=Path("configs/spike.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/v10_m4_phase_plan"))
    parser.add_argument("--grid-root", type=Path, default=Path("results/v10_m4_grid"))
    parser.add_argument("--phase-a-seeds", default="0")
    parser.add_argument("--phase-b-seeds", default="0,1,2,3,4")
    parser.add_argument("--batch-indices", default="0")
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--microbatch-size", type=int, default=32)
    parser.add_argument("--n-iter-fi", type=int, default=25000)
    parser.add_argument("--n-log-fi", type=int, default=5000)
    parser.add_argument("--bucket-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_a = phase_a_jobs(args)
    write_csv(output_dir / "phase_A_manifest.csv", phase_a)
    write_cmd(output_dir / "run_phase_A.cmd", phase_a)
    phase_b_template = phase_b_template_rows(args)
    write_csv(output_dir / "phase_B_template.csv", phase_b_template)
    summary = {
        "phase_A_jobs": len(phase_a),
        "phase_A_quantized_jobs": int(sum(job["job_kind"] == "quantized_m3_fia" for job in phase_a)),
        "phase_A_fp32_jobs": int(sum(job["job_kind"] == "fp32_truth_fia" for job in phase_a)),
        "phase_A_manifest": str(output_dir / "phase_A_manifest.csv"),
        "phase_A_cmd": str(output_dir / "run_phase_A.cmd"),
        "phase_B_template": str(output_dir / "phase_B_template.csv"),
        "notes": [
            "Phase A excludes deterministic rounding except future spot checks.",
            "Phase A uses one seed and stochastic rounding; Phase B fills boundary cells with five seeds.",
            "B=512 is emitted for both distinct and non-distinct labels to isolate duplicate-label effects.",
            "Each quantized job runs naive,A,A_oracle together; count method rows separately if needed.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote Phase A plan with {len(phase_a)} executable jobs to {output_dir}", flush=True)


def phase_a_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    index = 0
    seeds = parse_ints(args.phase_a_seeds)
    for seed in seeds:
        for B in (32, 128, 512):
            label_modes = [True, False] if B == 512 else [True]
            for distinct in label_modes:
                tag = run_tag(B=B, bits="fp32", seed=seed, scale="none", distinct=distinct)
                jobs.append(
                    job_row(
                        index,
                        phase="A",
                        run_tag=tag,
                        job_kind="fp32_truth_fia",
                        B=B,
                        bits="fp32",
                        rounding="none",
                        seed=seed,
                        scale_granularity="none",
                        require_distinct_labels=distinct,
                        command=fp32_command(args, B=B, seed=seed, run_tag=tag, distinct=distinct),
                    )
                )
                index += 1
                for bits in (2, 3, 4, 6, 8):
                    tag = run_tag(B=B, bits=bits, seed=seed, scale="per_row", distinct=distinct)
                    jobs.append(
                        job_row(
                            index,
                            phase="A",
                            run_tag=tag,
                            job_kind="quantized_m3_fia",
                            B=B,
                            bits=bits,
                            rounding="stochastic",
                            seed=seed,
                            scale_granularity="per_row",
                            require_distinct_labels=distinct,
                            command=quantized_command(
                                args,
                                B=B,
                                bits=bits,
                                seed=seed,
                                scale="per_row",
                                run_tag=tag,
                                distinct=distinct,
                            ),
                        )
                    )
                    index += 1
        for B in (32, 128):
            for bits in (3, 4):
                for scale in ("per_tensor", "per_bucket"):
                    tag = run_tag(B=B, bits=bits, seed=seed, scale=scale, distinct=True)
                    jobs.append(
                        job_row(
                            index,
                            phase="A",
                            run_tag=tag,
                            job_kind="quantized_m3_fia",
                            B=B,
                            bits=bits,
                            rounding="stochastic",
                            seed=seed,
                            scale_granularity=scale,
                            require_distinct_labels=True,
                            command=quantized_command(
                                args,
                                B=B,
                                bits=bits,
                                seed=seed,
                                scale=scale,
                                run_tag=tag,
                                distinct=True,
                            ),
                        )
                    )
                    index += 1
    return jobs


def phase_b_template_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for B in (32, 128, 512):
        for bits in (2, 3, 4, 6, 8):
            rows.append(
                {
                    "include": "fill_after_phase_A_if_boundary",
                    "B": B,
                    "bits": bits,
                    "rounding": "stochastic",
                    "scale_granularity": "per_row",
                    "seeds": str(args.phase_b_seeds),
                    "criterion": "near MP boundary, LPIPS transition, or Spearman transition",
                }
            )
    return rows


def run_tag(*, B: int, bits: int | str, seed: int, scale: str, distinct: bool) -> str:
    label = "distinct" if distinct else "mixed"
    return f"m4A_B{B}_bits{bits}_stoch_seed{seed}_scale{scale}_{label}"


def job_row(index: int, **kwargs: Any) -> dict[str, Any]:
    return {"job_index": int(index), **kwargs}


def fp32_command(args: argparse.Namespace, *, B: int, seed: int, run_tag: str, distinct: bool) -> str:
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
            f"data.require_distinct_labels={str(distinct).lower()}",
        ]
    )


def quantized_command(
    args: argparse.Namespace,
    *,
    B: int,
    bits: int,
    seed: int,
    scale: str,
    run_tag: str,
    distinct: bool,
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
            f"data.require_distinct_labels={str(distinct).lower()}",
            "--set",
            f"quantization.bits={bits}",
            "--set",
            "quantization.rounding=stochastic",
            "--set",
            f"quantization.scale_granularity={scale}",
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


def write_cmd(path: Path, jobs: list[dict[str, Any]]) -> None:
    lines = [
        "@echo off",
        "setlocal enabledelayedexpansion",
        "cd /d %~dp0\\..\\..",
        "echo Running reduced Milestone 4 Phase A",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


if __name__ == "__main__":
    main()
