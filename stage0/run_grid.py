#!/usr/bin/env python
"""Stage 0 job driver: enumerate and run the VGG-16 / ImageNet baseline grid.

Jobs are run sequentially (one GPU) and each is resumable: upstream
``AttackLog`` checkpoints into ``{exp_path}/{batch_size}_ckpt.pkl`` and reloads
it unless ``--fresh_start`` is passed, so re-running the same job continues
where it stopped.

Every job gets its own ``--exp_name``.  This is not cosmetic: upstream derives
``exp_path`` only from ``decor/T/tv/nv/l1``, so two jobs differing in seed,
init, or source tree would otherwise share a directory and silently inherit
each other's checkpoint *and* whitening cache
(see stage0/notes/cpa_code_reading.md 2 and 4.4).

Grids
-----
``repro``   headline batch size only, both source variants -> paper comparison
``sweep``   batch-size curve, both source variants
``wsrr``    K independent random inits at one batch size -> BRR / WSRR

Examples
--------
    python stage0/run_grid.py --grid repro --list
    python stage0/run_grid.py --grid repro
    python stage0/run_grid.py --grid sweep --n_batch 5
    python stage0/run_grid.py --grid wsrr --k 10 --batch_size 256 --n_batch 2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

STAGE0 = Path(__file__).resolve().parent
REPO_ROOT = STAGE0.parent
CONFIG = STAGE0 / "configs" / "vgg16_imagenet.yaml"
PYTHON = sys.executable


@dataclass
class Job:
    exp_name: str
    src: str
    batch_size: int
    seed: int = 42
    w_init: str = "eye"
    n_batch: int = 10
    extra: dict = field(default_factory=dict)

    def argv(self, fresh_start: bool) -> list[str]:
        cmd = [
            PYTHON,
            str(STAGE0 / "run_cpa.py"),
            "--config",
            str(CONFIG),
            "--step",
            "attack",
            "--src",
            self.src,
            "--seed",
            str(self.seed),
            "--w_init",
            self.w_init,
            "--num_workers",
            "0",
            "--set",
            f"batch_size={self.batch_size}",
            "--set",
            f"n_batch={self.n_batch}",
            "--set",
            f"exp_name={self.exp_name}",
        ]
        for key, value in self.extra.items():
            cmd += ["--set", f"{key}={value}"]
        if fresh_start:
            cmd += ["--set", "fresh_start=true"]
        return cmd

    def exp_path(self) -> Path:
        return (
            REPO_ROOT
            / "exp"
            / "imagenet"
            / "vgg16"
            / "attack"
            / "cp_direct"
            / "nodef"
            / self.exp_name
        )

    def rec_file(self) -> Path:
        return self.exp_path() / f"{self.batch_size}_rec.pkl"


def build_jobs(args) -> list[Job]:
    srcs = args.srcs.split(",")
    jobs: list[Job] = []

    if args.grid == "repro":
        for src in srcs:
            jobs.append(
                Job(
                    exp_name=f"stage0_{src}_b{args.batch_size}_seed42_weye",
                    src=src,
                    batch_size=args.batch_size,
                    n_batch=args.n_batch,
                    extra=dict(args.extra),
                )
            )

    elif args.grid == "sweep":
        for src in srcs:
            for bs in [int(b) for b in args.batch_sizes.split(",")]:
                jobs.append(
                    Job(
                        exp_name=f"stage0_{src}_b{bs}_seed42_weye",
                        src=src,
                        batch_size=bs,
                        n_batch=args.n_batch,
                        extra=dict(args.extra),
                    )
                )

    elif args.grid == "wsrr":
        # K independent runs. Upstream W_hat init is deterministic (init.eye_),
        # so varying the seed alone changes nothing in the GI stage -- the init
        # itself has to be randomized.  See notes/deviations.md 6.
        for src in srcs:
            for k in range(args.k):
                seed = 1000 + k
                jobs.append(
                    Job(
                        exp_name=(
                            f"stage0_{src}_b{args.batch_size}_seed{seed}_w{args.w_init}"
                        ),
                        src=src,
                        batch_size=args.batch_size,
                        seed=seed,
                        w_init=args.w_init,
                        n_batch=args.n_batch,
                        extra=dict(args.extra),
                    )
                )

    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True, choices=["repro", "sweep", "wsrr"])
    parser.add_argument("--srcs", default="official,negentropy")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--batch_sizes", default="8,16,32,64,128,256")
    parser.add_argument("--n_batch", type=int, default=10)
    parser.add_argument("--k", type=int, default=10, help="wsrr: independent runs")
    parser.add_argument("--w_init", default="randn", help="wsrr: init mode")
    parser.add_argument("--fresh_start", action="store_true")
    parser.add_argument(
        "--skip_done",
        action="store_true",
        help="skip jobs whose *_rec.pkl already exists",
    )
    parser.add_argument("--list", action="store_true", help="print jobs and exit")
    parser.add_argument(
        "--set",
        dest="extra_raw",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra attack override applied to every job (e.g. n_iter_fi=5000)",
    )
    args = parser.parse_args()

    args.extra = {}
    for item in args.extra_raw:
        key, value = item.split("=", 1)
        args.extra[key] = value

    jobs = build_jobs(args)
    log_dir = STAGE0 / "logs"
    log_dir.mkdir(exist_ok=True)

    print(f"[grid] {args.grid}: {len(jobs)} job(s)")
    for j in jobs:
        done = j.rec_file().exists()
        print(
            f"  {'DONE' if done else '    '}  {j.exp_name}"
            f"  src={j.src} bs={j.batch_size} seed={j.seed} w_init={j.w_init} n_batch={j.n_batch}"
        )
    if args.list:
        return

    for i, job in enumerate(jobs, 1):
        if args.skip_done and job.rec_file().exists():
            print(f"[grid] ({i}/{len(jobs)}) skip done: {job.exp_name}", flush=True)
            continue
        log_path = log_dir / f"attack_{job.exp_name}.log"
        cmd = job.argv(args.fresh_start)
        print(f"[grid] ({i}/{len(jobs)}) {job.exp_name}", flush=True)
        print(f"[grid]   log: {log_path}", flush=True)
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as fh:
            proc = subprocess.run(
                cmd, cwd=str(REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT
            )
        dt = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
        print(f"[grid]   {status} in {dt/60:.1f} min", flush=True)
        if proc.returncode != 0:
            print(f"[grid]   see {log_path}", flush=True)


if __name__ == "__main__":
    main()
