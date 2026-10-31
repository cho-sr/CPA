"""Run the authorized same80 R8 sparse sweep for keep 50/30/20 percent.

This is a single-file launcher.  It keeps the experiment implementation in
qacpa.run_paper_sparse_same80_r8 and executes all phases serially so the file
can be copied as the only launcher to another checkout of this project.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


KEEP_PERCENTS = (50, 30, 20)
CONFIG_TEMPLATE = "configs/cpa_fia_random_topk{percent}_same80_R8_S5k_K3_F25k.json"
RUNNER = "qacpa.run_paper_sparse_same80_r8"


def run(project: Path, args: list[str]) -> None:
    command = [sys.executable, "-u", "-m", *args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=project, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parent / "quantization_aware_cpa",
        help="Path to the quantization_aware_cpa checkout.",
    )
    args = parser.parse_args()
    project = args.project.resolve()

    required = [
        project / "qacpa" / "run_paper_sparse_same80_r8.py",
        project / "qacpa" / "report_paper_sparse_same80_r8.py",
        *(project / CONFIG_TEMPLATE.format(percent=p) for p in KEEP_PERCENTS),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing R8 project files:\n" + "\n".join(missing)
        )

    for phase in ("prepare", "migrate", "run"):
        for percent in KEEP_PERCENTS:
            config = CONFIG_TEMPLATE.format(percent=percent)
            run(project, [RUNNER, "--config", config, "--phase", phase])

    run(project, ["qacpa.report_paper_sparse_same80_r8"])
    print("R8 keep50/30/20 sweep and combined report completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
