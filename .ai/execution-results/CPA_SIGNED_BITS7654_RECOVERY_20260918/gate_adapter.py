"""Task-local adapter for the authenticated bit-sweep public gate.

The registered runner's public_gate function is unchanged.  This adapter only
normalizes its successful return value for a CLI caller because the historical
runner CLI indexed a task_id field that public_gate intentionally did not
return.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantization_aware_cpa.qacpa import signed_bits_screen3_runner as runner  # noqa: E402


class GateAdapterError(RuntimeError):
    pass


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def authenticate(bits: int, public_root: Path) -> dict[str, Any]:
    """Run the unchanged gate and return a strict task-bound result."""
    result = runner.public_gate(bits, Path(public_root))
    expected_task_id = runner.bit_task_id(bits)
    if (not isinstance(result, dict)
            or result.get("status") != "PUBLIC_GATE_PASSED"
            or result.get("bits") != bits
            or result.get("private_loaded") is not False
            or result.get("candidate_count") != 13
            or not isinstance(result.get("candidates"), dict)
            or len(result["candidates"]) != 13):
        raise GateAdapterError("unchanged public_gate returned an invalid success contract")
    normalized = dict(result)
    normalized["task_id"] = expected_task_id
    normalized["adapter_task_id_source"] = "runner.bit_task_id(authenticated_bits)"
    normalized["adapter_did_not_modify_runner"] = True
    return normalized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = authenticate(args.bits, args.public_root)
        write_json(args.output, result)
    except Exception as error:
        print(f"{runner.TASK_ID}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps({"task_id": result["task_id"], "status": result["status"],
                      "bits": result["bits"], "candidate_count": result["candidate_count"]},
                     allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
