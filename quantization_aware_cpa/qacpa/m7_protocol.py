"""Immutable protocol checks shared by the M7 control runners."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SPEC = PROJECT.parent / "codex" / "docs" / "M7_CONTROL_PREREGISTRATION.json"
LOCK = SPEC.with_name("M7_CONTROL_PREREGISTRATION.lock.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_protocol() -> tuple[dict, dict]:
    protocol = json.loads(SPEC.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8-sig"))
    if file_sha256(SPEC) != lock["sha256"]:
        raise RuntimeError("M7 preregistration was changed after locking; refusing to run")
    if protocol["protocol_id"] != lock["protocol_id"]:
        raise RuntimeError("M7 protocol ID mismatch")
    return protocol, lock


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)
