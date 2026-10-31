"""Scientific-validity checks for reports that consume withdrawn experiments.

Importing this module has no filesystem side effects. Execution completion and
scientific validity are separate; a completed run is not sufficient to report
an experiment that has been withdrawn.
"""
from __future__ import annotations

import json
from pathlib import Path


REGISTRY_PATH = Path(__file__).resolve().parents[1] / "results" / "result_validity.json"
NO_CPA_SOURCE_ID = "m7_no_cpa_control"


def require_valid_no_cpa_control() -> None:
    """Fail before regenerating scientific reports that require this control."""
    prefix = f"No-CPA-dependent report regeneration blocked for {NO_CPA_SOURCE_ID}"
    try:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"{prefix}: scientific-validity registry is missing: {REGISTRY_PATH}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{prefix}: cannot read the scientific-validity registry: {REGISTRY_PATH}") from exc
    invalid_sources = registry.get("invalid_sources") if isinstance(registry, dict) else None
    if not isinstance(invalid_sources, list) or any(not isinstance(entry, dict) for entry in invalid_sources):
        raise RuntimeError(f"{prefix}: registry must contain an invalid_sources list of source records: {REGISTRY_PATH}")
    records = [entry for entry in invalid_sources if entry.get("source_id") == NO_CPA_SOURCE_ID]
    if not records:
        raise RuntimeError(f"{prefix}: registry has no scientific-validity record for this withdrawn source: {REGISTRY_PATH}")
    if any(entry.get("scientific_validity") == "invalid" for entry in records):
        raise RuntimeError(
            f"{prefix}: the control experiment is scientifically invalid. Its dependent comparisons, "
            f"normalizations, thresholds, and success counts must not be regenerated. See {REGISTRY_PATH}"
        )
    if any(entry.get("scientific_validity") != "valid" for entry in records):
        raise RuntimeError(f"{prefix}: the registry does not explicitly establish scientific validity: {REGISTRY_PATH}")
