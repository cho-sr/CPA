from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def _import_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with `pip install pyyaml`.") from exc
    return yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    yaml = _import_yaml()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    data = resolve_paths(data, config_path)
    validate_config(data)
    return data


def resolve_paths(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    cfg = deepcopy(config)
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent

    paths = cfg.setdefault("paths", {})
    for key in ("cache_dir", "results_jsonl"):
        if key in paths:
            paths[key] = str(_resolve(project_root, paths[key]))

    data = cfg.setdefault("data", {})
    if "root" in data:
        data["root"] = str(_resolve(project_root, data["root"]))
    return cfg


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def validate_config(config: dict[str, Any]) -> None:
    data = config.get("data", {})
    quant = config.get("quantization", {})
    stage1 = config.get("stage1", {})

    config.setdefault("run_tag", "v3")
    batch_size = int(data.get("batch_size", 0))
    n_components = int(stage1.get("n_components", batch_size))
    if batch_size <= 0:
        raise ValueError("data.batch_size must be positive")
    if n_components != batch_size:
        raise ValueError("stage1.n_components must equal data.batch_size for Milestones 0-1")
    if quant.get("rounding") not in {"stochastic", "deterministic"}:
        raise ValueError("quantization.rounding must be stochastic or deterministic")
    bits = int(quant.get("bits", 0))
    if bits < 2:
        raise ValueError("quantization.bits must be >= 2")


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    cfg = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        cursor = cfg
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            next_value = cursor.setdefault(part, {})
            if not isinstance(next_value, dict):
                raise ValueError(f"cannot set nested override through scalar: {dotted_key}")
            cursor = next_value
        cursor[parts[-1]] = coerce_scalar(raw_value)
    validate_config(cfg)
    return cfg


def coerce_scalar(value: str) -> Any:
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value
