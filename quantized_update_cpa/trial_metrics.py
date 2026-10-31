"""Raw metric storage only: no scoring, ranking, selection, or oracle loading."""

from __future__ import annotations

import copy
import json


RAW_METRICS = (
    "cpa_objective", "mi", "negentropy", "sparsity", "sign_consistency",
    "source_diversity", "quantization_consistency", "scale_consistency",
)


def metric_record(value=None, *, status="NOT_COMPUTED", reason="Not executed", definition=None):
    """A computed raw value needs its definition; missing values need a reason.

    Nested JSON diagnostics are accepted. Nonfinite floats are rejected so a
    caller must explicitly store null plus its reason instead of emitting NaN.
    """
    if status not in {"COMPUTED", "NOT_COMPUTED", "UNKNOWN"}:
        raise ValueError("unknown metric status")
    if status == "COMPUTED":
        if value is None or not isinstance(definition, str) or not definition.strip():
            raise ValueError("computed metrics require value and definition")
        reason = None
    elif value is not None or not isinstance(reason, str) or not reason.strip():
        raise ValueError("missing metrics require null value and an explicit reason")
    json.dumps(value, allow_nan=False)
    return {"value": copy.deepcopy(value), "status": status,
            "reason": reason, "definition": definition}


def new_trial_metrics() -> dict:
    """Create distinct uncomputed public/source and oracle evaluation fields."""
    return {
        "raw_metrics": {name: metric_record() for name in RAW_METRICS},
        "evaluation_metrics": {
            "embedding_oracle": {"cosine": metric_record(reason="Oracle evaluation not executed")},
            "image_oracle": {name: metric_record(reason="Oracle evaluation not executed")
                             for name in ("lpips", "psnr", "ssim")},
        },
    }
