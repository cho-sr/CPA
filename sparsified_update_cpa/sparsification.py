"""Global Top-k sparsification for dense CPA update artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


SUPPORTED_SPARSE_POLICIES = {"global_topk"}
OBSERVED_ZERO_POLICY = "zeros_are_treated_as_true_observed_update_values"
RANDOMIZED_DESPARSE_POLICY = "dropped_sparse_coordinates_sampled_uniform_within_client_topk_threshold"


def format_keep_ratio(keep_ratio: float) -> str:
    return f"{keep_ratio:.6g}".replace(".", "p").replace("-", "m")


def require_global_topk(policy: str) -> None:
    if policy not in SUPPORTED_SPARSE_POLICIES:
        raise ValueError(f"Unsupported sparse policy {policy!r}; expected 'global_topk'.")


def validate_keep_ratio(keep_ratio: float) -> None:
    if not math.isfinite(float(keep_ratio)) or keep_ratio <= 0.0 or keep_ratio > 1.0:
        raise ValueError("keep_ratio must be in the interval (0, 1]")


def parse_keep_ratio_or_percent(value: str | float) -> float:
    """Parse keep ratio CLI values, accepting either fractions or percents."""
    keep_ratio = float(value)
    if keep_ratio > 1.0:
        keep_ratio = keep_ratio / 100.0
    validate_keep_ratio(keep_ratio)
    return float(keep_ratio)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _assert_matching_manifest(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.exists():
        return
    current = _read_json(path)
    if current != expected:
        raise RuntimeError(
            f"Manifest mismatch at {path}. Refusing to reuse a sparse artifact "
            "generated with different source data or sparsification settings."
        )


def _hash_tensor_sequence(values: Any) -> str:
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, Mapping):
            digest.update(f"mapping:{len(value)}".encode("ascii"))
            digest.update(b"\0")
            for key in sorted(value):
                digest.update(str(key).encode("utf-8"))
                digest.update(b"\0")
                update(value[key])
            return
        if isinstance(value, (list, tuple)):
            digest.update(f"sequence:{len(value)}".encode("ascii"))
            digest.update(b"\0")
            for item in value:
                update(item)
            return
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
        digest.update(b"\0")

    update(values)
    return digest.hexdigest()


def pack_bool_mask(mask: torch.Tensor) -> dict[str, Any]:
    mask_np = mask.detach().cpu().numpy().astype(np.bool_, copy=False)
    packed = np.packbits(mask_np.reshape(-1).astype(np.uint8, copy=False))
    return {
        "shape": list(mask_np.shape),
        "packed_bits": packed,
        "kept_elements": int(mask_np.sum()),
        "total_elements": int(mask_np.size),
    }


def unpack_bool_mask(packed_mask: Mapping[str, Any]) -> np.ndarray:
    shape = tuple(int(value) for value in packed_mask["shape"])
    total = int(np.prod(shape, dtype=np.int64))
    packed_bits = np.asarray(packed_mask["packed_bits"], dtype=np.uint8)
    flat = np.unpackbits(packed_bits, count=total).astype(np.bool_, copy=False)
    return flat.reshape(shape)


def randomized_desparsify_sparse_observation(
    observed: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    tau: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample dropped entries from U(-tau, tau), preserving observed entries."""
    observed_np = np.asarray(torch.as_tensor(observed).detach().cpu(), dtype=np.float32)
    mask_np = np.asarray(torch.as_tensor(mask).detach().cpu(), dtype=np.bool_)
    if observed_np.shape != mask_np.shape:
        raise ValueError(f"observed/mask shape mismatch: {observed_np.shape} vs {mask_np.shape}")
    tau_f = float(tau)
    if tau_f < 0.0 or not math.isfinite(tau_f):
        raise ValueError("tau must be a finite non-negative value")
    if bool(mask_np.all()):
        return observed_np.astype(np.float32, copy=True)
    if tau_f == 0.0:
        sampled = np.zeros(observed_np.shape, dtype=np.float32)
    else:
        sampled = rng.uniform(-tau_f, tau_f, size=observed_np.shape).astype(np.float32, copy=False)
    sampled[mask_np] = observed_np[mask_np]
    return sampled.astype(np.float32, copy=False)


def global_topk_attack_layer_observation(
    update: "OrderedDict[str, torch.Tensor]",
    *,
    keep_ratio: float,
    attack_layer_index: int,
) -> dict[str, Any]:
    """Return only the attack-layer slice of a full-model Global Top-k observation.

    This avoids materializing a full dense sparse model update when the caller
    only needs the fixed attack-FC observation used by randomized desparsification.
    The threshold and deterministic tie handling mirror ``sparsify_update_global_topk``.
    """
    validate_keep_ratio(keep_ratio)
    if not update:
        raise ValueError("update must contain at least one tensor")

    names = list(update.keys())
    if attack_layer_index < 0 or attack_layer_index >= len(names):
        raise ValueError("attack_layer_index is outside the update parameter mapping")
    tensors = [torch.as_tensor(update[name]).detach().cpu().float().contiguous() for name in names]
    total_elements = sum(int(tensor.numel()) for tensor in tensors)
    if total_elements < 1:
        raise ValueError("update tensors contain no elements")
    kept_elements = int(math.ceil(total_elements * keep_ratio))
    kept_elements = max(1, min(total_elements, kept_elements))

    if kept_elements == total_elements:
        attack_tensor = tensors[attack_layer_index]
        attack_mask = torch.ones(attack_tensor.shape, dtype=torch.bool)
        tau_value = min(float(tensor.abs().min().item()) for tensor in tensors if tensor.numel() > 0)
        observed_values = attack_tensor[attack_mask].detach().cpu().numpy().astype(np.float32, copy=True)
        return {
            "name": names[attack_layer_index],
            "mask": attack_mask,
            "values": observed_values,
            "tau": tau_value,
            "observed_elements": int(attack_mask.sum().item()),
            "total_elements": int(attack_mask.numel()),
            "global_total_elements": int(total_elements),
            "global_kept_elements": int(kept_elements),
            "selection_mode": "identity",
        }

    drop_elements = total_elements - kept_elements
    flat_abs = torch.empty(total_elements, dtype=torch.float32)
    offset = 0
    for tensor in tensors:
        numel = int(tensor.numel())
        torch.abs(tensor.reshape(-1), out=flat_abs[offset : offset + numel])
        offset += numel

    if drop_elements <= kept_elements:
        selection_mode = "drop_smallest"
        threshold_value = float(torch.kthvalue(flat_abs, k=drop_elements).values.item())
        strict_selected = int((flat_abs < threshold_value).sum().item())
        tie_needed = int(drop_elements - strict_selected)
    else:
        selection_mode = "keep_largest"
        kth_smallest = total_elements - kept_elements + 1
        threshold_value = float(torch.kthvalue(flat_abs, k=kth_smallest).values.item())
        strict_selected = int((flat_abs > threshold_value).sum().item())
        tie_needed = int(kept_elements - strict_selected)
    del flat_abs

    remaining_ties = tie_needed
    selected_min_value: float | None = None
    attack_mask: torch.Tensor | None = None
    for index, tensor in enumerate(tensors):
        abs_tensor = tensor.abs()
        tie_count = int((abs_tensor == threshold_value).sum().item())
        if selection_mode == "drop_smallest":
            drop_ties = min(int(remaining_ties), tie_count)
            if drop_ties < tie_count:
                selected_min_value = (
                    float(threshold_value)
                    if selected_min_value is None
                    else min(selected_min_value, float(threshold_value))
                )
            if index == attack_layer_index:
                mask = abs_tensor > threshold_value
                if tie_count:
                    tie_indices = torch.nonzero(abs_tensor.reshape(-1) == threshold_value, as_tuple=False).reshape(-1)
                    if drop_ties < tie_count:
                        mask.reshape(-1)[tie_indices[drop_ties:]] = True
                attack_mask = mask
            remaining_ties -= drop_ties
        else:
            keep_ties = min(int(remaining_ties), tie_count)
            if keep_ties > 0:
                selected_min_value = (
                    float(threshold_value)
                    if selected_min_value is None
                    else min(selected_min_value, float(threshold_value))
                )
            if index == attack_layer_index:
                mask = abs_tensor > threshold_value
                if keep_ties > 0:
                    tie_indices = torch.nonzero(abs_tensor.reshape(-1) == threshold_value, as_tuple=False).reshape(-1)
                    mask.reshape(-1)[tie_indices[:keep_ties]] = True
                attack_mask = mask
            remaining_ties -= keep_ties
        strict_selected_mask = abs_tensor > threshold_value
        if bool(strict_selected_mask.any().item()):
            strict_min = float(abs_tensor[strict_selected_mask].min().item())
            selected_min_value = strict_min if selected_min_value is None else min(selected_min_value, strict_min)
        del strict_selected_mask
        del abs_tensor

    if attack_mask is None:
        raise RuntimeError("failed to construct attack-layer Top-k observation mask")
    if remaining_ties != 0:
        raise RuntimeError(f"Global Top-k tie accounting ended with {remaining_ties} remaining ties")
    if selected_min_value is None:
        raise RuntimeError("Global Top-k selection produced no selected values")

    attack_tensor = tensors[attack_layer_index]
    observed_values = attack_tensor[attack_mask].detach().cpu().numpy().astype(np.float32, copy=True)
    return {
        "name": names[attack_layer_index],
        "mask": attack_mask,
        "values": observed_values,
        "tau": float(selected_min_value),
        "observed_elements": int(attack_mask.sum().item()),
        "total_elements": int(attack_mask.numel()),
        "global_total_elements": int(total_elements),
        "global_kept_elements": int(kept_elements),
        "selection_mode": selection_mode,
    }


def sparsify_update_global_topk(
    update: "OrderedDict[str, torch.Tensor]",
    *,
    keep_ratio: float,
    pack_masks: bool = True,
    return_masks: bool = False,
) -> (
    tuple["OrderedDict[str, torch.Tensor]", list[dict[str, Any]], dict[str, Any]]
    | tuple["OrderedDict[str, torch.Tensor]", list[dict[str, Any]], dict[str, Any], list[torch.Tensor]]
):
    """Return dense zero-filled tensors plus packed masks and statistics."""
    validate_keep_ratio(keep_ratio)
    if not update:
        raise ValueError("update must contain at least one tensor")

    names = list(update.keys())
    tensors = [torch.as_tensor(update[name]).detach().cpu().float().contiguous() for name in names]
    total_elements = sum(int(tensor.numel()) for tensor in tensors)
    if total_elements < 1:
        raise ValueError("update tensors contain no elements")
    kept_elements = int(math.ceil(total_elements * keep_ratio))
    kept_elements = max(1, min(total_elements, kept_elements))

    original_sq = torch.tensor(0.0)
    for tensor in tensors:
        original_sq = original_sq + tensor.square().sum()

    if kept_elements == total_elements:
        selection_mode = "identity"
        threshold_value = None
        tie_needed = 0
    else:
        drop_elements = total_elements - kept_elements
        flat_abs = torch.empty(total_elements, dtype=torch.float32)
        offset = 0
        for tensor in tensors:
            numel = int(tensor.numel())
            torch.abs(tensor.reshape(-1), out=flat_abs[offset : offset + numel])
            offset += numel

        if drop_elements <= kept_elements:
            # For high keep ratios, selecting the dropped complement avoids
            # allocating a giant top-k kept index vector.
            selection_mode = "drop_smallest"
            threshold_value = float(torch.kthvalue(flat_abs, k=drop_elements).values.item())
            strict_selected = int((flat_abs < threshold_value).sum().item())
            tie_needed = int(drop_elements - strict_selected)
        else:
            selection_mode = "keep_largest"
            kth_smallest = total_elements - kept_elements + 1
            threshold_value = float(torch.kthvalue(flat_abs, k=kth_smallest).values.item())
            strict_selected = int((flat_abs > threshold_value).sum().item())
            tie_needed = int(kept_elements - strict_selected)
        del flat_abs

    sparse_update: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    packed_masks: list[dict[str, Any]] = []
    raw_masks: list[torch.Tensor] = []
    per_tensor: list[dict[str, Any]] = []
    remaining_ties = tie_needed
    sparse_sq = torch.tensor(0.0)
    mask_kept_total = 0
    topk_threshold_abs: float | None = None
    for name, tensor in zip(names, tensors):
        numel = int(tensor.numel())
        if selection_mode == "identity":
            mask = torch.ones(tensor.shape, dtype=torch.bool)
        else:
            abs_tensor = tensor.abs()
            if selection_mode == "drop_smallest":
                mask = abs_tensor > threshold_value
                tie_mask = abs_tensor == threshold_value
                tie_indices = torch.nonzero(tie_mask.reshape(-1), as_tuple=False).reshape(-1)
                drop_ties = min(int(remaining_ties), int(tie_indices.numel()))
                if drop_ties < int(tie_indices.numel()):
                    mask.reshape(-1)[tie_indices[drop_ties:]] = True
                remaining_ties -= drop_ties
            else:
                mask = abs_tensor > threshold_value
                tie_mask = abs_tensor == threshold_value
                tie_indices = torch.nonzero(tie_mask.reshape(-1), as_tuple=False).reshape(-1)
                keep_ties = min(int(remaining_ties), int(tie_indices.numel()))
                if keep_ties > 0:
                    mask.reshape(-1)[tie_indices[:keep_ties]] = True
                remaining_ties -= keep_ties
            del abs_tensor

        sparse_tensor = tensor.clone()
        sparse_tensor.masked_fill_(~mask, 0.0)
        sparse_update[name] = sparse_tensor
        sparse_sq = sparse_sq + sparse_tensor.square().sum()
        if pack_masks:
            packed_masks.append({"name": name, **pack_bool_mask(mask)})
        if return_masks:
            raw_masks.append(mask)
        original_nonzero = int((tensor != 0).sum().item())
        stored_nonzero = int((sparse_tensor != 0).sum().item())
        mask_kept = int(mask.sum().item())
        mask_kept_total += mask_kept
        if mask_kept > 0:
            tensor_selected_min = float(tensor.abs()[mask].min().item())
            if topk_threshold_abs is None:
                topk_threshold_abs = tensor_selected_min
            else:
                topk_threshold_abs = min(topk_threshold_abs, tensor_selected_min)
        per_tensor.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "total_elements": numel,
                "mask_kept_elements": mask_kept,
                "stored_nonzero_elements": stored_nonzero,
                "original_nonzero_elements": original_nonzero,
                "mask_fraction": mask_kept / numel if numel else 0.0,
                "stored_nonzero_fraction": stored_nonzero / numel if numel else 0.0,
                "selected_original_zero_elements": max(0, mask_kept - stored_nonzero),
            }
        )

    if mask_kept_total != kept_elements:
        raise RuntimeError(
            "Global Top-k mask construction kept "
            f"{mask_kept_total} elements, expected {kept_elements}."
        )

    dropped_sq = (original_sq - sparse_sq).clamp_min(0.0)
    original_norm = torch.sqrt(original_sq)
    sparse_norm = torch.sqrt(sparse_sq)
    if float(original_norm) <= 0.0:
        relative_l2 = 0.0
        cosine = 1.0
        energy_retained = 1.0
    else:
        relative_l2 = float((torch.sqrt(dropped_sq) / original_norm.clamp_min(1e-12)).cpu())
        cosine = float((sparse_norm / original_norm.clamp_min(1e-12)).cpu())
        energy_retained = float((sparse_sq / original_sq.clamp_min(1e-12)).cpu())

    stored_nonzero_total = sum(int(row["stored_nonzero_elements"]) for row in per_tensor)
    stats = {
        "sparsification_policy": "global_topk",
        "keep_ratio": float(keep_ratio),
        "observed_zero_policy": OBSERVED_ZERO_POLICY,
        "total_elements": int(total_elements),
        "mask_kept_elements": int(kept_elements),
        "stored_nonzero_elements": stored_nonzero_total,
        "sparse_mask_fraction": float(kept_elements / total_elements),
        "sparse_nonzero_fraction": float(stored_nonzero_total / total_elements),
        "sparse_relative_l2": relative_l2,
        "sparse_cosine_similarity_to_fp32": cosine,
        "sparse_retained_l2_norm_ratio": cosine,
        "sparse_energy_retained": energy_retained,
        "topk_threshold_abs": topk_threshold_abs,
        "missing_value_abs_upper_bound": topk_threshold_abs,
        "per_tensor": per_tensor,
    }
    if return_masks:
        return sparse_update, packed_masks, stats, raw_masks
    return sparse_update, packed_masks, stats


def identity_global_topk_stats(update: "OrderedDict[str, torch.Tensor]") -> dict[str, Any]:
    """Stats for keep_ratio=1.0 without materializing a full all-true mask."""
    if not update:
        raise ValueError("update must contain at least one tensor")

    per_tensor = []
    total_elements = 0
    stored_nonzero_total = 0
    original_sq = torch.tensor(0.0)
    topk_threshold_abs: float | None = None
    for name, value in update.items():
        tensor = torch.as_tensor(value).detach().cpu().float().contiguous()
        numel = int(tensor.numel())
        original_nonzero = int((tensor != 0).sum().item())
        total_elements += numel
        stored_nonzero_total += original_nonzero
        original_sq = original_sq + tensor.square().sum()
        if numel > 0:
            tensor_min = float(tensor.abs().min().item())
            topk_threshold_abs = tensor_min if topk_threshold_abs is None else min(topk_threshold_abs, tensor_min)
        per_tensor.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "total_elements": numel,
                "mask_kept_elements": numel,
                "stored_nonzero_elements": original_nonzero,
                "original_nonzero_elements": original_nonzero,
                "mask_fraction": 1.0,
                "stored_nonzero_fraction": original_nonzero / numel if numel else 0.0,
                "selected_original_zero_elements": max(0, numel - original_nonzero),
            }
        )
    if total_elements < 1:
        raise ValueError("update tensors contain no elements")

    return {
        "sparsification_policy": "global_topk",
        "keep_ratio": 1.0,
        "observed_zero_policy": OBSERVED_ZERO_POLICY,
        "total_elements": int(total_elements),
        "mask_kept_elements": int(total_elements),
        "stored_nonzero_elements": int(stored_nonzero_total),
        "sparse_mask_fraction": 1.0,
        "sparse_nonzero_fraction": float(stored_nonzero_total / total_elements),
        "sparse_relative_l2": 0.0,
        "sparse_cosine_similarity_to_fp32": 1.0,
        "sparse_retained_l2_norm_ratio": 1.0,
        "sparse_energy_retained": 1.0 if float(original_sq) > 0.0 else 0.0,
        "topk_threshold_abs": topk_threshold_abs,
        "missing_value_abs_upper_bound": None,
        "per_tensor": per_tensor,
    }


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarize_client_sparse_stats(
    client_stats: list[dict[str, Any]], *, attack_layer_index: int
) -> dict[str, Any]:
    attack_rows = [row["per_tensor"][attack_layer_index] for row in client_stats]
    return {
        "num_clients": len(client_stats),
        "global_mask_fraction": _summary([float(row["sparse_mask_fraction"]) for row in client_stats]),
        "global_nonzero_fraction": _summary([float(row["sparse_nonzero_fraction"]) for row in client_stats]),
        "global_relative_l2": _summary([float(row["sparse_relative_l2"]) for row in client_stats]),
        "global_retained_l2_norm_ratio": _summary(
            [float(row["sparse_retained_l2_norm_ratio"]) for row in client_stats]
        ),
        "global_cosine_similarity_to_client_fp32": _summary(
            [float(row["sparse_cosine_similarity_to_fp32"]) for row in client_stats]
        ),
        "attack_layer_mask_fraction": _summary([float(row["mask_fraction"]) for row in attack_rows]),
        "attack_layer_nonzero_fraction": _summary([float(row["stored_nonzero_fraction"]) for row in attack_rows]),
    }


def _pair_stats_from_scalars(
    *,
    candidate_sq: torch.Tensor,
    reference_sq: torch.Tensor,
    diff_sq: torch.Tensor,
    dot: torch.Tensor,
    nonzero: int,
    total: int,
) -> dict[str, float]:
    candidate_norm = torch.sqrt(candidate_sq)
    reference_norm = torch.sqrt(reference_sq)
    diff_norm = torch.sqrt(diff_sq)
    if float(reference_norm) <= 0.0:
        relative_l2 = 0.0 if float(diff_norm) <= 0.0 else float("inf")
        retained_l2 = 1.0 if float(candidate_norm) <= 0.0 else float("inf")
    else:
        relative_l2 = float((diff_norm / reference_norm.clamp_min(1e-12)).cpu())
        retained_l2 = float((candidate_norm / reference_norm.clamp_min(1e-12)).cpu())
    if float(candidate_norm) <= 0.0 and float(reference_norm) <= 0.0:
        cosine = 1.0
    elif float(candidate_norm) <= 0.0 or float(reference_norm) <= 0.0:
        cosine = 0.0
    else:
        cosine = float(
            dot.div(candidate_norm.clamp_min(1e-12) * reference_norm.clamp_min(1e-12)).cpu()
        )
    return {
        "nonzero_fraction": float(nonzero / max(1, total)),
        "relative_l2_to_fp32": relative_l2,
        "retained_l2_norm_ratio": retained_l2,
        "cosine_similarity_to_fp32": cosine,
    }


def _pair_stats(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate = candidate.detach().cpu().float().reshape(-1)
    reference = reference.detach().cpu().float().reshape(-1)
    if candidate.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {tuple(candidate.shape)} != {tuple(reference.shape)}")
    diff = candidate - reference
    return _pair_stats_from_scalars(
        candidate_sq=candidate.square().sum(),
        reference_sq=reference.square().sum(),
        diff_sq=diff.square().sum(),
        dot=torch.dot(candidate, reference),
        nonzero=int((candidate != 0).sum().item()),
        total=int(candidate.numel()),
    )


def _pair_stats_many(
    candidates: list[torch.Tensor],
    references: list[torch.Tensor],
) -> dict[str, float]:
    if len(candidates) != len(references):
        raise ValueError("candidate/reference tensor counts do not match")
    candidate_sq = torch.tensor(0.0)
    reference_sq = torch.tensor(0.0)
    diff_sq = torch.tensor(0.0)
    dot = torch.tensor(0.0)
    nonzero = 0
    total = 0
    for candidate_value, reference_value in zip(candidates, references):
        candidate = candidate_value.detach().cpu().float()
        reference = reference_value.detach().cpu().float()
        if tuple(candidate.shape) != tuple(reference.shape):
            raise ValueError(f"Shape mismatch: {tuple(candidate.shape)} != {tuple(reference.shape)}")
        diff = candidate - reference
        candidate_sq = candidate_sq + candidate.square().sum()
        reference_sq = reference_sq + reference.square().sum()
        diff_sq = diff_sq + diff.square().sum()
        dot = dot + (candidate * reference).sum()
        nonzero += int((candidate != 0).sum().item())
        total += int(candidate.numel())
    return _pair_stats_from_scalars(
        candidate_sq=candidate_sq,
        reference_sq=reference_sq,
        diff_sq=diff_sq,
        dot=dot,
        nonzero=nonzero,
        total=total,
    )


def compare_update_to_reference(
    candidate: "OrderedDict[str, torch.Tensor]",
    reference: "OrderedDict[str, torch.Tensor]",
    *,
    attack_layer_index: int,
) -> dict[str, Any]:
    candidate_names = list(candidate)
    reference_names = list(reference)
    if candidate_names != reference_names:
        raise ValueError("candidate/reference parameter names do not match")
    attack_name = candidate_names[attack_layer_index]
    global_stats = _pair_stats_many(list(candidate.values()), list(reference.values()))
    attack_stats = _pair_stats(candidate[attack_name], reference[attack_name])
    return {
        "global": global_stats,
        "attack_layer": {"name": attack_name, **attack_stats},
    }


def attack_layer_spectrum_stats(tensor: torch.Tensor) -> dict[str, Any]:
    """Match the author's row-centered covariance used before whitening."""
    matrix = torch.as_tensor(tensor).detach().float()
    if matrix.ndim == 1:
        matrix = matrix.view(1, -1)
    elif matrix.ndim > 2:
        matrix = matrix.view(matrix.shape[0], -1)
    rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix = matrix.to(device)
    if cols > 1:
        centered = matrix - matrix.mean(dim=-1, keepdim=True)
    else:
        centered = matrix - matrix
    denom = max(cols - 1, 1)
    cov = torch.matmul(centered, centered.T) / denom
    eigvals = torch.linalg.eigvalsh(cov).float().clamp_min(0.0).flip(0).detach().cpu()
    singular_values = torch.sqrt(eigvals * denom).float()
    max_sv = float(singular_values.max().item()) if singular_values.numel() else 0.0
    eps = torch.finfo(torch.float32).eps
    rank_tol = float(max(rows, cols) * eps * max_sv)
    positive = singular_values[singular_values > rank_tol]
    numerical_rank = int(positive.numel())
    if numerical_rank > 0:
        min_positive = float(positive.min().item())
        condition_number = float(max_sv / max(min_positive, 1e-30))
    else:
        condition_number = None
    total_sv = singular_values.sum()
    if float(total_sv) > 0.0:
        probs = singular_values / total_sv.clamp_min(1e-30)
        effective_rank = float(torch.exp(-(probs * torch.log(probs.clamp_min(1e-30))).sum()).item())
    else:
        effective_rank = 0.0

    eig_list = [float(value) for value in eigvals.tolist()]
    sv_list = [float(value) for value in singular_values.tolist()]
    return {
        "row_zero_centered": True,
        "matrix_shape": [rows, cols],
        "covariance_shape": [rows, rows],
        "rank_tolerance": rank_tol,
        "numerical_rank": numerical_rank,
        "effective_rank": effective_rank,
        "condition_number": condition_number,
        "singular_values_descending": sv_list,
        "covariance_eigenvalues_descending": eig_list,
        "singular_values_summary": _summary(sv_list),
        "covariance_eigenvalues_summary": _summary(eig_list),
        "top_singular_values": sv_list[:32],
        "bottom_singular_values": sv_list[-32:],
        "top_covariance_eigenvalues": eig_list[:32],
        "bottom_covariance_eigenvalues": eig_list[-32:],
    }


def derive_sparse_update_pickle(
    *,
    fp32_update_file: Path,
    output_file: Path,
    keep_ratio: float,
    sparse_policy: str = "global_topk",
    author_source_hash: str | None = None,
) -> Path:
    """Derive a dense zero-filled sparse observation from a shared FP32 artifact."""
    require_global_topk(sparse_policy)
    validate_keep_ratio(keep_ratio)
    fp32 = _read_pickle(fp32_update_file)
    if fp32.get("metadata", {}).get("quant_bits") != 32:
        raise ValueError("Sparse artifacts must be derived from an FP32 collector artifact")

    source_hash = _sha256_file(fp32_update_file)
    manifest = {
        "schema_version": 1,
        "method": "global_topk_sparse_from_shared_fp32",
        "source_fp32_path": str(fp32_update_file.resolve()),
        "source_fp32_hash": source_hash,
        "sparse_policy": sparse_policy,
        "keep_ratio": float(keep_ratio),
        "observed_zero_policy": OBSERVED_ZERO_POLICY,
        "author_source_hash": author_source_hash,
    }
    manifest_file = output_file.parent / "manifest.json"
    if output_file.exists():
        _assert_matching_manifest(manifest_file, manifest)
        return output_file

    result = dict(fp32)
    result["metadata"] = copy.deepcopy(fp32["metadata"])
    result["grad"] = []
    result["quant_codes"] = None
    result["quant_stats"] = []
    result["sparse_masks"] = []
    result["sparse_stats"] = []

    names = list(result["param_names"])
    for trial_update in fp32["grad"]:
        update = OrderedDict((name, torch.as_tensor(value).float()) for name, value in zip(names, trial_update))
        sparse_update, packed_masks, stats = sparsify_update_global_topk(update, keep_ratio=keep_ratio)
        result["grad"].append([value.cpu().numpy().astype(np.float32, copy=False) for value in sparse_update.values()])
        result["sparse_masks"].append(packed_masks)
        result["sparse_stats"].append(stats)

    stat_rows = result["sparse_stats"]
    mean_nonzero_fraction = float(np.mean([row["sparse_nonzero_fraction"] for row in stat_rows]))
    mean_relative_l2 = float(np.mean([row["sparse_relative_l2"] for row in stat_rows]))
    mean_cosine = float(np.mean([row["sparse_cosine_similarity_to_fp32"] for row in stat_rows]))
    mean_mask_fraction = float(np.mean([row["sparse_mask_fraction"] for row in stat_rows]))
    result["metadata"].update(
        {
            "sparsified_observation": True,
            "sparsification_policy": sparse_policy,
            "keep_ratio": float(keep_ratio),
            "observed_zero_policy": OBSERVED_ZERO_POLICY,
            "sparse_mask_fraction": mean_mask_fraction,
            "sparse_nonzero_fraction": mean_nonzero_fraction,
            "sparse_relative_l2": mean_relative_l2,
            "sparse_cosine_similarity_to_fp32": mean_cosine,
            "source_fp32_update_file": str(fp32_update_file.resolve()),
            "source_fp32_update_hash": source_hash,
            "update_tensor_hash": _hash_tensor_sequence(result["grad"]),
            "sparse_mask_hash": _sha256_bytes(pickle.dumps(result["sparse_masks"], protocol=pickle.HIGHEST_PROTOCOL)),
            "sparse_stats_hash": _sha256_bytes(
                json.dumps(result["sparse_stats"], sort_keys=True, default=str).encode("utf-8")
            ),
        }
    )

    _write_pickle(output_file, result)
    _write_json(manifest_file, manifest)
    return output_file
