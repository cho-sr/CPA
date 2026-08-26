#!/usr/bin/env python3
"""Run FastICA on an int8-dequantized FedAvg aggregate update."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "int 8 fedavg"
    / "imagenet"
    / "vgg16"
    / "clients3_rounds20_spc8_seed42"
)
DEFAULT_INPUT = DEFAULT_RUN_DIR / "round_020_int8_dequant_aggregate_cpa.pickle"
FIRST_FC_WEIGHT_NAME = "classifier.0.weight"
FIRST_FC_WEIGHT_SHAPE = (4096, 25088)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover VGG16 first-FC input embeddings by applying sklearn FastICA "
            "to the stored int8-dequantized FedAvg aggregate update."
        )
    )
    parser.add_argument("--input_pickle", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--batch_index", type=int, default=0)
    parser.add_argument(
        "--attack_layer_index",
        type=int,
        default=None,
        help=(
            "Optional override for the VGG16 first FC weight tensor index. "
            "The selected tensor must be classifier.0.weight with shape (4096, 25088)."
        ),
    )
    parser.add_argument("--n_components", type=int, default=None)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument(
        "--whiten",
        type=str,
        default="unit-variance",
        choices=["unit-variance", "arbitrary-variance"],
    )
    parser.add_argument(
        "--fun",
        type=str,
        default="logcosh",
        choices=["logcosh", "exp", "cube"],
    )
    parser.add_argument(
        "--no_abs",
        action="store_true",
        help="Do not take abs() of FastICA sources before saving blind embeddings.",
    )
    parser.add_argument(
        "--fresh_start",
        action="store_true",
        help="Delete output_dir before writing new FastICA outputs.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_pickle(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_manifest(input_pickle: Path, explicit_manifest: Optional[Path]) -> Dict[str, Any]:
    manifest_path = resolve_path(explicit_manifest) if explicit_manifest else input_pickle.parent / "manifest.json"
    if not manifest_path.is_file():
        return {"manifest_path": str(manifest_path), "status": "missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def validate_dequantized_int8_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    checks = {
        "signal_is_int8_dequantized": manifest.get("signal")
        == "int8_fedavg_server_dequantized_aggregate_update",
        "quant_bits_is_8": manifest.get("quant_bits") == 8,
        "rounding_is_nearest": manifest.get("rounding") == "nearest",
        "server_input_is_dequantized": manifest.get("server_aggregation_input")
        == "dequantized_client_updates",
        "scale_scope_is_client_tensor": manifest.get("scale_scope")
        == "per_client_per_parameter_tensor",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "interpretation": (
            "This is a naive dequantized int8 FedAvg aggregate update"
            if all(checks.values())
            else "Manifest does not fully prove naive dequantized int8 FedAvg semantics"
        ),
    }


def get_batch_array(data: Dict[str, Any], key: str, batch_index: int) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing key {key!r} in input pickle")
    values = data[key]
    if not isinstance(values, list) or batch_index >= len(values):
        raise IndexError(f"Key {key!r} does not contain batch index {batch_index}")
    return np.asarray(values[batch_index])


def infer_first_fc_weight_index(
    manifest: Dict[str, Any],
    explicit_index: Optional[int],
) -> int:
    if explicit_index is not None:
        return int(explicit_index)
    param_names = manifest.get("param_names")
    if isinstance(param_names, list) and FIRST_FC_WEIGHT_NAME in param_names:
        return int(param_names.index(FIRST_FC_WEIGHT_NAME))
    if "attack_layer_index" in manifest:
        return int(manifest["attack_layer_index"])
    return 26


def load_attack_matrix(
    data: Dict[str, Any],
    *,
    batch_index: int,
    attack_layer_index: int,
    dtype: np.dtype,
) -> np.ndarray:
    grads = data.get("grad")
    if not isinstance(grads, list) or batch_index >= len(grads):
        raise KeyError(f"Input pickle has no grad batch {batch_index}")
    grad_batch = grads[batch_index]
    if attack_layer_index >= len(grad_batch):
        raise IndexError(
            f"attack_layer_index={attack_layer_index} exceeds grad tensors={len(grad_batch)}"
        )
    attack_matrix = np.asarray(grad_batch[attack_layer_index], dtype=dtype)
    if attack_matrix.ndim != 2:
        raise ValueError(f"Attack layer must be 2D, got shape={attack_matrix.shape}")
    return attack_matrix


def get_param_name(manifest: Dict[str, Any], layer_index: int) -> Optional[str]:
    param_names = manifest.get("param_names")
    if isinstance(param_names, list) and layer_index < len(param_names):
        return str(param_names[layer_index])
    return None


def validate_first_fc_weight_input(
    *,
    manifest: Dict[str, Any],
    layer_index: int,
    attack_matrix: np.ndarray,
) -> str:
    layer_name = get_param_name(manifest, layer_index)
    if layer_name is not None and layer_name != FIRST_FC_WEIGHT_NAME:
        raise ValueError(
            f"FastICA input must be {FIRST_FC_WEIGHT_NAME}, "
            f"but layer index {layer_index} is {layer_name}"
        )
    if tuple(attack_matrix.shape) != FIRST_FC_WEIGHT_SHAPE:
        raise ValueError(
            f"FastICA input must be VGG16 first FC weight update with "
            f"shape={FIRST_FC_WEIGHT_SHAPE}, got shape={tuple(attack_matrix.shape)}"
        )
    return layer_name or FIRST_FC_WEIGHT_NAME


def row_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return row_normalize(a) @ row_normalize(b).T


def hungarian_match(
    recovered: np.ndarray,
    true_z: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for oracle matching. Install it or omit true z from input."
        ) from exc

    sims = cosine_matrix(recovered.astype(np.float64), true_z.astype(np.float64))
    rec_ind, true_ind = linear_sum_assignment(-sims)
    order = rec_ind[np.argsort(true_ind)]
    sorted_true = np.sort(true_ind)
    if not np.array_equal(sorted_true, np.arange(len(sorted_true))):
        raise RuntimeError("Hungarian matching did not cover all true components")
    matched = recovered[order]
    matched_sims = np.diag(cosine_matrix(matched.astype(np.float64), true_z.astype(np.float64)))
    metrics = {
        "mean_cosine": float(np.mean(matched_sims)),
        "min_cosine": float(np.min(matched_sims)),
        "max_cosine": float(np.max(matched_sims)),
        "median_cosine": float(np.median(matched_sims)),
        "recovered_order_for_true_index": [int(x) for x in order.tolist()],
        "matched_cosines": [float(x) for x in matched_sims.tolist()],
    }
    return matched, metrics


def import_fastica():
    try:
        from sklearn.decomposition import FastICA
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for FastICA. Run: "
            r"C:\miniconda\envs\DLG\python.exe -m pip install scikit-learn"
        ) from exc
    return FastICA


def run_fastica(
    attack_matrix: np.ndarray,
    *,
    n_components: int,
    max_iter: int,
    tol: float,
    random_state: int,
    whiten: str,
    fun: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    FastICA = import_fastica()
    observations = attack_matrix.T
    ica = FastICA(
        n_components=n_components,
        algorithm="parallel",
        whiten=whiten,
        fun=fun,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    sources = ica.fit_transform(observations).T
    info = {
        "n_iter": int(getattr(ica, "n_iter_", -1)),
        "converged": bool(getattr(ica, "n_iter_", max_iter) < max_iter),
        "mixing_shape": list(getattr(ica, "mixing_", np.empty((0, 0))).shape),
        "components_shape": list(getattr(ica, "components_", np.empty((0, 0))).shape),
    }
    return sources.astype(np.float32), info


def main() -> None:
    args = parse_args()
    input_pickle = resolve_path(args.input_pickle)
    if not input_pickle.is_file():
        raise FileNotFoundError(input_pickle)
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else input_pickle.parent / "fastica_dequantized_aggregate"
    )
    if args.fresh_start and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(input_pickle, args.manifest)
    dequant_check = validate_dequantized_int8_manifest(manifest)
    if not dequant_check["passed"]:
        raise RuntimeError(dequant_check["interpretation"])

    data = read_pickle(input_pickle)
    attack_layer_index = infer_first_fc_weight_index(manifest, args.attack_layer_index)
    dtype = np.float32 if args.dtype == "float32" else np.float64
    attack_matrix = load_attack_matrix(
        data,
        batch_index=args.batch_index,
        attack_layer_index=attack_layer_index,
        dtype=dtype,
    )
    attack_layer_name = validate_first_fc_weight_input(
        manifest=manifest,
        layer_index=attack_layer_index,
        attack_matrix=attack_matrix,
    )
    x = get_batch_array(data, "x", args.batch_index)
    y = get_batch_array(data, "y", args.batch_index)
    true_z = get_batch_array(data, "z", args.batch_index) if data.get("z") else None

    n_components = args.n_components or int(x.shape[0])
    if n_components < 1:
        raise ValueError("n_components must be positive")
    if n_components > min(attack_matrix.shape):
        raise ValueError(
            f"n_components={n_components} exceeds min attack matrix dim={min(attack_matrix.shape)}"
        )

    raw_sources, fastica_info = run_fastica(
        attack_matrix,
        n_components=n_components,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.random_state,
        whiten=args.whiten,
        fun=args.fun,
    )
    blind_embeddings = raw_sources if args.no_abs else np.abs(raw_sources)

    oracle_embeddings = None
    oracle_metrics = None
    if true_z is not None and len(true_z) >= n_components:
        oracle_embeddings, oracle_metrics = hungarian_match(
            blind_embeddings,
            np.asarray(true_z[:n_components], dtype=np.float32),
        )

    output = {
        "blind_embeddings": blind_embeddings,
        "raw_sources": raw_sources,
        "oracle_reordered_embeddings": oracle_embeddings,
        "true_z": true_z,
        "x": x,
        "y": y,
        "metadata": {
            "input_pickle": str(input_pickle),
            "manifest": manifest,
            "dequant_check": dequant_check,
            "batch_index": args.batch_index,
            "attack_layer_index": attack_layer_index,
            "attack_layer_name": attack_layer_name,
            "fastica_input": (
                "Only grad[batch_index][attack_layer_index] is used. "
                "For this VGG16 run it is classifier.0.weight, the first FC weight update."
            ),
            "attack_matrix_shape": list(attack_matrix.shape),
            "n_components": n_components,
            "embedding_shape": list(blind_embeddings.shape),
            "abs_applied": not args.no_abs,
            "fastica": {
                "max_iter": args.max_iter,
                "tol": args.tol,
                "random_state": args.random_state,
                "whiten": args.whiten,
                "fun": args.fun,
                **fastica_info,
            },
        },
    }
    metrics = {
        "input_pickle": str(input_pickle),
        "output_dir": str(output_dir),
        "dequant_check": dequant_check,
        "attack_layer_index": attack_layer_index,
        "attack_layer_name": attack_layer_name,
        "attack_matrix_shape": list(attack_matrix.shape),
        "n_components": n_components,
        "blind_embedding_shape": list(blind_embeddings.shape),
        "fastica": output["metadata"]["fastica"],
        "oracle_matching": oracle_metrics,
    }

    write_pickle(output, output_dir / "fastica_embeddings.pkl")
    np.save(output_dir / "blind_embeddings.npy", blind_embeddings)
    if oracle_embeddings is not None:
        np.save(output_dir / "oracle_reordered_embeddings.npy", oracle_embeddings)
    write_json(metrics, output_dir / "fastica_metrics.json")

    print(f"FastICA embeddings saved: {output_dir / 'fastica_embeddings.pkl'}")
    print(f"FastICA metrics saved: {output_dir / 'fastica_metrics.json'}")
    if oracle_metrics:
        print(f"oracle mean cosine: {oracle_metrics['mean_cosine']:.6f}")


if __name__ == "__main__":
    main()
