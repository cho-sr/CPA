#!/usr/bin/env python3
"""Run Naive and randomized FastICA from the same client q,s artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from alignment import blind_global_sign_fix, blind_medoid_align_and_average
from assignment import linear_sum_assignment
from quant_bins import assert_within_bins, naive_dequantize, randomized_dequantize


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "int4_client_qs_randomized_ica"
    / "imagenet"
    / "vgg16"
    / "clients3_rounds20_spc8_seed42"
)
ATTACK_LAYER_NAME = "classifier.0.weight"
ATTACK_LAYER_SHAPE = (4096, 25088)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare deterministic naive dequantization and randomized exact-bin "
            "dequantization from the same final-round client q,s artifact."
        )
    )
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--R", type=int, default=64)
    parser.add_argument(
        "--smoke_R",
        type=int,
        default=None,
        help="Override R for smoke tests, e.g. 2 or 8. Output is written under randomized_R*_smoke.",
    )
    parser.add_argument("--noise_seed", type=int, default=1234)
    parser.add_argument("--fastica_random_state", type=int, default=42)
    parser.add_argument("--n_components", type=int, default=24)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument(
        "--whiten",
        type=str,
        default="unit-variance",
        choices=["unit-variance", "arbitrary-variance"],
    )
    parser.add_argument("--fun", type=str, default="logcosh", choices=["logcosh", "exp", "cube"])
    parser.add_argument("--fresh_start", action="store_true")
    parser.add_argument(
        "--skip_bin_assert",
        action="store_true",
        help="Skip post-sampling bin checks for faster full R runs.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def safe_rmtree(path: Path, *, allowed_root: Path) -> None:
    root = allowed_root.resolve()
    target = path.resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"refusing to delete outside {root}: {target}")
    if target.exists():
        shutil.rmtree(target)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_pickle(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def import_fastica():
    try:
        from sklearn.decomposition import FastICA
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for FastICA. Install scikit-learn in this environment."
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
    if attack_matrix.ndim != 2:
        raise ValueError(f"attack matrix must be 2D, got shape={attack_matrix.shape}")
    if tuple(attack_matrix.shape) != ATTACK_LAYER_SHAPE:
        raise ValueError(f"expected {ATTACK_LAYER_SHAPE}, got {tuple(attack_matrix.shape)}")
    if n_components > min(attack_matrix.shape):
        raise ValueError("n_components exceeds attack matrix dimensions")

    FastICA = import_fastica()
    ica = FastICA(
        n_components=n_components,
        algorithm="parallel",
        whiten=whiten,
        fun=fun,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    sources = ica.fit_transform(attack_matrix.T).T.astype(np.float32, copy=False)
    return sources, {
        "n_iter": int(getattr(ica, "n_iter_", -1)),
        "converged": bool(getattr(ica, "n_iter_", max_iter) < max_iter),
        "mixing_shape": list(getattr(ica, "mixing_", np.empty((0, 0))).shape),
        "components_shape": list(getattr(ica, "components_", np.empty((0, 0))).shape),
    }


def make_trial_seed_plan(R: int, noise_seed: int, fastica_random_state: int) -> List[Dict[str, Any]]:
    return [
        {
            "trial": trial,
            "dequantization_seed": int(noise_seed + 1_000_003 * trial),
            "fastica_random_state": int(fastica_random_state),
            "trial_varying_randomness": "dequantization_noise_only",
        }
        for trial in range(R)
    ]


def find_tensor_meta(meta: Dict[str, Any], client: Dict[str, Any], layer_name: str) -> Dict[str, Any]:
    matches = [row for row in client["tensors"] if row["name"] == layer_name]
    if len(matches) != 1:
        raise ValueError(
            f"client {client['client_id']} has {len(matches)} tensors named {layer_name}; "
            f"store_scope may not include the attack layer"
        )
    if meta["attack_layer"]["name"] != layer_name:
        raise ValueError(f"artifact attack layer is {meta['attack_layer']['name']}, not {layer_name}")
    return matches[0]


def aggregate_attack_matrix(
    *,
    codes_npz: Any,
    quant_meta: Dict[str, Any],
    layer_name: str,
    mode: str,
    rng: Optional[np.random.Generator],
    dtype: np.dtype,
    assert_bins: bool,
) -> np.ndarray:
    clients = quant_meta["clients"]
    aggregate: Optional[np.ndarray] = None
    for client in clients:
        tensor_meta = find_tensor_meta(quant_meta, client, layer_name)
        codes = codes_npz[tensor_meta["key"]]
        scale = float(tensor_meta["scale"])
        max_abs = float(tensor_meta["max_abs"])
        if mode == "naive":
            dequant = naive_dequantize(codes, scale)
        elif mode == "randomized":
            if rng is None:
                raise ValueError("randomized mode requires an rng")
            dequant = randomized_dequantize(codes, scale, max_abs, rng)
            if assert_bins:
                assert_within_bins(dequant, codes, scale, max_abs)
        else:
            raise ValueError(f"unknown mode: {mode}")

        weighted = dequant.astype(dtype, copy=False) * float(client["aggregation_weight"])
        aggregate = weighted if aggregate is None else aggregate + weighted

    if aggregate is None:
        raise ValueError("no clients found in q,s artifact")
    return aggregate.astype(dtype, copy=False)


def row_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def oracle_evaluate(recovered: np.ndarray, true_z: np.ndarray) -> Dict[str, Any]:
    recovered_f = np.asarray(recovered, dtype=np.float32)
    truth_f = np.asarray(true_z, dtype=np.float32)
    if recovered_f.shape != truth_f.shape:
        raise ValueError(f"oracle eval shape mismatch: {recovered_f.shape} vs {truth_f.shape}")

    similarity = row_normalize(truth_f.astype(np.float64)) @ row_normalize(
        recovered_f.astype(np.float64)
    ).T
    true_rows, recovered_cols = linear_sum_assignment(-np.abs(similarity))
    ordered_cols = np.zeros(len(true_rows), dtype=np.int64)
    signed_cosines = np.zeros(len(true_rows), dtype=np.float64)
    abs_cosines = np.zeros(len(true_rows), dtype=np.float64)
    signs = np.ones(len(true_rows), dtype=np.float64)
    for true_row, recovered_col in zip(true_rows, recovered_cols):
        value = float(similarity[true_row, recovered_col])
        ordered_cols[true_row] = int(recovered_col)
        signed_cosines[true_row] = value
        abs_cosines[true_row] = abs(value)
        signs[true_row] = 1.0 if value >= 0 else -1.0

    return {
        "policy": "oracle_hungarian_abs_cosine_eval_only",
        "oracle_inputs_used_only_for_final_eval": True,
        "mean_abs_cosine": float(np.mean(abs_cosines)),
        "median_abs_cosine": float(np.median(abs_cosines)),
        "min_abs_cosine": float(np.min(abs_cosines)),
        "max_abs_cosine": float(np.max(abs_cosines)),
        "mean_signed_cosine": float(np.mean(signed_cosines)),
        "per_sample_abs_cosine": abs_cosines.tolist(),
        "per_sample_signed_cosine": signed_cosines.tolist(),
        "recovered_order_for_true_index": ordered_cols.tolist(),
        "signs_for_eval_only": signs.tolist(),
    }


def load_true_z_for_final_eval(context: Dict[str, Any]) -> np.ndarray:
    """Evaluation-only access to true embeddings."""
    return np.asarray(context["z"], dtype=np.float32)


def branch_manifest(
    *,
    branch: str,
    q_s_artifact_hash: str,
    q_s_meta_hash: str,
    fastica_random_state: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "branch": branch,
        "source": "same final-round client q,s artifact",
        "q_s_codes_sha256": q_s_artifact_hash,
        "q_s_meta_sha256": q_s_meta_hash,
        "fastica_random_state": fastica_random_state,
        "true_z_policy": "not used for recovery, alignment, or final embedding generation; final eval only",
    }
    if extra:
        manifest.update(extra)
    return manifest


def update_root_manifest(run_dir: Path, updates: Dict[str, Any]) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    stages = dict(manifest.get("stages", {}))
    stages.update(updates)
    manifest["stages"] = stages
    write_json(manifest, manifest_path)


def main() -> None:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    client_qs_dir = run_dir / "client_qs"
    codes_path = client_qs_dir / "final_round_client_codes.npz"
    meta_path = client_qs_dir / "final_round_client_quant_meta.json"
    context_path = client_qs_dir / "final_round_context.pkl"
    if not codes_path.is_file():
        raise FileNotFoundError(codes_path)
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    if not context_path.is_file():
        raise FileNotFoundError(context_path)

    quant_meta = read_json(meta_path)
    context = read_pickle(context_path)
    n_context_samples = int(np.asarray(context["x"]).shape[0])
    if args.n_components != n_context_samples:
        raise ValueError(
            f"n_components={args.n_components} must match stored sample count={n_context_samples}"
        )

    dtype = np.float32 if args.dtype == "float32" else np.float64
    effective_R = int(args.smoke_R if args.smoke_R is not None else args.R)
    if effective_R < 1:
        raise ValueError("R must be >= 1")
    smoke_suffix = "_smoke" if args.smoke_R is not None else ""
    naive_dir = run_dir / "naive"
    randomized_dir = run_dir / f"randomized_R{effective_R}{smoke_suffix}"
    eval_dir = run_dir / "eval"
    if args.fresh_start:
        for target in (naive_dir, randomized_dir, eval_dir):
            safe_rmtree(target, allowed_root=run_dir)
    naive_dir.mkdir(parents=True, exist_ok=True)
    randomized_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    q_s_codes_hash = sha256_file(codes_path)
    q_s_meta_hash = sha256_file(meta_path)
    seed_plan = make_trial_seed_plan(effective_R, args.noise_seed, args.fastica_random_state)

    with np.load(codes_path) as codes_npz:
        print("Running naive branch from the shared client q,s artifact", flush=True)
        naive_attack = aggregate_attack_matrix(
            codes_npz=codes_npz,
            quant_meta=quant_meta,
            layer_name=ATTACK_LAYER_NAME,
            mode="naive",
            rng=None,
            dtype=dtype,
            assert_bins=False,
        )
        naive_raw, naive_info = run_fastica(
            naive_attack,
            n_components=args.n_components,
            max_iter=args.max_iter,
            tol=args.tol,
            random_state=args.fastica_random_state,
            whiten=args.whiten,
            fun=args.fun,
        )
        naive_embeddings, naive_sign_stats = blind_global_sign_fix(naive_raw)
        np.save(naive_dir / "naive_raw_sources.npy", naive_raw)
        np.save(naive_dir / "naive_embeddings.npy", naive_embeddings)
        write_pickle(
            {
                "raw_sources": naive_raw,
                "embeddings": naive_embeddings,
                "fastica": naive_info,
                "global_sign": naive_sign_stats,
            },
            naive_dir / "naive_embeddings.pkl",
        )
        naive_eval = oracle_evaluate(naive_embeddings, load_true_z_for_final_eval(context))
        write_json(naive_eval, eval_dir / "naive_oracle_eval.json")
        write_json(
            branch_manifest(
                branch="naive",
                q_s_artifact_hash=q_s_codes_hash,
                q_s_meta_hash=q_s_meta_hash,
                fastica_random_state=args.fastica_random_state,
                extra={
                    "dequantization": "deterministic q*s per client, then FedAvg",
                    "fastica": naive_info,
                    "global_sign": naive_sign_stats,
                },
            ),
            naive_dir / "manifest.json",
        )

        trial_sources: List[np.ndarray] = []
        trial_rows: List[Dict[str, Any]] = []
        for row in seed_plan:
            trial = int(row["trial"])
            print(
                f"Randomized trial {trial + 1:03d}/{effective_R:03d} | "
                f"dequant_seed={row['dequantization_seed']} | "
                f"fastica_seed={args.fastica_random_state}",
                flush=True,
            )
            rng = np.random.default_rng(int(row["dequantization_seed"]))
            attack_matrix = aggregate_attack_matrix(
                codes_npz=codes_npz,
                quant_meta=quant_meta,
                layer_name=ATTACK_LAYER_NAME,
                mode="randomized",
                rng=rng,
                dtype=dtype,
                assert_bins=not args.skip_bin_assert,
            )
            sources, info = run_fastica(
                attack_matrix,
                n_components=args.n_components,
                max_iter=args.max_iter,
                tol=args.tol,
                random_state=args.fastica_random_state,
                whiten=args.whiten,
                fun=args.fun,
            )
            trial_sources.append(sources)
            trial_rows.append({**row, **info})

    final_embeddings, aligned_stack, alignment_metrics = blind_medoid_align_and_average(trial_sources)
    raw_stack = np.stack(trial_sources, axis=0).astype(np.float32, copy=False)
    np.save(randomized_dir / "trial_raw_sources.npy", raw_stack)
    np.save(randomized_dir / "trial_aligned_sources.npy", aligned_stack)
    np.save(randomized_dir / "final_embeddings.npy", final_embeddings)
    write_pickle(
        {
            "trial_raw_sources": raw_stack,
            "trial_aligned_sources": aligned_stack,
            "final_embeddings": final_embeddings,
            "alignment_metrics": alignment_metrics,
            "trial_metrics": trial_rows,
        },
        randomized_dir / "randomized_fastica_embeddings.pkl",
    )
    write_csv(trial_rows, randomized_dir / "fastica_trial_metrics.csv")
    write_json(alignment_metrics, randomized_dir / "alignment_metrics.json")
    randomized_eval = oracle_evaluate(final_embeddings, load_true_z_for_final_eval(context))
    write_json(randomized_eval, eval_dir / "randomized_oracle_eval.json")
    write_json(
        branch_manifest(
            branch="randomized",
            q_s_artifact_hash=q_s_codes_hash,
            q_s_meta_hash=q_s_meta_hash,
            fastica_random_state=args.fastica_random_state,
            extra={
                "R": effective_R,
                "smoke": bool(args.smoke_R is not None),
                "noise_seed": args.noise_seed,
                "seed_plan": seed_plan,
                "dequantization": "uniform inside exact bounded quantization bins per client, then FedAvg",
                "alignment": "blind medoid permutation/sign/least-squares-scale alignment",
                "global_sign": "blind positive-energy rule after medoid average",
            },
        ),
        randomized_dir / "manifest.json",
    )

    update_root_manifest(
        run_dir,
        {
            f"fastica_comparison_R{effective_R}{smoke_suffix}": {
                "naive_dir": str(naive_dir.resolve()),
                "randomized_dir": str(randomized_dir.resolve()),
                "eval_dir": str(eval_dir.resolve()),
                "q_s_codes_sha256": q_s_codes_hash,
                "q_s_meta_sha256": q_s_meta_hash,
                "fastica_random_state": args.fastica_random_state,
                "noise_seed": args.noise_seed,
                "R": effective_R,
                "smoke": bool(args.smoke_R is not None),
            }
        },
    )

    print(f"Naive embeddings: {naive_dir / 'naive_embeddings.npy'}", flush=True)
    print(f"Randomized final embeddings: {randomized_dir / 'final_embeddings.npy'}", flush=True)
    print(
        "Oracle eval | naive mean_abs_cosine={:.6f} | randomized mean_abs_cosine={:.6f}".format(
            naive_eval["mean_abs_cosine"],
            randomized_eval["mean_abs_cosine"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
