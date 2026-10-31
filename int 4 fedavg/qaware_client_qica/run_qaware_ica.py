#!/usr/bin/env python3
"""Run client-wise bin-constrained QICA from final-round client int4 q/s."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from alignment import blind_centered_align_and_average
from qica_core import (
    ClientWiseBinConstrainedQICA,
    compute_fixed_whitening,
    finite_bin_bounds,
    oracle_embedding_eval,
    requantization_match,
    source_covariance_stats,
    weighted_aggregate,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "int 4 fedavg"
    / "imagenet"
    / "vgg16"
    / "clients3_rounds20_spc8_seed42"
)
ATTACK_LAYER_NAME = "classifier.0.weight"
ABLATIONS = {
    "qoff_neoff": {"use_q_loss": False, "use_ne": False},
    "qoff_neon": {"use_q_loss": False, "use_ne": True},
    "qon_neoff": {"use_q_loss": True, "use_ne": False},
    "qon_neon": {"use_q_loss": True, "use_ne": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Client-wise bin-constrained QICA from final-round client q/s artifacts."
    )
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--qica_ablation", type=str, default="all", choices=["all", *ABLATIONS])
    parser.add_argument("--qica_n_iter", type=int, default=1000)
    parser.add_argument("--qica_n_log", type=int, default=100)
    parser.add_argument("--qica_lr", type=float, default=1e-3)
    parser.add_argument("--qica_q_weight", type=float, default=1.0)
    parser.add_argument("--qica_columns_per_step", type=int, default=8192)
    parser.add_argument("--qica_eps", type=float, default=1e-6)
    parser.add_argument("--qica_trials", type=int, default=1)
    parser.add_argument("--n_components", type=int, default=None)
    parser.add_argument("--whitening_columns", type=int, default=None)
    parser.add_argument("--whitening_eig_floor_rel", type=float, default=1e-5)
    parser.add_argument("--whitening_rank_tol_rel", type=float, default=1e-5)
    parser.add_argument("--qica_train_observation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decor", type=float, default=5.3)
    parser.add_argument("--T", type=float, default=7.7)
    parser.add_argument("--l1", type=float, default=5.0)
    parser.add_argument("--nv", type=float, default=0.13)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--fresh_start", action="store_true")
    parser.add_argument("--run_fia", action="store_true")
    parser.add_argument("--n_sample_fi", type=int, default=-1)
    parser.add_argument("--n_iter_fi", type=int, default=25000)
    parser.add_argument("--n_log_fi", type=int, default=500)
    parser.add_argument("--fia_batch_size", type=int, default=4)
    parser.add_argument("--lr_fi", type=float, default=0.1)
    parser.add_argument("--tv", type=float, default=0.1)
    parser.add_argument("--fi", type=float, default=1.0)
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


def write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def print_progress(row: Dict[str, Any], *, n_iter: int) -> None:
    grad = row.get("x_client_grad_norm")
    grad_text = "None" if grad is None else f"{float(grad):.4g}"
    print(
        "[{ablation} trial={trial} iter={iter}/{n_iter}] "
        "loss={loss:.6g} q={loss_q:.3g} ne={loss_ne:.3g} "
        "bin={bin_satisfaction:.4f} req={requantization_match:.4f} "
        "preproj={pre_projection_violation:.3g} grad={grad}".format(
            ablation=row.get("ablation"),
            trial=row.get("trial"),
            iter=row.get("iter"),
            n_iter=n_iter,
            loss=float(row.get("loss", 0.0)),
            loss_q=float(row.get("loss_q", 0.0)),
            loss_ne=float(row.get("loss_ne", 0.0)),
            bin_satisfaction=float(row.get("bin_satisfaction", 0.0)),
            requantization_match=float(row.get("requantization_match", 0.0)),
            pre_projection_violation=float(row.get("pre_projection_violation", 0.0)),
            grad=grad_text,
        ),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def selected_ablations(value: str) -> Iterable[str]:
    return ABLATIONS.keys() if value == "all" else [value]


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compatible_legacy_client_qs_dir(run_dir: Path) -> Optional[Path]:
    parts = list(run_dir.resolve().parts)
    if "int 4 fedavg" not in parts:
        return None
    idx = parts.index("int 4 fedavg")
    rel_parts = parts[idx + 1 :]
    legacy = REPO_ROOT / "int4_client_qs_randomized_ica" / Path(*rel_parts) / "client_qs"
    if (legacy / "final_round_client_codes.npz").is_file():
        return legacy
    return None


def find_client_qs_dir(run_dir: Path) -> tuple[Path, Dict[str, Any]]:
    local = run_dir / "client_qs"
    if (local / "final_round_client_codes.npz").is_file():
        return local, {"source": "run_dir/client_qs", "client_qs_dir": str(local)}
    legacy = compatible_legacy_client_qs_dir(run_dir)
    if legacy is not None:
        return legacy, {
            "source": "compatible_existing_final_round_client_qs_artifact",
            "client_qs_dir": str(legacy),
            "note": "QICA outputs are written under int 4 fedavg run_dir; q/s source was the matching existing client-q/s artifact.",
        }
    raise FileNotFoundError(
        f"Missing client_qs under {run_dir}. Run collect_final_round_client_qs.py first."
    )


def find_tensor_meta(meta: Dict[str, Any], client: Dict[str, Any], layer_name: str) -> Dict[str, Any]:
    matches = [row for row in client["tensors"] if row["name"] == layer_name]
    if len(matches) != 1:
        raise ValueError(f"client {client.get('client_id')} has {len(matches)} tensors named {layer_name}")
    return matches[0]


def load_attack_layer_client_qs(
    client_qs_dir: Path,
    *,
    device: torch.device,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    codes_path = client_qs_dir / "final_round_client_codes.npz"
    meta_path = client_qs_dir / "final_round_client_quant_meta.json"
    context_path = client_qs_dir / "final_round_context.pkl"
    if not codes_path.is_file() or not meta_path.is_file() or not context_path.is_file():
        raise FileNotFoundError(f"incomplete client_qs artifact: {client_qs_dir}")
    meta = read_json(meta_path)
    context = read_pickle(context_path)
    if meta.get("quant_bits") != 4 or meta.get("rounding") != "nearest":
        raise ValueError("QICA requires nearest int4 client q/s artifacts")
    if meta.get("attack_layer", {}).get("name") != layer_name:
        raise ValueError(f"artifact attack layer is {meta.get('attack_layer')}, expected {layer_name}")

    codes_rows = []
    scales = []
    max_abs_values = []
    weights = []
    with np.load(codes_path) as npz:
        for client in meta["clients"]:
            tensor_meta = find_tensor_meta(meta, client, layer_name)
            codes_rows.append(np.asarray(npz[tensor_meta["key"]], dtype=np.int8))
            scales.append(float(tensor_meta["scale"]))
            max_abs_values.append(float(tensor_meta["max_abs"]))
            weights.append(float(client["aggregation_weight"]))

    codes_np = np.stack(codes_rows, axis=0)
    codes = torch.as_tensor(codes_np, dtype=torch.int8, device=device)
    scales_t = torch.as_tensor(scales, dtype=torch.float32, device=device)
    max_abs_t = torch.as_tensor(max_abs_values, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    weights_t = weights_t / weights_t.sum().clamp_min(torch.finfo(torch.float32).eps)
    return codes, scales_t, max_abs_t, weights_t, meta, context


def full_observation_metrics(
    client_values: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    max_abs: torch.Tensor,
    *,
    chunk_cols: int = 512,
) -> Dict[str, float]:
    total = 0
    satisfied = 0
    matched = 0
    violation_sum = 0.0
    for start in range(0, client_values.shape[-1], chunk_cols):
        end = min(start + chunk_cols, client_values.shape[-1])
        values = client_values[:, :, start:end]
        code_chunk = codes[:, :, start:end]
        lower, upper = finite_bin_bounds(code_chunk, scales, max_abs)
        below = torch.relu(lower - values)
        above = torch.relu(values - upper)
        dist = below + above
        total += int(values.numel())
        satisfied += int((dist <= 1e-8).sum().detach().cpu())
        violation_sum += float(dist.mean().detach().cpu()) * int(values.numel())
        q = torch.round(values / scales.view(-1, 1, 1).clamp_min(torch.finfo(values.dtype).eps)).clamp(-7, 7)
        q = torch.where(scales.view(-1, 1, 1) <= 0, torch.zeros_like(q), q).to(torch.int8)
        matched += int((q == code_chunk).sum().detach().cpu())
    return {
        "bin_satisfaction": satisfied / max(total, 1),
        "requantization_match": matched / max(total, 1),
        "mean_bin_violation": violation_sum / max(total, 1),
    }


def run_one_ablation(
    args: argparse.Namespace,
    *,
    ablation: str,
    output_root: Path,
    codes: torch.Tensor,
    scales: torch.Tensor,
    max_abs: torch.Tensor,
    weights: torch.Tensor,
    meta: Dict[str, Any],
    context: Dict[str, Any],
    source_info: Dict[str, Any],
) -> Path:
    flags = ABLATIONS[ablation]
    if flags["use_q_loss"] and not args.qica_train_observation:
        raise ValueError(f"{ablation} is Q-on and requires qica_train_observation=True")

    ablation_dir = output_root / ablation
    if args.fresh_start:
        safe_rmtree(ablation_dir, allowed_root=output_root)
    ablation_dir.mkdir(parents=True, exist_ok=True)
    log_path = ablation_dir / "losses.jsonl"
    if args.fresh_start and log_path.exists():
        log_path.unlink()

    k_clients, h_rows, d_cols = map(int, codes.shape)
    m_components = int(args.n_components or np.asarray(context["z"]).shape[0])
    whitening_columns = args.whitening_columns
    if whitening_columns is None:
        whitening_columns = args.qica_columns_per_step
    init_client = codes.to(torch.float32) * scales.view(-1, 1, 1)
    init_aggregate = weighted_aggregate(init_client, weights)
    whitening = compute_fixed_whitening(
        init_aggregate,
        n_components=m_components,
        eig_floor_rel=args.whitening_eig_floor_rel,
        rank_tol_rel=args.whitening_rank_tol_rel,
        eps=args.qica_eps,
        column_count=whitening_columns,
        seed=args.seed,
    )

    raw_sources: List[np.ndarray] = []
    trial_rows: List[Dict[str, Any]] = []
    for trial in range(args.qica_trials):
        runner = ClientWiseBinConstrainedQICA(
            codes=codes,
            scales=scales,
            max_abs=max_abs,
            weights=weights,
            whitening=whitening,
            n_components=m_components,
            use_q_loss=flags["use_q_loss"],
            use_ne=flags["use_ne"],
            train_observation=args.qica_train_observation,
            lr=args.qica_lr,
            eps=args.qica_eps,
            decor_weight=args.decor,
            decor_temperature=args.T,
            l1_weight=args.l1,
            nv_weight=args.nv,
            q_weight=args.qica_q_weight,
            seed=args.seed + 1_000_003 * trial,
        )
        last_loss: Dict[str, Any] = {}
        for iter_idx in range(args.qica_n_iter):
            loss_row = runner.step(iter_idx=iter_idx, columns_per_step=args.qica_columns_per_step)
            last_loss = loss_row
            if (iter_idx % args.qica_n_log == 0) or (iter_idx == args.qica_n_iter - 1):
                progress_row = {"ablation": ablation, "trial": trial, "iter": iter_idx, **loss_row}
                write_jsonl(log_path, progress_row)
                print_progress(progress_row, n_iter=args.qica_n_iter)
        source_np = runner.full_sources().detach().cpu().numpy().astype(np.float32, copy=False)
        raw_sources.append(source_np)
        x_final = runner._current_client().detach()
        obs_metrics = full_observation_metrics(x_final, codes, scales, max_abs)
        trial_rows.append({"trial": trial, **last_loss, **obs_metrics})
        torch.save(
            {
                "trial": trial,
                "ablation": ablation,
                "W_hat": runner.W_hat.detach().cpu(),
                "train_observation": runner.train_observation,
                "X_client_hat_saved": False,
                "note": "X_client_hat is omitted to keep checkpoints compact; recovered sources are saved separately.",
            },
            ablation_dir / f"batch_000_trial_{trial:03d}_qica.pt",
        )
        del runner
        if codes.device.type == "cuda":
            torch.cuda.empty_cache()

    final_embeddings, aligned_centered, aligned_mean_restored, alignment_metrics = blind_centered_align_and_average(raw_sources)
    raw_stack = np.stack(raw_sources, axis=0).astype(np.float32, copy=False)
    np.save(ablation_dir / "trial_raw_sources.npy", raw_stack)
    np.save(ablation_dir / "trial_aligned_centered_sources.npy", aligned_centered)
    np.save(ablation_dir / "trial_aligned_mean_restored_sources.npy", aligned_mean_restored)
    np.save(ablation_dir / "final_embeddings.npy", final_embeddings.astype(np.float32, copy=False))
    write_pickle(
        {
            "trial_raw_sources": raw_stack,
            "trial_aligned_centered_sources": aligned_centered,
            "trial_aligned_mean_restored_sources": aligned_mean_restored,
            "final_embeddings": final_embeddings,
            "alignment_metrics": alignment_metrics,
            "trial_metrics": trial_rows,
        },
        ablation_dir / "qica_embeddings.pkl",
    )
    write_json(alignment_metrics, ablation_dir / "alignment_metrics.json")

    qica_eval = {
        "trial_metrics": trial_rows,
        "source_covariance": source_covariance_stats(torch.as_tensor(final_embeddings)),
        "trial_alignment_stability": {
            key: alignment_metrics.get(key)
            for key in ("alignment_mean_abs_cosine", "alignment_min_abs_cosine", "members")
        },
    }
    write_json(qica_eval, ablation_dir / "qica_evaluation.json")
    oracle_eval = oracle_embedding_eval(final_embeddings, np.asarray(context.get("z"), dtype=np.float32))
    write_json(
        {
            "metric_note": "Oracle metrics are evaluation only and are not used for source generation/alignment.",
            "embedding": oracle_eval,
        },
        ablation_dir / "oracle_evaluation.json",
    )
    manifest = {
        "schema_version": 1,
        "method": "client-wise bin-constrained QICA",
        "ablation": ablation,
        "flags": flags,
        "source_info": source_info,
        "q_s_meta_sha256": sha256_file(Path(source_info["client_qs_dir"]) / "final_round_client_quant_meta.json"),
        "tensor_shapes": {
            "K_clients": k_clients,
            "H_attack_layer_rows": h_rows,
            "D_attack_layer_columns": d_cols,
            "M_components": m_components,
            "T_qica_trials": args.qica_trials,
            "X_client_hat": [k_clients, h_rows, d_cols],
            "X_agg_hat": [h_rows, d_cols],
            "whitening": [m_components, h_rows],
            "W_hat": [m_components, m_components],
            "final_source": [m_components, d_cols],
        },
        "quantization": {
            "qmin": -7,
            "qmax": 7,
            "endpoint_bins": "finite max-based bins: q=7 [6.5s, 7s], q=-7 [-7s, -6.5s]",
            "aggregate_requantization": False,
        },
        "whitening": {
            "policy": "fixed top-M low-rank whitening from initial alpha-weighted aggregate",
            "columns_used": int(d_cols if whitening.selected_columns is None else whitening.selected_columns.numel()),
            "eig_floor_rel": args.whitening_eig_floor_rel,
            "rank_tol_rel": args.whitening_rank_tol_rel,
            "numerical_rank": whitening.numerical_rank,
            "min_eigenvalue": whitening.min_eigvalue,
            "max_eigenvalue": whitening.max_eigvalue,
        },
        "settings": jsonable_args(args),
        "alignment": alignment_metrics,
        "true_z_policy": "not used for recovery, alignment, or final embedding generation; final eval only",
    }
    write_json(manifest, ablation_dir / "manifest.json")
    return ablation_dir


def run_fia(args: argparse.Namespace, *, run_dir: Path, source_info: Dict[str, Any], ablation_dir: Path) -> None:
    runner = REPO_ROOT / "int4_client_qs_randomized_ica" / "run_fia_from_embeddings.py"
    if not runner.is_file():
        raise FileNotFoundError(runner)
    fia_run_dir = Path(source_info["client_qs_dir"]).parent
    output_dir = ablation_dir / f"direct_fia_n{args.n_sample_fi}_iter{args.n_iter_fi}"
    cmd = [
        sys.executable,
        str(runner),
        "--run_dir",
        str(fia_run_dir),
        "--embedding_branch",
        "randomized",
        "--embedding_path",
        str(ablation_dir / "final_embeddings.npy"),
        "--output_dir",
        str(output_dir),
        "--n_sample",
        str(args.n_sample_fi),
        "--n_iter_fi",
        str(args.n_iter_fi),
        "--n_log_fi",
        str(args.n_log_fi),
        "--fia_batch_size",
        str(args.fia_batch_size),
        "--lr_fi",
        str(args.lr_fi),
        "--tv",
        str(args.tv),
        "--fi",
        str(args.fi),
        "--device",
        args.device,
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    args = parse_args()
    if args.qica_trials < 1:
        raise ValueError("--qica_trials must be >= 1")
    if args.qica_n_iter < 1:
        raise ValueError("--qica_n_iter must be >= 1")
    run_dir = resolve_path(args.run_dir)
    device = choose_device(args.device)
    client_qs_dir, source_info = find_client_qs_dir(run_dir)
    output_root = run_dir / "qaware_client_qica"
    output_root.mkdir(parents=True, exist_ok=True)

    codes, scales, max_abs, weights, meta, context = load_attack_layer_client_qs(
        client_qs_dir,
        device=device,
        layer_name=ATTACK_LAYER_NAME,
    )
    print(
        "client-wise bin-constrained QICA | "
        f"run_dir={run_dir} | q_s={client_qs_dir} | shape K,H,D={tuple(codes.shape)} | "
        f"device={device}",
        flush=True,
    )
    output_dirs = []
    for ablation in selected_ablations(args.qica_ablation):
        path = run_one_ablation(
            args,
            ablation=ablation,
            output_root=output_root,
            codes=codes,
            scales=scales,
            max_abs=max_abs,
            weights=weights,
            meta=meta,
            context=context,
            source_info=source_info,
        )
        output_dirs.append(path)
        print(f"QICA output: {path}", flush=True)
        if args.run_fia:
            run_fia(args, run_dir=run_dir, source_info=source_info, ablation_dir=path)
    write_json(
        {
            "schema_version": 1,
            "method": "client-wise bin-constrained QICA",
            "run_dir": str(run_dir),
            "source_info": source_info,
            "outputs": [str(path) for path in output_dirs],
        },
        output_root / "manifest.json",
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
