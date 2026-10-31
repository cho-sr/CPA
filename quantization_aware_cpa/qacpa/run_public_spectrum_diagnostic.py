"""CPU-only spectrum diagnostic of an existing public deterministic packet.

This command accepts a public observation, its registered plan and a new output
directory. It neither constructs a model nor accepts images, labels, checkpoint
or CPA output. M is formed from public integer symbols, not FP32 decoded values:
row_center(ideal_decode(q, r) / (2*r/15)) == row_center(q), for r > 0.
Only feature blocks are converted to float64; the full M is never materialized.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import psutil
import scipy
from scipy import linalg
from threadpoolctl import threadpool_info, threadpool_limits
import torch


EXPECTED_MANIFEST = "946f848bd8df76b8c276c2d7babafc66c823b1162f664a818e47e09560671364"
EXPECTED_FLOOR = 4.0987196067969e-6
PARTITIONS = ("positive_all", "at_or_above_floor", "below_floor")
_PLAN_FIXED = {
    "format": "public_deterministic_fc1_mp_diagnostic_v1",
    "scope": "public_packet_spectrum_only", "expected_manifest_sha256": EXPECTED_MANIFEST,
    "layer": "classifier.0.weight", "expected_shape": [4096, 25088],
    "device": "cpu", "dtype": "float64", "threads": 8, "feature_chunk": 1024,
    "expected_floor": EXPECTED_FLOOR, "partitions": list(PARTITIONS),
    "covariance_denominator": "d-1", "eigensolver": "scipy.linalg.eigh driver evr",
    "exclude_top": 16, "min_bulk": 32, "primary_fit_range": [.2, .8],
    "sensitivity_fit_ranges": [[.1, .9], [.3, .7]], "min_positive_fit_fraction": .9,
    "fit_error_metrics": ["relative_residual_rms", "relative_residual_p50", "relative_residual_p90"],
    "subspace_components": 16,
    "subspace_metrics": ["orthogonality_error", "eigen_residual", "pairwise_principal_cosines", "projector_overlap"],
    "memory_sampling_seconds": .05,
    "forbidden": ["private_truth_inputs", "new_observations", "training", "CPA", "FIA", "H_metric",
                  "image_reconstruction", "label_changes", "three_arm_comparison"],
}
_PLAN_TEXT = {
    "matrix_rule", "normalization_note", "floor_rule", "zero_radius_rule", "beta_rule",
    "covariance_method", "quantile_rank_rule", "fit_scale_rule", "mp_quantile_method",
    "negative_eigenvalue_tolerance", "degenerate_scale_rule", "no_fit_fallback",
    "subspace_rule", "reporting", "interpretation",
}


def validate_plan(plan: dict) -> None:
    """Reject unregistered/private input fields before reading any observation."""
    if not isinstance(plan, dict) or set(plan) != set(_PLAN_FIXED) | _PLAN_TEXT:
        raise ValueError("Plan must contain exactly the registered public diagnostic fields")
    for key, value in _PLAN_FIXED.items():
        if json.dumps(plan[key], sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ValueError(f"Unregistered diagnostic setting: {key}")
    if any(not isinstance(plan[key], str) or not plan[key] for key in _PLAN_TEXT):
        raise ValueError("Plan descriptions must be nonempty strings")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def validate_output(output: Path, observation: Path, plan_path: Path) -> Path:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    sources = (Path(observation).resolve(), Path(plan_path).resolve(), Path(__file__).resolve().parent)
    for source in sources:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Output must not overlap observation, plan or source code")
    return output


def _validate_codes_rows(codes: torch.Tensor, rows: np.ndarray, block_cols: int) -> torch.Tensor:
    if (not isinstance(codes, torch.Tensor) or codes.device.type != "cpu"
            or codes.dtype != torch.uint8 or codes.ndim != 2 or codes.shape[1] < 2):
        raise ValueError("codes must be a CPU uint8 matrix with at least two columns")
    if type(block_cols) is not int or block_cols <= 0:
        raise ValueError("block_cols must be a positive integer")
    rows = np.asarray(rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("row indices must be a one-dimensional integer array")
    if rows.size and (rows.min() < 0 or rows.max() >= codes.shape[0] or len(np.unique(rows)) != len(rows)):
        raise ValueError("row indices must be distinct and in range")
    return torch.from_numpy(rows.astype(np.int64, copy=True))


def blocked_centered_covariance(codes: torch.Tensor, row_indices: np.ndarray,
                                block_cols: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Return C=M M.T/(d-1) and row means, without allocating all of M."""
    rows = _validate_codes_rows(codes, row_indices, block_cols)
    m, d = rows.numel(), codes.shape[1]
    covariance = torch.zeros((m, m), dtype=torch.float64)
    sums = torch.zeros(m, dtype=torch.float64)
    for start in range(0, d, block_cols):
        block = codes[rows, start:start + block_cols].to(dtype=torch.float64)
        if bool((block > 15).any()):
            raise ValueError("codes exceed the public uint4 range")
        sums.add_(block.sum(dim=1))
        covariance.addmm_(block, block.T)
    means = sums.div_(d)
    covariance.addr_(means, means, alpha=-d).div_(d - 1)
    return covariance.numpy(), means.numpy()


def partition_rows(radius: torch.Tensor, scale_floor: float) -> dict[str, np.ndarray]:
    if (not isinstance(radius, torch.Tensor) or radius.device.type != "cpu" or radius.ndim != 1
            or not radius.is_floating_point() or not bool(torch.isfinite(radius).all())
            or bool(torch.signbit(radius).any())):
        raise ValueError("radius must be a finite nonnegative CPU vector")
    if not math.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("scale floor must be positive and finite")
    step = radius.double() * (2.0 / 15.0)
    masks = (step > 0, step >= scale_floor, (step > 0) & (step < scale_floor))
    return {key: torch.nonzero(mask, as_tuple=False).flatten().numpy()
            for key, mask in zip(PARTITIONS, masks)}


def _quantiles(values: np.ndarray) -> dict:
    probabilities = [0., .1, .25, .5, .75, .9, 1.]
    return {"probabilities": probabilities, "method": "linear_interpolation",
            "values": np.quantile(values, probabilities).tolist() if len(values) else None}


def analyze_partition(codes: torch.Tensor, full_covariance: np.ndarray,
                      positive_row_indices: np.ndarray, selected_row_indices: np.ndarray,
                      *, top_k: int = 16, block_cols: int = 1024) -> dict:
    """Use a principal covariance submatrix and return all raw eigenvalues + V.

    V has d rows in the original feature coordinates. The eigensolver is full
    scipy.linalg.eigh(evr); only up to top_k numerically positive left vectors
    are retained. Numerical checks concern this decomposition, not MP validity.
    """
    rows = _validate_codes_rows(codes, selected_row_indices, block_cols)
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be positive")
    positive = np.asarray(positive_row_indices)
    if full_covariance.shape != (len(positive), len(positive)):
        raise ValueError("covariance shape must match positive rows")
    position = {int(row): index for index, row in enumerate(positive)}
    if len(position) != len(positive) or any(int(row) not in position for row in selected_row_indices):
        raise ValueError("selected rows must be a subset of distinct positive rows")
    indices = np.array([position[int(row)] for row in selected_row_indices], dtype=np.int64)
    covariance = full_covariance if np.array_equal(indices, np.arange(len(positive))) else full_covariance[np.ix_(indices, indices)]
    m, d = len(indices), codes.shape[1]
    if not m:
        return {"eigenvalues_desc": np.empty(0), "right_vectors": np.empty((d, 0)),
                "left_vectors": np.empty((0, 0)), "row_indices": np.asarray(selected_row_indices),
                "checks": {"status": "empty_partition", "components": 0,
                           "negative_tolerance": 0., "orthogonality_error": None,
                           "eigen_residual": None, "right_relation_residual": None}}
    eigenvalues, eigenvectors = linalg.eigh(covariance, driver="evr", check_finite=True,
                                           overwrite_a=False)
    eigenvalues = eigenvalues[::-1].copy()
    tau = float(100 * np.finfo(np.float64).eps * max(m, d) * np.max(np.abs(eigenvalues)))
    k = min(top_k, int(np.count_nonzero(eigenvalues > tau)))
    left = eigenvectors[:, -k:][:, ::-1].copy() if k else np.empty((m, 0))
    del eigenvectors
    right = np.empty((d, k), dtype=np.float64)
    relation = torch.zeros((m, k), dtype=torch.float64)
    scales = np.sqrt((d - 1) * eigenvalues[:k])
    # Integer row sums are exactly representable at the registered dimensions.
    means = torch.zeros(m, dtype=torch.float64)
    for start in range(0, d, block_cols):
        means.add_(codes[rows, start:start + block_cols].double().sum(dim=1))
    means.div_(d)
    left_tensor = torch.from_numpy(left)
    scale_tensor = torch.from_numpy(scales)
    if k:
        for start in range(0, d, block_cols):
            block = codes[rows, start:start + block_cols].double().sub_(means[:, None])
            v_block = (block.T @ left_tensor).div_(scale_tensor[None, :])
            right[start:start + block.shape[1]] = v_block.numpy()
            relation.addmm_(block, v_block)
        orthogonality = float(np.linalg.norm(right.T @ right - np.eye(k), ord="fro"))
        eigen_residual = float(np.linalg.norm(covariance @ left - left * eigenvalues[:k], ord="fro")
                               / max(np.linalg.norm(covariance, ord="fro") * np.linalg.norm(left, ord="fro"), np.finfo(float).tiny))
        expected_relation = left * scales
        relation_residual = float(np.linalg.norm(relation.numpy() - expected_relation, ord="fro")
                                  / max(np.linalg.norm(expected_relation, ord="fro"), np.finfo(float).tiny))
        orthogonality_tolerance = float(math.sqrt(k) * tau / eigenvalues[k - 1])
        residual_tolerance = float(100 * np.finfo(float).eps * max(m, d))
        checks = {"status": "ok", "components": k, "negative_tolerance": tau,
                  "orthogonality_error": orthogonality, "eigen_residual": eigen_residual,
                  "right_relation_residual": relation_residual,
                  "orthogonality_tolerance": orthogonality_tolerance,
                  "residual_tolerance": residual_tolerance,
                  "numerical_checks_passed": bool(orthogonality <= orthogonality_tolerance
                      and eigen_residual <= residual_tolerance and relation_residual <= orthogonality_tolerance)}
        if not checks["numerical_checks_passed"]:
            checks["status"] = "numerical_check_failed"
    else:
        checks = {"status": "no_eigenvalues_above_numerical_tolerance", "components": 0,
                  "negative_tolerance": tau, "orthogonality_error": None,
                  "eigen_residual": None, "right_relation_residual": None}
    return {"eigenvalues_desc": eigenvalues, "right_vectors": right, "left_vectors": left,
            "row_indices": np.asarray(selected_row_indices), "checks": checks}


def compare_subspaces(left: np.ndarray, right: np.ndarray) -> dict:
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("right-vector matrices must share the original feature dimension")
    k_left, k_right = left.shape[1], right.shape[1]
    if not min(k_left, k_right):
        return {"status": "empty_subspace", "components_left": k_left, "components_right": k_right,
                "pairwise_principal_cosines": [], "principal_angles_degrees": [], "projector_overlap": None}
    # Orthonormalize to measure subspaces rather than amplification from finite
    # precision V normalization; raw V orthogonality is reported independently.
    q_left = linalg.qr(left, mode="economic")[0]
    q_right = linalg.qr(right, mode="economic")[0]
    cosines = np.clip(linalg.svdvals(q_left.T @ q_right), 0., 1.)
    return {"status": "ok", "components_left": k_left, "components_right": k_right,
            "pairwise_principal_cosines": cosines.tolist(),
            "principal_angles_degrees": np.degrees(np.arccos(cosines)).tolist(),
            "projector_overlap": float(np.square(cosines).sum() / min(k_left, k_right)),
            "projector_overlap_definition": "trace(P_left P_right)/min(k_left,k_right); QR bases of saved V"}


class _MemoryMonitor:
    def __init__(self, interval: float):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.samples = 0
        self.peak_rss = 0
        self.windows_peak_wset = None
        self.baseline_rss = self.process.memory_info().rss
        self.initial_available = psutil.virtual_memory().available
        self.minimum_available = self.initial_available
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        info = self.process.memory_info()
        self.samples += 1
        self.peak_rss = max(self.peak_rss, info.rss)
        peak = getattr(info, "peak_wset", None)
        if peak is not None:
            self.windows_peak_wset = max(self.windows_peak_wset or 0, peak)
        self.minimum_available = min(self.minimum_available, psutil.virtual_memory().available)

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self._sample()

    def start(self):
        self._sample()
        self.thread.start()

    def finish(self) -> dict:
        self.stop_event.set()
        self.thread.join()
        self._sample()
        return {"sampling_seconds": self.interval, "samples": self.samples,
                "baseline_rss_bytes": self.baseline_rss, "peak_sampled_rss_bytes": self.peak_rss,
                "windows_process_lifetime_peak_wset_bytes": self.windows_peak_wset,
                "initial_system_available_bytes": self.initial_available,
                "minimum_sampled_system_available_bytes": self.minimum_available,
                "note": "Sampled process RSS can miss shorter peaks; Windows peak_wset covers process lifetime including imports"}


@contextmanager
def _phase(name: str, timings: dict):
    started = time.monotonic()
    try:
        yield
    finally:
        timings[name] = time.monotonic() - started
        print(json.dumps({"phase": name, "seconds": timings[name]}), flush=True)


def run_diagnostic(*, observation: Path, plan_path: Path, output: Path) -> dict:
    observation, plan_path = Path(observation).resolve(), Path(plan_path).resolve()
    output = validate_output(output, observation, plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    if _digest(observation / "manifest.json") != EXPECTED_MANIFEST:
        raise ValueError("Public observation manifest differs from the registered input")
    # This reader is used only with None: it validates all six packets, without
    # evaluating its optional checkpoint/model/sender routines.
    from .deterministic_hybrid_setup import read_observation
    from .mp_spectrum import fit_mp_spectrum

    source_names = ("run_public_spectrum_diagnostic.py", "mp_spectrum.py",
                    "deterministic_hybrid_setup.py", "deterministic_observation.py",
                    "communication_codec.py", "full_update_observation.py", "full_update_pipeline.py")
    source_paths = {name: Path(__file__).parent / name for name in source_names}
    tracked = {"plan": plan_path, "manifest": observation / "manifest.json",
               **{f"source/{name}": path for name, path in source_paths.items()},
               **{f"observation/packet_{index:02d}.bin": observation / f"packet_{index:02d}.bin" for index in range(6)}}
    before = {name: _digest(path) for name, path in tracked.items()}
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "registered_plan.json", plan)
    _write_json(output / "provenance.json", {"plan_path": str(plan_path), "observation_path": str(observation),
                                           "hashes_before": before, "public_inputs_only": True})
    started = time.monotonic()
    timings = {}
    monitor = _MemoryMonitor(plan["memory_sampling_seconds"])
    old_threads = torch.get_num_threads()
    monitor.start()
    memory = None
    try:
        torch.set_num_threads(plan["threads"])
        with threadpool_limits(limits=plan["threads"]):
            environment = {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                           "torch": torch.__version__, "psutil": psutil.__version__,
                           "torch_cpu_threads": torch.get_num_threads(), "thread_pools": threadpool_info(),
                           "device": "cpu", "dtype": "float64"}
            with _phase("validate_and_read_all_six_packets", timings):
                manifest, observations = read_observation(observation, None)
            with _phase("public_floor_and_partitions", timings):
                positive_steps = torch.cat([obs.radius[obs.radius > 0].double() * (2.0 / 15.0)
                                            for obs in observations.values()])
                if positive_steps.numel() == 0:
                    raise ValueError("Public packets have no positive radius; the registered floor is unavailable")
                scale_floor = float(positive_steps.median())
                if scale_floor != plan["expected_floor"]:
                    raise ValueError("Public lower-median scale floor differs from registration")
                fc1 = observations[plan["layer"]]
                if list(fc1.codes.shape) != plan["expected_shape"]:
                    raise ValueError("FC1 codes shape differs from registration")
                partitions = partition_rows(fc1.radius, scale_floor)
                radius = fc1.radius.double().numpy()
                row_statistics = {"all_fc_packet_rows": sum(obs.radius.numel() for obs in observations.values()),
                                  "all_fc_positive_rows": positive_steps.numel(),
                                  "fc1_rows": len(radius), "fc1_zero_radius_rows": int(np.count_nonzero(radius == 0)),
                                  "public_scale_floor": scale_floor, "fc1_radius_quantiles": _quantiles(radius),
                                  "all_fc_positive_step_quantiles": _quantiles(positive_steps.numpy())}
                codes = fc1.codes
                del observations, fc1, positive_steps
            with _phase("positive_all_blocked_covariance", timings):
                covariance, means = blocked_centered_covariance(codes, partitions["positive_all"], plan["feature_chunk"])
            partition_reports, vectors = {}, {}
            for name, indices in partitions.items():
                with _phase(f"decompose_and_check_{name}", timings):
                    result = analyze_partition(codes, covariance, partitions["positive_all"], indices,
                                               top_k=plan["subspace_components"], block_cols=plan["feature_chunk"])
                with _phase(f"fit_and_save_{name}", timings):
                    eigenvalues = result["eigenvalues_desc"]
                    fit = fit_mp_spectrum(eigenvalues, len(indices), codes.shape[1],
                                          exclude_top=plan["exclude_top"], min_bulk=plan["min_bulk"])
                    vectors[name] = result["right_vectors"]
                    np.savez(output / f"{name}.npz", eigenvalues_desc=eigenvalues,
                             right_vectors=result["right_vectors"], left_vectors=result["left_vectors"],
                             row_indices=indices)
                    np.savetxt(output / f"{name}_eigenvalues.csv",
                               np.column_stack((np.arange(1, len(eigenvalues) + 1), eigenvalues)),
                               delimiter=",", header="descending_rank,eigenvalue", comments="", fmt=["%d", "%.18e"])
                    partition_reports[name] = {"m": len(indices), "d": codes.shape[1],
                                               "radius_quantiles": _quantiles(radius[indices]),
                                               "step_quantiles": _quantiles(radius[indices] * (2.0 / 15.0)),
                                               "decomposition": result["checks"], "mp_fit": fit,
                                               "arrays_file": f"{name}.npz", "spectrum_csv": f"{name}_eigenvalues.csv"}
                    del result
            with _phase("pairwise_right_subspaces", timings):
                pairs = {}
                for i, a in enumerate(PARTITIONS):
                    for b in PARTITIONS[i + 1:]:
                        invalid = [name for name in (a, b)
                                   if partition_reports[name]["decomposition"]["status"] == "numerical_check_failed"]
                        pairs[f"{a}__vs__{b}"] = (
                            {"status": "skipped_numerical_check_failed", "invalid_partitions": invalid,
                             "pairwise_principal_cosines": [], "principal_angles_degrees": [],
                             "projector_overlap": None}
                            if invalid else compare_subspaces(vectors[a], vectors[b]))
            with _phase("verify_input_and_source_hashes", timings):
                after = {name: _digest(path) for name, path in tracked.items()}
                if before != after:
                    raise RuntimeError("An input or diagnostic source changed during execution")
            memory = monitor.finish()
            report = {"status": "completed", "format": plan["format"], "plan": plan,
                      "input_manifest_sha256": before["manifest"], "hashes_before": before,
                      "hashes_after": after, "input_and_source_hashes_unchanged": True,
                      "communication": manifest["communication"], "row_statistics": row_statistics,
                      "partitions": partition_reports, "subspace_comparisons": pairs,
                      "runtime_seconds": time.monotonic() - started, "phase_seconds": timings,
                      "runtime_scope": "After input/source prehash and output registration through input/source posthash and memory-monitor finish; includes packet validation, Gram, decompositions, fitting and NPZ/CSV serialization; excludes imports, report.json serialization and plotting",
                      "environment": environment,
                      "process_memory": memory, "interpretation": plan["interpretation"],
                      "matrix_interpretation": plan["matrix_rule"] + "; " + plan["normalization_note"],
                      "scope_completed": "Public spectrum diagnostic only; no checkpoint loading, model forward/training, CPA, FIA, image or label optimization; strict reader may construct meta shapes"}
            _write_json(output / "report.json", report)
            return report
    except BaseException as error:
        if memory is None:
            memory = monitor.finish()
        _write_json(output / "failure.json", {"status": "failed", "error_type": type(error).__name__,
                                              "error": str(error), "phase_seconds": timings,
                                              "runtime_seconds": time.monotonic() - started,
                                              "process_memory": memory})
        raise
    finally:
        if memory is None:
            monitor.finish()
        torch.set_num_threads(old_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_diagnostic(observation=args.observation, plan_path=args.plan, output=args.output)


if __name__ == "__main__":
    main()
